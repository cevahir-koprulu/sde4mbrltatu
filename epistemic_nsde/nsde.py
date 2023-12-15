import jax
import jax.numpy as jnp
import haiku as hk

# # Store the SDE solvers and their corresponding names
from epistemic_nsde.sde_solver import sde_solver_name

# ELementary functions to manipulate dictionaries
from epistemic_nsde.dict_utils import set_values_all_leaves, update_same_struct_dict
from epistemic_nsde.dict_utils import get_penalty_parameters, get_non_negative_params

import copy

def compute_timesteps(params):
    """ Compute the timesteps for the numerical integration of the SDE.
    The timesteps are computed as follows:
        - params is a dictionary that must contains the keys: horizon, stepsize; 
            optional:  num_short_dt, short_step_dt, long_step_dt

        - horizon = total number of timesteps
        - stepsize = size of the timestep dt
        - The first num_short_dt timesteps are of size short_step_dt if given else stepsize
        - The last remaining timesteps are of size long_step_dt if given else stepsize
    
    Args:
        params (dict): The dictionary of parameters

    Returns:
        jnp.array: The array of timesteps
        
    """
    horizon = params['horizon']
    stepsize = params['stepsize']
    num_short_dt = params.get('num_short_dt', horizon)
    assert num_short_dt <= horizon, 'The number of short dt is greater than horizon'
    first_dt  = params.get('first_dt', 0)
    if first_dt > 0:
        first_dt = [first_dt]
    else:
        first_dt = []
    num_long_dt = horizon - num_short_dt - len(first_dt)
    # print(params)
    assert num_long_dt >= 0, 'The number of long dt is negative'
    assert num_short_dt + num_long_dt + len(first_dt) == horizon, 'The number of short and long dt does not match the horizon'
    short_step_dt = params.get('short_step_dt', stepsize)
    long_step_dt = params.get('long_step_dt', stepsize)
    return jnp.array(first_dt + [short_step_dt] * num_short_dt + [long_step_dt] * num_long_dt)


def sampling_strat_under_dataset_with_finer_steps(arr, strategy, rng_den):
    """ Given an array of size (horizon, num_steps2data, ...) and a sampling strategy,
    this function returns an array of size (horizon, ...) where each element is sampled from the dataset
    according to the sampling strategy.

    Args:
        arr (jnp.array): The array of size (horizon, num_steps2data, ...)
        strategy (str): The sampling strategy. Choose from first, mean, median, random
        rng_den (jax.random.PRNGKey): The random key

    Returns:
        jnp.array: The array of size (horizon, ...)
    """
    horizon = arr.shape[0]
    num_steps2data = arr.shape[1]

    if strategy == 'first':
        arr = arr[:,0,...] if len(arr.shape) > 2 else arr[:,0]
    elif strategy == 'mean':
        arr = arr.mean(axis=1)
    elif strategy == 'median':
        arr = jnp.median(arr, axis=1)
    elif strategy == 'random':
        rnd_indx = jax.random.randint(rng_den, shape=(horizon,), minval=0, maxval=num_steps2data)
        arr = arr[jnp.arange(horizon), rnd_indx, ...] if len(arr.shape) > 2 else arr[jnp.arange(horizon), rnd_indx]
    else:
        raise ValueError('Unknown u_sampling_strategy: {}. Choose from first, mean, median, random'.format(strategy))
    return arr


class ControlledSDE(hk.Module):
    """Define an SDE (stochastic differential equation) with observation and state (latent) variables
    and which is controlled via a control input. 

    Typically \dot{x} = f(x, u) dt + sigma(x;u) dW, where the SDE is defined by the drift f and the diffusion sigma.
    The SDE could be in the Ito or Stratonovich sense. 
    The choice of the sde solver will determine if the SDE is an Ito or Stratonovich SDE.

    This class implements several functions for deep NNs modeling and control of dynamical systems with conservative uncertainty estimate.
    By conservative uncertainty estimate, we mean in the sense of distance awareness from the training data.

    The class is designed to be inherited by the user to define the SDE model and the control policy.
    
    The class has the following functionalities:
        (a) Train an SDE to fit data given a priori knowledge on the dynamical systems as inductive bias 
        (b) The learned SDE must be able to provide uncertainty estimates on their predictions
        (c) We provide a way to control the SDE using a gradient-based MPC controller + NN policy + Value function
        (d) Fast sampling from the learned SDE model

    A user should create a class that inherits the properties of this class while redefining the functions below:

        - [Required] compositional_drift : Define the drift term of the SDE model to learn.
            The compositional drift is a function of the state and the control input.
            We write it as compositional_drift(x,u) = f(x,u) =  F(x, u, g_1(x), g_2(x), ..., g_n(x)), 
            where g_i are the functions that define the drift and that are parametrized by NNs in our framework.
            This form on the drift informs the learning with prior knowledge on the dynamical system.
            Such prior knowledge is typically obtained as an ODE from physics or engineering. And with such prior knowledge,
            we sometimes can characterize the uncertainty (or region where the ODE is wrong) in the state space. (see next poiint)

        - [Required] prior_diffusion : Define a prior diffusion on the stochastic distribution (SDE) to learn. 
            This is consdered as a priori knowledge on where in the state space the above physics knowledge could be uncertain and wrong. 
            It characterizes the approximations and limited knowledge that our prior ode model relies on.
            If such a prior is not known, then the prior should be set to a constant noise value.
            Typically, our SDE learning algorithm will learn distributions whose uncertainty is smaller than the prior around regionns of the state space
            where we have data. And increasingly revert back to the prior "as we move away from the data".
        
        - [Optional] prior_constraints : Define the constraints that may be known on the unknown terms of the compositional drift.
            These constraints, if present, are going to be enforced during the learning process.
            And they are tipically in the form of a function h(g_1, g_2, ..., g_n) <= 0 for x in a given region of the state space stored in params_model['constraints']

        - [Optional] init_encoder : 
            Initialize the encoder/decoder functions -> Provide a way to go from observation to state and its log probability
            This is useful when we have a latent state that is not directly observed and we want to learn it from the observations.
        
    """
    def __init__(self, params, name=None):
        """Initialize the parameterized sde prior and posterior dynamics

        Args:
            params (dict, optional): The set of params used when defining
                                     each of the parameters of the model.
                                     These parameters uniquely define a model and
                                     will usually be used to reset a model from a file
            name (None, optional): The name to prefix to the parameters description
        """
        # Need to be done for haiku initialization module
        super().__init__(name=name)

        # Save the parameters
        self.params = params

        # Some checks on the parameters
        assert 'n_x' in params and 'n_u' in params and 'n_y' in params, \
            'The number of state, control and observation dimensions must be specified in the params'
        assert 'horizon' in params, 'The prediction horizon must be specified in the params'
        assert 'stepsize' in params, 'The stepsize must be specified in the params'

        # Save some specific function parameters
        self.n_x = params['n_x'] # State dimension
        self.n_u = params['n_u'] # Control dimension
        self.n_y = params['n_y'] # Observation dimension
        self.num_substeps = params.get('num_substeps', 1) # Number of substeps for the SDE solver
        assert self.n_x == self.n_y, 'The state and observation dimensions must be the same'

        # The prediction horizon
        self.horizon = params.get('horizon', 1)

        # Compute the time steps for the SDE solver
        self.time_step = compute_timesteps(params)

        # Construct the diffusion density
        self.construct_diffusion_density_nn()

        # Initialize the SDE solver
        self.sde_solve = sde_solver_name[params.get('sde_solver', 'euler_maruyama')]


    def prior_diffusion(self, x, u, extra_args=None):
        """ Define the prior noise function over the knowledge of the dynamics of the system.
        This prior function defines the maimum noise that we expect to see in the system predictions outside the data region.

        The prior noise is a function of the state and possibly the control input.
        This code assumes diagonal noise, but the user can define a more complex noise function

        Args:
            x (TYPE): The current state of the system (aka latent state)
            u (TYPE, optional): The current control signal applied to the system
            extra_args (TYPE, optional): Extra arguments to pass to the function
        
        Returns:
            TYPE: A noise vector of the same size as the state vector (latent space)
        """
        raise NotImplementedError
    
    def compositional_drift(self, x, u, extra_args=None):
        """Define the drift term of the SDE model to learn.
        The compositional drift is a function of the state and the control input, which incorporates prior knowledge on the system.

        We write it as compositional_drift(x,u) = f(x,u) =  F(x, u, g_1(x), g_2(x), ..., g_n(x)), 
        where g_i are the functions that define the drift and that are parametrized by NNs in our framework.
        This form on the drift informs the learning with prior knowledge on the dynamical system.
        Such prior knowledge is typically obtained as an ODE from physics or engineering.

        The NNs' parameters should be initialized by the user in the init function of the custom class, and 
        can be used in the compositional_drift function to define the drift term of the SDE.

        Args:
            x (TYPE): The current state of the system (aka latent state)
            u (TYPE, optional): The current control signal applied to the system
            extra_args (TYPE, optional): Extra arguments to pass to the function
        
        Returns:
            TYPE: A vector of the same size as the state vector (latent space)
        """
        raise NotImplementedError


    def reduced_state(self, _x):
        """ User-defined function to reduce the state / latent space for noise estimation.
        This function is used to reduce the state dimensionality such that the distance aware noise estimation
        is depending only on states that are relevant for the system dynamics.

        If the function is not redefined by the user, identity function is used.

        Assumptiom: The size of the reduced state must match the size of the attribute indx_noise_in, which 
        is used to select the states in the subset of states that are used for noise estimation.
        For example, if indx_noise_in = [0, 1, 3], then the reduced state must be of size at least 4. And the distance
        aware noise estimation will be performed only on the states y[0], y[1] and y[3], where y = reduced_state(x).

        If the function is not identity, then the user must redefined indx_noise_in

        Args:
            _x (TYPE): The current state of the system (aka latent state)
        """
        return _x
    
    def construct_diffusion_density_nn(self):
        """ Define the neural network that parametrizes the diffusion's density (eta in the paper), distance awareness, over the training dataset.

            The density (eta in the paper) could be a function of the state and the control input, or only of the state.
            The density (eta in the paper)  is a scalar function that is used to compute the prior diffusion multiplier.
            Around the data, the density function would be close to zero or encloses the aleatoric uncertainty.
            And far from the data, the density function would be close to one and the noise would be close to the prior diffusion.
            
            density : \eta(x, u; \theta) or \eta(x, \theta) -> A real-valued function

            scaler: A neural network that parametrizesthat enable heteroscedastic noise estimation.
            
            Total diffusion : \sigma(x, u) = [ \eta(x, u; \theta) * scaler(\theta) ] \sigma_{prior}(x, u)

            Few parameters' names:
                - diffusion_density_nn : 
                    The name of the key with the parameters of the density NN and scaler. If not present, the system is considered as an ODE.
                
                - diffusion_density_nn -> indx_noise_in : 
                    The indexes of the reduced states that contribute to the noise (if not given, all states contribute)
                
                - diffusion_density_nn -> indx_noise_out : 
                    The indexes of the outputs that contribute to the noise (if not given, all outputs contribute)
                
                - diffusion_density_nn -> scaler_nn : 
                    The name of the key with the parameters of the scaler NN. If not present, the scaler is assumed to be 1
                
                - diffusion_density_nn -> scaler_nn -> type : 
                    {nn, scaler} default is scaler
                    [TODO] Add more scaler options here
                
                - diffusion_density_nn -> scaler_nn -> init_value : 
                    The parameters are uniformly initialized in the range [-init_value, init_value]
                
                - diffusion_density_nn -> scaler_nn -> activation_fn, hidden_layers -> The parameters of the NN in case type is nn
                
                - diffusion_density_nn -> density_nn : 
                    The name of the key with the parameters of the density NN

                - diffusion_density_nn -> density_nn -> init_value : 
                    The parameters are uniformly initialized in the range [-init_value, init_value]

                - diffusion_density_nn -> density_nn -> activation_fn, hidden_layers -> The parameters of the NN

        """

        # Define the default eta and scaler functions if they are not gievn in the params
        self.scaler_fn = None
        self.diff_density = lambda _x, _u : 1.0

        if 'diffusion_density_nn' not in self.params:
            return

        # Extract the indexes of the states that contribute to the noise (if not given, all states contribute)
        self.noise_inputs = jnp.array(self.params['diffusion_density_nn'].get('indx_noise_in', jnp.arange(self.n_x)))
        
        # Extract the indexes of the outputs that contribute to the noise (if not given, all outputs contribute)
        self.noise_outputs = jnp.array(self.params['diffusion_density_nn'].get('indx_noise_out', jnp.arange(self.n_x)))

        # Define the function that concatenates the relevant state and control input if needed for the density NN
        self.noise_relevant_state = lambda _x : self.reduced_state(_x)[self.noise_inputs] if self.noise_inputs.shape[0] < self.n_x else self.reduced_state(_x)
        self.noise_aug_state_ctrl = lambda _x, _u : jnp.concatenate([self.noise_relevant_state(_x), _u], axis=-1) if self.params.get('control_dependent_noise', False) else self.noise_relevant_state(_x)
        self.noise_aug_output = lambda _z : jnp.ones(self.n_x).at[self.noise_outputs].set(_z) if self.noise_outputs.shape[0] < self.n_x else _z
        
        # Lets design the scaler term first
        if 'scaler_nn' in self.params['diffusion_density_nn'] and self.params['diffusion_density_nn']['scaler_nn'].get('type', 'none') != 'none':
            scaler_type = self.params['diffusion_density_nn']['scaler_nn'].get('type', 'scaler')
            init_value = self.params['diffusion_density_nn']['scaler_nn'].get('init_value', 0.001)
            # This scaler return either a vector (a,b) where a and b are of size noise_outputs.shape[0] and represent the
            # the offset and the scale of the density term on each axis.
            if scaler_type == 'scaler':
                self.scaler_fn =  lambda _ : hk.get_parameter('scaler', shape=(self.noise_outputs.shape[0]*2,), 
                                                    init = hk.initializers.RandomUniform(-init_value, init_value))
            elif scaler_type == 'nn':
                _act_fn = self.params['diffusion_density_nn']['scaler_nn'].get('activation_fn', 'tanh')
                self.scaler_fn = hk.nets.MLP([*self.params['diffusion_density_nn']['scaler_nn']['hidden_layers'], self.noise_outputs.shape[0]*2],
                                    activation = getattr(jnp, _act_fn) if hasattr(jnp, _act_fn) else getattr(jax.nn, _act_fn),
                                    w_init=hk.initializers.RandomUniform(-init_value, init_value),
                                    name = 'scaler')
            else:
                raise ValueError('The scaler type should be either scaler or nn')
        
        # Now lets design the density term
        if 'density_nn' in self.params['diffusion_density_nn']:
            init_value = self.params['diffusion_density_nn']['density_nn'].get('init_value', 0.01)
            _act_fn = self.params['diffusion_density_nn']['density_nn'].get('activation_fn', 'tanh')
            self.density_net = hk.nets.MLP([*self.params['diffusion_density_nn']['density_nn']['hidden_layers'], 1],
                                    activation = getattr(jnp, _act_fn) if hasattr(jnp, _act_fn) else getattr(jax.nn, _act_fn),
                                    w_init=hk.initializers.RandomUniform(-init_value, init_value),
                                    name = 'density')
            
            # The choice of sigmoid is to make sure that the density is always between 0 and 1
            # The choice of -7.0 is to make sure that regularization (params close to 0) provides a value close to 0
            # The regularization of the parameters defining the density network
            self.density_function = lambda xu_red: jax.nn.sigmoid((self.density_net(xu_red)[0] - 7.0))
            
            # Create the diff_density function
            def __diff_density(_x, _u):

                # Augment the control and state if needed
                _xu_red = self.noise_aug_state_ctrl(_x, _u)

                # Estimate the density network which is unscaled
                _density = self.density_net(_xu_red)[0]

                # If the way to distribute the noise to each state is learned, we obtain the coefficients and offset
                if self.scaler_fn is not None:
                    # 1e-7 is to make sure that the scaler is close to 0 when the parameters are close to 0
                    # We make sure that the scaler is always positive to ensure increasing noise with the density
                    _scaler_val = jnp.exp(self.scaler_fn(_xu_red)) * 1e-7
                    scaler_coeff, scaler_offset = _scaler_val[:self.noise_outputs.shape[0]], _scaler_val[self.noise_outputs.shape[0]:]
                else:
                    scaler_coeff, scaler_offset = 0.0, 0.0

                # The final noise withour prior is given by:
                # Sigmoid ( _density -7.0  + _ scaler_coeff * _density + scaler_offset)
                # The +7.0 enforce that the density is close to 0 when the density network is close to 0
                # We can clearly see how the scaler_coeff and scaler_offset are learned and used to redistribute the noise
                
                # [TODO, Franck] Include other sort of noise here based on the paper description. 
                # Our experiments work well with this noise
                return jax.nn.sigmoid(scaler_coeff * _density - 7.0 + scaler_offset)
            
            # Utilities function to access 
            # a) The rescaled noise function
            # b) The actual neural network in the density estimation  without the scaler or the sigmoid
            # c) or the density function with sigmoid but without the scaler
            self.diff_density = lambda _x, _u : self.noise_aug_output(__diff_density(_x, _u))
            self.density_function_v2 =  lambda _x, _u : self.density_net(self.noise_aug_state_ctrl(_x, _u))[0]
            self.density_function_v1 =  lambda _x, _u : self.density_function(self.noise_aug_state_ctrl(_x, _u))

    
    def diffusion(self, x, u, extra_args=None):
        """ This function defines the total diffusion term of the SDE model to learn.
            It is a function of the user-defined, prior diffusion, 
            and the distance-aware, rescaled density distribution
        Args:
            x (TYPE): The current state of the system (aka latent state)
            u (TYPE, optional): The current control signal applied to the system
            extra_args (TYPE, optional): Extra arguments to pass to the function
        
        Returns:
            TYPE: A noise vector of the same size as the state vector (latent space)
        """
        return self.diff_density(x, u) * self.prior_diffusion(x, u, extra_args)
    

    def state_transform_for_loss(self, x):
        """ A function to transform the state for loss computation.
            E.g., if the state contains an angle, 
            maybe for loss computation its better to use sin and cos of the angle.
            This function returns the transformed state that is being used in the morm 2 loss

        Args:
            x (TYPE): The current state of the system

        Returns:
            TYPE: The transformed state of the system
        """
        return x
    
    def sample_sde(self, y0, uVal, rng, extra_scan_args=None, ret_aux=False):
        """Sample trajectory from the SDE distribution

        Args:
            y0 (TYPE): The observation of the system at time at initial time
            uVal (TYPE): The control signal applied to the system. This could be an array or a function of observation
            rng (TYPE): A random number generator key
            extra_scan_args (None, optional): Extra arguments to be passed to the scan function and principally the drift and diffusion functions

        Returns:

        """
        return self.sample_general(self.compositional_drift, self.diffusion, y0, uVal, rng, extra_scan_args, ret_aux)

    def sample_general(self, drift_term, diff_term, y0, uVal, rng_brownian, extra_scan_args=None, ret_aux=False):
        """A general function for sampling from a drift fn and a diffusion fn
        given times at which to sample solutions and control values to be applied at these times.

        Args:
            obs2state_fns (TYPE): A tuple of functions to convert observations to states and vice versa
            drift_term (TYPE): A function to compute the drift term of the SDE
            diff_term (TYPE): A function to compute the diffusion term of the SDE
            y0 (TYPE): The observation of the system at time at initial time
            uVal (TYPE): The control signal applied to the system. This could be an array or a function of observation
            rng_brownian (TYPE): A random number generator key
            extra_scan_args (None, optional): Extra arguments to be passed to the scan function and the drift and diffusion functions
        
        Returns:
            array: 2D array of shape (horizon+1, n_x) containing the sampled state trajectory
            array: 2D array of shape (horizon+1, n_y) containing the sampled observation trajectory
            array: 2D array of shape (horizon, n_u) containing the control input applied at each time step
        """

        # When the system is being initialized, we just need to run the functions once to initialize the parameters
        if hk.running_init():
            #[TODO, Franck] This assumes the u is given as a 2D or 1D array, not a function of the observation
            # Dummy return in this case -> This case is just to initialize NNs
            # Initialize the obs2state and state2obs parameters
            x0 = y0
            # Initialize the drift and diffusion parameters
            _ = drift_term(x0, uVal if uVal.ndim ==1 else uVal[0], extra_scan_args)
            _ = diff_term(x0, uVal if uVal.ndim ==1 else uVal[0], extra_scan_args)
            if not ret_aux:
                return jnp.zeros((self.params['horizon']+1, self.n_x)), jnp.zeros((self.params['horizon'], self.n_u))
            else:
                return jnp.zeros((self.params['horizon']+1, self.n_x)), jnp.zeros((self.params['horizon'], self.n_u)), {}
        else:
            # Solve the sde and return its output (latent space)
            if 'simpletic' in self.params['sde_solver']:
                return self.sde_solve(self.time_step, y0, uVal, 
                        rng_brownian, drift_term, diff_term,
                        self.num_pos, self.indx_pos_vel,
                        projection_fn= self.projection_fn if hasattr(self, 'projection_fn') else None,
                        extra_scan_args=extra_scan_args,
                        return_vector_field_aux=ret_aux,
                        num_substeps=self.num_substeps
                    )
            return self.sde_solve(self.time_step, y0, uVal, 
                        rng_brownian, drift_term, diff_term, 
                        projection_fn= self.projection_fn if hasattr(self, 'projection_fn') else None,
                        extra_scan_args=extra_scan_args,
                        return_vector_field_aux=ret_aux,
                        num_substeps=self.num_substeps
                    )
    
    def density_loss(self, y, u, rng, mu_coeff_fn):
        """ Given an observation, control, a random number generator key, and the function to compute strong convexity coefficient at (x,u),
            this function computes the following terms that define the loss on the density function and that incorporates distance awareness:
                1. The observation y is converted to a state x
                2. x and u are combined [x,u] then reduced to the relevant components as defined by noise_aug_state_ctrl
                3. The reduced state and control are passed to density_function to compute both the density and its gradient
                4. Now, params['density_loss'] contains 4 keys: ball_radius, mu_coeff, learn_mucoeff, ball_nsamples
                    a. ball_radius: The radius of the ball around the observation [x,u] where we sample points to enforce local density/ strong convexity
                    b. ball_nsamples: The number of points to sample from the ball to enforce the strong convexity term
                5. We generate a ball of radius ball_radius around [x,u] and sample ball_nsamples points from it
                6. We compute the density at each of the sampled points
            
            Args:
                y (TYPE): The observation of the system
                u (TYPE): The control signal applied to the system
                rng (TYPE): A random number generator key
                mu_coeff_fn (TYPE): A function that returns the local strong convexity coefficient at (x,u)

            Returns:
                TYPE: The gradient norm
                TYPE: the strong convexity loss
                TYPE: The density vector at (x,u)
                TYPE: The local coefficient of strong convexity at (x,u)
        """

        # Convert the observation to a state
        x = y

        # Combine the state and the control
        xu_red = self.noise_aug_state_ctrl(x, u)

        # Compute the density and its gradient
        den_xu, grad_den_xu = jax.value_and_grad(self.density_function)(xu_red)

        # Check if the ball_radius is an array or a scalar
        # [TODO] Make the radius to be a proportion of the magnitude of the state
        radius = jnp.array(self.params['density_loss']['ball_radius'])
        if radius.ndim > 0:
            assert radius.shape == xu_red.shape, "The ball_radius should be a scalar or an array of size xu_red.shape[0]"

        # Sample ball_nsamples points from the ball of radius ball_radius around xu_red
        ball_dist = jax.random.normal(rng, (self.params['density_loss']['ball_nsamples'], xu_red.shape[0])) * radius[None]
        xball = xu_red[None] + ball_dist

        # Compute the density at each of the sampled points
        den_xball = jax.vmap(self.density_function)(xball)
        mu_coeff = mu_coeff_fn(xu_red)

        # Compute the strong convexity loss given mu_coeff
        sconvex_cond = den_xball - den_xu - jnp.sum(grad_den_xu[None] * ball_dist, axis=1) - 0.5 * mu_coeff * jnp.sum(jnp.square(ball_dist), axis=1)
        sconvex_loss = jnp.sum(jnp.square(jnp.minimum(sconvex_cond, 0)))

        return jnp.sum(jnp.square(grad_den_xu)), sconvex_loss, den_xu, mu_coeff
    

    def mu_coeff_nn(self, aug_xu):
        """ This function returns the learned local quadratic/strong convexity coefficient of the density function
        Args:
            aug_xu (TYPE): The augmented state and control vector
        
        Returns:
            TYPE: A vector of the same size as the state vector (latent space)
        """
        type_mu = self.params['density_loss']['learn_mucoeff']['type']
        # If the strong convexity coefficient is constant, return the value stored in density_loss
        if type_mu == 'constant':
            return self.params['density_loss']['mu_coeff']
        # If the strong convexity coefficient is global, it is constant and the constant is being learned
        elif type_mu == 'global':
            return self.params['density_loss']['mu_coeff'] * jnp.exp(hk.get_parameter('mu_coeff', shape=(), init=hk.initializers.RandomUniform(-0.001, 0.001)))

        # In other cases, we use a neural network to learn the coefficient
        _act_fn = self.params['density_loss']['learn_mucoeff']['activation_fn']
        _init_value = self.params['density_loss']['learn_mucoeff'].get('init_value', 0.01)
        mu_net = hk.nets.MLP([*self.params['density_loss']['learn_mucoeff']['hidden_layers'], 1],
                                    activation = getattr(jnp, _act_fn) if hasattr(jnp, _act_fn) else getattr(jax.nn, _act_fn),
                                    w_init=hk.initializers.RandomUniform(-_init_value, _init_value),
                                    name = 'mu_coeff')
        
        # We make sure that the output is positive and we multiply it by the initial value
        return jnp.exp(mu_net(aug_xu)[0]) * self.params['density_loss']['mu_coeff']


    def sample_for_loss_computation(self, ymeas, uVal, rng, extra_scan_args=None):
        """ Sample the SDE and compute the loss between the estimated and the measured observation.
            This function is usually the objective to fit the SDE to the data.

        Args:
            ymeas (TYPE): The measured observation of the system
            uVal (TYPE): The control signal applied to the system. This is an array
            rng (TYPE): A random number generator key
            extra_scan_args (None, optional): Extra arguments to be passed to the scan function and the drift and diffusion functions
        
        Returns:
            TYPE: The estimated latent x0 at y0 = ymeas[0]
            TYPE: The error between the estimated and the measured observation
            TYPE: A dictionary containing extra information about the loss
        """

        # Check if the trajectory horizon matches the number of control inputs
        assert ymeas.shape[0] == uVal.shape[0]+1, 'Trajectory horizon must match'

        # Extract the number of integration steps to between two control inputs and measurements
        # This is greater than 1, in the case where the time step in the dataset is finer than the time step used for the integration
        # Typically, in MPC, we want shorter look-ahead horizons with longer time steps
        num_steps2data = self.params['num_steps2data']
        assert  self.params['horizon'] * num_steps2data == uVal.shape[0], 'Trajectory horizon and num_steps2data must match the number of control inputs'

        # Split the random number generator
        rng_brownian, rng_sample2consider, rng_density = jax.random.split(rng, 3)

        # uval is size (horizon * num_steps2data, num_ctrl), lets reshape it to (horizon, num_steps2data, num_ctrl)
        u_values = uVal.reshape((self.params['horizon'], num_steps2data, self.params['n_u']))

        # extra_scan_args is a tuple of arguments to pass the the integration scheme when not None.
        # We also need to reshape it so that it is of size (horizon, num_steps2data,)
        if extra_scan_args is not None:
            assert isinstance(extra_scan_args, tuple), 'extra_scan_args must be a tuple of arrays'
            # Now reshape the extra_scan_args depending on the number of arguments
            extra_scan_args = tuple([
                arg.reshape((self.params['horizon'], num_steps2data)) if arg.ndim == 1 else \
                arg.reshape((self.params['horizon'], num_steps2data, *arg.shape[1:])) \
                for arg in extra_scan_args
            ])

        # Let's get the actual y_values to fit the SDE to
        y_values = ymeas[::num_steps2data]

        #[TODO, Franck] More documentation on this
        # How do we pick u_values? Different strategies stored in params['u_sampling_strategy']
        # By default we pick the first control value, i.e. u_values[:,0,:]
        # Another strategy is the mean of all the control values, i.e. u_values.mean(axis=1)
        # Another strategy is the median of all the control values, i.e. jnp.median(u_values, axis=1)
        # Another strategy is a random control value
        u_sampling_strategy = self.params.get('u_sampling_strategy', 'first')
        u_values = sampling_strat_under_dataset_with_finer_steps(u_values, u_sampling_strategy, rng_sample2consider)
        if extra_scan_args is not None:
            extra_scan_args = tuple([
                sampling_strat_under_dataset_with_finer_steps(arg, u_sampling_strategy, rng_sample2consider) \
                for arg in extra_scan_args
            ])

        # Solve the SDE to obtain state and observation evolution
        y0 = y_values[0]
        ynext, _ = self.sample_sde(y0, u_values, rng_brownian, extra_scan_args)

        # Extract the state (ignore the initial state) used for loss computation
        ynext, meas_y = ynext[1:], y_values[1:]

        # Get indexes of the samples to consider in the logprob computation between the estimated and the measured observation
        # This essentially makes the fitting similar to a time series with irregular sampling and help with generalization
        _indx = None
        if 'num_sample2consider' in self.params:
            if self.params['num_sample2consider'] < ynext.shape[0]:
                _indx = jax.random.choice(rng_sample2consider, ynext.shape[0], 
                            shape=(self.params['num_sample2consider'],), replace=False)
                # Restrain the time indexes to be considered
                ynext = ynext[_indx]
                meas_y = meas_y[_indx]

        # Get the discount factor
        discount_factor = self.params.get('discount_pred', 1.0)
        discount_arr = jnp.array([discount_factor**i for i in range(ynext.shape[0])])

        # Compute the error between the estimated and the measured observation
        # We sum the error over the integration horizon and then average over the number of coordinates
        # _error_data = jnp.mean(jnp.sum(jnp.square(meas_y-ynext), axis=0))
        # meas_y_trans
        _error_data = jnp.sum( discount_arr * jnp.sum(jnp.square((self.state_transform_for_loss(meas_y)-self.state_transform_for_loss(ynext)) / jnp.array(self.params.get('obs_weights', 1.0))), axis=-1) )

        # Extra values to print or to penalize the loss function on
        extra = {}

        # Check if density_loss is in the parameters
        if 'density_loss' not in self.params:
            # Error on prediction, Gradient error, strong convexity error, and extra values
            return _error_data, 0.0, 0.0, 0.0, extra

        # Function to get the mu_coeff whether it is learnable or not
        my_density_loss = lambda _y, _u, _rng : self.density_loss(_y, _u, _rng, self.mu_coeff_nn)
        
        # Check if haiku is running initialization
        if hk.running_init():
            # Initialize the extra parameters if present
            grad_norm, sconvex, _density_val, _mu_coeff = my_density_loss(y0, u_values[0], rng_density)
            extra['density_val'] = _density_val
            # Error on prediction, Gradient error, strong convexity error, mu_coeff, and extra values
            return _error_data, grad_norm, sconvex, _mu_coeff, extra

        # Here haiku is not running initialization
        rng_density = jax.random.split(rng_density, ymeas.shape[0]-1)
        den_yinput = ymeas[:-1]
        den_uinput = uVal

        # Get the gradient and the convex loss
        grad_norm, sconvex, _density_val, _mu_coeff = jax.vmap(my_density_loss)(den_yinput, den_uinput, rng_density)
        extra['density_val'] = jnp.mean(_density_val)
        
        # Error on prediction, Gradient error, strong convexity error, mu_coeff, and extra values
        return _error_data, jnp.mean(grad_norm), jnp.sum(sconvex), jnp.mean(_mu_coeff), extra


def create_diffusion_fn(params_model, sde_constr=ControlledSDE, seed=0,
                        **extra_args_sde_constr):
    """ Return a function for estimating the diffusion term given an observation of the system.
    The function is a wrapper around the SDE class

    Args:
        params_model (dict): The parameters of the model
        sde_constr (TYPE, optional): The SDE class to use
        seed (int, optional): The seed for the random number generator
        **extra_args_sde_constr: Extra arguments for the constructor of the SDE class
        
    Returns:
        dict: A dictionary containing the initial parameter to compute the total diffusion and density diffusion
        function: The function that estimate the diffusion vector given an observation
                    The function takes as input the some hk model parameters, observation, control and a random key
                    and returns the estimated total diffusion term and the density diffusion term
                    diffusion_fn(params: dict, obs: ndarray, u: ndarray, rng: ndarray, extra_args) -> diffusion, density : ndarray, scalar
    """
    rng_zero = jax.random.PRNGKey(seed)
    yzero = jnp.zeros((params_model['n_y'],))
    uzero = jnp.zeros((params_model['n_u'],))

    def _diffusion(y0, u0, net=False, extra_args=None):
        m_model = sde_constr(params_model, **extra_args_sde_constr)
        # TODO: Maybe having this function independent of the encoder structure
        # Because the density function is going to be depending on the neural network structure of the encoder
        # Which might be quite actually quite good as it imposes a structure on the encoder too
        x0 = y0
        if not net:
            return m_model.diffusion(x0, u0, extra_args), m_model.density_function_v1(x0, u0)
        else:
            return m_model.diffusion(x0, u0, extra_args), m_model.density_function_v2(x0, u0)
    
    # Transform the function into a pure one
    diffusion_pure =  hk.without_apply_rng(hk.transform(_diffusion))
    nn_params = diffusion_pure.init(rng_zero, yzero, uzero)
    return nn_params, diffusion_pure.apply

def create_one_step_sampling(params_model, sde_constr= ControlledSDE, seed=0, 
                            num_samples=None, **extra_args_sde_constr):
    """Create a function that sample the next state

    Args:
        params_model (TYPE): The SDE solver parameters and model parameters
        sde_constr (TYPE): A class constructor that is child of ControlledSDE class
        seed (int, optional): The seed for the random number generator
        num_samples (int, optional): The number of samples to generate
        **extra_args_sde_constr: Extra arguments for the constructor of the SDE solver

    Returns:
        function: a function for one-step multi-sampling
                    The function takes as input the some hk model parameters, observation, control and a random key, and possibly extra arguments for drift and diffusion
                    and returns the next state or a number of particles of the next state
                    one_step_sampling(params: dict, y: ndarray, u: ndarray, rng: ndarray, extra_args: None or named args) -> next_x, next_y, u : ndarray
    """
    params_model = copy.deepcopy(params_model) # { k : v for k, v in params_model.items()}
    params_model['horizon'] = 1 # Set the horizon to 1

    # We remove num_short_dt, short_step_dt, and long_step_dt from the model as they are not used in the loss
    params_model.pop('num_short_dt', None)
    params_model.pop('short_step_dt', None)
    params_model.pop('long_step_dt', None)

    # Get the number of samples
    num_samples = params_model['num_particles'] if num_samples is None else num_samples

    # Some dummy initialization scheme
    #[TODO Franck] Maybe something more general in case these inputs are not valid
    rng_zero = jax.random.PRNGKey(seed)
    yzero = jnp.ones((params_model['n_y'],)) * 1e-1
    uzero = jnp.ones((params_model['n_u'],)) * 1e-1

    # Define the transform for the sampling function
    def sample_sde(y, u, rng, extra_args=None):
        """ Sampling function """
        m_model = sde_constr(params_model, **extra_args_sde_constr)
        return m_model.sample_sde(y, u, rng, extra_args)

    # Transform the function into a pure one
    sampling_pure =  hk.without_apply_rng(hk.transform(sample_sde))
    _ = sampling_pure.init(rng_zero, yzero, uzero, rng_zero)

    # Now define the n_sampling method
    def multi_sampling(_nn_params, y, u, rng, extra_args=None):
        assert rng.ndim == 1, 'RNG must be a single key for vmapping'
        m_rng = jax.random.split(rng, num_samples)
        vmap_sampling = jax.vmap(sampling_pure.apply, in_axes=(None, None, None, 0, None))
        yevol, uevol =  vmap_sampling(_nn_params, y, u, m_rng, extra_args)
        return (yevol[0], uevol[0]) if num_samples == 1 else (yevol, uevol)

    return multi_sampling

def create_sampling_fn(params_model, sde_constr= ControlledSDE, 
                       seed=0, num_samples=None, **extra_args_sde_constr):
    """Create a sampling function for prior or posterior distribution

    Args:
        params_model (TYPE): The SDE solver parameters and model parameters
        sde_constr (TYPE): A class constructor that is child of ControlledSDE class. it specifies the SDE model
        seed (int, optional): The seed for the random number generator
        num_samples (int, optional): The number of samples to generate
        **extra_args_sde_constr: Extra arguments for the constructor of the SDE solver

    Returns:
        dict: A dictionary containing the initial parameter models
        function: a function for multi-sampling on the posterior or prior
                    The function takes as input the some hk model parameters, observation, control and a random key, and possibly extra arguments for drift and diffusion
                    and returns the next state or a number of particles of the next state
                    sampling_fn(params: dict, y: ndarray, u: ndarray, rng: ndarray, extra_args: nor or named args) -> next_x, next_y, u : ndarray
    """
    # Some dummy initialization scheme
    #[TODO Franck] Maybe something more general in case these inputs are not valid
    # TODO: This function is almost the same as create_one_step_sampling. We should merge them
    rng_zero = jax.random.PRNGKey(seed)
    yzero = jnp.ones((params_model['n_y'],)) * 1e-1
    uzero = jnp.ones((params_model['n_u'],)) * 1e-1
    num_samples = params_model['num_particles'] if num_samples is None else num_samples

    # Define the transform for the sampling function
    def sample_sde(y, u, rng, extra_args=None):
        """ Sampling function """
        m_model = sde_constr(params_model, **extra_args_sde_constr)
        return m_model.sample_sde(y, u, rng, extra_args)

    # Transform the function into a pure one
    sampling_pure =  hk.without_apply_rng(hk.transform(sample_sde))
    nn_params = sampling_pure.init(rng_zero, yzero, uzero, rng_zero)

    # Now define the n_sampling method
    def multi_sampling(_nn_params, y, u, rng, extra_args=None):
        assert rng.ndim == 1, 'RNG must be a single key for vmapping'
        m_rng = jax.random.split(rng, num_samples)
        vmap_sampling = jax.vmap(sampling_pure.apply, in_axes=(None, None, None, 0, None))
        return vmap_sampling(_nn_params, y, u, m_rng, extra_args)

    return nn_params, multi_sampling

def create_model_loss_fn(model_params, loss_params, sde_constr=ControlledSDE, verbose=True,
                        **extra_args_sde_constr):
    """Create a loss function for evaluating the current model with respect to some
       pre-specified dataset

    Args:
        model_params (TYPE): The SDE model and solver parameters
        loss_params (TYPE): The pamaters used in the loss function computation. 
                            Typically penalty coefficient for the different losses.
        sde_constr (TYPE): A class constructor that is child of ControlledSDE class
        **extra_args_sde_constr: Extra arguments for the constructor of the SDE solver

    Returns:
        dict : A dictionary containing the initial parameters for the loss computation
        function: A function for computing the loss
                    The function takes as input the some hk model parameters, observation, control and a random key, and possibly extra arguments for drift and diffusion
                    and returns the loss value and a dictionary of the different losses
                    loss_fn(params: dict, y: ndarray, u: ndarray, rng: ndarray, extra_args: nor or named args) -> loss : float, losses: dict
        function: A function for projecting the special nonnegative parameters of the model
                    The function takes as input the some hk model parameters and returns the projected parameters
                    project_fn(params: dict) -> params: dict
        function: A function for sampling the learned model
    """

    # Verbose print function
    vprint = print if verbose else lambda *a, **k: None

    # Deep copy params_model
    params_model = model_params

    # The number of samples is given by the loss dictionary -> If not present, use the default value from the params-model
    num_sample = loss_params.get('num_particles', params_model.get('num_particles', 1) )
    params_model['num_particles'] = num_sample

    # The step size is given by the loss dictionary -> If not present, use the default value from the params-model
    step_size = loss_params.get('stepsize', params_model['stepsize'] )
    params_model['stepsize'] = step_size

    # Now let's get the data stepsize from the loss dictionary -> If not present, use the default value is the step_size
    data_stepsize = loss_params.get('data_stepsize', step_size)
    # Let's check if the stepsize is a multiple of the data_stepsize
    if abs (step_size - data_stepsize) <= 1e-6:
        num_steps2data = 1
    else:
        assert abs(step_size % data_stepsize) <= 1e-6, 'The data stepsize must be a multiple of the stepsize'
        # Let's get the number of steps between data points
        num_steps2data = int((step_size/data_stepsize) +0.5) # Hack to avoid numerical issues

    # Let's get the horizon of the loss
    horizon = loss_params.get('horizon', params_model.get('horizon', 1))
    # Let's get the actual actual horizon of the loss
    data_horizon = horizon * num_steps2data
    # Let's set the horizon in the params_model
    params_model['horizon'] = horizon
    params_model['num_steps2data'] = num_steps2data
    loss_params['data_horizon'] = data_horizon
    # Print the number of particles used for the loss
    vprint('Using [ N = {} ] particles for the loss'.format(num_sample))
    # Print the horizon used for the loss
    vprint('Using [ T = {} ] horizon for the loss'.format(params_model['horizon']))
    # Print the stepsize used for the loss
    vprint('Using [ dt = {} ] stepsize for the loss'.format(params_model['stepsize']))
    # Print the number of steps between data points
    vprint('Using [ num_steps2data = {} ] steps between data points'.format(num_steps2data))
    # SEt the discount factor for the loss
    params_model['discount_pred'] = loss_params.get('discount_pred', 1.0)

    # Extract the number of 
    if 'num_sample2consider' in loss_params:
        params_model['num_sample2consider'] = loss_params['num_sample2consider']
    
    if 'obs_weights' in loss_params:
        params_model['obs_weights'] = loss_params['obs_weights']
    
    if 'u_sampling_strategy' in loss_params:
        params_model['u_sampling_strategy'] = loss_params['u_sampling_strategy']

    # We remove num_short_dt, short_step_dt, and long_step_dt from the model as they are not used in the loss
    params_model.pop('num_short_dt', None)
    params_model.pop('short_step_dt', None)
    params_model.pop('long_step_dt', None)
    vprint ('Removed num_short_dt, short_step_dt, and long_step_dt from the model as they are not used in the loss and sde training')

    # Now check if the diffusion is parameterized by a neural network
    if 'diffusion_density_nn' not in params_model or 'density_nn' not in params_model['diffusion_density_nn']:
        loss_params.pop('density_loss', None)

    # Now we insert the density loss parameters if they are not present
    if 'density_loss' in loss_params:
        if 'type' not in loss_params['density_loss']['learn_mucoeff']:
            loss_params['density_loss']['learn_mucoeff']['type'] = 'constant'
        params_model['density_loss'] = loss_params['density_loss']
        if loss_params['density_loss']['learn_mucoeff']['type'] == 'constant':
            loss_params['pen_mu_coeff'] = 0.0
    else:
        loss_params['pen_mu_coeff'] = 0.0
        loss_params['pen_density_scvex'] = 0.0
        loss_params['pen_grad_density'] = 0.0

    # Define the transform for the sampling function
    def sample_loss(y, u, rng, extra_args=None):
        m_model = sde_constr(params_model, **extra_args_sde_constr)
        return m_model.sample_for_loss_computation(y, u, rng, extra_args)
    
    rng_zero = jax.random.PRNGKey(loss_params.get('seed', 0))
    yzero = jnp.ones((data_horizon+1,params_model['n_y'])) * 1e-2
    uzero = jnp.ones((data_horizon,params_model['n_u'],)) * 1e-2

    # Transform the function into a pure one
    sampling_pure =  hk.without_apply_rng(hk.transform(sample_loss))
    nn_params = sampling_pure.init(rng_zero, yzero, uzero, rng_zero)

    # Let's get nominal parameters values
    nominal_params_val = loss_params.get('nominal_parameters_val', {})
    default_params_val = loss_params.get('default_parameters_val', 0.) # This value imposes that the parameters should be minimized to 0
    _nominal_params_val = set_values_all_leaves(nn_params, default_params_val)
    nominal_params = _nominal_params_val if len(nominal_params_val) == 0 else update_same_struct_dict(_nominal_params_val, nominal_params_val)
    special_params_val = loss_params.get('special_parameters_val', {})
    nominal_params = get_penalty_parameters(nominal_params, special_params_val, None)
    # Print the resulting penalty coefficients
    vprint('Nominal parameters values: \n {}'.format(nominal_params))

    # Let's get the penalty coefficients for regularization
    special_parameters = loss_params.get('special_parameters_pen', {})
    default_weights = loss_params.get('default_weights', 1.)
    penalty_coeffs = get_penalty_parameters(nn_params, special_parameters, default_weights)
    # Print the resulting penalty coefficients
    vprint('Penalty coefficients: \n {}'.format(penalty_coeffs))

    # Get nonzero coefficient
    nonzero_params = get_penalty_parameters(nn_params, loss_params.get('nonneg_nonzero', {}), 0.0)

    # Nonnegative parameters of the problem
    nonneg_params = get_non_negative_params(nn_params, {k : True for k in params_model.get('noneg_params', []) })
    # Print the resulting nonnegative parameters
    vprint('Nonnegative parameters: \n {}'.format(nonneg_params))

    # Define a projection function for the parameters
    def nonneg_projection(_params):
        return jax.tree_map(lambda x, nonp, nzer : jnp.maximum(x, nzer) if nonp else x, _params, nonneg_params, nonzero_params)

    # Now define the n_sampling method
    def multi_sampling(_nn_params, y, u, rng, extra_args=None):
        assert rng.ndim == 1, 'RNG must be a single key for vmapping'
        m_rng = jax.random.split(rng, num_sample)
        vmap_sampling = jax.vmap(sampling_pure.apply, in_axes=(None, None, None, 0, None))
        return vmap_sampling(_nn_params, y, u, m_rng, extra_args)

    def loss_fn(_nn_params, y, u, rng, extra_args=None, include_diff=True):
        # CHeck if rng is given as  a ingle key
        assert rng.ndim == 1, 'THe rng key is splitted inside the loss function computation'

        # Split the rng key first
        rng = jax.random.split(rng, y.shape[0])

        # Do multiple step prediction of state and compute the logprob and KL divergence
        batch_vmap = jax.vmap(multi_sampling, in_axes=(None, 0, 0, 0, 0) if extra_args is not None else (None, 0, 0, 0, None))
        _error_data, grad_density, density_scvex, mu_coeff, extra_feat = batch_vmap(_nn_params, y, u, rng, extra_args)

        # COmpute the loss on fitting the trajectories
        loss_data = jnp.mean(jnp.mean(_error_data, axis=1))

        # Compute the loss on the gradient of the density
        loss_grad_density = jnp.mean(jnp.mean(grad_density, axis=1))

        # Compute the loss on the density local strong convexity
        loss_density_scvex = jnp.mean(jnp.mean(density_scvex, axis=1))

        # Compute the loss on the mu coefficient
        mu_coeff_mean = jnp.mean(jnp.mean(mu_coeff, axis=1)) # This is probably not needed
        loss_mu_coeff = 0.0

        # Weights penalization
        w_loss_arr = jnp.array( [jnp.sum(jnp.square(p - p_n)) * p_coeff \
                            for p, p_n, p_coeff in zip(jax.tree_util.tree_leaves(_nn_params), jax.tree_util.tree_leaves(nominal_params), jax.tree_util.tree_leaves(penalty_coeffs)) ]
                        )
        w_loss = jnp.sum(w_loss_arr)

        # Extra feature mean if there is any
        m_res = { k: jnp.mean(jnp.mean(v, axis=1)) for k, v in extra_feat.items()}

        # Multiplier for the diffusion aware terms
        pen_scvex_mult = loss_params.get('pen_scvex_mult', 1.0)

        # Compute the total sum
        total_sum = 0.0
        if loss_params.get('pen_data', 0) > 0:
            total_sum += loss_data * loss_params['pen_data']

        total_sum_unc = 0.0
        if loss_params.get('pen_grad_density', 0) > 0 and include_diff:
            total_sum_unc += loss_params['pen_grad_density'] * loss_grad_density  * pen_scvex_mult
        
        if loss_params.get('pen_density_scvex', 0) > 0 and include_diff:
            total_sum_unc += loss_params['pen_density_scvex'] * loss_density_scvex * pen_scvex_mult
        
        if loss_params.get('pen_mu_coeff', 0) > 0 and include_diff:
            # We seek to maximize the mu coefficient
            pen_mu_type = loss_params.get('pen_mu_type', 'quad_inv')
            if  pen_mu_type == 'quad_inv':
                loss_mu_coeff = 1.0 / mu_coeff_mean**2
            elif pen_mu_type == 'lin_inv':
                loss_mu_coeff = 1.0 / mu_coeff_mean
            elif pen_mu_type == 'exp_inv':
                loss_mu_coeff = jnp.exp(-mu_coeff_mean * loss_params['pen_mu_temp'])
            else:
                raise ValueError('Unknown pen_mu_type: {}. Choose from quad_inv, lin_inv, exp_inv'.format(pen_mu_type))
            total_sum_unc += loss_mu_coeff * loss_params['pen_mu_coeff'] * pen_scvex_mult

        total_sum += total_sum_unc
        if loss_params.get('pen_weights', 0) > 0:
            total_sum += loss_params['pen_weights'] * w_loss
            
        return total_sum, {'totalLoss' : total_sum, 'gradDensity' : loss_grad_density, 'lossMuCoeff' : loss_mu_coeff,
                            'densitySCVEX' : loss_density_scvex, 'weights' : w_loss, 'muCoeff' : mu_coeff_mean
                            , 'dataLoss' : loss_data, 'LossUnc' : total_sum_unc, **m_res}
    
    return nn_params, loss_fn, nonneg_projection