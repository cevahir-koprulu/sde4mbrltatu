import jax
import jax.numpy as jnp

import haiku as hk

from epistemic_nsde.train_sde import train_model
from epistemic_nsde.nsde import ControlledSDE, create_sampling_fn, compute_timesteps, create_diffusion_fn
from epistemic_nsde.dict_utils import load_yaml, apply_fn_to_allleaf, update_params, dump_yaml

from models.sde_models.utils_for_d4rl_mujoco import get_formatted_dataset_for_nsde_training

import numpy as np

import pickle
import os

from tqdm import tqdm

# Let's define constants for this environment
NUM_STATES = 17
NUM_ACTIONS = 6
NUM_OBSERVATIONS = 17
OBS_NAMES = ['rootz', 'rooty', 'bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot', 
             'Vrootx', 'Vrootz', 'Arooty', 'Abthigh', 'Abshin', 'Abfoot', 'Afthigh', 'Afshin', 'Affoot']
ANGLE_NAMES = ['rooty', 'bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
CONTROL_NAMES = ["Cbthigh", "Cbshin", "Cbfoot", "Cfthigh", "Cfshin", "Cffoot"]
TIMESTEP_ENV = 0.05 # skip_frames * xml_delta_t, time step in the data and in the environment
NUM_SUBSTEPS = 5

class HalfCheetahSDE(ControlledSDE):
    """ An SDE model for the HalfCheetah environment.
    """
    def __init__(self, params, name=None):
        # Define the params here if needed before initialization
        super().__init__(params, name=name)

        # Initialization of the residual networks
        # This function setup the parameters of the unknown neural networks and the residual network
        self.init_residual_networks()

        # Set the position to velocity indices
        # These are used by simpletic solver but not by normal euler or other solver
        self.num_pos = 8 # Where the position ends in the full state vector
        self.indx_pos_vel = jnp.array([1, 2, 3, 4, 5, 6, 7, 8]) # Where to identify the position velocity in the full velocity vector

        self.state_scaling = jnp.array(self.params.get('state_scaling', [1.0] * self.n_x))
        self.max_state_scaling = jnp.max(self.state_scaling)

        # In case scaling factor is give, we also need to ensure scaling diffusion network inputs
        if 'state_scaling' in self.params:
            # Include velocity only
            self.reduced_state = lambda x : x / self.max_state_scaling
    
    def prior_diffusion(self, x, u, extra_args=None):
        # Set the prior to a constant noise as defined in the yaml file
        return jnp.array(self.params['noise_prior_params'])

    # def projection_fn(self, x):
    #     """ A function to project the state to the original state space
    #     """
    #     return x

    def compositional_drift(self, x, u, extra_args=None, return_aux=False):
        """ The drift function of the SDE.
        """
        # Extract the state and control
        z = x[0:1]
        angles = x[1:8]
        sin_angles, cos_angles = jnp.sin(angles), jnp.cos(angles)
        vels = x[8:17]

        # The scaling factors
        z_scaling = self.state_scaling[0:1]
        vel_scaling = self.state_scaling[8:17]

        # Mass Matrix
        sin_cos = jnp.concatenate([sin_angles, cos_angles])
 
        # Coriolis Matrix
        C = self.CoriolisMatrixNN(sin_cos, vels/vel_scaling)

        # Actuator forces
        tau = self.ActuatorForcesNN(sin_cos, vels/vel_scaling, u)

        # External forces
        fext = self.ExternalForcesNN(sin_cos, vels/vel_scaling, u, z / z_scaling)

        # Gravity
        g = self.GravityNN(sin_cos)

        # Compute the acceleration
        veldot = (tau + C  + fext + g)
        # jnp.dot(invM, tau + jnp.dot(C, vels/vel_scaling) - fext)
        
        # Vx is ignored cause there's no x in the state dynamics
        if not return_aux:
            return jnp.concatenate([vels[1:], veldot])
        
        return jnp.concatenate([vels[1:], veldot]), {}
    
    def init_residual_networks(self):
        """Initialize the residual and other neural networks.
        """
        NUM_VELS = 9
        NUM_ACTUATORS = 6

        # The Coriolis matrix neural network
        # CorMainMat = lambda _shape : hk.get_parameter('CorMainMat', shape=[NUM_VELS, NUM_VELS, NUM_VELS, _shape], dtype=jnp.float32, init=hk.initializers.RandomUniform(-1e-4, 1e-4))
        _act_fn = self.params['coriolis_matrix']['activation_fn']
        self.coriolis_nn = hk.nets.MLP([*self.params['coriolis_matrix']['hidden_layers'], NUM_VELS*NUM_VELS],
                                            activation = getattr(jnp, _act_fn) if hasattr(jnp, _act_fn) else getattr(jax.nn, _act_fn),
                                            w_init=hk.initializers.RandomUniform(-1e-3, 1e-3),
                                            name = 'coriolis')
        
        def CoriolisMatrixNN(sin_cos, vels):
            # Compute the Coriolis matrix
            cor_Mat = self.coriolis_nn(jnp.concatenate([sin_cos, vels])).reshape((NUM_VELS, NUM_VELS))
            return jnp.dot(cor_Mat, vels)

            # return 0.0
        self.CoriolisMatrixNN = CoriolisMatrixNN

        # The actuator forces neural network
        _act_fn = self.params['actuator_forces']['activation_fn']
        self.actuator_nn = hk.nets.MLP([*self.params['actuator_forces']['hidden_layers'], NUM_VELS*NUM_ACTUATORS],
                                            activation = getattr(jnp, _act_fn) if hasattr(jnp, _act_fn) else getattr(jax.nn, _act_fn),
                                            w_init=hk.initializers.RandomUniform(-1e-3, 1e-3),
                                            name = 'actuator')
        
        def ActuatorForcesNN(sin_cos, vels, u):
            # Compute the actuator forces
            actMat = self.actuator_nn(sin_cos).reshape((NUM_VELS, NUM_ACTUATORS))
            return jnp.dot(actMat, u)
        self.ActuatorForcesNN = ActuatorForcesNN

        # Gravity
        _act_fn = self.params['gravity']['activation_fn']
        self.gravity_nn = hk.nets.MLP([*self.params['gravity']['hidden_layers'], NUM_VELS],
                                            activation = getattr(jnp, _act_fn) if hasattr(jnp, _act_fn) else getattr(jax.nn, _act_fn),
                                            w_init=hk.initializers.RandomUniform(-1e-3, 1e-3),
                                            name = 'gravity')
        
        def GravityNN(sin_cos,):
            # Compute the gravity
            return self.gravity_nn(sin_cos) * 9.1
        self.GravityNN = GravityNN

        # The external forces neural network
        _act_fn = self.params['residual_forces']['activation_fn']
        self.residual_nn = hk.nets.MLP([*self.params['residual_forces']['hidden_layers'], NUM_VELS],
                                            activation = getattr(jnp, _act_fn) if hasattr(jnp, _act_fn) else getattr(jax.nn, _act_fn),
                                            w_init=hk.initializers.RandomUniform(-1e-3, 1e-3),
                                            name = 'res_forces')
        def ExternalForcesNN(sin_cos, vels, u, z_val):
            # Compute the external forces
            res = self.residual_nn(jnp.concatenate([sin_cos, vels, u, z_val]))
            return res
        self.ExternalForcesNN = ExternalForcesNN

#################################################################################################

def load_learned_diffusion(model_path):
    """ Load the learned diffusion from the path
        Args:
            model_path (str): The path to the learned model
            num_samples (int, optional): The number of samples to generate
    """
    current_dir = os.path.expanduser(os.path.dirname(os.path.abspath(__file__)))

    learned_params_dir = current_dir + '/learned_sde_models/' + model_path

    # Load the pickle file
    with open(learned_params_dir, 'rb') as f:
        learned_params = pickle.load(f)
    # vehicle parameters
    _model_params = learned_params['nominal']
    # SDE learned parameters -> All information are saved using numpy array to facilicate portability
    # of jax accross different devices
    # These parameters are the optimal learned parameters of the SDE
    _sde_learned = apply_fn_to_allleaf(jnp.array, np.ndarray, learned_params['sde'])
    # Create the function to compute the diffusion
    _, m_diff_fn = create_diffusion_fn(_model_params, sde_constr=HalfCheetahSDE)
    noise_prior = jnp.array(_model_params['noise_prior_params'])
    diff_terms = jnp.array(_model_params['diffusion_density_nn']['indx_noise_out'])
    def diffusion_fn(y, u):
        # We use zero control input
        # return jax.vmap(m_diff_fn, in_axes=(None, None, None, 0, None))(_sde_learned, y, u, m_rng, net)
        # return (m_diff_fn(_sde_learned, y, u)[0]/noise_prior)[diff_terms]
        return jnp.full((diff_terms.shape[0],), m_diff_fn(_sde_learned, y, u)[1])
    return diffusion_fn

def load_predictor_function(learned_params_dir, prior_dist=False, nonoise=False, modified_params ={}, 
                            return_control=False, return_time_steps=False):
    """ Create a function to sample from the prior distribution or
        to sample from the posterior distribution
        Args:
            learned_params_dir (str): Directory where the learned parameters are stored
            prior_dist (bool): If True, the function will sample from the prior distribution
            nonoise (bool): If True, the function will return a function without diffusion term
            modified_params (dict): Dictionary of parameters to modify
        Returns:
            function: Function that can be used to sample from the prior or posterior distribution
    """
    # Load the pickle file
    current_dir = os.path.expanduser(os.path.dirname(os.path.abspath(__file__)))

    learned_params_dir = current_dir + '/learned_sde_models/' + learned_params_dir
    with open(os.path.expanduser(learned_params_dir), 'rb') as f:
        learned_params = pickle.load(f)

    # vehicle parameters
    _model_params = learned_params['nominal']
    # SDE learned parameters -> All information are saved using numpy array to facilicate portability
    # of jax accross different devices
    _sde_learned = apply_fn_to_allleaf(jnp.array, np.ndarray, learned_params['sde'])

    # Update the parameters with a user-supplied dctionary of parameters
    params_model = update_params(_model_params, modified_params)

    # If prior distribution, set the diffusion to zero
    if prior_dist:
        # TODO: Remove ground effect if present
        # Remove the learned density function
        params_model.pop('diffusion_density_nn', None)

    # If no_noise
    if nonoise:
        params_model['noise_prior_params'] = [0] * len(params_model['noise_prior_params'])
    
    # Compute the timestep of the model the extract the time evolution starting t0 = 0
    time_steps = compute_timesteps(params_model)
    time_evol = np.array([0] + jnp.cumsum(time_steps).tolist())

    # Create the model
    _prior_params, m_sampling = create_sampling_fn(params_model, sde_constr=HalfCheetahSDE)

    _sde_learned = _prior_params if prior_dist else _sde_learned
    if not return_time_steps:
        return lambda *x : m_sampling(_sde_learned, *x)[0] if not return_control else m_sampling(_sde_learned, *x)
    else:
        res_fn = lambda *x : m_sampling(_sde_learned, *x)[0] if not return_control else m_sampling(_sde_learned, *x)
        return (res_fn, time_evol)
    
def extract_training_and_test_data(env_dataset_dir, ratio_test, ratio_seed, remove_test_data=False):
    """Extract the training and testing data from the dataset
    Args:
        env_dataset_dir (str): The path to the dataset
        ratio_test (float): The ratio of testing data
        ratio_seed (int): The seed to use for the random split
        remove_test_data (bool, optional): Whether to remove the test data from the training data
    Returns:
        tuple: The training and testing data
    """
    data_dirs = [env_dataset_dir]
    
    # The full training dataset
    full_data = []
    min_traj_len = 0
    for data_dir in data_dirs:
        new_data_set = get_formatted_dataset_for_nsde_training(data_dir, min_traj_len)
        for _data in new_data_set:
            if _data['y'].shape[0] >= 2:
                full_data.append(_data)
                
    # Obtain the number of testing trajectories and make sure it is always greater than 1 and less than the total number of trajectories
    num_test_traj = max(1, min(int(ratio_test * len(full_data)), len(full_data)))
    np.random.seed(ratio_seed)
    # Pick indexes for the test data
    test_idx = np.random.choice(len(full_data), num_test_traj, replace=False)
    
    # [TODO] Make sure the test data is always a copied and do not share data with the train data/ full data
    test_data = [ { k : np.array(_v) for k, _v in full_data[i].items() } for i in test_idx]

    # Remove the test data from the full data if requested
    if remove_test_data:
        train_data = [full_data[i] for i in range(len(full_data)) if i not in test_idx]
    else:
        train_data = full_data

    return train_data, test_data


def train_sde(cfg_file_name):
    """ Main function to train the SDE model.
    """
    # Get this file full path
    current_dir = os.path.expanduser(os.path.dirname(os.path.abspath(__file__)))
    
    # Load the config file
    cfg_file_name = current_dir + '/' + cfg_file_name
    cfg_train = load_yaml(cfg_file_name)

    # Obtain the path to the log data
    env_dataset = cfg_train['data_dir']

    training_datset_dir = current_dir + '/training_dataset'
    if not os.path.exists(training_datset_dir):
        os.makedirs(training_datset_dir)
    
    # Path where to save a simpler version of the dataset
    env_dataset = training_datset_dir + '/' + env_dataset + '_dataset.pkl'

    # Check if the dataset exists
    if not os.path.exists(env_dataset):
        # Extract the training and testing data
        train_data, test_data = extract_training_and_test_data(cfg_train['data_dir'], cfg_train['ratio_test'], cfg_train['ratio_seed'], cfg_train.get('remove_test_data', False))
        # Save the dataset
        with open(env_dataset, 'wb') as f:
            pickle.dump({'train_data': train_data, 'test_data': test_data}, f)
    
    # Load the dataset
    with open(env_dataset, 'rb') as f:
        dataset = pickle.load(f)
        train_data = dataset['train_data']
        test_data = dataset['test_data']

    print('\nNumber of testing trajectories: {}'.format(len(test_data)))
    print('Number of training trajectories: {}\n'.format(len(train_data)))

    # Set the dimension of the problem
    cfg_train['model']['n_x'] = NUM_STATES
    cfg_train['model']['n_u'] = NUM_ACTIONS
    cfg_train['model']['n_y'] = NUM_OBSERVATIONS
    cfg_train['model']["num_substeps"] = NUM_SUBSTEPS

    # Check if the control input match the model
    assert cfg_train['model']['n_u'] == train_data[0]['u'].shape[-1], 'The control input dimension does not match the model'
    assert cfg_train['model']['n_u'] == test_data[0]['u'].shape[-1], 'The control input dimension does not match the model'
    assert cfg_train['model']['n_y'] == train_data[0]['y'].shape[-1], 'The state dimension does not match the model'
    assert cfg_train['model']['n_y'] == test_data[0]['y'].shape[-1], 'The state dimension does not match the model'

    # Obtain the angles names
    obs_names = cfg_train['model'].get('obs_names', OBS_NAMES)
    angle_names = cfg_train['model'].get('angle_names', ANGLE_NAMES)
    angle_idx = [obs_names.index(_name) for _name in angle_names]

    print("Observation names: {}".format(obs_names))
    print("Angle names: {}".format(angle_names))
    print("Angle indexes: {}\n".format(angle_idx))

    # Check if state and gradient scaling is needed
    if cfg_train['model'].get('data_state_scaling', False):

        # Compute scales based on the training data
        _state_scaling = np.ones(train_data[0]['y'][0,:].shape)*-np.inf
        for x in tqdm(train_data):
            _state_scaling = np.maximum(np.max(np.abs(x['y']), axis=0), _state_scaling)

        # Set the angle scaling to 1
        for _idx in angle_idx:
            _state_scaling[_idx] = 1.0
        
        # Set the state scaling
        cfg_train['model']['state_scaling'] = _state_scaling
        cfg_train['sde_loss']['obs_weights'] = [float(_v) for _v in cfg_train['model']['state_scaling']]
        print('State scaling is set to {}\n'.format(cfg_train['model']['state_scaling']))
        cfg_train['model']['state_scaling'] = [float(_v) for _v in cfg_train['model']['state_scaling']]
    

    # CHeck if state_scaling is in the model
    if 'state_scaling' not in cfg_train['model']:
        cfg_train['model']['state_scaling'] = [1.0] * cfg_train['model']['n_x']
        cfg_train['sde_loss']['obs_weights'] = [1.0] * cfg_train['model']['n_x']
    print('Prior Observation weights are set to \n {}\n'.format(cfg_train['sde_loss']['obs_weights']))

    # Specific penalization terms for fitting certain states better than others (due to order of magnitude etc...)
    if 'updated_obs_weights' in cfg_train['sde_loss']:
        for _i, _v in cfg_train['sde_loss']['updated_obs_weights'].items():
            cfg_train['sde_loss']['obs_weights'][_i] *= _v
    print('Updated observation weights are set to \n {}\n'.format(cfg_train['sde_loss']['obs_weights']))
    
    # Read the output_file
    output_file = cfg_train['output_file']

    # Create the directory to save the config files used to train the model
    model_dir = current_dir + '/learned_sde_models/'
    model_config_dir = model_dir + '/config_models'
    if not os.path.exists(model_config_dir):
        os.makedirs(model_config_dir)

    output_file = output_file + "_hr-{}_dt-{}".format(
        cfg_train['sde_loss']['horizon'], cfg_train['sde_loss']['stepsize'],)
    
    # Save the config file
    dump_yaml(model_config_dir + '/' + output_file + '_config.yaml', cfg_train)
    output_file = model_dir + output_file

    # Train the model
    train_model(cfg_train, train_data, test_data, output_file, HalfCheetahSDE)