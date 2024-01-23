import jax
import jax.numpy as jnp

from epistemic_nsde.nsde import ControlledSDE
from models.sde_models.utils_loading_n_training_sde_d4rl import get_mlp_from_params, generic_load_predictor_function
from models.sde_models.utils_loading_n_training_sde_d4rl import generic_train_sde, generic_load_learned_diffusion


# Let's define constants for this environment
NUM_STATES = 17
NUM_ACTIONS = 6
NUM_OBSERVATIONS = 17
NUM_VELS = 9
OBS_NAMES = ['rootz', 'rooty', 'bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot', 
             'Vrootx', 'Vrootz', 'Arooty', 'Abthigh', 'Abshin', 'Abfoot', 'Afthigh', 'Afshin', 'Affoot']
ANGLE_NAMES = ['rooty', 'bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
CONTROL_NAMES = ["Cbthigh", "Cbshin", "Cbfoot", "Cfthigh", "Cfshin", "Cffoot"]
TIMESTEP_ENV = 0.008 # skip_frames * xml_delta_t, time step in the data and in the environment
NUM_SUBSTEPS = 4 # The number of substeps to use for the integration of the SDE

class WalkerSDE(ControlledSDE):
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

        # Extract the state scaling factors
        self.state_scaling = jnp.array(self.params.get('state_scaling', [1.0] * self.n_x))
        self.max_state_scaling = jnp.max(self.state_scaling)

        # In case scaling factor is give, we also need to ensure scaling diffusion network inputs
        if 'state_scaling' in self.params:
            # Include velocity only
            self.reduced_state = lambda x : x / self.max_state_scaling
    
    def prior_diffusion(self, x, u, extra_args=None):
        # Set the prior to a constant noise as defined in the yaml file
        return jnp.array(self.params['noise_prior_params'])

    def projection_fn(self, x):
        """ A function to project the state to the original state space
        """
        return jnp.concatenate((x[:8], jnp.clip(x[8:], -10, 10)), axis=-1)

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

        # Position derivative correction
        pos_dot = self.PositionCorrectionNN(sin_cos, vels, u, z)
        
        # Vx is ignored cause there's no x in the state dynamics
        if not return_aux:
            return jnp.concatenate([pos_dot, veldot])
        
        return jnp.concatenate([pos_dot, veldot]), {}
    
    def init_residual_networks(self):
        """Initialize the residual and other neural networks.
        """
        # The Coriolis matrix neural network
        # CorMainMat = lambda _shape : hk.get_parameter('CorMainMat', shape=[NUM_VELS, NUM_VELS, NUM_VELS, _shape], dtype=jnp.float32, init=hk.initializers.RandomUniform(-1e-4, 1e-4))
        self.coriolis_nn = get_mlp_from_params(self.params['coriolis_matrix'], NUM_VELS*NUM_VELS, 'coriolis')
        def CoriolisMatrixNN(sin_cos, vels):
            # Compute the Coriolis matrix
            cor_Mat = self.coriolis_nn(jnp.concatenate([sin_cos, vels])).reshape((NUM_VELS, NUM_VELS))
            return jnp.dot(cor_Mat, vels)
        self.CoriolisMatrixNN = CoriolisMatrixNN

        # The actuator forces neural network
        self.actuator_nn = get_mlp_from_params(self.params['actuator_forces'], NUM_VELS*NUM_ACTIONS, 'actuator')
        def ActuatorForcesNN(sin_cos, vels, u):
            # Compute the actuator forces
            actMat = self.actuator_nn(sin_cos).reshape((NUM_VELS, NUM_ACTIONS))
            return jnp.dot(actMat, u)
        self.ActuatorForcesNN = ActuatorForcesNN

        # Gravity
        self.gravity_nn = get_mlp_from_params(self.params['gravity'], NUM_VELS, 'gravity')
        def GravityNN(sin_cos,):
            # Compute the gravity
            return self.gravity_nn(sin_cos) * 9.1
        self.GravityNN = GravityNN

        # The external forces neural network
        self.residual_nn = get_mlp_from_params(self.params['residual_forces'], NUM_VELS, 'res_forces')
        def ExternalForcesNN(sin_cos, vels, u, z_val):
            # Compute the external forces
            if self.params['residual_forces'].get('include_control', True):
                res = self.residual_nn(jnp.concatenate([sin_cos, vels, u, z_val]))
            else:
                res = self.residual_nn(jnp.concatenate([sin_cos, vels, z_val]))
            return res
        self.ExternalForcesNN = ExternalForcesNN

        # Add pos prediction correction term
        if 'position_correction' in self.params:
            self.position_correction_nn = get_mlp_from_params(self.params['position_correction'], NUM_STATES-NUM_VELS, 'pos_correction')
            def PositionCorrectionNN(sin_cos, vels, u, z_val):
                args_val = vels
                if self.params['position_correction'].get('include_z', False):
                    args_val = jnp.concatenate([args_val, z_val])
                if self.params['position_correction'].get('include_control', False):
                    args_val = jnp.concatenate([args_val, u])
                if self.params['position_correction'].get('include_sin_cos', False):
                    args_val = jnp.concatenate([args_val, sin_cos])
                return self.position_correction_nn(args_val)
            self.PositionCorrectionNN = PositionCorrectionNN
        else:
            self.PositionCorrectionNN = lambda sin_cos, vels, u, z_val: jnp.zeros(NUM_STATES-NUM_VELS) if 'simpletic' in self.params['sde_solver'] else vels[1:]

# Load predictor function
load_predictor_function = lambda *x, **y: generic_load_predictor_function(WalkerSDE, *x, **y)

# Train SDE
model_class_params = {
    'NUM_STATES': NUM_STATES,
    'NUM_ACTIONS': NUM_ACTIONS,
    'NUM_OBSERVATIONS': NUM_OBSERVATIONS,
    'NUM_VELS': NUM_VELS,
    'OBS_NAMES': OBS_NAMES,
    'ANGLE_NAMES': ANGLE_NAMES,
    'CONTROL_NAMES': CONTROL_NAMES,
    'TIMESTEP_ENV': TIMESTEP_ENV,
    'NUM_SUBSTEPS': NUM_SUBSTEPS,
}
train_sde = lambda *x, **y: generic_train_sde(WalkerSDE, model_class_params, *x, **y)

# Load learned diffusion
load_learned_diffusion = lambda *x, **y: generic_load_learned_diffusion(WalkerSDE, *x, **y)