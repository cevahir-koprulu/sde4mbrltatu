import numpy as np

import jax
import jax.numpy as jnp

import haiku as hk

from epistemic_nsde.nsde import create_sampling_fn, compute_timesteps, create_diffusion_fn
from epistemic_nsde.train_sde import train_model
from epistemic_nsde.dict_utils import load_yaml, apply_fn_to_allleaf, update_params, dump_yaml
from models.sde_models.utils_for_d4rl_mujoco import get_formatted_dataset_for_nsde_training

from tqdm import tqdm

import pickle
import os

def get_mlp_from_params(params, out_num, name):
    """ Create an MLP from the parameters
    """
    _act_fn = params['activation_fn']
    return hk.nets.MLP([*params['hidden_layers'], out_num],
                                            activation = getattr(jnp, _act_fn) if hasattr(jnp, _act_fn) else getattr(jax.nn, _act_fn),
                                            w_init=hk.initializers.RandomUniform(-1e-3, 1e-3),
                                            name = name)


def extract_training_and_test_data(env_dataset_dir, ratio_test, ratio_seed, remove_test_data=False, min_horizon=6):
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
    failed_traj = 0
    for data_dir in data_dirs:
        new_data_set = get_formatted_dataset_for_nsde_training(data_dir, min_traj_len)
        for _data in new_data_set:
            if _data['y'].shape[0] >= min_horizon:
                full_data.append(_data)
            else:
                failed_traj += 1
                
    print("Number of too short trajectories: {}".format(failed_traj))
                
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

def generic_load_predictor_function(model_class, learned_params_dir, prior_dist=False, nonoise=False, modified_params ={}, 
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
    print("Loading learned parameters from {}".format(learned_params_dir))
    with open(os.path.expanduser(learned_params_dir), 'rb') as f:
        learned_params = pickle.load(f)

    # vehicle parameters
    _model_params = learned_params['nominal']
    # SDE learned parameters -> All information are saved using numpy array to facilicate portability
    # of jax accross different devices
    _sde_learned = apply_fn_to_allleaf(jnp.array, np.ndarray, learned_params['sde'])
    # print("Loaded learned parameters from \n {}".format(_sde_learned))

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
    _prior_params, m_sampling = create_sampling_fn(params_model, sde_constr=model_class)

    _sde_learned = _prior_params if prior_dist else _sde_learned
    if not return_time_steps:
        return lambda *x : m_sampling(_sde_learned, *x)[0] if not return_control else m_sampling(_sde_learned, *x)
    else:
        res_fn = lambda *x : m_sampling(_sde_learned, *x)[0] if not return_control else m_sampling(_sde_learned, *x)
        return (res_fn, time_evol)


def generic_load_learned_diffusion(model_class, model_path):
    """ Load the learned diffusion from the path
        Args:
            model_path (str): The path to the learned model
            model_class (class): The class of the model (a child of ControlledSDE)
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
    _, m_diff_fn = create_diffusion_fn(_model_params, sde_constr=model_class)
    noise_prior = jnp.array(_model_params['noise_prior_params'])
    diff_terms = jnp.array(_model_params['diffusion_density_nn']['indx_noise_out'])
    def diffusion_fn(y, u):
        """ This function returns only the density term (real-valued) but broadcasted to the number of states
        """
        # We use zero control input
        # return jax.vmap(m_diff_fn, in_axes=(None, None, None, 0, None))(_sde_learned, y, u, m_rng, net)
        # return (m_diff_fn(_sde_learned, y, u)[0]/noise_prior)[diff_terms]
        return jnp.full((diff_terms.shape[0],), m_diff_fn(_sde_learned, y, u)[1])
    return diffusion_fn


def generic_train_sde(model_class, model_class_params, cfg_file_name):
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
    NUM_STATES, NUM_ACTIONS, NUM_OBSERVATIONS, NUM_SUBSTEPS, OBS_NAMES, ANGLE_NAMES = \
                [model_class_params[_k] for _k in ['NUM_STATES', 'NUM_ACTIONS', 'NUM_OBSERVATIONS', 'NUM_SUBSTEPS', 'OBS_NAMES', 'ANGLE_NAMES']]
    
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

        # # Set the angle scaling to 1
        # for _idx in angle_idx:
        #     _state_scaling[_idx] = 1.0
        
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

    # if cfg_train['sde_loss'].get('scale_loss', False):
    #     max_obs_weights = np.max(cfg_train['sde_loss']['obs_weights'])
    #     cfg_train['sde_loss']['obs_weights'] = [ max_obs_weights for _ in cfg_train['sde_loss']['obs_weights']]
    
    print('Updated observation weights are set to \n {}\n'.format(cfg_train['sde_loss']['obs_weights']))

    # Read the output_file
    output_file = cfg_train['output_file']

    # Create the directory to save the config files used to train the model
    model_dir = current_dir + '/learned_sde_models/'
    model_config_dir = model_dir + '/config_models'
    if not os.path.exists(model_config_dir):
        os.makedirs(model_config_dir)

    num_steps = cfg_train['sde_loss']['horizon'] * NUM_SUBSTEPS
    base_dt = cfg_train['sde_loss']['stepsize'] / NUM_SUBSTEPS
    output_file = output_file + "_hr-{}_dt-{:0.3f}".format(num_steps, base_dt)
    
    # Save the config file
    dump_yaml(model_config_dir + '/' + output_file + '_config.yaml', cfg_train)
    output_file = model_dir + output_file

    # Train the model
    train_model(cfg_train, train_data, test_data, output_file, model_class)