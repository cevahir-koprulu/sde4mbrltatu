"""
Main code to train the neural SDEs models
"""

import os
from typing import Any, Tuple, Dict, List, Callable
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

import flax
flax.config.update('flax_return_frozendict', False)

import optax

from tqdm.auto import tqdm

from nsdes_dynamics.base_nsdes import(
    BaseNeuralSDE,
)
from nsdes_dynamics.load_learned_nsdes import (
    load_system_model,
    load_system_model_by_name,
    SDE_MODELS_PATH
)

from nsdes_dynamics.losses_trajectories import batch_sequence_loss
from nsdes_dynamics.losses_diffusion import create_dad_loss

from nsdes_dynamics.dataset_op import (
    split_dataset,
    pick_batch_transitions_as_array,
    sequential_loader_full_dataset
)

from nsdes_dynamics.parameter_op import (
    create_gaussian_regularization_loss,
    load_yaml,
    pretty_print_config
)

from nsdes_dynamics.logging_utils import (
    TrainCheckpoints
)

from nsdes_dynamics.utils_for_d4rl_mujoco import (
    get_environment_infos_from_name,
    load_dataset_for_nsdes
)

CONFIG_TRAINING_PATH = os.path.join(SDE_MODELS_PATH, "configs")

def get_train_and_test_dataset(
    dataset_config : Dict[str, Any],
    min_traj_length: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float]]:
    """ 
    Get the training and test data
    
    Args:
        dataset_config: The configuration of the dataset
            dict
    
    Returns:
        training_data: The training data
            dict
        test_data: The testing dataset
            dict
        max_fields: The maximum values for the relevant fields
            dict
    """
    # Load the full dataset
    dataset_name = dataset_config['name']
    full_dataset = load_dataset_for_nsdes(
        dataset_name,
        min_traj_len=min_traj_length,
        save_always = False
    )
    # Extract the maximum values for the all the fields
    max_fields_dict = full_dataset['max_values_per_field']
    min_fields_dict = full_dataset['min_values_per_field']
    mean_fields_dict = full_dataset['mean_values_per_field']
    median_fields_dict = full_dataset['median_values_per_field']
    print("\nMax fields :\n", max_fields_dict)
    print("\nMin fields :\n", min_fields_dict)
    print("\nMean fields :\n", mean_fields_dict)
    print("\nMedian fields :\n", median_fields_dict)
    
    # Split the dataset
    test_ratio = dataset_config['test_ratio']
    seed_split = dataset_config['seed']
    train_data, test_data = split_dataset(
        full_dataset, test_ratio, seed_split
    )
    return train_data, test_data, mean_fields_dict

def setup_system_dataset_and_nsde(
    config: Dict[str, Any],
    seed: int,
    ckpt_name: str = "",
):
    """ 
    Setup the dataset and the neural SDE model
    
    Args:
        config: The configuration of the training
            dict
        seed: The seed for the random number generator
            int
        ckpt_name: The name of the checkpoint
    
    Returns:
        BaseNeuralSDE: The neural SDE model
        Dict[str, Any]: Model parameters
        Dict[str, Any]: The training data
        Dict[str, Any]: The test data
    """
    # Extract the evnvironment name
    env_name = config["dataset"]["name"]
    config["env_name"] = env_name
    # Load the environment infos
    env_infos = get_environment_infos_from_name(env_name)
    names_states, names_controls = \
        env_infos['names_states'], env_infos['names_controls']
    sde_config = config['model']
    time_steps = env_infos["stepsize"]

    diff_model_name = sde_config.get('diffusion_term', {}).get('model_name', '')
    is_diff_distance_aware = 'DistanceAwareDiffusion' in diff_model_name

    # There's no feature parameters to use
    if is_diff_distance_aware:
        diff_args = sde_config['diffusion_term']['args']
        diff_args["feature_parameters_to_use"] = []
        diff_args["default_feature_values"] = []

    # Get the minimum length of a trajectory from the sampling config
    sampling_config = config['loss_definitions']['loss_traj_train']["sampling"]
    min_traj_len = \
        sampling_config["stepsize_range"][1] * (sampling_config["horizon"] + 1)
    print("\nMinimum trajectory length: ", min_traj_len)

    # Load the dataset, while keeping only relevant fields
    dataset_config = config['dataset']
    train_data, test_data, max_fields = get_train_and_test_dataset(
        dataset_config, min_traj_len
    )
    print("\nScaling Factor to Use :\n", max_fields)
    print("\n")

    # Let's extract the mean values for the relevant fields and scale
    mean_data_fields = train_data['mean_data_fields']
    scale_data_fields = train_data['scale_data_fields']
    percentile_data_fields = train_data['95th_percentile_data_fields']
    print("\nMean Data Fields :\n", mean_data_fields)
    print("\nScale Data Fields :\n", scale_data_fields)
    print("\n95th Percentile Data Fields :\n", percentile_data_fields)
    _mean_states = np.array([mean_data_fields[k] for k in names_states])
    _scale_states = np.array([scale_data_fields[k] for k in names_states])
    _mean_controls = np.array([mean_data_fields[k] for k in names_controls])
    _scale_controls = np.array([scale_data_fields[k] for k in names_controls])
    percentile_states = np.array([percentile_data_fields[k] for k in names_states])


    if is_diff_distance_aware:
        # Extract the scaling factor for density NN.
        u_dependent = diff_args['diffusion_is_control_dependent']
        fields_to_scale = names_states + (names_controls if u_dependent else [])
        scaling_factor = np.array([max_fields[f] for f in fields_to_scale])
        if dataset_config.get("normalize_data", True):
            diff_args['feature_density_scaling'] = jnp.ones(scaling_factor.shape)
        else:
            diff_args['feature_density_scaling'] = scaling_factor
        if "upper_bound_diffusion_scale" in diff_args:
            max_stds = diff_args.pop("upper_bound_diffusion_scale") * \
                percentile_states
            # Include the stepsize
            upper_bound_diffusion = max_stds / np.sqrt(time_steps)
            diff_args["upper_bound_diffusion"] = upper_bound_diffusion
            print ("\nUpper Bound Diffusion : \n", upper_bound_diffusion)            

    # Let's set the names of the states and controls for the model loader
    args_drift = sde_config['drift_term']['args']
    args_drift['_names_states'] = names_states
    args_drift['_names_controls'] = names_controls
    args_drift['_names_positions'] = env_infos['names_positions']
    args_drift["_names_angles"] = env_infos['names_angles']
    if dataset_config.get("normalize_data", True):
        args_drift["_mean_states"] = _mean_states
        args_drift["_scale_states"] = _scale_states
        args_drift["_mean_controls"] = _mean_controls
        args_drift["_scale_controls"] = _scale_controls
    # Let's fill up missing terms
    required_attributes = \
        ["residual_forces_nn", "coriolis_forces_nn", "gravity_forces_nn",
         "actuator_forces_nn", "mass_matrix_nn"
        ]
    for attr in required_attributes:
        if attr not in args_drift:
            args_drift[attr] = {}
        else:
            if "features" not in args_drift[attr]:
                args_drift[attr]["features"] = \
                    ["positions", "velocities", "controls"]
            else:
                assert len(args_drift[attr]["features"]) > 0,\
                    "features should not be empty"

    # Create the neural SDE model
    if ckpt_name != "" and ckpt_name is not None:
        sde_model, sde_params, _ = \
            load_system_model_by_name(env_name, ckpt_name, step=-1)
    else:                
        sde_model, sde_params = load_system_model(
            sde_config, seed_init=seed, verbose=True
        )

    # Update the configuration with the relevant fields
    config['extra_infos'] = {
        'max_fields': max_fields,
        'is_diff_distance_aware': is_diff_distance_aware,
        'names_states': sde_model.names_states,
        'names_controls': sde_model.names_controls,
        'names_positions': env_infos['names_positions'],
    }
    return sde_model, sde_params, train_data, test_data

def fill_gaps_in_config(
    sde_model : BaseNeuralSDE,
    config : Dict[str, Any]
) -> Dict[str, Any]:
    """ Fill in some of the gaps in the configuration dictionary
    that are specific to the neural SDE model
    
    Args:
        sde_params: The parameters of the neural SDE model
            dict
        config: The configuration dictionary
            dict
    """
    # Extract the default model/vehicle parameters
    name_states = sde_model.names_states
    # Start with the regularization loss
    loss_config = config['loss_definitions']

    # Update the likehood noise scale
    likehood_config = {
        'noise_scale': np.ones(len(name_states)),
        'discount_factor': \
            loss_config['loss_traj_train'].get('discount_factor', 1.0),
        'nll_type': 'gauss_approx'
    }
    loss_config['loss_traj_train']['likehood'] = likehood_config
    # likehood_config = loss_config['loss_traj_train']['likehood']
    # use_data_scaling = likehood_config['use_data_scaling']
    # specials_scale = likehood_config.get('specials', {})
    # for state_n in specials_scale:
    #     assert state_n in name_states, \
    #         f"State {state_n} not found in {name_states}"
    # scaling_factor = np.array(
    #     [specials_scale.get(k, 1.0) for k in name_states]
    # )
    # if use_data_scaling:
    #     # Extract the maximum values for the relevant fields
    #     max_fields = config['extra_infos']['max_fields']
    #     state_max = np.array([max_fields[k] for k in name_states])
    #     scaling_factor = scaling_factor * state_max
    # likehood_config['noise_scale'] = scaling_factor
    # pretty_print_config(likehood_config)

    # Update the sampling strategy
    train_sampling = loss_config['loss_traj_train']['sampling']
    if "action_sampling_strategy" not in train_sampling:
        train_sampling["action_sampling_strategy"] = {"default": "first"}
    test_sampling = loss_config['loss_traj_train']['validation_sampling']
    if "action_sampling_strategy" not in test_sampling:
        test_sampling["action_sampling_strategy"] = {"default": "first"}

    # Default loss weights
    default_weights = {
        "RegLoss": 0.0001,
        "DataLoss": 1.0
    }
    loss_weights = loss_config.get("loss_weights", default_weights)
    for k in default_weights:
        if k not in loss_weights:
            loss_weights[k] = default_weights[k]
    loss_config["loss_weights"] = loss_weights
    print("\nLoss Weights :")
    pretty_print_config(loss_weights)

    # Default regularization loss
    default_loss_reg = {
        "default_mean" : 0.0,
        "default_scale" : 100.0,
        "specials": {}
    }
    loss_reg = loss_config.get("loss_reg", default_loss_reg)
    for k in default_loss_reg:
        if k not in loss_reg:
            loss_reg[k] = default_loss_reg[k]
        if k == "specials":
            for name in loss_reg[k]:
                if "mean" not in loss_reg[k][name]:
                    loss_reg[k][name]["mean"] = 0.0
                if "scale" not in loss_reg[k][name]:
                    loss_reg[k][name]["scale"] = 100.0
    loss_config["loss_reg"] = loss_reg
    print("\nRegularization Loss :")
    pretty_print_config(loss_reg)

    # Diffusion loss defaults
    if "loss_diffusion" not in loss_config:
        return

    # Default diffusion loss terms
    loss_diffusion = loss_config["loss_diffusion"]
    loss_diffusion["diff_loss_config"]["cvx_coeff_loss_type"] = "quad_inv"
    loss_diffusion["diff_loss_config"]["min_grad_density"] = 1.0e-3
    # loss_diffusion["diff_loss_config"]["weight_diff_loss"]["gradient_loss"] = 0
    loss_diffusion["diff_loss_config"]["weight_min_grad_density"]= 1000.0
    if "cvx_coeff_config" not in loss_diffusion:
        loss_diffusion["cvx_coeff_config"] = {
            "is_cvx_coeff_learned" : True,
            "cvx_coeff_params" : {
                "is_constant" : True,
                "coef_init" : 100.0
            }
        }
    print("\nDiffusion Loss :")
    pretty_print_config(loss_diffusion)

def build_optimizer(
    opt_config : Dict[str, Any]
) -> optax.GradientTransformation:
    """ Build the optimizer from a configuration dictionary
    
    Args:
        opt_config: The configuration of the optimizer
            dict
    
    Returns:
        optax.GradientTransformation: The optimizer
    """
    chain_list = []
    for elem in opt_config:
        name_elem = elem['name']
        m_fn = getattr(optax, name_elem)
        m_params = elem.get('params', {})
        print(f'Function : {name_elem} | params : {m_params}')
        if elem.get('scheduler', False):
            m_params = m_fn(**m_params)
            chain_list.append(optax.scale_by_schedule(m_params))
        else:
            chain_list.append(m_fn(**m_params))
    # Return the optimizer
    return optax.chain(*chain_list)

@partial(jax.jit, static_argnames=("loss_fn", "opt"))
def map_gradient_update_params(
    params : Dict[str, Any],
    opt_state : Any,
    state : jnp.ndarray,
    control : jnp.ndarray,
    time_dependent_parameters : Dict[str, Any],
    rng_key : jnp.ndarray,
    extra_params : Dict[str, Any],
    loss_fn : Callable,
    opt : optax.GradientTransformation
) -> Tuple[Dict[str, Any], Any, Dict[str, jnp.ndarray]]:
    """ Compute the gradient and update the parameters
    
    Args:
        params: The parameters of the neural SDE model
            dict
        opt_state: The state of the optimizer
            Any
        state: The state of the system
            (n,) array
        control: The control of the system
            (m,) array
        time_dependent_parameters: The time dependent parameters
            dict
        rng_key: The random number generator key
            (n,) array
        loss_fn: The loss function
            callable
        opt: The optimizer
            optax.GradientTransformation
    
    Returns:
        Dict[str, Any]: The updated parameters
        Any: The updated optimizer state
        Dict[str, jnp.ndarray]: The feature values
    """
    # By default only differentiate with respect to params
    grads, featvals = jax.grad(loss_fn, has_aux=True)(
        params, state, control, time_dependent_parameters,
        rng_key, extra_params
    )
    # jax.debug.print("Grads : \n {}", grads)
    # jax.debug.print("Drift params : \n {}", params['drift'])
    # jax.debug.print("state : \n {}", state)
    # jax.debug.print("control : \n {}", control)
    # jax.debug.print("time_dependent_parameters : \n {}", time_dependent_parameters)
    # jax.debug.print("feat vals : \n {}", featvals)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, featvals


def train_general_nsdes(
    sde_model : BaseNeuralSDE,
    sde_params : Dict[str, Any],
    train_data : Dict[str, Any],
    test_data : Dict[str, Any],
    config : Dict[str, Any],
    seed : int,
    output_name : str
):
    """ Train the neural SDE model
    
    Args:
        sde_model: The neural SDE model
            BaseNeuralSDE
        train_data: The training data
            dict
        test_data: The test data
            dict
        config: The dictionary to extract training parameters
            dict
        seed: The seed for the random number generator
            int
        output_name: The name of the output
            str
    """
    # Random number generator
    rng_key = jax.random.PRNGKey(seed)
    np.random.seed(seed)
    config['seed'] = seed

    # Extract the loss definitions
    loss_def = config['loss_definitions']

    # Create the regularization loss
    # reg_loss, reg_dict = create_gaussian_regularization_loss(
    #     sde_params['drift'], loss_def['loss_reg']
    # )
    reg_loss, reg_dict = create_gaussian_regularization_loss(
        sde_params, loss_def['loss_reg']
    )
    print("\n")
    print(f"Regularization dictionary:\n {reg_dict}")

    # Extract the loss weights
    loss_weights = loss_def['loss_weights']

    # Parameters for the training loss
    likehood_config = loss_def['loss_traj_train']['likehood']
    noise_scale = likehood_config['noise_scale']
    discount_factor = likehood_config['discount_factor']
    nll_type = likehood_config['nll_type']
    if not sde_model.is_sde:
        assert nll_type != 'gauss_approx', \
            "Gaussian approximation Loss not supported for ODEs"

    num_substeps = loss_def['loss_traj_train']['num_substeps']
    sampling_config = loss_def['loss_traj_train']['sampling']
    integration_method = sampling_config['integration_method']
    num_samples = sampling_config['num_samples']
    train_horizon = sampling_config['horizon']
    train_stepsize_range = sampling_config['stepsize_range']
    action_sampling_strategy = sampling_config['action_sampling_strategy']

    # Parameters for the diffusion loss
    is_distance_aware = config['extra_infos']['is_diff_distance_aware']
    loss_diffusion_config = loss_def['loss_diffusion']
    if is_distance_aware:
        rng_key, key = jax.random.split(rng_key)
        dad_loss_fn, dad_loss_params = create_dad_loss(
            sde_model.diffusion_term,
            key,
            loss_diffusion_config['diff_loss_config'],
            loss_diffusion_config['cvx_coeff_config']
        )
        if len(dad_loss_params) > 0:
            sde_params = {**sde_params, **dad_loss_params}
            print("\n")
            print(f"DAD MU Coeff parameters : \n{dad_loss_params}")

    @jax.jit
    def train_dad_loss(
        params : Dict[str, Any],
        state: jnp.ndarray,
        control: jnp.ndarray,
        time_dependent_parameters: Dict[str, Any],
        key: jnp.ndarray,
        extra_params : Dict[str, Any]= {}
    ) -> Tuple[float, Dict[str, jnp.ndarray]]:
        """ Compute the loss of the distance-aware diffusion model
        """
        state = state[:dad_batch_size, :, :-1]
        control = control[:dad_batch_size]
        state_dad = sde_model.transform_states(state[:,0])
        control_dad = sde_model.transform_controls(control[:,0])
        time_dependent_parameters_dad = jax.tree_util.tree_map(
            lambda x: x[:dad_batch_size, :, 0], time_dependent_parameters
        )
        # We need to update the full parameters with the current parameters
        full_dad_params = {
            "diffusion" : { "params" : params["diffusion"]},
            "mu_coeff_nn": params.get("mu_coeff_nn", {})
        }
        
        total_diff_loss, dad_out_dict = dad_loss_fn(
            full_dad_params, state_dad, control_dad,
            time_dependent_parameters_dad,
            key
        )
        return total_diff_loss, dad_out_dict
    
    @jax.jit
    def train_loss(
        params : Dict[str, Any],
        state: jnp.ndarray,
        control: jnp.ndarray,
        time_dependent_parameters: Dict[str, Any],
        key: jnp.ndarray,
        dad_params : Dict[str, Any]
    ) -> Tuple[float, Dict[str, jnp.ndarray]]:
        """ Compute the loss of the training data
        """
        # Extract the time from the state
        time_val = state[..., -1]
        # print("TIME SHAPE : ", time_val.shape, state.shape)
        time_steps = time_val[..., 1:] - time_val[..., :-1]
        print("Time steps : ", time_steps.shape, state.shape)
        state = state[..., :-1]
        # jax.debug.print("Time steps : {}", time_steps)

        # # Compute the diffusion loss
        # dad_out_dict = {}
        # total_diff_loss = 0.0
        # if is_distance_aware:
        #     key, key_dad = jax.random.split(key)
        #     state_dad = sde_model.transform_states(state[:,0])
        #     control_dad = sde_model.transform_controls(control[:,0])
        #     time_dependent_parameters_dad = jax.tree_util.tree_map(
        #         lambda x: x[:,0], time_dependent_parameters
        #     )
        #     total_diff_loss, dad_out_dict = dad_loss_fn(
        #         params, state_dad, control_dad,
        #         time_dependent_parameters_dad,
        #         key_dad
        #     )
        full_params = {
            "drift": params["drift"],
            "diffusion": {
                "params": {
                    **dad_params['diffusion'],
                    **params['diffusion'],
                }
            }
        }

        # Compute the loss
        traj_loss, out_dict = batch_sequence_loss(
            sde_model, full_params, time_steps, state, control,
            time_dependent_parameters, key, num_samples, integration_method,
            noise_scale, discount_factor, reg_loss, loss_weights, nll_type,
            num_substeps=num_substeps
        )
        # total_loss = traj_loss + total_diff_loss * loss_weights['DiffLoss']
        total_loss = traj_loss
        out_dict['TotalLoss'] = total_loss
        # return total_loss, {**out_dict, **dad_out_dict}
        return total_loss, out_dict

    # Split the parameters
    density_related_names = ["density_nn",]
    dad_sde_params = {
        "diffusion": {
            k : v for k, v in sde_params["diffusion"]["params"].items() \
                if k in density_related_names
        },
        "mu_coeff_nn": sde_params.get("mu_coeff_nn", {})
    }
    dad_free_sde_params = {
        "drift": sde_params["drift"],
        "diffusion": {
            k : v for k, v in sde_params["diffusion"]["params"].items() \
                if k not in density_related_names
        },
    }
    # Let's print the parameters with the values of each key
    print("\n")
    print("DAD SDE Parameters :")
    pretty_print_config(jax.tree_map(lambda x: list(x.shape), dad_sde_params))
    print("\n")
    print("DAD Free SDE Parameters :")
    pretty_print_config(jax.tree_map(lambda x: list(x.shape), dad_free_sde_params))
    print("\n")
    
    
    # Define the optimizer
    print("\n")
    opt = build_optimizer(config['model_optimizer'])
    opt_state = opt.init(dad_free_sde_params)
    print("\n")

    # optimizer for the dad loss
    if is_distance_aware:
        print("\n")
        dad_opt = build_optimizer(config['dad_optimizer'])
        dad_opt_state = dad_opt.init(dad_sde_params)
        print("\n")

    # Define the problem configuration dictionary used to extract
    # from the dataset structured information for the training
    tdep_params = []
    problem_config_for_dataset_extraction = {
        'names_states': sde_model.names_states + ['time'], 
        'names_controls': sde_model.names_controls,
        'time_dependent_parameters': tdep_params,
    }
    print("Problem config for dataset extraction : \n",
          problem_config_for_dataset_extraction)
    print("\n")

    model_training_config = config['model_training']
    train_batch = model_training_config['train_batch']
    test_batch = model_training_config['test_batch']
    test_freq = model_training_config['test_freq']
    save_freq = model_training_config['save_freq']
    num_gradient_steps = model_training_config['num_gradient_steps']
    dad_batch_size = model_training_config.get('dad_batch_size', 128)
    freq_update_dad = model_training_config.get('freq_update_dad', 1)

    def evaluation_metrics(
        params : Dict[str, Any],
        key : jnp.ndarray,
        test_data : Dict[str, Any]
    ) -> Dict[str, float]:
        """ Compute the evaluation metrics on the testing dataset
        """
        # COnfiguration for the validation dataset
        validation_cfg = loss_def['loss_traj_train'].get(
            'validation_sampling', sampling_config
        )
        horizon_test = validation_cfg['horizon']
        stepsize_range_test = validation_cfg['stepsize_range']
        sampling_strategy_test = validation_cfg['action_sampling_strategy']

        # Iterate through the test data
        res = {}
        num_test_batches = model_training_config['test_num_steps']
        for n_iter in tqdm(range(num_test_batches), leave=False):
            # Get the current batch of training data
            test_batch_data = pick_batch_transitions_as_array(
                test_data, test_batch, stepsize_range_test,
                horizon_test, problem_config_for_dataset_extraction,
                sampling_strategy_test
            )
            # Compute the loss
            key, key_loss = jax.random.split(key)
            _, test_feat_dict = train_loss(
                params, *test_batch_data, key_loss,
                dad_params = dad_sde_params
            )
            if len(res) == 0:
                res = { _key : np.zeros(num_test_batches) \
                        for _key in test_feat_dict
                    }
            for _key, val in test_feat_dict.items():
                res[_key][n_iter] = val

        # Return the average of the results
        return { k : np.mean(v) for k, v in res.items() }

    # Create the checkpoint manager
    ckpt_model = TrainCheckpoints(
        os.path.join(SDE_MODELS_PATH, config["env_name"]),
        output_name,
        config['track_n_checkpoints'],
        best_mode = 'min',
        writer_on = True,
        extra_config_to_save_as_yaml=config,
        saving_freq = save_freq
    )

    # Main training loop
    count_nan_failur = 0
    grad_step = 0
    # best_dad_params = dad_sde_params
    # best_dad_loss = np.inf
    # epochs_since_best = 0
    load_batch, traj_indexes, num_batches = \
        sequential_loader_full_dataset(
            train_data, train_batch, train_stepsize_range,
            train_horizon, problem_config_for_dataset_extraction,
            action_sampling_strategy
        )

    for curr_epoch in tqdm(range(num_gradient_steps)):

        # # Get the current batch of training data
        # train_batch_data = pick_batch_transitions_as_array(
        #     train_data, train_batch, train_stepsize_range,
        #     train_horizon, problem_config_for_dataset_extraction,
        #     action_sampling_strategy
        # )
        # shuffle traj_indexes
        np.random.shuffle(traj_indexes)
        for n_batch in tqdm(range(num_batches), leave=False):
            train_batch_data = load_batch(traj_indexes, n_batch)
            # Is it time to evaluate the model?
            eval_metrics = {}
            if grad_step % test_freq == 0:
                # tqdm.write(f"Model parameters: {sde_params}")
                rng_key, rng_key_test = jax.random.split(rng_key)
                eval_metrics = evaluation_metrics(
                    dad_free_sde_params, rng_key_test, test_data
                )
                eval_metrics = {f"Test/{k}" : v for k, v in eval_metrics.items()}

            # Apply gradient updates
            rng_key, key = jax.random.split(rng_key)
            temp_sde_params, temp_opt_state, train_metrics = \
                map_gradient_update_params(
                    dad_free_sde_params, opt_state, *train_batch_data, key, 
                    dad_sde_params, train_loss, opt
                )
            train_metrics = {f"Train/{k}" : np.array(v) \
                                for k, v in train_metrics.items()
                            }

            # Check if there is any NaN in the pytree parameters
            flatten_params = jax.tree_util.tree_flatten(temp_sde_params)[0]
            if np.any([np.any(np.isnan(v)) for v in flatten_params]):
                tqdm.write("NaN detected in the parameters")
                tqdm.write(f"{train_metrics}")
                count_nan_failur += 1
                if count_nan_failur >= 10:
                    tqdm.write("Stopping the training")
                    break
                continue
            else:
                count_nan_failur = 0
                dad_free_sde_params = temp_sde_params
                opt_state = temp_opt_state
                grad_step += 1

            dad_metrics = {}
            if grad_step % freq_update_dad == 0 and is_distance_aware:
                # Let's update the dad loss
                rng_key, key = jax.random.split(rng_key)
                dad_sde_params, dad_opt_state, dad_metrics = \
                    map_gradient_update_params(
                        dad_sde_params, dad_opt_state, *train_batch_data, key,
                        {}, train_dad_loss, dad_opt
                    )
                # # Extract the total loss
                # dad_loss = dad_metrics['TotalDiffLoss']
                # if dad_loss < best_dad_loss:
                #     best_dad_loss = dad_loss
                #     best_dad_params = dad_sde_params
                # # dad_sde_params = _dad_sde_params

                dad_metrics = {f"DAD/{k}" : np.array(v) \
                                for k, v in dad_metrics.items()}

            # Metrics for checkpoint best
            metrics_save = {**eval_metrics, **train_metrics, **dad_metrics}
            full_sde_params = {
                "drift": dad_free_sde_params["drift"],
                "diffusion": {
                    "params": {
                        **dad_sde_params["diffusion"],
                        **dad_free_sde_params["diffusion"]
                    }
                }
            }
            # Dictionary to save
            save_dict = {
                "sde_params": full_sde_params,
                "sde_params_full": \
                    {**full_sde_params,
                     "mu_coeff_nn" : dad_sde_params.get("mu_coeff_nn", {})
                    },
                "train_config": config,
                "current_step": grad_step,
                "test_metrics": eval_metrics,
                "train_metrics": train_metrics
            }

            # Write the checkpoint
            ckpt_model.write_checkpoint_and_log_data(save_dict, metrics_save)

        # Check if we need to stop the training
        best_step = ckpt_model.get_best_step()
        best_step_epochs = best_step // num_batches
        if (curr_epoch - best_step_epochs) > model_training_config.get('early_stopping_epochs', -1):
            tqdm.write("Early stopping")
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description = "Script to train neural SDEs/ODEs models",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str, required=True,
        help="The name of the configuration file. Assumed to be in training_configs/"
    )
    parser.add_argument(
        "--seed",
        type=int, default=10,
        help="The seed for the random number generator"
    )
    parser.add_argument(
        "--output_name",
        type=str, required=True,
        help="The name of the file to save the output"
    )
    parser.add_argument(
        "--ckpt_name",
        type=str, default="",
        help="The name of the checkpoint to load"
    )

    EXAMPLE_CMD_LINE = """
    Example command line:
    
    - Train the NSDE model given the configuration in 
        nsdes_dynamics/model_parameters/nsde_config.yaml and save the output in 
        nsdes_dynamics/model_parameters/#nv_name#/output_name
    
    python train_nsde.py --config nsde_config --seed 10 --output_name test
    """
    parser.epilog = EXAMPLE_CMD_LINE
    args = parser.parse_args()

    # Parse the config yaml file
    config_yaml = args.config.split(".yaml")[0] + ".yaml"
    config_yaml = os.path.join(CONFIG_TRAINING_PATH, config_yaml)
    config_dict = load_yaml(config_yaml)

    # Setup the model
    _sde_model, _sde_params, _train_data, _test_data = \
        setup_system_dataset_and_nsde(config_dict, args.seed, args.ckpt_name)

    # Fill in the gaps in the configuration
    fill_gaps_in_config(_sde_model, config_dict)

    # Train the model
    train_general_nsdes(
        _sde_model, _sde_params, _train_data, _test_data, config_dict,
        args.seed, args.output_name
    )