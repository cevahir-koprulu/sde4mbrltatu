"""
Main code to load the learned neural SDEs models from configuration,
checkpoints, or yaml files.
"""

import os

from typing import Any, Dict, Tuple, Callable

from functools import partial

import jax
import jax.numpy as jnp

from nsdes_dynamics.parameter_op import (
    pretty_print_config,
    modify_entry_from_config_with_dict
)

from nsdes_dynamics.base_nsdes import (
    BaseNeuralSDE, DiffusionTerm, sample_sde
)
from nsdes_dynamics.rigid_body_dynamics_drift import (
    models_by_name,
    RBD_Drift
)
from nsdes_dynamics.distance_aware_diffusion import diffusion_terms_by_name
from nsdes_dynamics.logging_utils import load_saved_data_from_checkpoint

# Extract the path of the current file and folder containing the parameters
_file = os.path.abspath(__file__)
_path = os.path.dirname(_file)
SDE_MODELS_PATH = os.path.join(_path, "model_parameters")

def load_drift_term(
    drift_term_config: Dict[str, Any],
) -> Tuple[RBD_Drift, Dict[str, Any]]:
    """ 
    Load the drift term of the vehicle model.
    
    Args:
        drift_term_config: config dictionary of the drift model
            dict
        vehicle_params: dictionary containing the default parameters
        of the vehicle
            dict
    
    Returns:
        drift_term: the drift term of the vehicle model
            VehicleDriftTerm
        drift_term_params: the arguments used to construct an 
        instance of the drift term
            dict
    """
    model_name = drift_term_config['model_name']
    assert model_name in models_by_name, \
        f"Unknown vehicle model name: {model_name} in {models_by_name}"
    model_class = models_by_name[model_name]
    # Every fields must be used
    args_drift_term = drift_term_config.get('args', {})
    if 'reward_nn' not in args_drift_term:
        args_drift_term['reward_nn'] = {}
    drift_term = model_class(**args_drift_term)
    return drift_term, args_drift_term

def load_diffusion_term(
    diffusion_term_config: Dict[str, Any],
    extra_args: Dict[str, Any],
) -> Tuple[DiffusionTerm, Dict[str, Any]]:
    """
    Load the diffusion term of the vehicle model.
    
    Args:
        diffusion_term_config: config dictionary of the diffusion term
            dict
        
    Returns:
        diffusion_term: the diffusion term of the vehicle model
            DiffusionTerm
    """
    model_name = diffusion_term_config['model_name']
    assert model_name in diffusion_terms_by_name, \
        f"Unknown diffusion name: {model_name} in {diffusion_terms_by_name}"
    model_class = diffusion_terms_by_name[model_name]
    args_diff_term = diffusion_term_config.get('args', {})
    args_diff_term['_num_states'] = extra_args['num_states']
    args_diff_term['_num_controls'] = extra_args['num_controls']
    diffusion_term = model_class(**args_diff_term)
    return diffusion_term, args_diff_term

def initialize_system_model(
    rng_key: jax.random.PRNGKey,
    drift_term: RBD_Drift,
    diffusion_term: DiffusionTerm
) -> Dict[str, Any]:
    """
    Initialize the vehicle model.
    
    Args:
        rng_key: random number generator key
            (2,) array
        drift_term: drift term of the vehicle model
            VehicleDriftTerm
        diffusion_term: diffusion term of the vehicle model
            DiffusionTerm
    
    Returns:
        vehicle_model_params: parameters of the vehicle model
            dict
    """
    num_states = drift_term.num_states
    num_controls = drift_term.num_controls

    # The input does not really matters here, even if they are infeasible
    state_init = jax.random.normal(rng_key, (num_states,))
    control_init = jax.random.normal(rng_key, (num_controls,))

    # Drift term parameters
    drift_params  = drift_term.init(
        rng_key, state_init, control_init,
        {}
    )

    # Diffusion term parameters
    diffusion_params = {}
    if diffusion_term is not None:
        diffusion_params = diffusion_term.init(
            rng_key, state_init, control_init,
            {}
        )
    # Full set of parameters of the NSDE model
    return {'drift': drift_params, 'diffusion': diffusion_params}

def load_system_model(
    config_dict: Dict[str, Any],
    seed_init : int = 0,
    ignore_initialization: bool = False,
    modified_model_parameters: Dict[str, Any] = {},
    verbose: bool = False,
) -> Tuple[BaseNeuralSDE, Dict[str, Any]]:
    """
    Load the vehicle model from a configuration dictionary.

    Args:
        config_dict: dictionary containing the parameters to load the NSDE
        or string name representing the vehicle model or trained model name.
            dict
        seed_init: seed used to initialize the vehicle model
            int
        ignore_initialization: whether to ignore the initialization of the
        the neural networks or parameters in the vehicle model
            bool
        modified_model_parameters: modified model parameters. These are a
        set of parameters that will be used to update the default vehicle
        parameters and the parameters of the drift and diffusion terms.
            dict
        verbose: whether to print some information
            bool
            
    Returns:
        vehicle_model: the system model
            BaseNeuralSDE
        learnable_parameters: parameters of the vehicle model
            dict
    """
    # Load the drift term of the model
    drift_term_config = config_dict['drift_term']
    # print("\nUpdating the drift term config with modified model parameters")
    modify_entry_from_config_with_dict(
        drift_term_config, modified_model_parameters
    )
    drift_term, args_drift_term = load_drift_term(drift_term_config)

    # Diffusion term
    diffusion_term_config = config_dict.get('diffusion_term', None)
    diffusion_term, args_diff_term = None, {}
    extra_args_diff_term = {'num_states': drift_term.num_states,
                            'num_controls': drift_term.num_controls
                            }
    if diffusion_term_config is not None:
        # print("\nUpdating the diffusion term config with modified parameters")
        modify_entry_from_config_with_dict(
            diffusion_term_config, modified_model_parameters
        )
        diffusion_term, args_diff_term = load_diffusion_term(
            diffusion_term_config, extra_args_diff_term
        )

    # Let's initialize both models
    learnable_params = {}
    if not ignore_initialization:
        rng_key = jax.random.PRNGKey(seed_init)
        learnable_params = initialize_system_model(
            rng_key, drift_term, diffusion_term
        )

    # Load the vehicle model -> Return nothing by default
    nsde_model = BaseNeuralSDE(drift_term, diffusion_term)

    if verbose:
        print("\n")
        print("Drift term initialization arguments:")
        pretty_print_config(args_drift_term)
        print("\n")
        print("Diffusion term initialization arguments:")
        pretty_print_config(args_diff_term)
        print("\n")

    return nsde_model, learnable_params

def load_sde_model_parameters_from_ckpt(
    env_name: str,
    model_name: str,
    step: int = -2,
    verbose: bool = False
) -> Tuple[Dict[str, Any], Dict[str,Any]]:
    """
    Load the optimal parameters of the SDE and the configuration to
    load the model.
    
    Args:
        model_name: name of the model
            str
        step: The step of the checkpoint to restore. -1 means the latest,
        -2 means the best, -3 means the second best and so on, and any other
        non-negative integer means the corresponding step of the checkpoint
            int
    
    Returns:
        sde_params: parameters of the SDE model
            Dict[str, Any]
        model_cfg: configuration of the model that will be used to
        load the model
            Dict[str, Any]
    """
    config, path_ckpt = load_saved_data_from_checkpoint(
        os.path.join(SDE_MODELS_PATH, env_name),
        model_name,
        best_mode = 'min',
        step = step,
    )

    # Print the path of the loaded checkpoint
    if verbose:
        print(f"Loaded checkpoint from: {path_ckpt}")

    # Extract relevant parameters
    sde_params = config['sde_params']
    train_config = config['train_config']
    loss_train_def = train_config['loss_definitions']['loss_traj_train']
    sde_model = train_config['model']
    saved_env_name = train_config['env_name']
    assert saved_env_name == env_name, \
        f"Loaded model from {saved_env_name} but expected {env_name}"

    if verbose:
        print(f"Loaded parameters for the model\n {sde_model}\n")
        print(f"Loaded parameters for the SDE\n {sde_params}\n")

    return sde_params, \
        {"model": sde_model, "train_config" : train_config,
         "num_substeps" : loss_train_def['num_substeps'],
         "integration_method" : loss_train_def['sampling']['integration_method']
        }


def load_system_model_by_name(
    env_name: str,
    model_name: str,
    seed_init: int = 0,
    ignore_initialization: bool = False,
    modified_model_parameters: Dict[str, Any] = {},
    verbose: bool = False,
    step: int = -2
) -> Tuple[BaseNeuralSDE, Dict[str, Any]]:
    """
    Load the vehicle model by its name or the name of the learned model.
    This is a wrapper around load_vehicle_model that works for the base
    platform models and learned models.

    Args:
        env_name: name of the environment
            str
        model_name: name of the model
            str
        seed_init: seed used to initialize the vehicle model
            int
        ignore_initialization: whether to ignore the initialization of the
        the neural networks or parameters in the vehicle model
            bool
        modified_model_parameters: modified model parameters. These are a
        set of parameters that will be used to update the default vehicle
        parameters and the parameters of the drift and diffusion terms.
            dict
        verbose: whether to print some information
            bool
        step: step of the checkpoint to restore. -1 means the latest,
        -2 means the best, -3 means the second best and so on, and any other
        non-negative integer means the corresponding step of the checkpoint
            int
    
    Returns:
        vehicle_model: the vehicle model
            BaseNeuralSDE
        learnable_parameters: parameters of the vehicle model
            dict
    """
    learnable_params, temp_config = load_sde_model_parameters_from_ckpt(
        env_name, model_name, step = step, verbose = verbose
    )
    config_dict = temp_config['model']
    sde_model, _ = \
        load_system_model(config_dict, seed_init = seed_init,
            ignore_initialization = ignore_initialization,
            modified_model_parameters = modified_model_parameters,
            verbose = verbose
        )
    return sde_model, learnable_params, temp_config

def load_system_sampler_from_model_name(
    env_name: str,
    model_name: str,
    stepsize: float,
    horizon: int,
    num_particles: int,
    integration_method: str = 'euler',
    step: int = -2,
    verbose: bool = True,
    return_sde_model: bool = False
) -> Callable:
    """
    Load the system sampler from the model name.

    Args:
        env_name: name of the environment
            str
        model_name: name of the model
            str
        stepsize: stepsize of the integration
            float
        horizon: horizon of the integration
            int
        num_particles: number of particles
            int
        integration_method: method used to integrate the system
            str
        step: step of the checkpoint to restore. -1 means the latest,
        -2 means the best, -3 means the second best and so on, and any other
        non-negative integer means the corresponding step of the checkpoint
            int
        verbose: whether to print some information
            bool
    
    Returns:
        system_sampler: the system sampler
            Callable
    """
    sde_model, learnable_params, _train_config = load_system_model_by_name(
        env_name, model_name, step = step, verbose = verbose
    )
    if integration_method == "" or integration_method is None:
        # Let's extract the integration method from the config
        integration_method = _train_config['integration_method']
        print(f"Integration method: {integration_method}")
        
    model_sampler = partial(
        sample_sde,
        sde_model = sde_model,
        time_dependent_parameters = {},
        learnable_parameters = learnable_params,
        time_steps = jnp.array([ stepsize for _ in range(horizon)]),
        num_samples = num_particles,
        integration_method = integration_method,
        scan_over_learnable_parameters = False,
        num_substeps = _train_config['num_substeps']
    )
    if return_sde_model:
        return model_sampler, sde_model
    return model_sampler
        
        