""" Define/Approximate the log probability of the neural SDEs trajectories.
"""

from typing import Any, Tuple, Dict, Callable

import jax
import jax.numpy as jnp

from nsdes_dynamics.base_nsdes import(
    BaseNeuralSDE,
    sample_sde,
)

def nll_sdes_with_fixed_cov(
    states: jnp.ndarray,
    sde_states: jnp.ndarray,
    noise_scale: jnp.ndarray,
    discount_factor: float = 1.0
) -> Tuple[float, jnp.ndarray]:
    """
    Compute the negative log-likelihood of the neural SDEs 
    when the noise scale is fixed. This assumes that the 
    likehood of observing a state in the dataset is a Gaussian 
    with fixed noise scale [No direct involvement of the
    diffusion term in the likelihood computation.]
    
    Args:
        states: states of the model (horizon, dim)
            jnp.ndarray
        sde_states: SDE states of the model
        (horizon, dim) or (num_particles, horizon, dim)
            jnp.ndarray
        noise_scale: noise scale of the model (dim,)
            jnp.ndarray
            
    Returns:
        nlog_likelihood: negative log-likelihood of the trajectory
            float
        discounted_diff_states: discounted difference between 
        the data states and predicted states of the SDE (dim,)
            jnp.ndarray
    """
    # Some dimension checks
    assert states.ndim == 2, \
        f"Wrong shape dimension {states.ndim}"
    assert states.shape[-2:] == sde_states.shape[-2:], \
        f"Wrong shape {states.shape} != {sde_states.shape}"
    assert noise_scale.ndim == 1 and noise_scale.shape[0] == states.shape[-1], \
        f"Wrong shape dimension {noise_scale.shape}"

    # Reshape the SDE states if it doesn't contain the number of particles
    if sde_states.ndim in (2, 3):
        sde_states = sde_states[None] \
            if sde_states.ndim == 2 else sde_states
    else:
        raise ValueError(f"Wrong shape dimension {sde_states.ndim}")

    # Difference between the states and the SDE states
    diff_states = (sde_states - states[None]) ** 2

    # Noise scale
    noise_scale_sqre = noise_scale ** 2
    # jax.debug.print("Noise scale {}", noise_scale_sqre)

    # Discount array
    discount_array = discount_factor ** jnp.arange(states.shape[0])

    discounted_diff_states = diff_states * discount_array.reshape(1, -1, 1)
    discounted_diff_states = discounted_diff_states.sum(axis=1).mean(axis=0)
    log_likelihood = 0.5 * (discounted_diff_states / noise_scale_sqre).sum() # sum
    return log_likelihood, discounted_diff_states

def nll_sdes_with_std_gaussian_approx(
    states: jnp.ndarray,
    sde_states: jnp.ndarray,
    noise_scale: jnp.ndarray,
    discount_factor: float = 1.0
) -> Tuple[float, jnp.ndarray]:
    """
    Compute the negative log-likelihood of the neural SDEs 
    when particles at each time step are approximated by 
    a Gaussian distribution with a sample-based kernel 
    density estimator.
    
    Args:
        states: states of the model (horizon, dim)
            jnp.ndarray
        sde_states: SDE states of the model (num_particles, horizon, dim)
            jnp.ndarray
        noise_scale: A priori scale to use in addition 
        to the kernel estimate (dim,)
            jnp.ndarray
        discount_factor: discount factor
            float
            
    Returns:
        log_likelihood: negative log-likelihood of the trajectory
            jnp.ndarray
        discounted_diff_states: discounted difference between
        the data states and predicted states of the SDE
            jnp.ndarray
    """
    assert states.ndim == 2, \
        f"Wrong shape dimension {states.ndim}"
    assert states.shape[-2:] == sde_states.shape[-2:], \
        f"Wrong shape {states.shape} != {sde_states.shape}"
    assert noise_scale.ndim == 1 and noise_scale.shape[0] == states.shape[-1], \
        f"Wrong shape dimension {noise_scale.shape}"
    assert sde_states.ndim == 3 and sde_states.shape[0] > 1, \
        f"Wrong shape dimension {sde_states.shape}"

    # Compute the mean and standard deviation of the SDE states
    sde_states_mean = sde_states.mean(axis=0)
    sde_states_std = sde_states.std(axis=0)
    noise_scale_sqre = (sde_states_std * noise_scale.reshape(1,-1)) ** 2

    # Compute the NLL of the SDEs
    diff_states = (sde_states_mean - states) ** 2
    nll = 0.5  * (diff_states / noise_scale_sqre) + \
        0.5 * jnp.log(2 * jnp.pi * noise_scale_sqre)

    discount_array = discount_factor ** jnp.arange(states.shape[0])
    nll_discounted = (nll * discount_array.reshape(-1, 1)).sum()
    return nll_discounted, \
        (diff_states * discount_array.reshape(-1, 1)).sum(axis=0)

def nll_sdes_with_kernel_gaussian_approx(
    states: jnp.ndarray,
    sde_states: jnp.ndarray,
    noise_scale: jnp.ndarray,
    discount_factor: float = 1.0
) -> Tuple[float, jnp.ndarray]:
    """ Compute the negative log-likehood of the neural SDEs 
    by approximating the state at each time step of the trajectory predicted by
    the SDEs with a Gaussian distribution centered around the mean particles
    and with standard deviation given by the diffusion term.
    
    Args:
        states: states of the model (horizon, dim)
            jnp.ndarray
        sde_states: SDE states of the model (num_particles, horizon, dim)
            jnp.ndarray
        noise_scale: A priori scale to use in addition 
        to the kernel estimate (dim,)
            jnp.ndarray
        discount_factor: discount factor
            float
            
    Returns:
        log_likelihood: negative log-likelihood of the trajectory
            jnp.ndarray
        discounted_diff_states: discounted difference between
        the data states and predicted states of the SDE
            jnp.ndarray
    """
    assert states.ndim == 2, \
        f"Wrong shape dimension {states.ndim}"
    assert states.shape[-2:] == sde_states.shape[-2:], \
        f"Wrong shape {states.shape} != {sde_states.shape}"
    assert noise_scale.ndim == 3 and noise_scale.shape[-1] == states.shape[-1], \
        f"Wrong shape dimension {noise_scale.shape}"

    # Compute the mean and standard deviation of the SDE states
    sde_states_mean = sde_states.mean(axis=0)
    noise_scale_mean = noise_scale.mean(axis=0) ** 2
    diff_states = (sde_states_mean - states) ** 2
    nll = (diff_states / noise_scale_mean) + jnp.log(noise_scale_mean)
    discount_arr = discount_factor ** jnp.arange(states.shape[0])
    nll_discounted = (nll * discount_arr.reshape(-1, 1)).sum()
    return nll_discounted, \
        (diff_states * discount_arr.reshape(-1, 1)).sum(axis=0)
    

def single_sequence_loss(
    sde_model : BaseNeuralSDE,
    params: Dict[str, Any],
    time_steps: jnp.ndarray,
    state: jnp.ndarray,
    control: jnp.ndarray,
    time_dependent_parameters: Dict[str, Any],
    key: jnp.ndarray,
    num_samples: int,
    integration_method: str,
    likehood_noise : jnp.ndarray,
    discount_factor: float,
    nll_type: str,
    num_substeps: int = 1,
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    """ 
    Compute the loss of a single trajectory without 
    parameter regularization
    """
    # Check the dimension
    assert state.ndim == 2 and control.ndim == 2, \
        "The state and control should have dimension 2"
    assert state.shape[0] == control.shape[0]+1, \
        "The state and control should have the same length"
    assert time_steps.ndim == 1 and time_steps.shape[0] == state.shape[0]-1, \
        f"The time steps shape {time_steps.shape} is incorrect"

    # Sample the SDE
    state_evol, extra_ret = sample_sde(
        sde_model, state[0],
        control, time_dependent_parameters,
        params, time_steps, key,
        num_samples,
        scan_over_learnable_parameters=False,
        integration_method=integration_method,
        num_substeps=num_substeps
    )

    # Compute the negative log-likelihood
    if nll_type == 'fixed_cov':
        nll_fn = nll_sdes_with_fixed_cov
    elif nll_type == 'std_gauss_approx':
        nll_fn = nll_sdes_with_std_gaussian_approx
    elif nll_type == 'gauss_approx':
        nll_fn = nll_sdes_with_kernel_gaussian_approx
        # assert state_evol.shape[0] == 1, \
        #     "The Gaussian approximation works only with a single particle"
        likehood_noise = extra_ret["diffusion_value"]
    else:
        raise ValueError(f"Unknown nll type {nll_type}")

    # Compute the loss
    nll_loss, diff_states = nll_fn(
        state[1:], state_evol[:, 1:, :], likehood_noise, discount_factor
    )
    extra_ret = {
        _k : jnp.sum(jnp.mean(_v, axis=0)) \
            for _k, _v in extra_ret.items()
    }

    # Return per state loss
    out_dict = { f'Loss-{sname}' : diff_states[idx] \
                for idx, sname in enumerate(sde_model.names_states)
                }
    out_dict["StateLoss"] = jnp.sum(diff_states)
    return nll_loss, {**out_dict, **extra_ret}

def batch_sequence_loss(
    sde_model : BaseNeuralSDE,
    params: Dict[str, Any],
    time_steps: jnp.ndarray,
    state: jnp.ndarray,
    control: jnp.ndarray,
    time_dependent_parameters: Dict[str, Any],
    key: jnp.ndarray,
    num_samples: int,
    integration_method: str,
    likehood_noise : jnp.ndarray,
    discount_factor: float,
    reg_fn : Callable,
    pen_dict : Dict[str, Any],
    nll_type: str,
    batch_handle_fn = jnp.mean,
    num_substeps: int = 1,
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    """ 
    Compute the loss of a batch of trajectories.
    """
    assert state.ndim == 3, "The state should have dimension 3"
    assert control.ndim == 3, "The control should have dimension 3"

    # Split the random key
    key = jax.random.split(key, state.shape[0])

    # Define the loss function to be vmapped
    def loss_lambda(_t, _s, _c, _tp, _rng) :
        """ Wrapper around the trajectory loss function"""
        return single_sequence_loss(
            sde_model, params, _t, _s, _c, _tp, _rng, num_samples,
            integration_method, likehood_noise, discount_factor, nll_type,
            num_substeps
        )

    # Compute the loss
    loss, out_dict = jax.vmap(loss_lambda)(
        time_steps, state, control, time_dependent_parameters, key
    )

    # Compute the mean loss over the batch
    mean_loss = batch_handle_fn(loss)
    out_dict = { k : batch_handle_fn(v) for k, v in out_dict.items() }
    out_dict['DataLoss'] = mean_loss

    # Regularization loss
    reg_val = reg_fn(params)
    out_dict['RegLoss'] = reg_val

    # Standard deviation of the SDE min and max values
    log_std_width = out_dict['log_std_width']
    extra_reg = 0.0
    if 'VarBoundLoss' in pen_dict:
        var_bound_loss = pen_dict['VarBoundLoss'] * log_std_width
        extra_reg += var_bound_loss

    # loss_terms = {'DataLoss' : mean_loss, 'RegLoss' : reg_val}
    total_loss = mean_loss * pen_dict['DataLoss'] + reg_val * pen_dict['RegLoss']
    total_loss += extra_reg
    # # jax.debug.print("Loss terms {}", pen_dict)
    # loss_arr = jnp.array([pen_dict[k] * v for k, v in loss_terms.items()])
    # total_loss = jnp.sum(loss_arr)
    out_dict['TotalLoss'] = total_loss

    return total_loss, out_dict
