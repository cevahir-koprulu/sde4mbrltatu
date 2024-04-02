"""
Utilities functions for numerically integrating deterministic and stochastic
differential equations.

All functions here do not include time dependencies in the SDE/ODEs.
A way to incorporate time dependency is by augmenting the state with a time.
"""
from typing import Callable, Any, Tuple, Dict

import jax
import jax.numpy as jnp


def euler(
    state: jnp.ndarray,
    control: jnp.ndarray,
    de_terms: Callable,
    time_steps: jnp.ndarray,
    extra_scan_args: Any,
    num_substeps: int = 1
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Euler integration scheme for ODEs
    
    Args:
        state: the state of the system.
            (n,) array
        control: the control of the system.
            (m,) or (horizon, m) array
        de_terms: function that returns the vectorfield of the ODE
            Callable (state, control, *extra_scan_args) -> (n,) array, dict
        time_steps: array of time steps for integration.
            (horizon,) array
        extra_scan_args: extra arguments to pass to the de_terms function.
            Any
        
    Returns:
        next_state: the next state of the system.
            (n,) array
        extra: additional information that might be needed for the
        integration of the ODE or other purposes.
            (str, Any) dictionary
    """
    # Format timesteps if needed
    time_steps = jnp.array(time_steps)
    if time_steps.ndim == 0:
        time_steps = time_steps[None]

    # Format control if needed and check it is of the right shape
    if control.ndim == 1:
        control = control[None]
    assert control.shape[0] == time_steps.shape[0], \
        "The control {control.shape} should have the same number of rows as " + \
        "the time steps {time_steps.shape}"

    # Check on the state dimenstions
    assert state.ndim == 1, "The state {state.shape} should be a vector"

    if num_substeps > 1:
        # Repeat the control input
        control = jnp.repeat(control, num_substeps, axis=0)
        # Repeat the time step
        time_steps = jnp.repeat(time_steps, num_substeps, axis=0) / num_substeps
        # Repeat the extra scan args
        extra_scan_args = jax.tree_map(
            lambda x: jnp.repeat(x, num_substeps, axis=0), extra_scan_args)
        print('NUM STEPS: ', time_steps.shape[0])

    def euler_step(curr_state, extra):
        """One step Euler update
        """
        dt, curr_control, vf_args = extra
        dstate, features = de_terms(curr_state, curr_control, *vf_args)
        state_next = curr_state + dt * dstate
        return state_next, (state_next, features)

    # Scan over to compute trajectories
    carry_init = state
    xs = (time_steps, control, extra_scan_args)
    _, (state_traj, extra) = jax.lax.scan(euler_step, carry_init, xs)
    xevol = jnp.concatenate((state[None], state_traj))
    if num_substeps > 1:
        xevol = xevol[::num_substeps]
        extra = jax.tree_map(lambda x: x[::num_substeps], extra)
    return xevol, extra

def heun(
    state: jnp.ndarray,
    control: jnp.ndarray,
    de_terms: Callable,
    time_steps: jnp.ndarray,
    extra_scan_args: Any
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Heun integration scheme for ODEs
    
    Args:
        state: the state of the system.
            (n,) array
        control: the control of the system.
            (m,) or (horizon, m) array
        de_terms: function that returns the vectorfield of the ODE
            Callable (state, control, *extra_scan_args) -> (n,) array, dict
        time_steps: array of time steps for integration.
            (horizon,) array
        extra_scan_args: extra arguments to pass to the de_terms function.
            Any
        
    Returns:
        next_state: the next state of the system.
            (n,) array
        extra: additional information that might be needed for the
        integration of the ODE or other purposes.
            (str, Any) dictionary
    """
    # Format timesteps if needed
    time_steps = jnp.array(time_steps)
    if time_steps.ndim == 0:
        time_steps = time_steps[None]

    # Format control if needed and check it is of the right shape
    if control.ndim == 1:
        control = control[None]
    assert control.shape[0] == time_steps.shape[0], \
        "The control {control.shape} should have the same number of rows as " + \
        "the time steps {time_steps.shape}"

    # Check on the state dimenstions
    assert state.ndim == 1, "The state {state.shape} should be a vector"

    def heun_step(curr_state, extra):
        """One step Heun update
        """
        dt, curr_control, vf_args = extra
        dstate, features = de_terms(curr_state, curr_control, *vf_args)
        dstate2, _ = de_terms(curr_state + dt * dstate, curr_control, *vf_args)
        state_next = curr_state + dt / 2 * (dstate + dstate2)
        return state_next, (state_next, features)

    # Scan over to compute trajectories
    carry_init = state
    xs = (time_steps, control, extra_scan_args)
    _, (state_traj, extra) = jax.lax.scan(heun_step, carry_init, xs)
    return jnp.concatenate((state[None], state_traj)), extra

def rk4(
    state: jnp.ndarray,
    control: jnp.ndarray,
    de_terms: Callable,
    time_steps: jnp.ndarray,
    extra_scan_args: Any
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Runge-Kutta 4 integration scheme for ODEs
    
    Args:
        state: the state of the system.
            (n,) array
        control: the control of the system.
            (m,) or (horizon, m) array
        de_terms: function that returns the vectorfield of the ODE
            Callable (state, control, *extra_scan_args) -> (n,) array, dict
        time_steps: array of time steps for integration.
            (horizon,) array
        extra_scan_args: extra arguments to pass to the de_terms function.
            Any
        
    Returns:
        next_state: the next state of the system.
            (n,) array
        extra: additional information that might be needed for the
        integration of the ODE or other purposes.
            (str, Any) dictionary
    """
    # Format timesteps if needed
    time_steps = jnp.array(time_steps)
    if time_steps.ndim == 0:
        time_steps = time_steps[None]

    # Format control if needed and check it is of the right shape
    if control.ndim == 1:
        control = control[None]
    assert control.shape[0] == time_steps.shape[0], \
        "The control {control.shape} should have the same number of rows as " + \
        "the time steps {time_steps.shape}"

    # Check on the state dimenstions
    assert state.ndim == 1, "The state {state.shape} should be a vector"

    def rk4_step(curr_state, extra):
        """One step rk4 method
        """
        dt, curr_control, vf_args = extra
        _x = curr_state
        dt_2 = dt / 2
        _k1, _features = de_terms(_x, curr_control, *vf_args)
        _k2, _ = de_terms(_x + dt_2 * _k1, curr_control, *vf_args)
        _k3, _ = de_terms(_x + dt_2 * _k2, curr_control, *vf_args)
        _k4, _ = de_terms(_x + dt * _k3, curr_control, *vf_args)
        _xnext = _x + dt / 6 * (_k1 + 2 * _k2 + 2 * _k3 + _k4)
        return _xnext, (_xnext, _features)

    # Define the carry and the scan arguments
    carry_init = state
    xs = (time_steps, control, extra_scan_args)
    # Run the scan
    _, (state_traj, extra) = jax.lax.scan(rk4_step, carry_init, xs)
    return jnp.concatenate((state[None], state_traj)), extra

def euler_maruyama(
    rng_key: jnp.ndarray,
    state: jnp.ndarray,
    control: jnp.ndarray,
    de_terms: Tuple[Callable, Callable],
    time_steps: jnp.ndarray,
    extra_scan_args: Any,
    num_substeps: int = 1
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Euler-Maruyama integration scheme for SDEs
    
    Args:
        rng_key: random number generator key.
            (2,) array
        state: the state of the system.
            (n,) array
        control: the control of the system.
            (m,) or (horizon, m) array
        de_terms: tuple of functions that return respectively
        the drift and diffusion terms of the SDE
            Tuple[Callable, Callable] 
            (state, control, *extra_scan_args) -> (n,) array, dict
        time_steps: array of time steps for integration.
            (horizon,) array
        extra_scan_args: extra arguments to pass to the de_terms functions.
            Any
        
    Returns:
        next_state: the next state of the system.
            (n,) array
        extra: additional information that might be needed for the
        integration of the SDE or other purposes.
            (str, Any) dictionary
    """
    # Format timesteps if needed
    time_steps = jnp.array(time_steps)
    if time_steps.ndim == 0:
        time_steps = time_steps[None]

    # Format control if needed and check it is of the right shape
    if control.ndim == 1:
        control = control[None]
    assert control.shape[0] == time_steps.shape[0], \
        "The control {control.shape} should have the same number of rows as " + \
        "the time steps {time_steps.shape}"

    # Check on the state dimenstions
    assert state.ndim == 1, "The state {state.shape} should be a vector"

    # Unpack the drift and diffusion terms
    drift_term, diffusion_term = de_terms

    if num_substeps > 1:
        # Repeat the control input
        control = jnp.repeat(control, num_substeps, axis=0)
        # Repeat the time step
        time_steps = jnp.repeat(time_steps, num_substeps, axis=0) / num_substeps
        # Repeat the extra scan args
        extra_scan_args = jax.tree_map(
            lambda x: jnp.repeat(x, num_substeps, axis=0), extra_scan_args)
        print('NUM STEPS: ', time_steps.shape[0])

    # Generate the brownian increments
    num_step = time_steps.shape[0]
    dw  = jax.random.normal(key=rng_key, shape=(num_step, state.shape[0]))
    dw *= jnp.sqrt(time_steps.reshape(-1, 1))

    def euler_maruyama_step(curr_state, extra):
        """One step Euler-Maruyama update
        """
        dt, curr_control, vf_args, curr_dw = extra
        drift, features = drift_term(curr_state, curr_control, *vf_args)
        diffusion, feats_diff = diffusion_term(curr_state, curr_control, *vf_args)
        features = {**features, **feats_diff}
        state_next = curr_state + drift * dt + curr_dw * diffusion # scaled already
        return state_next, (state_next, features)

    # Scan over to compute trajectories
    carry_init = state
    xs = (time_steps, control, extra_scan_args, dw)
    _, (state_traj, extra) = jax.lax.scan(euler_maruyama_step, carry_init, xs)
    xevol = jnp.concatenate((state[None], state_traj))
    if num_substeps > 1:
        xevol = xevol[::num_substeps]
        extra = jax.tree_map(lambda x: x[::num_substeps], extra)
    return xevol, extra


# List of ODEs solvers
ODE_SOLVER_LIST = {
    "euler": euler,
    "heun": heun,
    "rk4" : rk4,
}

# List of SDEs solvers
SDE_SOLVER_LIST = {
    "euler_maruyama": euler_maruyama,
}

ODE_TO_SDE_SOLVER = {
    "euler": "euler_maruyama",
    # "heun": "euler_maruyama",
    "rk4": "euler"
}

SDE_TO_ODE_SOLVER = { v : k for k, v in ODE_TO_SDE_SOLVER.items() }

def get_solver_by_name(solver_name: str, is_sde: bool) -> Callable:
    """
    Returns the solver function given its name.
    This function raises an error if the solver is not found or if 
    the solver is not compatible with the type of differential equation.

    Args:
        solver_name: name of the solver.
            str
        is_sde: whether the solver is for an SDE or an ODE.
            bool

    Returns:
        solver: the solver function.
            Callable
    """
    solver_list = SDE_SOLVER_LIST if is_sde else ODE_SOLVER_LIST
    res = solver_list.get(solver_name, None)

    if res is not None:
        return res

    where_to_look = ODE_TO_SDE_SOLVER if is_sde else SDE_TO_ODE_SOLVER
    if solver_name in where_to_look:
        print(f"Using {where_to_look[solver_name]} instead of {solver_name}")
        return solver_list[where_to_look[solver_name]]

    raise ValueError(f"The solver {solver_name} is not found in the list")
