import jax
import jax.numpy as jnp

# All of functions here do not include time dependencies in the SDE
# A way to incorporate time dependency is by augmenting the state with a time
# factor. The choice is rather simple as in most dynamical systems of interest,
# the time does not appear in the dynamics.


def euler_maruyama(time_step, y0, us, rng_brownian, drift_fn, diffusion_fn, 
                                projection_fn=None, extra_scan_args=None, 
                                return_vector_field_aux=False, num_substeps=1):
    """Implement Euler-Maruyama method for Ito SDEs -> us can be a function or a sequence of control inputs
        Args:
            time_step (TYPE): The time indexes at which the integration is done
            y0 (TYPE): The initial observation of the solver
            us (TYPE): The control input given as a function of y or as an array
            rng_brownian (TYPE): A random key generator for the brownian noise
            drift_fn (TYPE): The drift function of the dynamics
            diffusion_fn (TYPE): The diffusion function of the dynamics
            projection_fn (TYPE): A projection function to project the state back or in to the manifold
            extra_scan_args (TYPE): Extra arguments to pass to the scan function
    """
    # Check of us is a function or an array
    if hasattr(us, 'ndim'):
        # If us is of dimension 1 then add an axis
        if us.ndim == 1:
            us = us[None]
        # Check the dimension properties
        assert time_step.shape[0] == us.shape[0], "Dimension mismatch on the input of the sde solver"
    
    # Check the initial observation dimensions
    assert y0.ndim == 1 and rng_brownian.ndim == 1, "Not the right dimension for the initial observation and the random key"

    # The number of steps
    num_step = time_step.shape[0]

    # Check if the number of substeps is > 1 then we need to update us, time_step and extra_scan_args
    if num_substeps > 1:
        # Repeat the control input
        us = jnp.repeat(us, num_substeps, axis=0)
        # Repeat the time step
        time_step = jnp.repeat(time_step, num_substeps, axis=0) / num_substeps
        # Repeat the extra scan args
        if extra_scan_args is not None:
            extra_scan_args = {k : jnp.repeat(v, num_substeps, axis=0) for k, v in extra_scan_args.items()}
        # Redefine the number of steps
        num_step = time_step.shape[0]
        print('NUM STEPS: ', num_step)

    # Convert the initial observation to the initial state
    x0 = y0

    # Build the brownian motion for this integration
    dw  = jax.random.normal(key=rng_brownian, shape=(num_step, x0.shape[0]))

    def euler_step(_stateObs, extra):
        """ One step euler method for Ito integrals
        """
        _dw, dt, _maybe_u, _e_args = extra
        _x = _stateObs
        sqrt_dt = jnp.sqrt(dt)
        
        # Extract the control input
        _u = us(_x) if _maybe_u is None else _maybe_u

        # Drift and diffusion at current time and current state
        drift_t = drift_fn(_x, _u, extra_args=_e_args, return_aux=return_vector_field_aux)
        diff_t = diffusion_fn(_x, _u, extra_args=_e_args)

        extra_ret =  {}
        if return_vector_field_aux:
            drift_t, aux = drift_t
            extra_ret = {**aux, 'diff' : diff_t**2}
        
        # Now the next state can be computed
        _xnext = _x + drift_t * dt + diff_t * _dw * sqrt_dt

        # Do the projection
        _xnext = projection_fn(_xnext) if projection_fn is not None else _xnext

        return _xnext, (_xnext, _u, extra_ret)
    
    # Define the carry and the scan arguments
    carry_init = x0
    xs = (dw, time_step, None if not hasattr(us, 'ndim') else us, None if extra_scan_args is None else extra_scan_args)

    # Do the scan
    _, (xevol, uevol, extra_ret) = jax.lax.scan(euler_step, carry_init, xs)
    xevol = jnp.concatenate((x0[None], xevol))

    # Reshape the output to ignore the substeps
    if num_substeps > 1:
        xevol = xevol[::num_substeps]
        uevol = uevol[::num_substeps]
        if extra_scan_args is not None:
            extra_ret = {k : v[::num_substeps] for k, v in extra_ret.items()}
        # Check size
        assert xevol.shape[0] == uevol.shape[0]+1, "Dimension mismatch on the output of the sde solver"

    if return_vector_field_aux:
        return xevol, uevol, extra_ret
    return xevol, uevol

def simpletic_euler_maruyama(time_step, y0, us, rng_brownian, drift_fn, diffusion_fn,
                                num_pos, indx_pos_vel,
                                projection_fn=None, extra_scan_args=None,
                                return_vector_field_aux=False, num_substeps=1):
    """Implement Euler-Maruyama method for Ito SDEs -> us can be a function or a sequence of control inputs
    """
    # Check of us is a function or an array
    if hasattr(us, 'ndim'):
        # If us is of dimension 1 then add an axis
        if us.ndim == 1:
            us = us[None]
        # Check the dimension properties
        assert time_step.shape[0] == us.shape[0], "Dimension mismatch on the input of the sde solver"
    
    # Check the initial observation dimensions
    assert y0.ndim == 1 and rng_brownian.ndim == 1, "Not the right dimension for the initial observation and the random key"

    # The number of steps
    num_step = time_step.shape[0]

    # Check if the number of substeps is > 1 then we need to update us, time_step and extra_scan_args
    if num_substeps > 1:
        # Repeat the control input
        us = jnp.repeat(us, num_substeps, axis=0)
        # Repeat the time step
        time_step = jnp.repeat(time_step, num_substeps, axis=0) / num_substeps
        # Repeat the extra scan args
        if extra_scan_args is not None:
            extra_scan_args = {k : jnp.repeat(v, num_substeps, axis=0) for k, v in extra_scan_args.items()}
        # Redefine the number of steps
        num_step = time_step.shape[0]
        print('NUM STEPS: ', num_step)

    # Convert the initial observation to the initial state
    x0 = y0

    # Build the brownian motion for this integration
    dw  = jax.random.normal(key=rng_brownian, shape=(num_step, x0.shape[0]))

    def euler_step(_stateObs, extra):
        """ One step euler method for Ito integrals
        """
        _dw, dt, _maybe_u, _e_args = extra
        _x = _stateObs
        sqrt_dt = jnp.sqrt(dt)
        
        # Extract the control input
        _u = us(_x) if _maybe_u is None else _maybe_u

        # Drift and diffusion at current time and current state
        drift_t = drift_fn(_x, _u, extra_args=_e_args, return_aux=return_vector_field_aux)
        diff_t = diffusion_fn(_x, _u, extra_args=_e_args)
        diff_vel = diff_t[num_pos:]
        diff_pos = diff_t[:num_pos]

        extra_ret =  {}
        if return_vector_field_aux:
            drift_t, aux = drift_t
            extra_ret = {**aux, 'diff' : diff_t**2}
        
        # Now the next state can be computed
        _xnext_vel = _x[num_pos:] + drift_t[num_pos:] * dt + diff_vel * _dw[num_pos:] * sqrt_dt
        _xnext_pos = _x[:num_pos] + dt * _xnext_vel[indx_pos_vel] + diff_pos * _dw[:num_pos] * sqrt_dt
        # _xnext = _x + drift_t * dt + diff_t * _dw
        # _xnext_pos = _x[:num_pos] + dt * drift_t[:num_pos]
        _xnext = jnp.concatenate((_xnext_pos, _xnext_vel))

        # Do the projection
        _xnext = projection_fn(_xnext) if projection_fn is not None else _xnext

        return _xnext, (_xnext, _u, extra_ret)
    
    # Define the carry and the scan arguments
    carry_init = x0
    xs = (dw, time_step, None if not hasattr(us, 'ndim') else us, None if extra_scan_args is None else extra_scan_args)

    # Do the scan
    _, (xevol, uevol, extra_ret) = jax.lax.scan(euler_step, carry_init, xs)
    xevol = jnp.concatenate((x0[None], xevol))

    # Reshape the output to ignore the substeps
    if num_substeps > 1:
        xevol = xevol[::num_substeps]
        uevol = uevol[::num_substeps]
        if extra_scan_args is not None:
            extra_ret = {k : v[::num_substeps] for k, v in extra_ret.items()}
        # Check size
        assert xevol.shape[0] == uevol.shape[0]+1, "Dimension mismatch on the output of the sde solver"

    if return_vector_field_aux:
        return xevol, uevol, extra_ret
    return xevol, uevol

# A dictionary to map string keys to sde solver functions
sde_solver_name ={
    'euler_maruyama': euler_maruyama,
    'simpletic_euler_maruyama': simpletic_euler_maruyama
}