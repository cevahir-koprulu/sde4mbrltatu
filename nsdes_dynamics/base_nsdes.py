"""
Base class for neural SDEs dynamics with either a fixed diffusion term,
a fully parameterized diffusion term, or a distance-aware diffusion term.

This script also provides a function to sample trajectories from the SDE model.
"""
from typing import Any, Dict, Tuple, List
from functools import partial

import jax
import jax.numpy as jnp

import flax.linen as nn

from nsdes_dynamics.integration_schemes import get_solver_by_name

class DiffusionTerm(nn.Module):
    """
    Base class for the diffusion term of a neural SDE.
    nn.Module are data structures / dataclasses 
    so this class does not have any init method.
    """
    _num_states: int
    _num_controls: int
    def diffusion(
        self,
        state: jnp.ndarray,
        control: jnp.ndarray,
        time_dependent_parameters: Dict[str, Any],
        learnable_parameters: Dict[str, Any]
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Returns the diffusion term of the SDE.

        Args:
            state: the state of the system.
                (n,) array
            control: the control of the system.
                (m,) array
            time_dependent_parameters: updated model parameters that depend 
            on time or that needs to be propagated during integration.
                (str, Any) dictionary
            learnable_parameters: module parameters that are learnable.
                (str, Any) dictionary or None

        Returns:
            diffusion: the diffusion term of the SDE.
                (n, ) array
        """
        raise NotImplementedError

    @property
    def num_states(self) -> int:
        """Returns the number of state variables."""
        return self._num_states

    @property
    def num_controls(self) -> int:
        """Returns the number of control variables."""
        return self._num_controls

class DriftTerm(nn.Module):
    """
    Base class for the drift term of a neural SDE.
    nn.Module are data structures / dataclasses 
    so this class does not have any init method.
    """
    def drift(
        self,
        state: jnp.ndarray,
        control: jnp.ndarray,
        time_dependent_parameters: Dict[str, Any],
        learnable_parameters: Dict[str, Any]
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Returns the drift term of the SDE.

        Args:
            state: the state of the system.
                (n,) array
            control: the control of the system.
                (m,) array
            time_dependent_parameters: updated model parameters that depend 
            on time or that needs to be propagated during integration.
                (str, Any) dictionary
            learnable_parameters: module parameters that are learnable.
                (str, Any) dictionary or None

        Returns:
            drift: the drift term of the SDE.
                (n,) array
            extra: Additional information that might be needed for the
            integration of the SDE or other purposes.
                (str, Any) dictionary
        """
        raise NotImplementedError

    @property
    def names_states(self) -> List[str]:
        """Returns the names of the state variables."""
        raise NotImplementedError

    @property
    def names_controls(self) -> List[str]:
        """Returns the names of the control variables."""
        raise NotImplementedError

    @property
    def num_states(self) -> int:
        """Returns the number of state variables."""
        return len(self.names_states)

    @property
    def num_controls(self) -> int:
        """Returns the number of control variables."""
        return len(self.names_controls)

    def transform_states(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Transforms the state variables if needed.
        """
        raise NotImplementedError

    def transform_controls(self, control: jnp.ndarray) -> jnp.ndarray:
        """
        Transforms the control variables if needed.
        """
        raise NotImplementedError

class BaseNeuralSDE:
    """
    Base class for neural SDEs.
    """
    def __init__(
        self,
        drift_term: DriftTerm,
        diffusion_term: DiffusionTerm
    ):
        """
        Initializes the neural SDE model.

        Args:
            drift_term: the drift term of the SDE.
                DriftTerm instance
            diffusion_term: the diffusion term of the SDE.
                DiffusionTerm instance
        """
        self._drift_term = drift_term
        self._diffusion_term = diffusion_term
        self._is_sde = self._diffusion_term is not None

    @property
    def names_states(self) -> List[str]:
        """Returns the names of the state variables."""
        return self._drift_term.names_states

    @property
    def names_controls(self) -> List[str]:
        """Returns the names of the control variables."""
        return self._drift_term.names_controls

    @property
    def num_states(self) -> int:
        """Returns the number of state variables."""
        return self._drift_term.num_states

    @property
    def num_controls(self) -> int:
        """Returns the number of control variables."""
        return self._drift_term.num_controls

    def __getattr__(self, name: str) -> Any:
        """
        Returns the attribute of the class.
        This is mainly a wrapper around the drift term of the SDE.

        Args:
            name: the name of the attribute.
                str

        Returns:
            attribute: the attribute of the class.
                Any
        """
        if name != "apply":
            return getattr(self._drift_term, name)

        # TODO: Incorporate the learned parameters and the apply method if needed
        def apply_fn(learnable_parameters: Dict[str, Any], *args, **kwargs):
            drift_params = learnable_parameters['drift']
            return self._drift_term.apply(
                drift_params,
                *args,
                **kwargs
            )
        return apply_fn

    def drift_fn(
        self,
        state: jnp.ndarray,
        control: jnp.ndarray,
        time_dependent_parameters: Dict[str, Any],
        learnable_parameters: Dict[str, Any]
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Returns the drift term of the SDE.
        
        Args:
            state: the state of the system.
                (n,) array
            control: the control of the system.
                (m,) array
            time_dependent_parameters: updated model parameters that depend 
            on time or that need to be propagated during integration.
                (str, Any) dictionary
            learnable_parameters: module parameters that are learnable.
                (str, Any) dictionary
                
        Returns:
            drift: the drift term of the SDE.
                (n,) array
            extra: Additional information that might be needed for the
            integration of the SDE or other purposes.
                (str, Any) dictionary
        """
        # Extract the drift parameters from the learnable parameters
        drift_params = learnable_parameters['drift']
        dritf_val, features =  self._drift_term.drift(
            state, control, time_dependent_parameters, drift_params
        )
        return dritf_val, features

    def diffusion_fn(
        self,
        state: jnp.ndarray,
        control: jnp.ndarray,
        time_dependent_parameters: Dict[str, Any],
        learnable_parameters: Dict[str, Any]
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Returns the diffusion term of the SDE.
        
        Args:
            state: the state of the system.
                (n,) array
            control: the control of the system.
                (m,) array
            time_dependent_parameters: updated model parameters that depend 
            on time or that need to be propagated during integration.
                (str, Any) dictionary
            learnable_parameters: module parameters that are learnable.
                (str, Any) dictionary or None
                
        Returns:
            diffusion: the diffusion term of the SDE.
                (n,) array
        """
        if not self._is_sde:
            raise ValueError("The model is not an SDE.")
        # Transform the state and control if needed
        state = self._drift_term.transform_states(state)
        control = self._drift_term.transform_controls(control)
        # Extract the diffusion parameters from the learnable parameters
        diffusion_params = learnable_parameters['diffusion']
        diffusion_value, features = self._diffusion_term.diffusion(
            state, control, time_dependent_parameters, diffusion_params
        )
        return diffusion_value, features

    @property
    def is_sde(self) -> bool:
        """
        Returns True if the model is an SDE and False if an ODE.
        """
        return self._is_sde

    @property
    def drift_term(self) -> DriftTerm:
        """
        Returns the drift term of the SDE.
        """
        return self._drift_term

    @property
    def diffusion_term(self) -> DiffusionTerm:
        """
        Returns the diffusion term of the SDE.
        """
        return self._diffusion_term

def sample_sde(
    sde_model: BaseNeuralSDE,
    state: jnp.ndarray,
    control: jnp.ndarray,
    time_dependent_parameters: Dict[str, Any],
    learnable_parameters: Dict[str, Any],
    time_steps: jnp.ndarray,
    rng_key: jnp.ndarray = jax.random.PRNGKey(0), # Default value for ODEa
    num_samples: int = 1,
    scan_over_learnable_parameters: bool = False,
    integration_method: str = "euler_maruyama",
    num_substeps: int = 1
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Return sample from an SDE model.
    THis function will raise an error if the model is an ODE and a stochastic
    solver is given through the integration_method argument.
    This function will sample only a single trajectory if the model is an ODE.
    
    Args:
        sde_model: the neural SDE model.
            BaseNeuralSDE instance
        state: the state of the system.
            (n,) array
        control: the control of the system.
            (horizon, m) array
        time_dependent_parameters: updated model parameters that depend
        on time or updated  subset of model_parameters.
        The leaves of the dictionary are assumed to be of dim (horizon, ...)
            (str, Any) dictionary
        learnable_parameters: module parameters that are learnable.
        When scan_over_learnable_parameters is True, the leaves of the dictionary
        are assumed to be of dim (horizon, ...)
            (str, Any) dictionary
        time_steps: the time steps for integration.
            (horizon,) array
        rng_key: random number generator key.
            (2,) array
        num_samples: number of samples to draw.
            int
        scan_over_learnable_parameters: whether to also scan over the learnable.
        parameters or not.
            bool
        integration_method: the integration method to use.
            str
            
    Returns:
        samples: the samples from the SDE.
            (num_samples, horizon+1, n) array
        extra: Additional information that might be needed for the
        integration of the SDE or other purposes. The leaves of the dictionary
        are assumed to be of dim (num_samples, horizon, ...)
            (str, Any) dictionary
    """
    # Set the number of samples to 1 if the model is an ODE
    if not sde_model.is_sde:
        num_samples = 1

    # In case we want to scan over the learnable parameters
    if not scan_over_learnable_parameters:
        # Get the learnable parameters
        drift_fn = partial(
            sde_model.drift_fn,
            learnable_parameters=learnable_parameters
        )
        diffusion_fn = partial(
            sde_model.diffusion_fn,
            learnable_parameters=learnable_parameters
        )
        extra_scan_args = (time_dependent_parameters, )
    else:
        drift_fn = sde_model.drift_fn
        diffusion_fn = sde_model.diffusion_fn
        extra_scan_args = (time_dependent_parameters, learnable_parameters)

    # Extract the integration method
    solver = get_solver_by_name(integration_method, is_sde=sde_model.is_sde)
    de_terms = (drift_fn, diffusion_fn) if sde_model.is_sde else drift_fn

    # Reduced solver function
    reduced_solver = partial(
        solver,
        state = state,
        control = control,
        de_terms=de_terms,
        time_steps=time_steps,
        extra_scan_args=extra_scan_args,
        num_substeps=num_substeps
    )

    def single_sample_fn(key):
        """Sample a single particle wrapper.
            When the problem is an ODE, the solver does not need the key.
        """
        return reduced_solver(key) if sde_model.is_sde else reduced_solver()

    # Split the rng key
    rng_key = jax.random.split(rng_key, num_samples)
    return jax.vmap(single_sample_fn)(rng_key)
    