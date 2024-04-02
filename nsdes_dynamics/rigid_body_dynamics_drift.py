"""
Drift term for the SDE that encodes prior knowledge from the rigid body dynamics
equations of motion.
"""

from typing import List, Dict, Any, Tuple, Union

import jax
import jax.numpy as jnp
import flax.linen as nn

from nsdes_dynamics.base_nsdes import (
    DriftTerm,
)

from nsdes_dynamics.networks import (
    load_network_from_config,
)

class RBD_Drift(DriftTerm):
    """
    Drift term for the SDE that encodes prior knowledge from the rigid body dynamics
    equations of motion.
    """
    residual_forces_nn: Dict[str, Any]
    coriolis_forces_nn: Dict[str, Any]
    gravity_forces_nn: Dict[str, Any]
    actuator_forces_nn: Dict[str, Any]
    mass_matrix_nn: Dict[str, Any]
    _names_states: List[str]
    _names_controls: List[str]
    _names_positions: List[str]
    _names_angles: List[str]
    _mean_states: Union[jnp.ndarray, None] = None
    _scale_states: Union[jnp.ndarray, None] = None
    _mean_controls: Union[jnp.ndarray, None] = None
    _scale_controls: Union[jnp.ndarray, None] = None
    include_pos_to_vel_relation: bool = True

    @property
    def names_states(self) -> List[str]:
        """Returns the names of the state variables."""
        return self._names_states

    @property
    def names_controls(self) -> List[str]:
        """Returns the names of the control variables."""
        return self._names_controls

    @property
    def names_positions(self) -> List[str]:
        """Returns the names of the position variables."""
        return self._names_positions

    @property
    def names_velocities(self) -> List[str]:
        """Returns the names of the velocity variables."""
        return self._names_states[self.num_positions:]

    @property
    def num_positions(self) -> int:
        """Returns the number of position variables."""
        return len(self.names_positions)

    @property
    def num_velocities(self) -> int:
        """Returns the number of velocity variables."""
        return len(self.names_velocities)

    @property
    def angles_indexes(self) -> jnp.ndarray:
        """Returns the indexes of the angle variables."""
        return jnp.array([
            self.names_states.index(name) for name in self._names_angles
        ])

    @property
    def mean_states(self) -> jnp.ndarray:
        """Returns the mean of the states."""
        if self._mean_states is None:
            return jnp.zeros(self.num_states)
        return jnp.array(self._mean_states)

    @property
    def mean_controls(self) -> jnp.ndarray:
        """Returns the mean of the controls."""
        if self._mean_controls is None:
            return jnp.zeros(self.num_controls)
        return jnp.array(self._mean_controls)

    @property
    def scale_controls(self) -> jnp.ndarray:
        """Returns the scale of the controls."""
        if self._scale_controls is None:
            return jnp.ones(self.num_controls)
        return jnp.array(self._scale_controls)

    @property
    def scale_states(self) -> jnp.ndarray:
        """Returns the scale of the states."""
        if self._scale_states is None:
            return jnp.ones(self.num_states)
        return jnp.array(self._scale_states)

    def transform_states(self, states: jnp.ndarray) -> jnp.ndarray:
        """Transform the states."""
        return (states - self.mean_states) / self.scale_states

    def transform_controls(self, controls: jnp.ndarray) -> jnp.ndarray:
        """Transform the controls."""
        return (controls - self.mean_controls) / self.scale_controls

    def setup(self):
        """ Initialize the learnable parameters of the model.
        """
        self.create_residual_forces()
        self.create_coriolis_forces()
        self.create_gravity_forces()
        self.create_actuator_forces()
        self.create_mass_matrix()

    def create_networks(
        self,
        config_dict: Dict[str, Any],
        num_out: int,
    ) -> nn.Module:
        """
        Temporary function to create a neural network given 
        a configuration.
        """
        if len(config_dict) <= 0:
            return None
        # Construct the neural network
        nn_type = config_dict.get("type", "MLP")
        # features for the NN
        features_nn = config_dict["features"]
        assert len(features_nn) > 0, \
            "Empty features for the neural network"
        # Create the network
        return load_network_from_config(
            nn_type,
            output_dimension = num_out,
            **config_dict["args"]
        )

    def create_residual_forces(self,):
        """Create the residual and other neural networks."""
        num_out = self.num_states
        residual_forces = self.create_networks(
            self.residual_forces_nn,
            num_out
        )
        if residual_forces is not None:
            self.residual_forces = residual_forces

    def get_residual_forces(
        self,
        features: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """Compute the residual forces."""
        if len(self.residual_forces_nn) <= 0:
            return 0
        in_array = jnp.concatenate(
            [ features[name] for name in self.residual_forces_nn["features"] ],
            axis=-1
        )
        return self.residual_forces(in_array)

    def create_coriolis_forces(self,):
        """Create the coriolis forces neural network."""
        num_out = self.num_velocities * self.num_velocities
        coriolis_forces = self.create_networks(
            self.coriolis_forces_nn,
            num_out
        )
        if coriolis_forces is not None:
            self.coriolis_forces = coriolis_forces
        
    def get_coriolis_forces(
        self,
        features: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """Compute the coriolis forces."""
        if len(self.coriolis_forces_nn) <= 0:
            return 0
        in_array = jnp.concatenate(
            [ features[name] for name in self.coriolis_forces_nn["features"] ],
            axis=-1
        )
        coriolis_matrix = self.coriolis_forces(in_array).reshape(
            (self.num_velocities, self.num_velocities)
        )
        vel = features["velocities"]
        return jnp.dot(coriolis_matrix, vel)

    def create_gravity_forces(self,):
        """Create the gravity forces neural network."""
        num_out = self.num_velocities
        gravity_forces = self.create_networks(
            self.gravity_forces_nn,
            num_out
        )
        if gravity_forces is not None:
            self.gravity_forces = gravity_forces

    def get_gravity_forces(
        self,
        features: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """Compute the gravity forces."""
        if len(self.gravity_forces_nn) <= 0:
            return 0
        in_array = jnp.concatenate(
            [ features[name] for name in self.gravity_forces_nn["features"] ],
            axis=-1
        )
        return self.gravity_forces(in_array)

    def create_actuator_forces(self,):
        """Create the actuator forces neural network."""
        num_out = self.num_velocities * self.num_controls
        actuator_forces = self.create_networks(
            self.actuator_forces_nn,
            num_out
        )
        if actuator_forces is not None:
            self.actuator_forces = actuator_forces

    def get_actuator_forces(
        self,
        features: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """Compute the actuator forces."""
        if len(self.actuator_forces_nn) <= 0:
            return 0
        in_array = jnp.concatenate(
            [ features[name] for name in self.actuator_forces_nn["features"] ],
            axis=-1
        )
        actuator_matrix = self.actuator_forces(in_array).reshape(
            (self.num_velocities, self.num_controls)
        )
        return jnp.dot(actuator_matrix, features["controls"])

    def get_position_correction(
        self,
        vel : jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the position correction."""
        if not self.include_pos_to_vel_relation:
            return 0

        pos_damper = self.param(
            "pos_damper",
            nn.initializers.normal(),
            (self.num_positions,)
        )
        exp_pos = jnp.exp(pos_damper)
        return exp_pos * vel

    def create_mass_matrix(self,):
        """Create the mass matrix neural network."""
        num_out = self.num_velocities * self.num_velocities
        mass_matrix = self.create_networks(
            self.mass_matrix_nn,
            num_out
        )
        if mass_matrix is not None:
            self.mass_matrix = mass_matrix

    def get_local_to_global(
        self,
        v_local : jnp.ndarray,
        features: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """Compute the local to global transformation using the Mass matrix."""
        if len(self.mass_matrix_nn) <= 0:
            return v_local
        in_array = jnp.concatenate(
            [ features[name] for name in self.mass_matrix_nn["features"] ],
            axis=-1
        )
        mass_matrix = self.mass_matrix(in_array)
        return jnp.dot(mass_matrix, v_local)

    def vectorfield(
        self,
        state: jnp.ndarray,
        control: jnp.ndarray,
        time_dependent_parameters: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Returns the drift term of the SDE.
        
        Args:
            state: the state of the system.
                (n,) array
            control: the control of the system.
                (n,) array
            time_dependent_parameters: updated model parameters that depend
            on time or a subset of model_parameters that need to be updated.
                (str, Any) dictionary
            learnable_parameters: module parameters that are learnable.
                (str, Any) dictionary or None
        
        Returns:
            drift: the drift term of the SDE.
                (n, ) array
            extra: the extra return arguments.
                (str, Any) dictionary
        """
        # Extract the angles first
        indexes_angles = self.angles_indexes
        angles_val = state[indexes_angles]
        angles_val_mean = self.mean_states[indexes_angles]
        # TODO: This assumes that the velocity with no positions
        # are stacked at the beginning of the velocity vector
        vel_to_pos = state[-self.num_positions:]

        # Transform the states
        state = self.transform_states(state)
        control = self.transform_controls(control)
        # Let's separate the state into position and velocity
        pos = state[:self.num_positions]
        vels = state[self.num_positions:]
        # Let's create the features vector
        feats = {
            "positions": pos,
            "velocities": vels,
            "controls": control,
            "cos_angles": (jnp.cos(angles_val) - jnp.cos(angles_val_mean)),
            "sin_angles": (jnp.sin(angles_val) - jnp.sin(angles_val_mean)),
            **{
                state_name: jnp.array([state[indx],]) \
                for indx, state_name in enumerate(self.names_states)
            }
        }
        # force scale factor
        _force_scale = 1 # Account for the time step
        # Compute the residual forces
        res_forces = self.get_residual_forces(feats)
        res_forces_pos = res_forces[:self.num_positions]
        res_forces_vels = res_forces[self.num_positions:]
        # Compute the coriolis forces
        cor_forces = self.get_coriolis_forces(feats)
        # Compute the gravity forces
        grav_forces = self.get_gravity_forces(feats)
        # Compute the actuator forces
        act_forces = self.get_actuator_forces(feats)
        # Compute the position correction
        pos_dot = self.get_position_correction(vel_to_pos) + \
            res_forces_pos * _force_scale
        # Sum all the forces and convert to global
        forces = res_forces_vels + cor_forces + grav_forces + act_forces
        forces = forces * _force_scale
        # jax.debug.print("forces {}, {}, {}", res_forces, cor_forces, grav_forces)
        vels_dot = self.get_local_to_global(forces, feats)
        return jnp.concatenate([pos_dot, vels_dot], axis=-1), {}

    @nn.compact
    def __call__(
        self,
        state: jnp.ndarray,
        control: jnp.ndarray,
        time_dependent_parameters: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Returns the drift term of the SDE.
        
        Args:
            state: the state of the system.
                (4,) array
            control: the control of the system.
                (4,) array
            time_dependent_parameters: updated model parameters that depend
            on time or a subset of model_parameters that need to be updated.
                (str, Any) dictionary
        
        Returns:
            drift: the drift term of the SDE.
                (4, ) array
            extra: the extra return arguments.
                (str, Any) dictionary
        """
        return self.vectorfield(state, control, time_dependent_parameters)

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
                (4,) array
            control: the control of the system.
                (4,) array
            time_dependent_parameters: updated model parameters that depend
            on time or a subset of model_parameters that need to be updated.
                (str, Any) dictionary
            learnable_parameters: module parameters that are learnable.
                (str, Any) dictionary or None
        
        Returns:
            drift: the drift term of the SDE.
                (4, ) array
            extra: the extra return arguments.
                (str, Any) dictionary
        """
        # Now the assumption is that the model has already being initialized
        # and we can simply call the apply method of the model
        return self.apply(learnable_parameters,
                          state,
                          control,
                          time_dependent_parameters
                        )

class RBD_Drift_Simple(RBD_Drift):
    """
    Simple RBD dRIFT WHERE THE VECTOR FIELD IS A FULLY CONNECTED NETWORK
    """
    def create_residual_forces(self):
        """Create the residual and other neural networks."""
        num_out = self.num_states
        residual_forces = self.create_networks(
            self.residual_forces_nn,
            num_out
        )
        if residual_forces is not None:
            self.residual_forces = residual_forces
    
    def vectorfield(
        self,
        state: jnp.ndarray,
        control: jnp.ndarray,
        time_dependent_parameters: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        # Let's separate the state into position and velocity
        pos = state[:self.num_positions]
        vels = state[self.num_positions:]
        # TODO: This assumes that the velocity with no positions
        # are stacked at the beginning of the velocity vector
        vel_to_pos = vels[-self.num_positions:]
        # Get the cos and sin of the angles
        angles_val = state[self.angles_indexes]
        # Let's create the features vector
        feats = {
            "positions": pos,
            "velocities": vels,
            "vel_to_pos": vel_to_pos,
            "controls": control,
            "cos_angles": jnp.cos(angles_val),
            "sin_angles": jnp.sin(angles_val),
            **{
                state_name: jnp.array([state[indx],]) \
                for indx, state_name in enumerate(self.names_states)
            }
        }
        # Compute the residual forces
        res_forces = self.get_residual_forces(feats)
        return res_forces, {}


# Define the models by their names
models_by_name = {
    "RBD_Drift": RBD_Drift,
    "RBD_Drift_Simple": RBD_Drift_Simple
}