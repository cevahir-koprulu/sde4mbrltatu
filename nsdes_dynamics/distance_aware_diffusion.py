"""
Main code to define and encode diffusion terms that characterize some 
notion of distance awareness to a training dataset.
"""

from typing import Dict, Any, List, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from nsdes_dynamics.base_nsdes import DiffusionTerm
from nsdes_dynamics.networks import (
    MLP,
    get_activation_fn_from_name
)

class BasicDistanceAwareDiffusionTerm(DiffusionTerm):
    """
    Class for the distance-aware diffusion term of a neural SDE.
    The parameter dictionary of the diffusion term should contain:
        - 
    """
    upper_bound_diffusion: jnp.ndarray
    density_nn_params: Dict[str, Any]
    density_free_nn_params: Dict[str, Any]
    diffusion_is_control_dependent: bool
    feature_parameters_to_use: List[str]
    default_feature_values: List[float]
    feature_density_scaling: jnp.ndarray

    def setup(self):
        """ Initialize the learnable parameters of the model.
        """
        self.density_nn = self.create_parameterized_density_function() # noqa
        self.density_free_nn = self.create_density_free_nn() # noqa

    def diffusion(
        self,
        state: jnp.ndarray,
        control: jnp.ndarray,
        time_dependent_parameters: Dict[str, Any],
        learnable_parameters: Dict[str, Any]
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        return self.apply(
            learnable_parameters, state, control, time_dependent_parameters
        )

    @nn.compact
    def __call__(
        self,
        state: jnp.ndarray,
        control: jnp.ndarray,
        time_dependent_parameters: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:

        # Upper bound on the density-based diffusion term
        upper_bound_diffusion = jnp.array(self.upper_bound_diffusion)

        # Do some checks
        assert upper_bound_diffusion.shape[-1] == self.num_states, \
            "The upper bound term {upper_bound_diffusion.shape}"  + \
                "should have the same dimension as the state {self.num_states}"
        assert state.shape[-1] == self.num_states, \
            "The state {state.shape} must match {self.num_states}"
        assert control.shape[-1] == self.num_controls, \
            "The control {control.shape} must match {self.num_controls}"

        # Extract the features to be used as input to the density nn
        feature_density = self.extract_features_for_density_nn(
            state, control, time_dependent_parameters
        )

        # Compute the heterogeneous noise term
        heterogeneous_noise_term, density_term = \
            self.compute_heterogeneous_noise_term(feature_density)
        density_based_diff = upper_bound_diffusion * heterogeneous_noise_term

        # Diffusion free term
        density_free_diff_term, (center_range, width_range) = \
            self.get_density_free_diffusion_term(feature_density)

        # Total diffusion term
        # total_diffusion = (1 - density_term) * density_free_diff_term + \
        #     density_based_diff
        total_diffusion = density_free_diff_term + \
            density_based_diff

        # Extra return dictionary
        ret_dict = {
            "diffusion_value" : total_diffusion,
            "dad_free_diff" : density_free_diff_term,
            "dad_based_diff" : density_based_diff,
            "diff_density" : density_term,
            "log_std_center" : center_range,
            "log_std_width" : width_range,
        }
        return total_diffusion, ret_dict

    def create_parameterized_density_function(self):
        """
        Create the neural networks that parameterize the density function
        used to approximate the distance to the training dataset.
        """
        # Extract the neural network parameters from the model_parameters
        density_nn_params = self.density_nn_params
        act_fn_name = density_nn_params['activation_fn']
        layers_archictecture = density_nn_params['layers_archictecture']
        initial_value_range = density_nn_params['initial_value_range']

        # Extract the activation function
        act_fn = get_activation_fn_from_name(act_fn_name)

        # Create the neural network
        return MLP(
            output_dimension=1,
            initial_value_range=initial_value_range,
            activation_fn=act_fn,
            layers_archictecture=layers_archictecture,
        )

    def create_density_free_nn(self):
        """
        Create the neural networks that parameterize the density free
        terms of the diffusion term. This density free terms is designed to
        capture the stochasticity of the dynamics without being influenced
        by the distance-aware term.
        """
        if len(self.density_free_nn_params) == 0:
            return
        density_free_nn_params = self.density_free_nn_params
        act_fn_name = density_free_nn_params['activation_fn']
        layers_archictecture = density_free_nn_params['layers_archictecture']
        initial_value_range = density_free_nn_params['initial_value_range']
        # Extract the activation function
        act_fn = get_activation_fn_from_name(act_fn_name)
        return MLP(
            output_dimension=self.num_states,
            initial_value_range=initial_value_range,
            activation_fn=act_fn,
            layers_archictecture=layers_archictecture,
        )

    def get_density_free_mid_and_width(self,):
        """ Return the parameters for the center and width of the interval in
        which the density free term is bounded.
        """
        center_n_width = self.param(
            "diff_center_n_width",
            nn.initializers.uniform(),
            (2 * self.num_states,)
        )
        return center_n_width[:self.num_states], \
            jnp.abs(center_n_width[self.num_states:])

    def get_density_free_diffusion_term(
        self,
        features_model: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute the density free term of the diffusion term.
        
        Args:
            features_model: the features to be used as input to the density
            free neural network.
                (k,) array
                
        Returns:
            density_free_term: the density free term of the diffusion term.
                (n,) array
        """
        if len(self.density_free_nn_params) == 0:
            return jnp.zeros(self.num_states)
        density_free_term = self.density_free_nn(features_model)
        # Put the density free term in the interval [-1, 1]
        scaled_diff_term = jnp.tanh(density_free_term)
        center, width = self.get_density_free_mid_and_width()
        log_std_val =  center + width * scaled_diff_term
        std_val = jnp.exp(log_std_val)
        return std_val, (center, width)

    def create_heterogeneous_noise_scaling_term(self):
        """
        Create the parameterized term that scales the density function
        to obtain an heteregeneous noise term.
        
        For this basic implementation, we use a constant vector to scale the
        density function in a way that is bijective and keeps the diffusion
        term bounded by 1. ( 1 values should be attained outside the training)
        """
        heteregeneous_scaler = self.param(
            "heteregeneous_scaler",
            nn.initializers.uniform(
                scale=0.001 # Small value to initialize it close to 1
            ),
            (2 * self.num_states,)
        )
        # Make the multiplicative scaler non-negative
        heteregeneous_scaler = jnp.exp(heteregeneous_scaler)
        mult_scaler_term = heteregeneous_scaler[:self.num_states]
        add_scaler_term = heteregeneous_scaler[self.num_states:]
        return mult_scaler_term, add_scaler_term

    @staticmethod
    def static_extract_features_for_density_nn(
        state : jnp.ndarray,
        control : jnp.ndarray,
        time_dependent_parameters: Dict[str, Any],
        diffusion_is_control_dependent: bool,
        feat_params_to_use: List[str],
        default_values: List[float]
    ) -> jnp.ndarray:
        """
        Extract the features to be used as input to the density neural network.
        
        Args:
            state: the state of the system.
                (n,) array
            control: the control of the system.
                (m,) array
            time_dependent_parameters: some model parameters that could be used
            as input to the density term.
                (str, Any) dictionary
            diffusion_is_control_dependent: a boolean indicating whether the 
            diffusion term is control dependent.
                bool
            feat_params_to_use: a dictionary containing the model parameters
            and their default values to be used as input to the density term.
                (str, jnp.ndarray) dictionary
            default_values: the default values of the parameters to be used
            as input to the density term.
                (p,) array, where p is the sum of dimenion of parameters
                
        Returns:
            feature_density: the features to be used as input to the density
            neural network.
                (k <= n + m + p,) array, where p is the sum of dimenion 
                of parameters
        """
        feature_density = state

        # Check if the diffusion term is control dependent
        if diffusion_is_control_dependent:
            feature_density = jnp.concatenate((feature_density, control))

        # Check if the diffusion term depends on model parameters
        # Typically used when trained on multimodal datasets
        for parameter, default_val in zip(feat_params_to_use, default_values):
            param_value = time_dependent_parameters.get(
                parameter,
                default_val
            )
            param_value = jnp.array(param_value)
            param_value = \
                param_value[None] \
                if param_value.ndim == 0 \
                else param_value
            feature_density = jnp.concatenate((feature_density, param_value))

        return feature_density

    def extract_features_for_density_nn(
        self,
        state : jnp.ndarray,
        control : jnp.ndarray,
        time_dependent_parameters: Dict[str, Any],
    ) -> jnp.ndarray:
        """
        Extract the features to be used as input to the density neural network.
        
        Args:
            state: the state of the system.
                (n,) array
            control: the control of the system.
                (m,) array
            time_dependent_parameters: updated model parameters that depend 
            on time or a subset of model_parameters that need to be updated.
                (str, Any) dictionary
                
        Returns:
            feature_density: the features to be used as neural network inputs.
                (k,) array,
        """
        feature_density = self.static_extract_features_for_density_nn(
            state, control, time_dependent_parameters,
            self.diffusion_is_control_dependent,
            self.feature_parameters_to_use,
            self.default_feature_values,
        )
        # Scale the features according to fixed values in the model parameters
        feature_density = feature_density / self.feature_density_scaling

        return feature_density

    def compute_density_term(
        self,
        feature_density: jnp.ndarray,
    ) -> float:
        """
        Compute the density term approximating the distance to the training
        dataset.
        
        Args:
            feature_density: the features to be used as neural network inputs.
                (k,) array
                
        Returns:
            density_term: approximation of the distance to the training dataset.
                float
        """
        # Compute the distance to the training dataset
        density = self.density_nn(feature_density)[...,0] # (..., 1) array
        return jax.nn.sigmoid(density - 7.0) # Offset s.t. sigmoid is ~0 at 0

    def compute_heterogeneous_noise_term(
        self,
        feature_density: jnp.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute the heterogeneous noise term.
        
        Args:
            feature_density: the features to be used as neural network inputs.
                (k,) array
                
        Returns:
            heterogeneous_noise_term: the heterogeneous noise term.
                float
        """
        # Compute the heterogeneous noise term
        mult_scaler_term, add_scaler_term = \
            self.create_heterogeneous_noise_scaling_term()

        # Compute the density
        density = self.density_nn(feature_density)[...,0]

        # Compute the heterogeneous noise term
        unbounded_term =  (1 + mult_scaler_term) * density -7.0 + add_scaler_term
        actual_density = jax.nn.sigmoid(density - 7.0)

        return jax.nn.sigmoid(unbounded_term), actual_density

# Define the diffusion terms by name
diffusion_terms_by_name ={
    "BasicDistanceAwareDiffusionTerm": BasicDistanceAwareDiffusionTerm
}