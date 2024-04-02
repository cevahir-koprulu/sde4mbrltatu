"""
Script to define the loss functions or terms used in training the diffusion
term of a neural SDE model.
"""

from typing import Any, Dict, Tuple, Callable

import jax
import jax.numpy as jnp
import flax.linen as nn

from nsdes_dynamics.networks import MLP
from nsdes_dynamics.distance_aware_diffusion import (
    BasicDistanceAwareDiffusionTerm,
)

def density_loss(
    diff_nn_params: Dict[str, Any],
    diffusion_term: BasicDistanceAwareDiffusionTerm,
    features: jnp.ndarray,
    cvx_coeff: float,
    rng_key: jnp.ndarray,
    ball_radius: jnp.ndarray,
    ball_num_samples: int,
) -> Dict [str, float]:
    """
    Loss function to train the diffusion term of a neural SDE model.

    Args:
        diff_nn_params: parameters of the diffusion term neural network.
            (str, Any) dictionary
        diffusion_term: diffusion term of the neural SDE model.
            BasicDistanceAwareDiffusionTerm instance
        features: The features used as input to the density term.
            (n,) array
        cvx_coeff: coefficient used to enforce the local
        strong convexity property of the density.
            float
        rng_key: random number generator key.
            (2,) array
        ball_radius: radius of the ball used to compute the density. We sample
        from this ball to enforce the local strong convexity property.
            (n,) array or float
        ball_num_samples: number of sample points to use to enforce the local
        strong convexity property.
            int

    Returns:
        loss: dictionary containing the loss values
            (str, float) dictionary
    """
    # Some checks
    radius = jnp.array(ball_radius)
    if radius.ndim > 0:
        assert radius.ndim == 1, "The radius should be a scalar or a vector"
        assert radius.shape == features.shape, \
            "The radius of shape {radius.shape} should be the same as the " + \
            "features of shape {features.shape}"

    # Sample points from a ball around the features
    ball_dist = jax.random.normal(
        rng_key,
        (ball_num_samples, features.shape[0])
    )
    ball_dist = ball_dist * radius[None]

    # Define the sample points from the ball
    samples_points = features[None] + ball_dist

    # Wrapper around the density term
    def feature_density(x: jnp.ndarray) -> float:
        dad_term = diffusion_term.apply(
            diff_nn_params,
            x,
            method=diffusion_term.compute_density_term
        )
        return dad_term

    # Evalue the value and gradient of the feature density
    density_feat, grad_density_feat = \
        jax.value_and_grad(feature_density)(features)

    # Estimate the density at the sample points
    density_samples_points = jax.vmap(feature_density)(samples_points)

    # Compute the local convexity condition, (n_samples,) array
    local_convex_cond = density_samples_points - density_feat - \
        jnp.sum(grad_density_feat[None] * ball_dist, axis=1) - \
        0.5 * cvx_coeff * jnp.sum(jnp.square(ball_dist), axis=1)

    # Enforce the local convexity condition
    local_convex_loss = jnp.sum(jnp.square(jnp.minimum(local_convex_cond, 0)))

    # Enforce one values for sampled points
    density_sample_points_equal_one = \
        jnp.mean(jnp.square(density_samples_points - 1.0))

    # Return the results
    return {
        'local_convex_loss': local_convex_loss,
        'gradient_loss': jnp.sum(jnp.square(grad_density_feat)),
        'density_value': density_feat,
        'density_set_one': density_sample_points_equal_one
    }

class StrongConvexityCoef(nn.Module):
    """ Learnable strong convexity coefficient for the density term.
    """
    coef_init: float
    is_constant: bool = True
    activation_name: str = 'swish'
    layers_archictecture: Tuple = (16, 16)

    @nn.compact
    def __call__(self, features) -> jnp.ndarray:
        """ Forward pass of the local strong convexity coefficient.
        
        Args:
            features: input to the network
                (n,) array
                
        Returns:
            coef: output of the network
                float
        """
        if self.is_constant:
            coef = self.param(
                'strong_convexity_coef', 
                nn.initializers.ones, ()
            ) * self.coef_init
        else:
            activation_fn = \
                getattr(jnp, self.activation_name) \
                if hasattr(jnp, self.activation_name) \
                else getattr(jax.nn, self.activation_name)
            coef = MLP(
                output_dimension=1,
                initial_value_range=0.1,
                activation_fn=activation_fn,
                layers_archictecture=self.layers_archictecture
            )(features)[...,0] + self.coef_init
        return jnp.abs(coef)

    @staticmethod
    def loss_convexity_coef(type_loss, coef):
        """ Loss function for the strong convexity coefficient.
            This assumes that the optimizer is trying to minimize the loss.
        
        Args:
            type_loss: type of loss function to use.
                str
            coef: strong convexity coefficient
                float or (n,) array
                
        Returns:
            loss: value of the loss
                float
        """
        if type_loss == 'quad_inv':
            loss = 1.0 / coef**2
        elif type_loss == 'lin_inv':
            loss = 1.0 / coef
        elif type_loss == 'exp_inv':
            loss = jnp.exp(-coef)
        else:
            raise ValueError("Unknown type_loss: {loss_type}" + \
                "Choose from quad_inv, lin_inv, exp_inv")
        return loss

def create_dad_loss(
    diff_model: BasicDistanceAwareDiffusionTerm,
    key: jnp.ndarray,
    diff_loss_config: Dict[str, float],
    cvx_coeff_config: Dict[str, Any],
) -> Tuple[Callable, Dict[str, Any]]:
    """ Create the loss function for the diffusion term.

    Args:
        diff_model: the diffusion term model
            BasicDistanceAwareDiffusionTerm instance
        key: random number generator key
            (2,) array
        diff_loss_config: configuration of the diffusion loss
            (str, float) dictionary
        cvx_coeff_config: configuration of the strong convexity coefficient
            (str, Any) dictionary

    Returns:
        dad_loss: the loss function for the diffusion term
            Callable
        cvx_coef_params: parameters of the strong convexity coefficient
            (str, Any) dictionary
    """
    # Extract the configuration
    is_cvx_coeff_learned = cvx_coeff_config['is_cvx_coeff_learned']
    cvx_coeff_term = cvx_coeff_config["cvx_coeff_params"]["coef_init"]
    cvx_coeff_learnable_params = None

    # If the coefficient is learned, we need to create the parameters
    # of the model to be learned
    if is_cvx_coeff_learned:
        cvx_coeff_term = StrongConvexityCoef(
            **cvx_coeff_config["cvx_coeff_params"]
        )

        # Initialize the NN module and its parameters
        state_init = jnp.ones((diff_model.num_states,)) * 1.0e-3
        control_init = jnp.ones((diff_model.num_controls,)) * 1.0e-3
        time_dependent_init = \
            { key : 1.0e-3 for key in diff_model.feature_parameters_to_use }
        feature_init = diff_model.extract_features_for_density_nn(
            state_init, control_init, time_dependent_init
        )

        # Initialize the learnable parameters
        cvx_coeff_learnable_params = cvx_coeff_term.init(
            key, feature_init
        )

    # Create the loss function
    def dad_loss(
        nn_params : Dict[str, Any],
        state: jnp.ndarray,
        control: jnp.ndarray,
        time_dependent_parameters: Dict[str, Any],
        rng_key: jnp.ndarray
    ) -> Tuple[float, Dict[str, jnp.ndarray]]:
        """ Loss function for the diffusion term.
        """
        # Compute the features
        features = diff_model.extract_features_for_density_nn(
            state, control, time_dependent_parameters
        )

        # Extract the diffusion term parameters
        diff_params = nn_params['diffusion']
        if is_cvx_coeff_learned:
            convexity_coeff_params = nn_params['mu_coeff_nn']
            cvx_coefficient = cvx_coeff_term.apply(
                convexity_coeff_params,
                features,
            )
        else:
            cvx_coefficient = cvx_coeff_term

        # Compute the loss
        dict_result = density_loss(
            diff_params,
            diff_model,
            features,
            cvx_coefficient,
            rng_key,
            diff_loss_config["ball_radius"],
            diff_loss_config["ball_num_samples"]
        )

        # If the loss on the cvx_coefficient is learned, we need to update
        # the dictionaary loss
        if is_cvx_coeff_learned:
            cvx_coeff_loss = cvx_coeff_term.loss_convexity_coef(
                diff_loss_config['cvx_coeff_loss_type'], cvx_coefficient
            )
            dict_result['cvx_coeff_loss'] = cvx_coeff_loss
            dict_result['cvx_coefficient'] = cvx_coefficient

        # Return the loss
        return dict_result

    def vmap_dad_loss(
        nn_params : Dict[str, Any],
        state: jnp.ndarray,
        control: jnp.ndarray,
        time_dependent_parameters: Dict[str, Any],
        rng_key: jnp.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """ Loss function for the diffusion term with vmap option
        
        Args:
            nn_params: parameters of the diffusion term
                (str, Any) dictionary
            state: state of the system
                (n, d) array
            control: control of the system
                (n, k) array
            time_dependent_parameters: time dependent parameters
                (str, Any) dictionary
            rng_key: random number generator key
                (2,) array
        
        Returns:
            total_loss: total loss
                (float,) array
            loss_dict: dictionary of the loss values
                (str, float) dictionary
        """
        # Split the random key and compute the loss
        rng_key = jax.random.split(rng_key, state.shape[0])
        loss_dict = jax.vmap(dad_loss, in_axes=(None, 0, 0, 0, 0))(
            nn_params, state, control, time_dependent_parameters, rng_key
        )

        # Mean over the samples
        loss_dict = {k: jnp.mean(v) for k, v in loss_dict.items()}

        # Let's penalize gradient values that are too small -> constant density
        gradient_loss = loss_dict["gradient_loss"]
        sum_gradient_loss = jnp.mean(gradient_loss)
        gradient_loss_min_constraint = jnp.where(
            sum_gradient_loss < diff_loss_config['min_grad_density'],
            - jnp.log(sum_gradient_loss),
            0.0,
        )

        # Compute the total diffusion loss
        weight_diff = diff_loss_config['weight_diff_loss']
        total_loss = \
            jnp.array([loss_dict[k] * weight_diff[k] for k in weight_diff]).sum()

        # Add the min constraint on the gradient
        total_loss += gradient_loss_min_constraint * \
            diff_loss_config.get('weight_min_grad_density', 1.0)

        # Store and return the results
        loss_dict['TotalDiffLoss'] = total_loss
        loss_dict['grad_min_constr'] = gradient_loss_min_constraint
        return total_loss, loss_dict

    # Package the learnable parameters
    cvx_coeff_learnable_params = {} if cvx_coeff_learnable_params is None \
        else {'mu_coeff_nn': cvx_coeff_learnable_params}

    # Return the results
    return vmap_dad_loss, cvx_coeff_learnable_params
