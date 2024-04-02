"""
Define any neural network architectures that could be used in the definition of
the neural SDEs models. 
"""
from typing import Tuple, Union

import jax
import jax.numpy as jnp

import flax.linen as nn

class MLP(nn.Module):
    """ A MLP class
    """
    output_dimension: int
    initial_value_range: float = 0.01
    activation_fn: Union[str, callable] = "tanh"
    layers_archictecture: Tuple = (16, 16)

    @nn.compact
    def __call__(self, x : jnp.ndarray) -> jnp.ndarray:
        """ Forward pass of the MLP
        
        Args:
            x: input to the network
                (...,n) array
        Returns:
            x: output of the network
                (output_dimension,) array
        """
        if isinstance(self.activation_fn, str):
            activation_fn = get_activation_fn_from_name(self.activation_fn)
        else:
            activation_fn = self.activation_fn
        for feat_dimension in self.layers_archictecture:
            x = nn.Dense(
                feat_dimension,
                kernel_init=nn.initializers.uniform(
                    scale=self.initial_value_range
                    )
                )(x)
            x = activation_fn(x)
        x = nn.Dense(self.output_dimension)(x)
        return x

def get_activation_fn_from_name(
    act_fn_name : str
) -> callable:
    """ Get the activation function from its name.
    
    Args:
        act_fn_name: name of the activation function
            str
    Returns:
        act_fn: the activation function
            callable
    """
    if hasattr(jnp, act_fn_name):
        act_fn = getattr(jnp, act_fn_name)
    else:
        act_fn = getattr(jax.nn, act_fn_name)
    return act_fn

def load_network_from_config(
    name : str,
    **kwargs
) -> nn.Module:
    """ Load a network from a configuration dictionary.
    
    Args:
        name: name of the network to load
            str
        kwargs: dictionary containing the parameters to initialize the
        network structure
            dict
    
    Returns:
        network: the network
            nn.Module
    """
    if name == "MLP":
        return MLP(**kwargs)

    raise ValueError(f"Unknown network name {name}")
