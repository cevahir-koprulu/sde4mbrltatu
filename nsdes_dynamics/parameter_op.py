"""
Utility script to perform some basic operations like loading, saving,
structuring, pretty printing, etc. of the parameters of the neural SDE
"""

import os
import copy

from typing import Any, Dict, Callable, Tuple, List
from collections.abc import Mapping

import yaml

import numpy as np

import jax.numpy as jnp
from jax.tree_util import tree_leaves

def full_path(path : str) -> str:
    """ Complete the path with the user home directory if needed

    Args:
        path (str): The path to complete

    Returns:
        str: The completed path
    """
    return os.path.expanduser(path)

def load_yaml(config_path : str) -> Dict[str, Any]:
    """Load a yaml file from a given path

    Args:
        config_path: The path to the yaml file
            str

    Returns:
        yaml_dict: The dictionary containing the configuration in the yaml file
            dict
    """
    yml_file = open(full_path(config_path), 'r', encoding="utf-8")
    yml_byte = yml_file.read()
    cfg_train = yaml.load(yml_byte, yaml.SafeLoader)
    yml_file.close()
    return cfg_train

def dump_yaml(
    config_path : str,
    dict_val : Dict[str, Any]
):
    """Dump a dictionary to a yaml file

    Args:
        config_path: The path to the yaml file
            str
        dict_val: The dictionary to dump
            dict
    """
    with open(config_path, 'w', encoding="utf-8") as f:
        yaml.dump(dict_val, f)

def convert_dict_of_array_to_dict_of_list(
    config : Dict[str, Any]
) -> Dict[str, Any]:
    """Convert a dictionary of arrays to a dictionary of lists

    Args:
        config: The configuration dictionary
            dict

    Returns:
        dict_list: The configuration dictionary with all the arrays 
        converted to lists
            dict
    """
    if not isinstance(config, Mapping):
        return config

    new_config = {}
    for key, value in config.items():
        if isinstance(value, Mapping):
            new_config[key] = \
                convert_dict_of_array_to_dict_of_list(value)
        elif isinstance(value, list):
            new_config[key] = \
                [convert_dict_of_array_to_dict_of_list(v) for v in value]
        elif isinstance(value, tuple):
            new_config[key] = \
                tuple([convert_dict_of_array_to_dict_of_list(v) for v in value])
        else:
            new_config[key] = \
                value.tolist() if hasattr(value, 'tolist') else value
    return new_config

def pretty_print_config(config : Dict[str, Any]):
    """Pretty print a configuration dictionary

    Args:
        config : The configuration dictionary
            dict
    """
    new_config = convert_dict_of_array_to_dict_of_list(config)
    print(yaml.dump(new_config, allow_unicode=True, default_flow_style=False))

def are_parameters_empty(
    config : Dict[str, Any]
) -> bool:
    """Check if the config dict empty

    Args:
        config: The configuration dictionary
            dict

    Returns:
        is_empty: True if the parameters are empty, False otherwise
            bool
    """
    if not config:
        return True
    if not isinstance(config, Mapping):
        return False
    for value in config.values():
        if not are_parameters_empty(value):
            return False
    return True

def is_key_in_config(
    config : Dict[str, Any],
    key : str
) -> bool:
    """Check if the key is in the configuration dictionary

    Args:
        config : The configuration dictionary
            dict
        key : The key to check
            str

    Returns:
        key_in_dict: True if the key is in the configuration 
        dictionary, False otherwise
            bool
    """
    if not isinstance(config, Mapping):
        return False
    for k, v in config.items():
        if k == key:
            return True
        if isinstance(v, Mapping) and is_key_in_config(v, key):
            return True
    return False

def set_values_all_leaves(
    config : Dict[str, Any],
    value : Any,
    same_dim_out : bool = False
) -> Dict[str, Any]:
    """Set the value of all leaves of the configuration dictionary

    Args:
        config: The configuration dictionary
            dict
        value: The value to set
            any
        same_dim_out: If True, the output will have the same dimension as the
            input, otherwise it will be a scalar
            bool
    
    Returns:
        new_config: The new configuration dictionary with the value set
            dict
    """
    config = copy.deepcopy(config)
    for k, v in config.items():
        if isinstance(v, Mapping):
            config[k] = set_values_all_leaves(v, value, same_dim_out)
        else:
            config[k] = value * np.ones_like(v) if same_dim_out else value
    return config

def set_values_matching_keys(
    config : Dict[str, Any],
    keys_with_config_to_update : Dict[str, Any],
    feature_extractor : Callable,
    same_dim_out : bool = False
) -> Dict[str, Any]:
    """Set the value of the leaves of the configuration dictionary that match
    the keys in the dictionary keys_with_config_to_update.
    The value to be set is obtain by calling the feature_extractor on the
    value of the keys_with_config_to_update.

    Args:
        config: The configuration dictionary
            dict
        keys_with_config_to_update: The dictionary with the keys to match and
            the configuration to update
            dict
        feature_extractor: The function to extract the feature from the
            keys_with_config_to_update
            callable
        same_dim_out: If True, the output will have the same dimension as the
            input, otherwise it will be a scalar
            bool

    Returns:
        new_config: The new configuration dictionary with the value set
            dict
    """
    config = copy.deepcopy(config)
    for k, v in config.items():
        if k in keys_with_config_to_update:
            value_to_set = feature_extractor(keys_with_config_to_update[k])
            if isinstance(v, Mapping):
                config[k] = set_values_all_leaves(
                    v,
                    value_to_set,
                    same_dim_out
                )
            else:
                config[k] = value_to_set * np.ones_like(v) \
                    if same_dim_out else value_to_set
        elif isinstance(v, Mapping):
            config[k] = set_values_matching_keys(
                v,
                keys_with_config_to_update,
                feature_extractor,
                same_dim_out
            )
    return config

def find_all_values_matching_key(
    config : Dict[str, Any],
    key : str
) -> List[Any]:
    """
    Find all the values in the configuration dictionary that match the key
    
    Args:
        config: The configuration dictionary
            dict
        key: The key to match
            str
            
    Returns:
        values: The list of values that match the key
            list
    """
    values = []
    for k, v in config.items():
        if k == key:
            values.append(v)
        elif isinstance(v, Mapping):
            values.extend(find_all_values_matching_key(v, key))
    return values

def modify_entry_from_config(
    config : Dict[str, Any],
    key : str,
    value : Any
) -> Dict[str, Any]:
    """ Modify the value of the key in the configuration dictionary.
    This is a modification in place. This function assumes the key is unique.

    Args:
        config: The configuration dictionary
            dict
        key: The key to modify
            str
        value: The value to set
            any
    """
    for k, v in config.items():
        if k == key:
            config[k] = value
        elif isinstance(v, Mapping):
            config[k] = modify_entry_from_config(v, key, value)
    return config

def modify_entry_from_config_with_dict(
    config : Dict[str, Any],
    modified_entries : Dict[str, Any]
) -> Dict[str, Any]:
    """ Modify the value of the keys in the configuration dictionary
    that matches the keys in the modified_entries dictionary.
    
    Args:
        config: The configuration dictionary
            dict
        modified_entries: The dictionary with the keys to modify and the
            values to set
            dict
    """
    for k, v in modified_entries.items():
        # Let's find all the values that match the key
        current_values = find_all_values_matching_key(config, k)
        if len(current_values) == 0:
            # print(f"\nCannot modify the key {k} with value {v} because\n" + \
            #     "it is is not in the configuration dictionary"
            # )
            continue
        elif len(current_values) > 1:
            raise ValueError(f"\nMany values found for key {k} in the\n" + \
                f"configuration dictionary: {current_values}"
            )
        else:
            print(f"\nModifying key {k} with value {current_values[0]} to\n" + \
                f"key {k} value {v}"
            )
            config = modify_entry_from_config(config, k, v)
    return config

def create_gaussian_regularization_loss(
    params: Dict[str, Any],
    reg_config: Dict[str, Any]
) -> Tuple[Callable, Dict[str, Any]]:
    """ Create the gaussian regularization loss
    
    Args:
        params: The parameters of the model
            dict
        reg_config: The configuration of the regularization
            dict
    
    Returns:
        Callable: A function that takes as input the parameters of the model,
        specifically the drift terms and return the regularization loss
            callable
        Dict[str, Any]: The dictionary of mean and scale values used for 
        computing the Gaussian regularization loss
            dict
    """
    # Extract the default mean and scale for the initial prior
    default_mean = reg_config.get('default_mean', 0.0)
    default_scale = reg_config.get('default_scale', 1.0)

    # Create a dictionary with the same structure as drift_params but with
    # the default mean and scale
    mean_params_dict = set_values_all_leaves(
        params,
        default_mean
    )
    scale_params_dict = set_values_all_leaves(
        params,
        default_scale
    )

    # Extract the specials parameter keys
    specials = reg_config.get('specials', {})
    if len(specials) > 0: # We need to update the mean and scale for some keys
        mean_params_dict = \
            set_values_matching_keys(
                mean_params_dict,
                specials,
                lambda x : x['mean']
            )
        scale_params_dict = \
            set_values_matching_keys(
                scale_params_dict,
                specials,
                lambda x : x['scale']
            )

    # Create the gaussian regularization loss
    def reg_loss(
        params : Dict[str, Any]
    ) -> jnp.array:
        """ Compute the gaussian regularization loss

        Args:
            params: The parameters of the model
                dict

        Returns:
            reg_loss: The regularization loss
                jnp.array
        """
        # Flatten the parameters
        val_reg = jnp.array(
            [
                jnp.sum(jnp.square((p - m) / s if s > 0 else jnp.zeros_like(p)))
                    for p, m, s in zip(
                        tree_leaves(params),
                        tree_leaves(mean_params_dict),
                        tree_leaves(scale_params_dict)
                    )
            ]
        )
        return jnp.sum(val_reg) / 2.0

    return reg_loss, {'mean': mean_params_dict, 'scale': scale_params_dict}
