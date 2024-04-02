"""
Utility functions for dataset operations such as loading,
sampling, preprocessing, etc.
"""

from typing import Any, Dict, List, Tuple, Union, Callable
import numpy as np

def split_dataset(
    dataset : Dict[str, Any],
    test_ratio : float = 0.1,
    seed : int = 0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Split the dataset into a training and testing dataset.
    
    Args:
        dataset: The dataset
            dict
        test_ratio: The ratio of the testing dataset in terms 
        of the total number of trajectories in the dataset
            float
        seed: The seed for the random number generator. This is for 
        reproducibility of the testing and training datasets
    
    Returns:
        training_dataset: The training dataset
            dict
        testing_dataset: The testing dataset
            dict
    """
    # Extract the number of trajectories
    num_trajs = len(dataset["trajectories"])
    non_trajetory_infos = \
        { k : v for k, v in dataset.items() \
            if k not in ["trajectories", "trajectories_info"]
        }

    # Create the random number generator
    rng = np.random.default_rng(seed)

    # Number of testing trajectories
    num_test_trajs = int(test_ratio * num_trajs)
    num_test_trajs = max(1, num_test_trajs)
    num_train_trajs = num_trajs - num_test_trajs
    num_train_trajs = max(1, num_train_trajs)

    # shuffle the indices
    idx = rng.permutation(num_trajs)

    # Split the dataset
    training_dataset = {
        "trajectories": [
            dataset["trajectories"][i] for i in idx[:num_train_trajs]
        ],
        "trajectories_info": [
            dataset["trajectories_info"][i] for i in idx[:num_train_trajs]
        ],
        **non_trajetory_infos
    }
    # Testing dataset
    testing_dataset = {
        "trajectories": [
            dataset["trajectories"][i] for i in idx[num_train_trajs:]
        ],
        "trajectories_info": [
            dataset["trajectories_info"][i] for i in idx[num_train_trajs:]
        ],
        **non_trajetory_infos
    }
    return training_dataset, testing_dataset

def pick_batch_transitions_from_trajectory_as_array(
    trajectory : Dict[str, Any],
    trajectory_info : Dict[str, Any],
    index_start : int,
    stepsize_values : Union[List[int], np.ndarray],
    problem_config : Dict[str, Any],
    strategy : Dict[str, str],
    ignore_last: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Pick a batch of states from the trajectory, and regroup them into
    arrays suitable for neural sde training/eval using the configuration of
    the problem.
    
    Args:
        trajectory: The trajectory
            dict
        trajectory_info: The trajectory information
            dict
        iindex_start: The start index of the transition
            int
        stepsize_values: The stepsize values
            list of int
        problem_config: The configuration of the problem containing the names
        of the states, controls, and extra arguments for the vector field or
        features.
            dict
        strategy: The strategy employed to pick the control or
        time_dependent_parameters when they are not constant over two points
        in the transition separated by a stepsize > 1
            dict
            
    Returns:
        states: The states of dimension (horizon+1, num_states)
            np.ndarray
        controls: The controls of dimension (horizon, num_controls)
            np.ndarray
        time_dependent_parameters: dictionary with each key being of dimension
        (horizon)
            dict
    """
    special_keys = problem_config['names_controls'] + \
                    problem_config['time_dependent_parameters']
    transitions = pick_transitions(
        trajectory, [index_start], np.array([stepsize_values]),
        special_keys, strategy
    )[0]
    states, controls, time_dependent_parameters = \
        convert_transitions_to_array(
            [transitions], [trajectory_info], problem_config,
            ignore_last=ignore_last
        )
    return states[0], controls[0], \
        {key: value[0] for key, value in time_dependent_parameters.items()}

def pick_batch_transitions_as_array(
    dataset : Dict[str, Any],
    batch_size : int,
    stepsize_range : Tuple[int, int],
    horizon : int,
    problem_config : Dict[str, Any],
    strategy : Dict[str, str],
    pick_from_single_traj : bool = False,
    indx_trajectory : Union[List[int], np.ndarray] = None,
    indx_transitions : Union[List[int], np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Pick a batch of transitions from the dataset, and regroup 
    them into arrays suitable for neural sde training using 
    the configuration of the problem.
    
    Args:
        dataset: The dataset
            dict
        batch_size: The batch size
            int
        stepsize_range: The range of the stepsize between two consecutive
        data points in a transition
            tuple of int
        horizon: The horizon of the transitions >= 2
            int
        problem_config: The configuration of the problem containing 
        the names of the states, controls, and extra arguments 
        for the vector field or features.
            dict
        strategy: The strategy employed to pick the control or 
        time_dependent_parameters when they are not constant over
        two points in the transition separated by a stepsize > 1
            dict
        pick_from_single_traj: Whether to pick the transitions from a single
        trajectory or not
            bool
            
    Returns:
        states: The states of dimension (batch_size, horizon+1, num_states)
            np.ndarray
        controls: The controls of dimension (batch_size, horizon, num_controls)
            np.ndarray
        time_dependent_parameters: dictionary with each key being of dimension
        (batch_size, horizon)
            dict
    """
    # Extract the special keys -> Typically the time dependent parameters
    special_keys = problem_config['names_controls'] + \
                    problem_config['time_dependent_parameters']

    # Pick the bacth of transiitons
    transitions, traj_infos = pick_batch_transitions_as_dict(
        dataset, batch_size, stepsize_range, horizon+1,
        special_keys, strategy, pick_from_single_traj,
        indx_trajectory = indx_trajectory, indx_transitions = indx_transitions
    )
    return convert_transitions_to_array(
        transitions, traj_infos, problem_config
    )

def sequential_loader_full_dataset(
    dataset : Dict[str, Any],
    batch_size : int,
    stepsize_range : Tuple[int, int],
    horizon : int,
    problem_config : Dict[str, Any],
    strategy : Dict[str, str],
) -> Callable[[], Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]]:
    """ A sequential loader for the full dataset.
    """
    # Define an array with the number of trajectories and transitions
    max_horizon = stepsize_range[1] * (horizon + 1)
    num_trans = \
        [ len(traj["time"]) - max_horizon for traj in dataset["trajectories"] ]
    num_trans_arr = np.array(num_trans)
    cum_sum_len = np.cumsum(num_trans_arr)
    shifted_cum_sum_len = np.concatenate(([0], cum_sum_len))

    # Define a 1D array with all the trajectories indexes
    traj_indexes_list = []
    for num in num_trans:
        indx_start = len(traj_indexes_list)
        traj_indexes_list.extend((indx_start + np.arange(num)).tolist())
    traj_indexes = np.array(traj_indexes_list)

    # Define the number of batches
    num_batches = len(traj_indexes) // batch_size
    print(f"Number of batches: {num_batches}")
    print(f"Number of transitions: {len(traj_indexes)}")

    # Define the function to load the batch
    def load_batch(m_traj_indexes : np.ndarray, n_batch : int = 0):

        # Pick the index of start of the batch
        start_idx = n_batch * batch_size
        end_idx = (n_batch + 1) * batch_size
        batch_indexes = m_traj_indexes[start_idx:end_idx]

        # Extract the actual trajectory index for 
        # batch_traj_indexes through cum_prod_len
        batch_traj_indexes = np.array(
            [ np.where(cum_sum_len > idx)[0][0] for idx in batch_indexes ]
        )
        batch_trans_start = np.array(
            [ idx - shifted_cum_sum_len[traj_idx] \
                for idx, traj_idx in zip(batch_indexes, batch_traj_indexes) ]
        )

        # Pick the batch of transitions
        return pick_batch_transitions_as_array(
            dataset, batch_size, stepsize_range,
            horizon, problem_config,
            strategy, pick_from_single_traj=False,
            indx_trajectory = batch_traj_indexes,
            indx_transitions = batch_trans_start
        )
    return load_batch, traj_indexes, num_batches


def convert_transitions_to_array(
    transitions : List[Dict[str, Any]],
    transitions_info : List[Dict[str, Any]], # Unused in this function
    problem_config : Dict[str, Any],
    ignore_last : bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Convert the list of transitions dictionary to relevant state, control, and
    time_dependent_parameters that typically used in the neural SDE model.
    
    Args:
        transitions: The transitions
            list of dict
        transitions_info: The trajectory information
            list of dict
        problem_config: The configuration of the problem containing the names
        of the states, controls, and extra arguments for the vector field or
        features.
            dict
        ignore_last: Whether to ignore the last control and 
        time_dependent_parameters. This is useful when predicting "horizon+1"
        states from "horizon" controls and time_dependent_parameters
            
    Returns:
        states: The states
            np.ndarray
        controls: The controls
            np.ndarray
        extra_args: The extra arguments
            dict
    """
    # Extract the state names
    names_states = problem_config['names_states']
    state_arr = np.array(
        [ np.array([ trans[state_v] for state_v in names_states ]).T \
            for trans in transitions
        ]
    )

    # Extract the control names
    names_controls = problem_config['names_controls']
    hor = -1 if ignore_last else state_arr.shape[1]
    control_arr = np.array(
        [ np.array([ trans[state_v][:hor] for state_v in names_controls ]).T \
            for trans in transitions
        ]
    )

    # Extract the extra arguments
    time_dependent_parameters_names = problem_config['time_dependent_parameters']
    time_dependent_parameters = {
        key : np.array([trans[key][:hor] for trans in transitions])
        for key in time_dependent_parameters_names
    }

    return state_arr, control_arr, time_dependent_parameters

def pick_trajectory(
    dataset : Dict[str, Any],
    idx : Union[List[int], np.array],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Pick a list of trajectory from the dataset.
    
    Args:
        dataset: The dataset
            dict
        idx: The index of the trajectory
            list of int
            
    Returns:
        trajectory: The trajectory
            list of dict
        trajectory_info: The trajectory information
            dict
    """
    # Pick the trajectory
    trajectory = [dataset["trajectories"][i] for i in idx]
    trajectory_info = [dataset["trajectories_info"][i] for i in idx]
    return trajectory, trajectory_info

def pick_random_trajectory(
    dataset : Dict[str, Any],
    num_samples : int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Pick a random trajectory from the dataset.
    
    Args:
        dataset: The dataset
            dict
        num_samples: The number of trajectories to pick
            int
            
    Returns:
        trajectory: The trajectory
            list of dict
        trajectory_info: The trajectory information
            dict
    """
    # Pick the trajectory
    num_trajs = len(dataset["trajectories"])
    idx = np.random.randint(0, num_trajs, num_samples)
    return pick_trajectory(dataset, idx)

def pick_transitions(
    trajectory : Dict[str, Any],
    idx : Union[List[int], np.ndarray],
    step_sizes : np.ndarray,
    special_keys : List[str],
    strategy : Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Pick a transition from the trajectory.
    
    Args:
        trajectory: The trajectory
            dict
        idx: The start indexes of the transitions to pick -> The indexes are 
        of right size and within the range of the trajectory
            list of int
        step_sizes: The cumulative number of steps to take from the start
        indexes in order to get the transitions
            array of int [len(idx), horizon]
        special_keys: The special keys for which we need to extract data
        differently according to a given strategy
            list of str
        strategy: The strategy to pick the data. Dictionary with a 'default' as
        a mandatory key and other keys as the special keys
            
    Returns:
        transitions: The transitions
            list of dict
    """
    # Pick the transitions
    transitions = [
        {
            key: value[i+step_indx] if key not in special_keys else \
                sampling_under_dataset_with_finer_steps(
                    value, strategy.get(key, strategy['default']), i+step_indx
                )
            for key, value in trajectory.items()
        }
        for i, step_indx in zip(idx, step_sizes)
    ]
    return transitions

def pick_random_transitions(
    trajectory : Dict[str, Any],
    num_transitions : int,
    stepsize_range : Tuple[int, int],
    horizon : int,
    special_keys : List[str],
    strategy : Dict[str, str],
    idx_transitions : Union[List[int], np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Pick random transitions from the trajectory.
    
    Args:
        trajectory: The trajectory
            dict
        num_transitions: The number of transitions to pick
            int
        stepsize_range: The range of the stepsize between two consecutive
        data points in a transition
            tuple of int
        horizon: The horizon of the transitions >= 2
            int
        special_keys: The special keys for which we need to extract data
        differently according to the strategy
            list of str
        strategy: The strategy to pick the data. Dictionary with a 'default' as
        a mandatory key and other keys as the special keys
        
    Returns:
        transitions: The transitions
            list of dict
    """
    # Pick the transitions
    total_num_steps = trajectory["time"].shape[0]
    step_indx_value = np.random.randint(
        stepsize_range[0], stepsize_range[1]+1, size=(num_transitions,horizon-1)
    )

    step_indx_value = np.concatenate(
        (np.zeros((num_transitions, 1), dtype=int),
         np.cumsum(step_indx_value, axis=1)
         ), axis=1
    )

    if idx_transitions is None:
        idx_transitions = np.random.randint(
            0, total_num_steps - step_indx_value[:,-1], num_transitions
        )

    return pick_transitions(
        trajectory, idx_transitions, step_indx_value, special_keys, strategy
    )

def pick_batch_transitions_as_dict(
    dataset : Dict[str, Any],
    batch_size : int,
    stepsize_range : Tuple[int, int],
    horizon : int,
    special_keys : List[str],
    strategy : Dict[str, str],
    pick_from_single_traj : bool = False,
    indx_trajectory : Union[List[int], np.ndarray] = None,
    indx_transitions : Union[List[int], np.ndarray] = None,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, Any]]]:
    """
    Pick a batch of transitions from the dataset.
    
    Args:
        dataset: The dataset
            dict
        batch_size: The batch size
            int
        stepsize_range: The range of the stepsize between two consecutive
        data points in a transition
            tuple of int
        horizon: The horizon of the transitions >= 2
            int
        special_keys: The special keys for which we need to extract data
        differently according to the strategy
            list of str
        strategy: The strategy to pick the data. Dictionary with a 'default' as
        a mandatory key and other keys as the special keys
            dict
        pick_from_single_traj: Whether to pick the transitions from a single
        trajectory or not
    
    Returns:
        transitions: The transitions
            list of dict
        trajs_info: The trajectory information
            dict
    """
    # Pick batch_size trajectories
    if indx_trajectory is None:
        if not pick_from_single_traj:
            trajs, trajs_info = pick_random_trajectory(dataset, batch_size)
        else:
            trajs, trajs_info = pick_random_trajectory(dataset, 1)
            trajs = [trajs[0] for _ in range(batch_size)]
            trajs_info = [trajs_info[0] for _ in range(batch_size)]
    else:
        trajs, trajs_info = pick_trajectory(dataset, indx_trajectory)

    transitions = \
    [
        pick_random_transitions(
            traj, 1, stepsize_range, horizon, special_keys, strategy,
            idx_transitions = None if indx_transitions is None else [indx_transitions[_i],]
        )[0] for _i, traj in enumerate(trajs)
    ]
    return transitions, trajs_info

# def pick_batch_transitions_as_dict_from_indexes(
#     dataset : Dict[str, Any],
#     batch_size : int,
#     stepsize_range : Tuple[int, int],
#     horizon : int,
#     special_keys : List[str],
#     strategy : Dict[str, str],
#     pick_from_single_traj : bool = False,
# ) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, Any]]]:
#     """
#     Pick a batch of transitions from the dataset.
    
#     Args:
#         dataset: The dataset
#             dict
#         batch_size: The batch size
#             int
#         stepsize_range: The range of the stepsize between two consecutive
#         data points in a transition
#             tuple of int
#         horizon: The horizon of the transitions >= 2
#             int
#         special_keys: The special keys for which we need to extract data
#         differently according to the strategy
#             list of str
#         strategy: The strategy to pick the data. Dictionary with a 'default' as
#         a mandatory key and other keys as the special keys
#             dict
#         pick_from_single_traj: Whether to pick the transitions from a single
#         trajectory or not
    
#     Returns:
#         transitions: The transitions
#             list of dict
#         trajs_info: The trajectory information
#             dict
#     """
#     # Pick batch_size trajectories
#     if not pick_from_single_traj:
#         trajs, trajs_info = pick_random_trajectory(dataset, batch_size)
#     else:
#         trajs, trajs_info = pick_random_trajectory(dataset, 1)
#         trajs = [trajs[0] for _ in range(batch_size)]
#         trajs_info = [trajs_info[0] for _ in range(batch_size)]

#     transitions = \
#     [
#         pick_random_transitions(
#             traj, 1, stepsize_range, horizon, special_keys, strategy
#         )[0] for traj in trajs
#     ]
#     return transitions, trajs_info


def sampling_under_dataset_with_finer_steps(
    arr : np.ndarray,
    strategy : str,
    target_indx : np.ndarray,
) -> np.ndarray:
    """
    Given an array of size (len, ...), target indexes that represent
    the indexes to pick, and a sampling strategy, this function returns
    the values between each two indexes according to some strategy.
    Assumption: Make sure target_indx is sorted
    
    Args:
        arr: array of size (horizon, num_extra_steps, ...)
            onp.ndarray
        strategy: sampling strategy
            str
        target_indx: target indexes to pick
    
    Returns:
        arr: array of size (horizon, ...)
            onp.ndarray
    """
    if strategy == 'first':
        arr = arr[target_indx]
    elif strategy == 'mean':
        target_indx = np.concatenate((target_indx, target_indx[-1:]+1))
        arr = np.array(
            [ arr[idx_start:idx_end].mean(axis=0) \
                for idx_start, idx_end in zip(target_indx[:-1], target_indx[1:])
            ]
        )
    elif strategy == 'median':
        target_indx = np.concatenate((target_indx, target_indx[-1:]+1))
        arr = np.array(
            [ np.median(arr[idx_start:idx_end], axis=0) \
                for idx_start, idx_end in zip(target_indx[:-1], target_indx[1:])
            ]
        )
    elif strategy == 'last':
        target_indx = np.concatenate((target_indx, target_indx[-1:]+1))
        arr = arr[target_indx[1:] - 1]
    elif strategy == 'random':
        target_indx = np.concatenate((target_indx, target_indx[-1:]+1))
        arr = np.array(
            [ arr[np.random.randint(idx_start, idx_end)] \
                for idx_start, idx_end in zip(target_indx[:-1], target_indx[1:])
            ]
        )
    else:
        raise ValueError(f'Unknown strategy: {strategy}.')

    return arr
