import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from typing import Any, Dict, Tuple, List

import copy
import pickle
import time
import gym
import d4rl
import jax
import argparse
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


from nsdes_dynamics.load_learned_nsdes import (
    load_system_sampler_from_model_name,
)
from nsdes_dynamics.utils_for_d4rl_mujoco import (
    load_dataset_for_nsdes,
    get_environment_infos_from_name,
)
from nsdes_dynamics.dataset_op import (
    pick_batch_transitions_from_trajectory_as_array
)
from nsdes_dynamics.parameter_op import (
    pretty_print_config
)

from models.tf_dynamics_models.bnn import BNN
from models.tf_dynamics_models.constructor import construct_model


def load_dataset(env_name, verbose=False):
    """ Load the dataset and print infos about the different environments
    """
    dataset = load_dataset_for_nsdes(env_name)
    if not verbose:
        return dataset

    print(f"\tNumber of trajectories: {len(dataset['trajectories'])}")
    trajectory_infos = dataset["trajectories_info"]
    print("\tTrajectory information:")
    for info in trajectory_infos:
        for _k in info.keys():
            print(f"\t\t{_k}: {info[_k]}")
    # Maximum values per field
    max_value_per_field = dataset["max_values_per_field"]
    min_value_per_field = dataset["min_values_per_field"]
    mean_value_per_field = dataset["mean_values_per_field"]
    median_value_per_field = dataset["median_values_per_field"]
    print("\tMaximum values per field:")
    for k in max_value_per_field.keys():
        print(f"\t\t{k}: {max_value_per_field[k]}")
    print("\tMinimum values per field:")
    for k in min_value_per_field.keys():
        print(f"\t\t{k}: {min_value_per_field[k]}") 
    print("\tMean values per field:")
    for k in mean_value_per_field.keys():
        print(f"\t\t{k}: {mean_value_per_field[k]}")
    print("\tMedian values per field:")
    for k in median_value_per_field.keys():
        print(f"\t\t{k}: {median_value_per_field[k]}")
    # Print the fields in the dataset
    print(f"\tFields in the dataset: {dataset['data_fields']}")
    return dataset

# Function to do model prediction and error computation
def model_xpred_from_trajs(
    sampling_fn,
    trajectory: Dict[str, np.ndarray],
    trajectory_info: Dict[str, Any],
    rng_key: jnp.ndarray,
    names_states: List[str],
    names_controls: List[str],
    horizon_pred: int,
    num_splits: int = 5,
    num_steps: int = 1,
    in_batch: bool = False,
) -> Dict[str, jnp.ndarray]:
    """
    Predict trajectories using the given model and the trajectory.
    
    Args:
        sampling_fn: The sampling function
            Callable
        trajectory: The trajectory
            Dict[str, np.ndarray]
        trajectory_info: The trajectory information
            Dict[str, Any]
        rng_key: The random key
            jnp.ndarray
        names_states: The names of the states
            List[str]
        names_controls: The names of the controls
            List[str]
    
    Returns:
        dict_xevol: The predicted trajectories.
            Dict[str, jnp.ndarray]
    """
    # Problem config to extract relevant info from dataset
    problem_config = {
        "names_states" : names_states + ["time"],
        "names_controls" : names_controls,
        "time_dependent_parameters" : []
    }

    # Some parameters needed for extracting sequences from the trajectory
    data_num_points = len(trajectory["time"])
    num_points_predicton = num_steps * horizon_pred
    # num_sequences = (data_num_points - 1) // num_points_predicton
    num_sequences = data_num_points - 1 - num_points_predicton
    valid_num_traj_points = num_sequences * num_points_predicton

    # Some checks
    assert valid_num_traj_points > 0, \
        f"valid_num_traj_points {valid_num_traj_points} should be greater than 0"
    assert num_splits > 0, \
        f"num_splits {num_splits} should be greater than 0"
    assert num_splits <= num_sequences, \
        f"num_splits {num_splits} should be less than " + \
        f"num_sequences {num_sequences}"

    # print(f"valid_num_traj_points: {valid_num_traj_points}")
    # print(f"num_splits: {num_splits}")
    # print(f"num_sequences: {num_sequences}")
    # print(f"num_points_predicton: {num_points_predicton}")

    # Iterate through the number of split trajectory and perform the prediction
    num_sequences_split = num_sequences // num_splits

    # print(f"num_sequences_split: {num_sequences_split}")
    result_list = []
    for n_split in range(num_splits):
        # print(f"Split {n_split}")
        # How many sequences to use for this iteration based on the split
        num_splits_for_iter = \
            num_sequences_split if n_split < num_splits - 1 else \
            (num_sequences - n_split * num_sequences_split)
        # num_splits_for_iter = num_sequences_split

        states_list = []
        controls_list = []

        for i in range(num_splits_for_iter):
            start_idx = n_split * num_sequences_split + i
            stepsizes = np.array(
                [ num_steps * j for j in range(horizon_pred+1)]
            )
            # Extract the sequence
            states, controls, _ = \
                pick_batch_transitions_from_trajectory_as_array(
                    trajectory, trajectory_info, start_idx, stepsizes,
                    problem_config, {"default": "first"}
                )

            # Append the states, controls and time_dependent_parameters
            states_list.append(states)
            controls_list.append(controls)

        # Merge the states, controls, and time_dependent_parameters
        states = np.stack(states_list, axis=0)
        controls = np.stack(controls_list, axis=0)

        # Sample the trajectories
        if in_batch:
            x_pred, _ = sampling_fn(
                state=states[:,0,:-1], control=controls, rng_key=None
            )
        else:
            x_pred_list = []
            for _x, _u in zip(states, controls):
                rng_key, temp_key = jax.random.split(rng_key)
                x_pred, _ = sampling_fn(
                    state=_x[0,:-1], control=_u, rng_key=temp_key
                )
                x_pred_list.append(x_pred)
            x_pred = np.stack(x_pred_list, axis=0)

        # Separate the time and the states from ground truth
        time_gt = states[...,-1]
        states_gt = states[...,:-1]
        states_dictionary = {}
        for _k, name in enumerate(names_states):
            states_dictionary[name+"_gt"] = states_gt[..., _k]
            states_dictionary[name+"_pred"] = x_pred[..., _k]
        states_dictionary["time_pred"] = time_gt
        result_list.append(states_dictionary) # Append the results
    

    # Stack the results along the batch dimension
    stacked_results = {
        key: np.concatenate([res[key] for res in result_list], axis=0)
        for key in result_list[0].keys()
    }
    return stacked_results


def compute_error_metrics(
    infos: Dict[str, np.ndarray],
    names_states: List[str],
) -> Dict[str, np.ndarray]:
    """
    Compute the error metrics for the predicted trajectories.
    
    Args:
        infos: The information about the predicted trajectories.
            Dict[str, np.ndarray]
        names_states: The names of the states.
            List[str]
    
    Returns:
        error_metrics: The error metrics for the predicted trajectories.
            Dict[str, np.ndarray]
    """
    # Compute the error metrics
    res_dict = {}
    total_error = None

    for name in names_states:
        state_gt = infos[name + "_gt"] # [B, H]
        state_pred = infos[name + "_pred"] # [B, Particle, H]

        # Two type of errors. One is the error on the mean trajectory and
        # the other is the mean of error over particle trajectories
        mean_pred_state = np.mean(state_pred, axis=1)
        error_on_mean = np.cumsum(
            np.abs(mean_pred_state - state_gt), axis=-1
        )

        mean_on_error_val = np.abs(state_pred - \
            state_gt.reshape((state_gt.shape[0], 1, state_gt.shape[1])))
        if total_error is None:
            total_error = np.zeros_like(state_pred)
        total_error += np.power(mean_on_error_val, 2) 

        # mean_on_error_std = np.cumsum(
        #     np.std(mean_on_error_val, axis=1), axis=-1
        # )
        # mean_on_error_val = np.cumsum(
        #     np.mean(mean_on_error_val, axis=1), axis=-1
        # )
        # res_dict[name + "_mean_of_error"] = mean_on_error_val
        # res_dict[name + "_std_of_error"] = mean_on_error_std
        # res_dict[name + "_error_of_mean"] = error_on_mean

    # Compute the total error
    total_error = np.sqrt(total_error)
    # mean of total error over horizon
    res_dict["total_error"] = np.cumsum(total_error, axis=-1)
    res_dict["total_error_mean"] = np.mean(total_error, axis=1)
    res_dict["total_error_std"] = np.std(total_error, axis=1)

    return res_dict

def load_ensemble(load_dir, task='halfcheetah-random-v2', algo='tatu_mopo'):
    model_dir = os.path.join('log', task, algo, load_dir, 'dynamics_model')
    env = gym.make(task)
    obs_shape = env.observation_space.shape
    action_dim = np.prod(env.action_space.shape)
    dynamics_model = construct_model(
        obs_dim=np.prod(obs_shape),
        act_dim=action_dim,
        hidden_dim=200,
        num_networks=7,
        num_elites=5,
        model_type="mlp",
        separate_mean_var=True,
        load_dir=model_dir,
        name="BNN_0",
    )
    return dynamics_model

def predict_with_ensemble(dynamics_model, num_traj=1, deterministic=False):
    def ensemble_sampling_fn(state, control, rng_key):
        x  = state
        us = control
        x = np.repeat(x, num_traj, axis=0)
        xs = [x]
        stds = [np.zeros_like(x)]        
        for t in range(us.shape[1]):
            u = np.repeat(us[:,t], num_traj, axis=0)
            inputs = np.concatenate((x, u), axis=-1)
            ens_model_means, ens_model_vars = dynamics_model.predict(inputs, factored=True)
            ens_model_means = ens_model_means[:,:,1:] + x # Remove reward and move
            ens_model_stds = np.sqrt(ens_model_vars[:,:,1:])
            if deterministic:
                ens_samples = ens_model_means
                samples = np.mean(ens_samples, axis=0)
                model_means = np.mean(ens_model_means, axis=0)
                model_stds = np.mean(ens_model_stds, axis=0)
            else:
                ens_samples = ens_model_means + np.random.normal(size=ens_model_means.shape) * ens_model_stds
                #### choose one model from ensemble
                num_models = ens_model_means.shape[0]
                model_inds = np.random.choice(num_models, size=ens_model_means.shape[1])
                samples = np.array([ens_samples[model_ind,i,:] for i, model_ind in enumerate(model_inds)])
                model_means = np.array([ens_model_means[model_ind,i,:] for i, model_ind in enumerate(model_inds)])
                model_stds = np.array([ens_model_stds[model_ind,i,:] for i, model_ind in enumerate(model_inds)])
            x = samples
            xs.append(x)
            stds.append(model_stds)
        xs = np.array(xs).transpose(1,0,2)
        stds = np.array(stds).transpose(1,0,2)
        return xs.reshape(-1,num_traj,xs.shape[1],xs.shape[2]),  stds.reshape(-1,num_traj,stds.shape[1],stds.shape[2])
    return ensemble_sampling_fn

def get_models_dict(task, models_to_evaluate, data_stepsize, data_horizon=10, num_steps=1, num_particles=10):
    models_dict = {}
    # Load the different models
    for i, model in enumerate(models_to_evaluate):
        print(f"{i+1}. Loading model {model['model_name']}\n")
        model_name = model["model_name"]
        model_plot_name = model["plot_name"]
        model_step = model.get("step", -2)
        is_gaussian = model.get("is_gaussian", False)
        if not is_gaussian:
            sampling_fn = load_system_sampler_from_model_name(
                env_name = task, model_name = model_name,
                stepsize = data_stepsize*num_steps, horizon = data_horizon,
                step = model_step, integration_method="euler_maruyama",
                num_particles=num_particles
            )
            jit_sampling = jax.jit(sampling_fn)
        else:
            gaussian_model = load_ensemble(
                model_name, task=task, algo='tatu_mopo'
            )
            jit_sampling = predict_with_ensemble(
                dynamics_model=gaussian_model, num_traj=num_particles,
                deterministic=False
            )
        models_dict[model_plot_name] = {
            "sampling_fn" : jit_sampling,
            "is_gaussian" : is_gaussian
        }
        print(f"Model {model_name} loaded\n")
    return models_dict

def compute_results(task, results_path, models_to_evaluate, data_horizon=50, num_steps=1, num_particles=10, num_splits=1, eval_traj_type=1):
    dataset = load_dataset(task, verbose=False)
    env_infos = get_environment_infos_from_name(task)
    models_dict = get_models_dict(task, models_to_evaluate, env_infos["stepsize"], 
                                  data_horizon, num_steps, num_particles)   

    names_states = env_infos["names_states"]
    names_controls = env_infos["names_controls"] 
    traj_indexes_to_evaluate = get_traj_ids_to_evaluate(len(dataset["trajectories"]), eval_traj_type)
    rng_key = jax.random.PRNGKey(10)
    error_dict = {}
    for traj_id in traj_indexes_to_evaluate:
        curr_trajectory = dataset["trajectories"][traj_id]
        curr_trajectory_info = dataset["trajectories_info"][traj_id]
        print("\nTraj id: ", traj_id)
        rng_key, model_key = jax.random.split(rng_key)
        for model_name, model_dict in models_dict.items():
            start_time = time.time()
            pred_trajectory = model_xpred_from_trajs(
                model_dict["sampling_fn"],
                curr_trajectory,
                curr_trajectory_info,
                rng_key=model_key,
                names_states=names_states,
                names_controls=names_controls,
                horizon_pred=data_horizon,
                num_steps=num_steps,
                num_splits=num_splits,
                in_batch=model_dict["is_gaussian"],
            )
            end_time = time.time()
            error_metrics = compute_error_metrics(
                pred_trajectory, names_states
            )
            total_error_mean = np.mean(error_metrics['total_error_mean'],axis=0)[-1]
            total_error_std = np.mean(error_metrics['total_error_std'],axis=0)[-1]
            print(f"Model: {model_name} - Total error: {total_error_mean} - Std: {total_error_std} - Time taken: {end_time - start_time}")
            if model_name not in error_dict:
                error_dict[model_name] = {error_key: error_arr for error_key, error_arr in error_metrics.items()}
            else:
                error_dict[model_name] = {error_key: np.concatenate((error_dict[model_name][error_key], error_metrics[error_key])) for error_key in error_metrics.keys()}

    results_dict = {}
    for model_name, error_dict_model in error_dict.items():
        results_dict[model_name] = {}
        for error_key, error_arr in error_dict_model.items():
            results_dict[model_name][f'{error_key}_mean'] =  np.mean(error_arr, axis=0)
            results_dict[model_name][f'{error_key}_std'] = np.std(error_arr, axis=0)

    with open(results_path, "wb") as f:
        pickle.dump(results_dict, f)
    return results_dict

def get_traj_ids_to_evaluate(dataset_len, eval_traj_type):
    if eval_traj_type == 0:
        return list(range(dataset_len))
    elif eval_traj_type < 0:
        if -1 * eval_traj_type > dataset_len:
            raise ValueError("Invalid eval_traj_type")
        return [-1 * eval_traj_type]
    else:
        return list(range(eval_traj_type))
    
def plot_results(results_dict, task, fig_path):
    plt.figure(figsize=(10, 6))
    for model_name, result_dict in results_dict.items():
        mean_error_vs_horizon = result_dict['total_error_mean_mean']
        std_error_vs_horizon = result_dict['total_error_mean_std']
        plt.errorbar(
            np.arange(mean_error_vs_horizon.shape[0]),
            mean_error_vs_horizon,
            yerr=std_error_vs_horizon,
            label=model_name
            )

    plt.xlabel("Horizon")
    plt.ylabel("Cumulative error")
    plt.yscale('log')
    plt.legend()
    plt.title(f"Prediction error vs horizon for {task}")
    plt.savefig(f"{fig_path}.png")

# main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="halfcheetah-medium-expert-v2")
    parser.add_argument("--results-dir", type=str, default="model_analysis_results")
    parser.add_argument("--load-results", default=False, action='store_true')
    parser.add_argument("--plot-results", default=False, action='store_true')
    parser.add_argument("--data-horizon", type=int, default=50)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--num-particles", type=int, default=10)
    parser.add_argument("--num-splits", type=int, default=1)
    parser.add_argument("--eval-traj-type", type=int, default=0) # 0 for all, <0 for specific, >0 for aranged list
    args= parser.parse_args()

    models_to_evaluate = {
        "halfcheetah-medium-expert-v2": [
            { 
                "model_name" : "hc_me_final",
                "plot_name" : "NSDE",
                "step" : -2, # The best model
            },
            {
                "model_name" : "critic_num_2_seed_32_0518_155659-halfcheetah_medium_expert_v2_tatu_mopo",
                "plot_name" : "Ens",
                "is_gaussian" : True,
            }
        ],
        "halfcheetah-medium-replay-v2": [
            { 
                "model_name" : "hc_mr_final",
                "plot_name" : "NSDE",
                "step" : -2, # The best model
            },
            {
                "model_name" : "critic_num_2_seed_32_0518_155620-halfcheetah_medium_replay_v2_tatu_mopo",
                "plot_name" : "Ens",
                "is_gaussian" : True,
            }
        ],
        "halfcheetah-medium-v2": [
            { 
                "model_name" : "hc_m_final",
                "plot_name" : "NSDE",
                "step" : -2, # The best model
            },
            {
                "model_name" : "critic_num_2_seed_32_0518_155450-halfcheetah_medium_v2_tatu_mopo",
                "plot_name" : "Ens",
                "is_gaussian" : True,
            }
        ],
        "halfcheetah-random-v2": [
            { 
                "model_name" : "hc_rand_final",
                "plot_name" : "NSDE",
                "step" : -2, # The best model
            },
            {
                "model_name" : "critic_num_2_seed_32_0518_155708-halfcheetah_random_v2_tatu_mopo",
                "plot_name" : "Ens",
                "is_gaussian" : True,
            }
        ],
    }
    
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    
    results_path = f"{args.results_dir}/results_{args.task}_type={args.eval_traj_type}.pkl"

    if os.path.exists(results_path) and args.load_results:
        with open(results_path, "rb") as f:
            results_dict = pickle.load(f)
    else:
        results_dict = compute_results(
            task=args.task, results_path=results_path,
            models_to_evaluate=models_to_evaluate[args.task],
            data_horizon=args.data_horizon, num_steps=args.num_steps,
            num_particles=args.num_particles, num_splits=args.num_splits,
            eval_traj_type=args.eval_traj_type
        )
    
    if args.plot_results:
        fig_path = f"{args.results_dir}/fig_{args.task}_type={args.eval_traj_type}"
        plot_results(results_dict, args.task, fig_path)