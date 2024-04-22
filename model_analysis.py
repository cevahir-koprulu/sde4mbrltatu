# %%
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from typing import Any, Dict, Tuple, List

import numpy as np

import copy
import pickle
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import jax
import jax.numpy as jnp

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


import matplotlib.pyplot as plt


# %%
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

def plot_dataset(
    dataset : Dict[str, Any],
    indx_traj : int,
    xaxis_name : str = "time",
    per_subplot_figsize : Tuple[float, float] = (3.5, 3.5)
) -> None:
    """
    Plot the dataset.
    
    Args:
        dataset: The dataset
            dict
        indx_traj: The index of the trajectory to plot
            int
    """
    print(f"Plotting the trajectory {indx_traj} of the dataset")
    print(dataset["trajectories_info"][indx_traj])
    trajectories = dataset["trajectories"]
    num_traj = len(trajectories)
    if indx_traj >= num_traj:
        raise ValueError(
            f"The index of the trajectory should be less than {num_traj}"
        )

    # Field to plot
    field_to_plot = [field for field in dataset["data_fields"] if field != xaxis_name]
    x_array = trajectories[indx_traj][xaxis_name]

    # Create the figure with the subplots
    num_cols = 4
    num_rows = len(field_to_plot) // num_cols
    if len(field_to_plot) % num_cols != 0:
        num_rows += 1
    single_subplot_fig_size = per_subplot_figsize
    _, axs = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * single_subplot_fig_size[0],
        num_rows * single_subplot_fig_size[1]),
        sharex=True
    )

    # Plot the data
    for i, field in enumerate(field_to_plot):
        row = i // num_cols
        col = i % num_cols
        axs[row, col].plot(x_array, trajectories[indx_traj][field])
        axs[row, col].set_ylabel(field)
        if row == num_rows - 1:
            axs[row, col].set_xlabel(xaxis_name)
        axs[row, col].grid(True)

# %%
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
    num_steps: int = 1
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
    num_sequences = (data_num_points - 1) // num_points_predicton
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

    result_list = []
    for n_split in range(num_splits):
        # How many sequences to use for this iteration based on the split
        num_splits_for_iter = \
            num_sequences_split if n_split < num_splits - 1 else \
            (num_sequences - n_split * num_sequences_split)
        # num_splits_for_iter = num_sequences_split

        # print("num_splits_for_iter: ", num_splits_for_iter)
        states_list = []
        controls_list = []

        for i in range(num_splits_for_iter):
            # Where to start the sequence
            start_idx = n_split * num_sequences_split * num_points_predicton + \
                i * num_points_predicton

            # The stepsizes to use for the sequence
            stepsizes = np.array(
                [ num_steps * j for j in range(horizon_pred+1)]
            )
            # print("start_idx: ", start_idx)
            # print("stepsizes: ", stepsizes)

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
        # print("states and controls: ", states.shape, controls.shape)

        # Sample the trajectories
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
        mean_on_error_std = np.cumsum(
            np.std(mean_on_error_val, axis=1), axis=-1
        )
        mean_on_error_val = np.cumsum(
            np.mean(mean_on_error_val, axis=1), axis=-1
        )
        res_dict[name + "_mean_of_error"] = mean_on_error_val
        res_dict[name + "_std_of_error"] = mean_on_error_std
        res_dict[name + "_error_of_mean"] = error_on_mean

    return res_dict

# %%
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from models.tf_dynamics_models.bnn import BNN
from models.tf_dynamics_models.constructor import construct_model

import gym
import d4rl

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
        x = np.ones((num_traj,x.shape[0]))*x 
        xs = [x]
        stds = [np.zeros_like(x)]        
        for u in us:
            u_ = np.ones((num_traj, u.shape[0])) * u
            inputs = np.concatenate((x, u_), axis=-1)
            ens_model_means, ens_model_vars = dynamics_model.predict(inputs, factored=True)
            ens_model_means = ens_model_means[:,:,1:] + x # Remove reward and move
            ens_model_stds = np.sqrt(ens_model_vars[:,:,1:])
            # Let's print the stds
            # print("stds:\n", ens_model_stds[0,:2])
            if deterministic:
                ens_samples = ens_model_means
                samples = np.mean(ens_samples, axis=0)
                # model_means = np.mean(ens_model_means, axis=0)
                model_stds = np.mean(ens_model_stds, axis=0)
            else:
                ens_samples = ens_model_means + np.random.normal(size=ens_model_means.shape) * ens_model_stds
                #### choose one model from ensemble
                num_models = ens_model_means.shape[0]
                model_inds = np.random.choice(num_models, size=num_traj)
                samples = np.array([ens_samples[model_ind,i,:] for i, model_ind in enumerate(model_inds)])
                # model_means = np.array([ens_model_means[model_ind,i,:] for i, model_ind in enumerate(model_inds)])
                model_stds = np.array([ens_model_stds[model_ind,i,:] for i, model_ind in enumerate(model_inds)])
            x = samples
            xs.append(x)
            stds.append(model_stds)
        return np.array(xs).transpose(1,0,2), np.array(stds).transpose(1,0,2)
    return ensemble_sampling_fn

# %%
# Load a dataset
dataset_name = "halfcheetah-medium-expert-v2"
# dataset_name = "halfcheetah-random-v2"
# dataset_name = "halfcheetah-medium-replay-v2"
dataset_name_ens = "halfcheetah-medium-expert-v2"

dataset = load_dataset(dataset_name, verbose=False)

# %%
# indx_traj = 0
# plot_dataset(dataset, indx_traj, 
#              xaxis_name =  "time",
#              per_subplot_figsize= (3,3),
# )

# ensemble_dynamics = load_ensemble(load_dir=ensemble_model_name, task=dataset, algo='tatu_mopo')
#     ensemble_sampling_fn = predict_with_ensemble(ensemble_dynamics, num_traj=num_sample, deterministic=deterministic_ensemble)

# %%
# Define the models to evaluate hc_rand_v2_
models_to_evaluate = \
[
    # { # A learned model
    #     "model_name" : "hc_mr_new3_",
    #     "plot_name" : "Learned v0",
    #     "step" : -1, # The best model
    #     "task_name" : "halfcheetah-random-v2"
    # },
    { # A learned model
        "model_name" : "hc_mr_new6__",
        "plot_name" : "Learned v1",
        "step" : -2, # The best model
        "task_name" : "halfcheetah-random-v2"
    },
    { # A learned model
        "model_name" : "hc_rand_v13___",
        "plot_name" : "Learned v2",
        "step" : -2, # The best model
        "task_name" : "halfcheetah-random-v2"
    },
    # {
    #     "model_name" : "critic_num_2_seed_32_0128_215130-halfcheetah_medium_expert_v2_tatu_mopo",
    #     "plot_name" : "Ens",
    #     "is_gaussian" : True,
    # },
    # { # A learned model
    #     "model_name" : "hc_me_v7",
    #     "plot_name" : "Learned v1",
    #     "step" : -2, # The best model
    #     "task_name" : "halfcheetah-medium-expert-v2"
    # },
    # { # A learned model
    #     "model_name" : "hc_me_new",
    #     "plot_name" : "Learned v2",
    #     "step" : -2, # The best model
    #     "task_name" : "halfcheetah-medium-expert-v2"
    # },
]
env_infos = get_environment_infos_from_name(dataset_name)
data_stepsize = env_infos["stepsize"]
data_horizon = 10
num_steps = 1 # 1 means stepsize is 0.05, 2 means stepsize is 0.1 etc.
num_particles = 5
names_states = env_infos["names_states"]
names_controls = env_infos["names_controls"]

# %%
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
            env_name = model.get("task_name", dataset_name),
            model_name = model_name,
            stepsize = data_stepsize*num_steps, horizon = data_horizon,
            step = model_step, integration_method="euler_maruyama",
            num_particles=num_particles
        )
        jit_sampling = jax.jit(sampling_fn)
    else:
        gaussian_model = load_ensemble(
            model_name, task=dataset_name_ens, algo='tatu_mopo'
        )
        jit_sampling = predict_with_ensemble(
            dynamics_model=gaussian_model, num_traj=num_particles,
            deterministic=False
        )
    models_dict[model_plot_name] = jit_sampling
    print(f"Model {model_name} loaded\n")

# %%
# Let's compute model predictions and model errors
traj_indexes_to_evaluate = [8,4,5,6,7,1,2,9,10,11,12,13,14,15,16,17,18,19,20,21]
traj_indexes_to_evaluate = [ i for i in range(100)]
pred_trajectory_res = []
num_splits = 5 # For vmapping accross the computation -> the lower the more parallelism
rng_key = jax.random.PRNGKey(10)
for traj_id in traj_indexes_to_evaluate:
    curr_trajectory = dataset["trajectories"][traj_id]
    curr_trajectory_info = dataset["trajectories_info"][traj_id]
    pretty_print_config(curr_trajectory_info)
    temp_dict = {}
    rng_key, model_key = jax.random.split(rng_key)
    for model_name, model_sample in models_dict.items():
        print(f"Computing model predictions for model {model_name}...")
        pred_trajectory = model_xpred_from_trajs(
            model_sample,
            curr_trajectory,
            curr_trajectory_info,
            rng_key=model_key,
            names_states=names_states,
            names_controls=names_controls,
            horizon_pred=data_horizon,
            num_steps=num_steps,
            num_splits=num_splits,
        )
        # Compute error metrics
        error_metrics = compute_error_metrics(
            pred_trajectory, names_states
        )
        temp_dict[model_name] = {"pred" : pred_trajectory,
                                "error_metrics" : error_metrics
                                }
        print(f"Done\n")
        # exit(0)
    pred_trajectory_res.append((temp_dict, traj_id))

# %%
# List of colors for the plots
colors = ["b", "r", "g", "c", "m", "y", "k"]
color_per_model = {model_name : colors[i] for i, model_name in enumerate(models_dict.keys())}
# Let's display


# %%
# Let's plot the full trajectory prediction of the states
fields_to_plot = names_states
xaxis_toplot = "time"
# Lets plot some of the forces
per_subplot_figsize = (3,3)
max_num_cols = 4
num_rows = len(fields_to_plot) // max_num_cols
num_rows = num_rows if len(fields_to_plot) % max_num_cols == 0 else num_rows + 1
num_cols = len(fields_to_plot) \
    if len(fields_to_plot) < max_num_cols else max_num_cols
for (_pred_traj, _traj_id) in pred_trajectory_res[:1]:
    fig, axs = plt.subplots(num_rows, num_cols, 
                            figsize=(num_cols*per_subplot_figsize[0],
                                        num_rows*per_subplot_figsize[1]),
                            constrained_layout=True,
                            sharex=True
    )
    axs = axs.flatten()
    for i, field_name in enumerate(fields_to_plot):
        ax = axs[i]
        # Plot the ground truth first
        ax.plot(dataset["trajectories"][_traj_id]["time"], 
                dataset["trajectories"][_traj_id][field_name], 
                label="Ground truth" if i == 0 else None,
                c="k", linestyle="--",
                zorder=1000
        )
        # # Just plot the controls for the ground truth
        # if field_name in ["handwheel_angle",
        #         "rear_roadwheel_motor_torque",
        #         "front_roadwheel_brake_torque", 
        #         "rear_roadwheel_brake_torque"
        #         ]:
        #     ax.set_xlabel("time")
        #     ax.set_ylabel(field_name)
        #     ax.grid(True)
        #     continue
        for model_name, pred in _pred_traj.items():
            prediction_model = pred["pred"]
            color = color_per_model[model_name]
            xaxis_name = xaxis_toplot + "_pred"
            yaxis_name = field_name + "_pred"
            # Iteration over the different splits
            first_val = i == 0
            for _xval, _yval in zip(prediction_model[xaxis_name], 
                                    prediction_model[yaxis_name]):
                # Iteration over the different samples
                for _yval_sample in _yval:
                    ax.plot(_xval, _yval_sample, 
                            label=model_name if first_val else None,
                            c=color, linestyle="-"
                    )
                    first_val = False
        ax.set_xlabel(xaxis_toplot)
        ax.set_ylabel(field_name)
        ax.grid(True)
        ax.legend()




# Make and histogram of the errors for all trajectories. By using the error at the end of the horizon
fields_to_plot = names_states
type_plot = "mean_of_error" # "mean_of_error" # or error_of_mean
num_steps_error = -1 # -1 for all the steps

# Lets plot some of the forces
per_subplot_figsize = (3,3)
max_num_cols = 4
num_rows = len(fields_to_plot) // max_num_cols
num_rows = num_rows if len(fields_to_plot) % max_num_cols == 0 else num_rows + 1
num_cols = len(fields_to_plot) \
    if len(fields_to_plot) < max_num_cols else max_num_cols

# Let's merge all the data first
error_metrics_all = {}
for (_pred_traj, _traj_id) in pred_trajectory_res:
    for model_name, pred in _pred_traj.items():
        error_model = pred["error_metrics"]
        if model_name not in error_metrics_all:
            error_metrics_all[model_name] = {}
        for field_name in fields_to_plot:
            if field_name not in error_metrics_all[model_name]:
                error_metrics_all[model_name][field_name] = []
            suffix_name = "_mean_of_error" \
                if type_plot == "mean_of_error" else "_error_of_mean"
            value_error = error_model[field_name + suffix_name][:,num_steps_error]
            error_metrics_all[model_name][field_name].extend(
                value_error.tolist() # Sum of the errors till horizon
            )

# Now let's plot the histograms
fig, axs = plt.subplots(num_rows, num_cols, 
                        figsize=(num_cols*per_subplot_figsize[0],
                                    num_rows*per_subplot_figsize[1]),
                        constrained_layout=True,
)

# USe seaborn kdeplot for histogram
import seaborn as sns

axs = axs.flatten()
for i, field_name in enumerate(fields_to_plot):
    ax = axs[i]
    for model_name, error_model in error_metrics_all.items():
        color = color_per_model[model_name]
        sns.kdeplot(np.log(error_model[field_name]), 
                    label=model_name,
                    color=color, alpha=0.5,
                    ax=ax,
                    fill=True
        )
        # ax.hist(error_model[field_name], 
        #         label=model_name,
        #         color=color, alpha=0.5,
        #         # bins=50
        # )
    ax.set_xlabel(f"Error {field_name}")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    ax.legend()

plt.show()