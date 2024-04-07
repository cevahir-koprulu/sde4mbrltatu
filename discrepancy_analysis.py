# import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np

import jax
import jax.numpy as jnp

from tqdm.auto import tqdm

from nsdes_dynamics.load_learned_nsdes import (
    load_system_sampler_from_model_name,
)

from nsdes_dynamics.utils_for_d4rl_mujoco import (
    load_dataset_for_nsdes,
    get_environment_infos_from_name,
)


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

def load_diffusion_and_disc(
    model_name: str,
    step: int,
    task_name: str,
    num_particles: int,
    stepsize: float,
    horizon: int,
    rollout_batch_size : int,
    integration_method = None,
    pert_scale = 0.1,
):
    """ Createa function to analyze the diffusion term and compute the discrepancy
    over the training dataset
    """
    # Load the model sampler
    sampler_fn = load_system_sampler_from_model_name(
        task_name,
        model_name = model_name,
        stepsize = stepsize,
        horizon = horizon,
        num_particles = num_particles,
        step = step,
        integration_method=integration_method
    )

    @jax.jit
    def _my_pred_fn(
        x : jax.numpy.ndarray,
        u : jax.numpy.ndarray,
        rng : jax.random.PRNGKey
    ):
        # Some checks
        assert x.ndim == 2 and u.ndim == 3 and \
            x.shape[0] == u.shape[0], f"Invalid shapes: {x.shape} {u.shape}"
        assert u.shape[1] == horizon, f"Invalid u horizon: {u.shape[1]}"

        # Get matching rng keys
        rng, rng_pert = jax.random.split(rng)
        rng = jax.random.split(rng, x.shape[0])

        rng_pert, rng_pert_u = jax.random.split(rng_pert)
        rng_pert = jax.random.split(rng_pert, x.shape[0])

        # Compute the pertubated control inputs
        u_pert = u + pert_scale * jax.random.normal(rng_pert_u, u.shape)

        # Define the vmap sampler function
        _temp_sampler =  lambda _x, _u, _rng: \
            sampler_fn(state=_x, control=_u, rng_key=_rng)

        # Compute the predictions
        pred_states, pred_feats = \
            jax.vmap(_temp_sampler)(x, u, rng)
        pred_states_pert, pred_feats_pert = \
            jax.vmap(_temp_sampler)(x, u_pert, rng_pert)
        # Let's remove the first state
        pred_states = pred_states[:, :, 1:, :]
        pred_states_pert = pred_states_pert[:, :, 1:, :]

        diffs = pred_states - jnp.expand_dims(jnp.mean(pred_states, axis=1), axis=1)
        disc = jnp.linalg.norm(diffs, axis=-1)
        disc = jnp.cumsum(disc, axis=-1)

        diffs_pert = pred_states_pert - jnp.expand_dims(jnp.mean(pred_states_pert, axis=1), axis=1)
        disc_pert = jnp.linalg.norm(diffs_pert, axis=-1)
        disc_pert = jnp.cumsum(disc_pert, axis=-1)

        # Get the diffusion term
        pred_feats = {
            k : jnp.cumsum(jnp.linalg.norm(v, axis=-1), axis=-1) \
                for k, v in pred_feats.items() \
                    if k in ["diffusion_value", "dad_free_diff",
                             "dad_based_diff", "diff_density"]
        }
        pred_feats_pert = {
            f'{k}_pert' : jnp.cumsum(jnp.linalg.norm(v, axis=-1), axis=-1) \
                for k, v in pred_feats_pert.items() \
                    if k in ["diffusion_value", "dad_free_diff",
                             "dad_based_diff", "diff_density"]
        }

        result_dict = {
            "disc" : disc,
            "disc_pert" : disc_pert,
            **pred_feats,
            **pred_feats_pert
        }
        return result_dict

    def augmented_pred_fn(x, u , rng):
        """ Wrapper around prediction function to varying 
        batch sizes.
        """
        batch_x = x.shape[0]
        last_x = x[-1]
        last_u = u[-1]
        # Complte x, u to have a batch size of rollout_batch_size
        # so that to avoid recompilation
        if batch_x < rollout_batch_size:
            last_x = np.repeat(last_x[None], rollout_batch_size-batch_x, axis=0)
            last_u = np.repeat(last_u[None], rollout_batch_size-batch_x, axis=0)
            x = np.concatenate((x, last_x), axis=0)
            u = np.concatenate((u, last_u), axis=0)
        res = _my_pred_fn(x, u, rng)
        res = { k : np.array(v) for k, v in res.items()}
        # if batch_x < rollout_batch_size:
        res  = {
            k : v[:batch_x].reshape((-1,v.shape[-1])) for k, v in res.items()
        }
        return res

    return augmented_pred_fn

def compute_discrepancy_on_full_dataset(
    _dataset,
    pred_fn,
    horizon,
    rollout_batch_size: int,
    _names_states,
    _names_controls,
    seed = 10
):
    """ Compute the discrepancy on the full dataset
    """
    trajectories = _dataset["trajectories"]
    rng = jax.random.PRNGKey(seed)
    res_list = []
    for traj in tqdm(trajectories):
        # Now we want to every batch of size rollout_batch_size
        # to compute the discrepancy
        num_transitions = (len(traj["time"]) - horizon) // horizon
        num_batches = num_transitions // rollout_batch_size
        num_batches = num_batches + 1 if num_transitions % rollout_batch_size != 0 else num_batches
        # print(f"num_batches {num_batches}, num_transitions {num_transitions}")
        for indx_batch in range(2):
            start_indx = indx_batch * rollout_batch_size * horizon
            end_indx = (indx_batch+1) * rollout_batch_size * horizon
            if indx_batch == num_batches - 1:
                end_indx = num_transitions * horizon
            # print(f"start_indx: {start_indx}, end_indx: {end_indx}")
            states = np.array(
                [traj[name_state][start_indx:end_indx:horizon] for name_state in _names_states]
            ).T
            # print(states.shape)
            controls = np.array(
                [ [traj[name_control][i:i+horizon] for i in range(start_indx, end_indx, horizon)] \
                    for name_control in _names_controls
                ]
            ).transpose((1,2,0))
            # Compute the discrepancy
            rng, _rng = jax.random.split(rng)
            res = pred_fn(states, controls, _rng)
            res_list.append(res)
    res_names = res_list[0].keys()
    stacked_results = {
        k : np.concatenate([r[k] for r in res_list], axis=0) for k in res_names
    }
    for k in res_names:
        print(f"Shape of {k}: {stacked_results[k].shape}")
    return stacked_results

dataset_name = "halfcheetah-medium-expert-v2"
# dataset_name = "halfcheetah-random-v2"
dataset = load_dataset(dataset_name, verbose=False)

# Define the models to evaluate hc_rand_v2_
models_to_evaluate = \
[
    # { # A learned model
    #     "model_name" : "hc_rand_v7",
    #     "plot_name" : "Learned v1",
    #     "step" : -2, # The best model
    #     "task_name" : "halfcheetah-random-v2"
    # },
    # { # A learned model
    #     "model_name" : "hc_rand_v13___",
    #     "plot_name" : "Learned v2",
    #     "step" : -2, # The best model
    #     "task_name" : "halfcheetah-random-v2"
    # },
    # {
    #     "model_name" : "critic_num_2_seed_32_0128_215130-halfcheetah_medium_expert_v2_tatu_mopo",
    #     "plot_name" : "Ens",
    #     "is_gaussian" : True,
    # },
    # { # A learned model
    #     "model_name" : "hc_me_v31",
    #     "plot_name" : "Learned v1",
    #     "step" : -2, # The best model
    #     "task_name" : "halfcheetah-medium-expert-v2"
    # },
    { # A learned model
        "model_name" : "hc_me_v11",
        "plot_name" : "Learned v2",
        "step" : -2, # The best model
        "task_name" : "halfcheetah-medium-expert-v2"
    },
]
env_infos = get_environment_infos_from_name(dataset_name)
data_stepsize = env_infos["stepsize"]
data_horizon = 5 # Rollout length for example
num_steps = 1 # 1 means stepsize is 0.05, 2 means stepsize is 0.1 etc.
num_particles = 5
names_states = env_infos["names_states"]
names_controls = env_infos["names_controls"]
pert_scale = 0.2
rollout_batch_discr = 150


models_dict = {}
# Load the different models
for i, model in enumerate(models_to_evaluate):
    print(f"{i+1}. Loading model {model['model_name']}\n")
    model_name = model["model_name"]
    model_plot_name = model["plot_name"]
    model_step = model.get("step", -2)
    discr_fn = load_diffusion_and_disc(
        model_name = model_name,
        step = model_step,
        task_name = model.get("task_name", dataset_name),
        num_particles = num_particles,
        stepsize = data_stepsize,
        horizon = data_horizon,
        rollout_batch_size = rollout_batch_discr,
        pert_scale = pert_scale
    )
    models_dict[model_plot_name] = discr_fn
    print(f"{i+1}. Loaded model {model_name}\n")

# Compute the discrepancy
discrepancy_results = {}
for model_name, discr_fn in models_dict.items():
    print(f"Computing discrepancy for model {model_name}")
    discrepancy_results[model_name] = compute_discrepancy_on_full_dataset(
        dataset,
        discr_fn,
        data_horizon,
        rollout_batch_discr,
        names_states,
        names_controls
    )
    print(f"Computed discrepancy for model {model_name}")


# Let's do some plotting
# List of colors for the plots
colors = ["b", "r", "g", "c", "m", "y", "k"]
color_per_model = {model_name : colors[i] for i, model_name in enumerate(models_dict.keys())}

import seaborn as sns
import matplotlib.pyplot as plt

num_steps_error = -1 # -1 Errot analysis till the last step. 1 means error analysis till the first step
fields_to_plot = discrepancy_results[list(discrepancy_results.keys())[0]].keys()

# Lets plot some of the forces
per_subplot_figsize = (3,3)
max_num_cols = 4
num_rows = len(fields_to_plot) // max_num_cols
num_rows = num_rows if len(fields_to_plot) % max_num_cols == 0 else num_rows + 1
num_cols = len(fields_to_plot) \
    if len(fields_to_plot) < max_num_cols else max_num_cols

# Now let's plot the histograms
for model_name, discr_res in discrepancy_results.items():
    fig, axs = plt.subplots(num_rows, num_cols, 
                        figsize=(num_cols*per_subplot_figsize[0],
                                    num_rows*per_subplot_figsize[1]),
                        constrained_layout=True,
    )
    axs = axs.flatten()
    for i, field_name in enumerate(fields_to_plot):
        sns.histplot(
            discr_res[field_name][:, num_steps_error],
            ax=axs[i], color=color_per_model[model_name],
            label=model_name, kde=False,
            # stat="density",
            fill=True, alpha=0.8
        )
        axs[i].set_xlabel(f"{field_name}")
        axs[i].grid(True)
        axs[i].legend()

# for i, field_name in enumerate(fields_to_plot):
#     for model_name, discr_res in discrepancy_results.items():
#         sns.histplot(
#             discr_res[field_name][:, num_steps_error],
#             ax=axs[i], color=color_per_model[model_name],
#             label=model_name, kde=True, stat="density",
#             fill=True, alpha=0.8
#         )
#     # axs[i].set_title(f"{field_name}")
#     axs[i].set_xlabel(f"{field_name}")
#     axs[i].grid(True)
#     axs[i].legend()

plt.show()