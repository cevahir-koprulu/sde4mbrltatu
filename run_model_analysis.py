import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from typing import Any, Dict, Tuple, List

import copy
import pickle
import time
import gym
import d4rl
import jax
from tqdm import tqdm
import argparse
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from pathlib import Path

# os path to parent directory
# import sys
# sys.path.append("../")

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

def load_ensemble(load_dir, task='halfcheetah-random-v2', algo='tatu_mopo'):
    # model_dir = os.path.join(Path.cwd().parent,
    #     'log', task, algo, load_dir, 'dynamics_model')

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

def get_ensemble_pred_fn(dynamics_model, num_traj=1, deterministic=False):
    def ensemble_pred_fn(state, control, rng_key):
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
        return xs.reshape(-1,num_traj,xs.shape[1],xs.shape[2])[:,:,1:,:]
    return ensemble_pred_fn

def get_nsde_pred_fn(sampling_fn, is_reward_included, batch_size, horizon=10):
    def _est_fn(
            x : jax.numpy.ndarray,
            u : jax.numpy.ndarray,
            rng : jax.random.PRNGKey
        ):
        assert x.ndim == 2 and u.ndim == 3 and \
            x.shape[0] == u.shape[0], f"Invalid shapes: {x.shape} {u.shape}"
        assert u.shape[1] == horizon, f"Invalid u.shape[1]={u.shape[1]} vs horizon={horizon}"
        
        if is_reward_included:
            zero_last_dim = jax.numpy.zeros((x.shape[0], 1))
            x = jax.numpy.concatenate((x, zero_last_dim), axis=-1)

        rng = jax.random.split(rng, x.shape[0])
        _temp_sampler =  lambda _x, _u, _rng: \
            sampling_fn(state=_x, control=_u, rng_key=_rng)
        
        pred_states, _ = \
            jax.vmap(_temp_sampler)(x, u, rng)
        pred_states = pred_states[:, :, 1:, :]
        if is_reward_included:
            pred_states = pred_states[..., :-1]
        return pred_states
    jit_est_fn = jax.jit(_est_fn)

    def augmented_pred_fn(x, u , rng):
        batch_x = x.shape[0]
        last_x = x[-1]
        last_u = u[-1]
        if batch_x < batch_size:
            last_x = np.repeat(last_x[None], batch_size-batch_x, axis=0)
            last_u = np.repeat(last_u[None], batch_size-batch_x, axis=0)
            x = np.concatenate((x, last_x), axis=0)
            u = np.concatenate((u, last_u), axis=0)
        pred_traj = jit_est_fn(x, u, rng)
        return pred_traj[:batch_x] 
    
    return augmented_pred_fn
    
def run_analysis(
    dataset,
    pred_fn,
    horizon: int,
    skip_step: int,
    batch_size: int,
    _names_states,
    _names_controls,
    rng
):
    """ Compute the discrepancy on the full dataset
    """
    trajectories = dataset["trajectories"]
    result = {}
    num_short_trajectories = 0
    pbar = tqdm(trajectories, desc=f"Short traj= {num_short_trajectories/len(trajectories)*100}%")
    for traj in pbar:
        if len(traj["time"]) < horizon:
            # print(f"Trajectory too short: {len(traj['time'])}")
            num_short_trajectories += 1
            continue
        pbar.set_description(f"Short traj= {num_short_trajectories/len(trajectories)*100}%")
        # print(f"Short trajectory ratio: {num_short_trajectories/len(trajectories)*100}%")
        num_transitions = (len(traj["time"]) - horizon) // skip_step
        num_batches = num_transitions // batch_size
        num_batches = num_batches + 1 \
            if num_transitions % batch_size != 0 \
                else num_batches
        for indx_batch in range(num_batches):
            start_indx = indx_batch * batch_size * skip_step
            end_indx = (indx_batch+1) * batch_size * skip_step
            if indx_batch == num_batches - 1:
                end_indx = num_transitions * skip_step
            states = np.array(
                [traj[name_state][start_indx:end_indx:skip_step] \
                    for name_state in _names_states]
            ).T
            controls = np.array(
                [ [traj[name_control][i:i+horizon] \
                    for i in range(start_indx, end_indx, skip_step)] \
                        for name_control in _names_controls
                ]
            ).transpose((1,2,0))
            gt_traj = np.array(
                [ [traj[name_state][i:i+horizon] \
                    for i in range(start_indx, end_indx, skip_step)] \
                        for name_state in _names_states
                ]
            ).transpose((1,2,0)).reshape(-1, 1, horizon, len(_names_states))
            rng, _rng = jax.random.split(rng)
            pred_traj = pred_fn(states, controls, _rng)
            res = compute_error(pred_traj, gt_traj)
            for k in res.keys():
                if k not in result:
                    result[k] = res[k]
                else:
                    result[k] = np.concatenate(
                        (result[k], res[k]),
                        axis=0
                    )
    return result

def compute_error(pred, gt):
    error = np.linalg.norm(pred - gt, axis=-1)
    return {
        "total_error_mean": np.mean(error, axis=1),
        "total_error_std": np.std(error, axis=1),
    }

def get_models_dict(task, models, batch_size, stepsize, num_steps, horizon, num_particles=10):
    models_dict = {}
    # Load the different models
    for i, model in enumerate(models):
        print(f"{i+1}. Loading model {model['model_name']}\n")
        model_name = model["model_name"]
        model_plot_name = model["label"]
        model_step = model.get("step", -2)
        is_gaussian = model.get("is_gaussian", False)
        if not is_gaussian:
            sampling_fn, sde_model = load_system_sampler_from_model_name(
                env_name = task, 
                model_name = model_name,
                stepsize = stepsize*num_steps, 
                horizon = horizon,
                step = model_step, 
                integration_method=None, #"euler_maruyama",
                num_particles=num_particles,
                verbose=False,
                return_sde_model= True
            )
            pred_fn = get_nsde_pred_fn(
                sampling_fn=sampling_fn,
                is_reward_included=sde_model.drift_term.has_reward,
                batch_size=batch_size,
                horizon=horizon
            )
        else:
            gaussian_model = load_ensemble(
                model_name, task=task, algo='tatu_mopo'
            )
            pred_fn = get_ensemble_pred_fn(
                dynamics_model=gaussian_model, num_traj=num_particles,
                deterministic=False
            )
        models_dict[model_plot_name] = {
            "pred_fn" : pred_fn,
        }
        print(f"Model {model_name} loaded\n")
    return models_dict

def get_results(source_task, target_task, models, horizon, num_particles, skip_step, batch_size, results_path):
    dataset = load_dataset(target_task, verbose=False) 
    env_infos = get_environment_infos_from_name(source_task)
    names_states = env_infos["names_states"]
    names_controls = env_infos["names_controls"] 
    stepsize = env_infos["stepsize"]

    # # Plot a histogram of the trajectory lengths
    # lengths = [len(traj["time"]) for traj in dataset["trajectories"]]
    # plt.hist(lengths)
    # plt.title("Histogram of trajectory lengths")
    # plt.savefig(f"traj_len_hist/{target_task}.png")
    # exit()
    
    num_steps = 1
    models_dict = get_models_dict(
        source_task, models, batch_size, stepsize, num_steps, horizon, num_particles
    )
    for model_name, model_dict in models_dict.items():
        results_filename = f"{results_path}_skip={skip_step}_h={horizon}_{model_name}.pkl"
        if os.path.exists(results_filename):
            continue
        pred_fn = model_dict["pred_fn"]
        rng = jax.random.PRNGKey(0)
        model_analysis = run_analysis(
            dataset=dataset,
            pred_fn=pred_fn,
            horizon=horizon,
            skip_step=skip_step,
            batch_size=batch_size,
            _names_states=names_states,
            _names_controls=names_controls,
            rng=rng
        )
            
        summary = {
            "mean_mean": np.mean(model_analysis["total_error_mean"], axis=0),
            "mean_std": np.std(model_analysis["total_error_mean"], axis=0),
            "std_mean": np.mean(model_analysis["total_error_std"], axis=0),
            "std_std": np.std(model_analysis["total_error_std"], axis=0),
        }
        # save summary
        with open(results_filename, "wb") as f:
            pickle.dump(summary, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-task", type=str, default="halfcheetah-random-v2")
    parser.add_argument("--target-task", type=str, default="halfcheetah-random-v2")
    parser.add_argument("--results-dir", type=str, default="model_analysis_results")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--num-particles", type=int, default=10)
    parser.add_argument("--skip-step", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=100)
    args= parser.parse_args()

    results_filename = f"results_{args.source_task}_in_{args.target_task}"

    models = {
        # halfcheetah
        "halfcheetah-medium-expert-v2": [
            { 
                "model_name" : "hc_me_final_rew",
                "label" : "nsde_rew",
                "step" : -2,
            },
            { 
                "model_name" : "hc_me_final",
                "label" : "nsde",
                "step" : -2,
            },
            {
                "model_name" : "critic_num_2_seed_32_0518_155659-halfcheetah_medium_expert_v2_tatu_mopo",
                "label" : "ensemble",
                "is_gaussian" : True,
            }
        ],
        "halfcheetah-medium-replay-v2": [
            { 
                "model_name" : "hc_mr_final_rew",
                "label" : "nsde_rew",
                "step" : -2, 
            },
            { 
                "model_name" : "hc_mr_final",
                "label" : "nsde",
                "step" : -2, 
            },
            {
                "model_name" : "critic_num_2_seed_32_0518_155620-halfcheetah_medium_replay_v2_tatu_mopo",
                "label" : "ensemble",
                "is_gaussian" : True,
            }
        ],
        "halfcheetah-medium-v2": [
            { 
                "model_name" : "hc_m_final_rew",
                "label" : "nsde_rew",
                "step" : -2, 
            },
            { 
                "model_name" : "hc_m_final",
                "label" : "nsde",
                "step" : -2, 
            },
            {
                "model_name" : "critic_num_2_seed_32_0518_155450-halfcheetah_medium_v2_tatu_mopo",
                "label" : "ensemble",
                "is_gaussian" : True,
            }
        ],
        "halfcheetah-random-v2": [
            { 
                "model_name" : "hc_rand_final_rew",
                "label" : "nsde_rew",
                "step" : -2, 
            },
            { 
                "model_name" : "hc_rand_final",
                "label" : "nsde",
                "step" : -2, 
            },
            {
                "model_name" : "critic_num_2_seed_32_0518_155708-halfcheetah_random_v2_tatu_mopo",
                "label" : "ensemble",
                "is_gaussian" : True,
            }
        ],
        # walker2d
        "walker2d-medium-expert-v2": [
            { 
                "model_name" : "wk_me_final_rew",
                "label" : "nsde_rew",
                "step" : -2,
            },
            { 
                "model_name" : "wk_me_final",
                "label" : "nsde",
                "step" : -2,
            },
            {
                "model_name" : "critic_num_2_seed_32_0921_155850_pc=2.0-walker2d_medium_expert_v2_tatu_mopo",
                "label" : "ensemble",
                "is_gaussian" : True,
            }
        ],
        "walker2d-medium-replay-v2": [
            { 
                "model_name" : "wk_mr_final_rew",
                "label" : "nsde_rew",
                "step" : -2, 
            },
            { 
                "model_name" : "wk_mr_final",
                "label" : "nsde",
                "step" : -2, 
            },
            {
                "model_name" : "critic_num_2_seed_32_0921_155758_pc=2.0-walker2d_medium_replay_v2_tatu_mopo",
                "label" : "ensemble",
                "is_gaussian" : True,
            }
        ],
        "walker2d-medium-v2": [
            { 
                "model_name" : "wk_m_final_rew",
                "label" : "nsde_rew",
                "step" : -2, 
            },
            { 
                "model_name" : "wk_m_final",
                "label" : "nsde",
                "step" : -2, 
            },
            {
                "model_name" : "critic_num_2_seed_32_0920_200636_pc=2.0-walker2d_medium_v2_tatu_mopo",
                "label" : "ensemble",
                "is_gaussian" : True,
            }
        ],
        "walker2d-random-v2": [
            { 
                "model_name" : "wk_rand_final_rew",
                "label" : "nsde_rew",
                "step" : -2, 
            },
            { 
                "model_name" : "wk_rand_final",
                "label" : "nsde",
                "step" : -2, 
            },
            {
                "model_name" : "critic_num_2_seed_32_0920_200510_pc=2.0-walker2d_random_v2_tatu_mopo",
                "label" : "ensemble",
                "is_gaussian" : True,
            }
        ],
        # hopper
        "hopper-medium-expert-v2": [
            { 
                "model_name" : "hop_me_final_rew",
                "label" : "nsde_rew",
                "step" : -2,
            },
            { 
                "model_name" : "hop_me_final",
                "label" : "nsde",
                "step" : -2,
            },
            {
                "model_name" : "critic_num_2_seed_32_0920_195817_pc=3.5-hopper_medium_expert_v2_tatu_mopo",
                "label" : "ensemble",
                "is_gaussian" : True,
            }
        ],
        "hopper-medium-replay-v2": [
            { 
                "model_name" : "hop_mr_final_rew",
                "label" : "nsde_rew",
                "step" : -2, 
            },
            { 
                "model_name" : "hop_mr_final",
                "label" : "nsde",
                "step" : -2, 
            },
            {
                "model_name" : "critic_num_2_seed_32_0920_195703_pc=2.0-hopper_medium_replay_v2_tatu_mopo",
                "label" : "ensemble",
                "is_gaussian" : True,
            }
        ],
        "hopper-medium-v2": [
            { 
                "model_name" : "hop_m_final_rew",
                "label" : "nsde_rew",
                "step" : -2, 
            },
            { 
                "model_name" : "hop_m_final",
                "label" : "nsde",
                "step" : -2, 
            },
            {
                "model_name" : "critic_num_2_seed_32_0919_153900_pc=3.5-hopper_medium_v2_tatu_mopo",
                "label" : "ensemble",
                "is_gaussian" : True,
            }
        ],
        "hopper-random-v2": [
            { 
                "model_name" : "hop_rand_final_rew",
                "label" : "nsde_rew",
                "step" : -2, 
            },
            { 
                "model_name" : "hop_rand_final",
                "label" : "nsde",
                "step" : -2, 
            },
            {
                "model_name" : "critic_num_2_seed_32_0919_153111_pc=2.0-hopper_random_v2_tatu_mopo",
                "label" : "ensemble",
                "is_gaussian" : True,
            }
        ],
    }

    cwd = Path.cwd()
    results_dir = os.path.join(cwd, args.results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    results_path = f"{results_dir}/{results_filename}"
    get_results(
        source_task=args.source_task,
        target_task=args.target_task,
        models=models[args.source_task],
        horizon=args.horizon,
        num_particles=args.num_particles,
        skip_step=args.skip_step,
        batch_size=args.batch_size,
        results_path=results_path
    )
    