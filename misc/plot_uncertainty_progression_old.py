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

from tqdm import tqdm

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

def parse_progress_file(filepath):
    progress_f = os.path.join(filepath, "progress.txt")
    with open(progress_f, 'r') as f:
        lines = f.readlines()

    line_no = 0
    results_dict = {}
    while line_no < len(lines):
        line_splitted = lines[line_no].split()
        if "Epoch" in line_splitted:
            if "eval_unc_stats" in line_splitted[2]:
                unc_stat_type = line_splitted[2][len("eval_unc_stats/"):-1]
                unc_stat_mean = float(line_splitted[3])
                unc_stat_std = float(line_splitted[5])
                if unc_stat_type not in results_dict:
                    results_dict[unc_stat_type] = {"mean" : [], 
                                                    "std" : []}
                results_dict[unc_stat_type]["mean"].append(unc_stat_mean)
                results_dict[unc_stat_type]["std"].append(unc_stat_std)
        line_no += 1

    results_dict = {
        k : {
            "mean" : np.array(v["mean"]),
            "std" : np.array(v["std"])
        } for k, v in results_dict.items()
    }
    return results_dict


def load_uncertainty_estimation_fn(
    env_name: str, model: Dict[str, Any],
    stepsize: int = 1, num_particles: int = 5,
    decision_var: str = "diff_density",
):
    print("\n################################################")
    print("Initializing the uncertainty computation")
    print("################################################\n")
    sampler_fn, sde_model = load_system_sampler_from_model_name(
        env_name,
        model_name = model['model_name'],
        stepsize = stepsize,
        horizon = 1,
        num_particles = num_particles,
        step = model.get('ckpt_step', -2),
        integration_method=None,
        verbose=False,
        return_sde_model= True
    )
    is_reward_included = sde_model.drift_term.has_reward

    def _uncertainty_est_fn(
        x : jax.numpy.ndarray,
        u : jax.numpy.ndarray,
        rng : jax.random.PRNGKey
    ):
        assert len(x.shape) == len(u.shape) == 2

        if is_reward_included:
            zero_last_dim = jax.numpy.zeros((x.shape[0], 1))
            x = jax.numpy.concatenate((x, zero_last_dim), axis=-1)

        # Get matching rng keys
        next_rng, rng = jax.random.split(rng, 2)
        rng = jax.random.split(rng, x.shape[0])

        u = jax.numpy.expand_dims(u, axis=1)

        # Define the vmap sampler function
        _temp_sampler =  lambda _x, _u, _rng: \
            sampler_fn(state=_x, control=_u, rng_key=_rng)

        # Compute the predictions
        pred_states, pred_feats = \
            jax.vmap(_temp_sampler)(x, u, rng)
        
        pred_states = pred_states[:, :, 1, :]
        pred_feats["diff_density"] = pred_feats["diff_density"][...,None]

        if is_reward_included:
            pred_states = pred_states[..., :-1]
            pred_feats =  jax.tree_map(
                lambda z : z[..., :-1] if z.shape[-1] == x.shape[-1] else z,
                pred_feats
            )

        diffs = pred_states - jnp.expand_dims(jnp.mean(pred_states, axis=1), axis=1)
        disc = jnp.expand_dims(jnp.linalg.norm(diffs, axis=-1), axis=2)
        
        if decision_var not in pred_feats:
            raise KeyError(f"Invalid threshold_decision_var: {decision_var}")

        # Get the diffusion term
        pred_feats = {
            k : jnp.linalg.norm(v, axis=-1) \
                for k, v in pred_feats.items() \
                    if k in ["diffusion_value", "dad_free_diff",
                            "dad_based_diff", "diff_density"]
        }
        diffusion_value = pred_feats[decision_var]
        penalty = jnp.mean(diffusion_value, axis=1)

        result_dict = {
            "disc" : disc,
            'penalty': penalty,
            **pred_feats
        }
        return result_dict, next_rng

    # Jit the uncertainty estimation function
    # jit_uncertainty_est_fn = jax.jit(_uncertainty_est_fn, backend='cpu')
    jit_uncertainty_est_fn = jax.jit(_uncertainty_est_fn)
    batch_size = model.get('batch_size_trunc_thresh', 100)

    def augmented_pred_fn(x, u , rng):
        batch_x = x.shape[0]
        last_x = x[-1]
        last_u = u[-1]
        # Complte x, u to have a batch size of rollout_batch_size
        # so that to avoid recompilation
        if batch_x < batch_size:
            last_x = np.repeat(last_x[None], batch_size-batch_x, axis=0)
            last_u = np.repeat(last_u[None], batch_size-batch_x, axis=0)
            x = np.concatenate((x, last_x), axis=0)
            u = np.concatenate((u, last_u), axis=0)
        res, next_rng = jit_uncertainty_est_fn(x, u, rng)
        res = { k : np.array(v) for k, v in res.items()}
        # if batch_x < batch_size:
        res  = {
            k : v[:batch_x].reshape((-1,v.shape[-1])) \
                for k, v in res.items()
        }
        return res, next_rng
    
    return augmented_pred_fn

def compute_uncertainty_distribution(action_type, action_space, dataset, uncertainty_est_fn):
    observations = dataset["observations"]
    actions = dataset["actions"]

    mini_batch = 5000
    slice_num = len(observations)// mini_batch
    alone_num = len(observations)% mini_batch
    unc_dicts = []
    rng = jax.random.PRNGKey(0)
    for i in tqdm(range(slice_num), desc=f"Computing Uncertainty for action_type={action_type}"):
        obs = observations[i*mini_batch:(i+1)*mini_batch,:]
        if action_type == "random":
            act = np.random.uniform(low=action_space.low, 
                                    high=action_space.high, 
                                    size=(obs.shape[0], action_space.shape[0]))
        elif action_type == "data":
            act = actions[i*mini_batch:(i+1)*mini_batch,:]
        unc_dict, rng = uncertainty_est_fn(obs,act,rng)
        unc_dicts.append(unc_dict)
        
    if alone_num != 0:
        obs = observations[slice_num*mini_batch:,:]
        if action_type == "random":
            act = np.random.uniform(low=action_space.low, 
                                    high=action_space.high, 
                                    size=(obs.shape[0], action_space.shape[0]))
        elif action_type == "data":
            act = actions[slice_num*mini_batch:,:]
        unc_dict, rng = uncertainty_est_fn(obs,act,rng)
        unc_dicts.append(unc_dict)
    unc_dict_keys = unc_dicts[0].keys()
    unc_dict_stacked = {
        k : np.concatenate([r[k] for r in unc_dicts], axis=0) \
            for k in unc_dict_keys
    }
    unc_stats = {
        k : {
            "mean" : np.mean(v),
            "std" : np.std(v)
        } for k, v in unc_dict_stacked.items()
    }   
    return unc_stats

def compute_rnd_and_data_results(
    task: str, models_to_evaluate: List[Dict[str, Any]],
    num_steps: int, num_particles: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    results_dict = {}
    env = gym.make(task)
    action_space = env.action_space
    dataset = d4rl.qlearning_dataset(env)
    for model in models_to_evaluate:
        uncertainty_est_fn = load_uncertainty_estimation_fn(env_name=task, 
                                                            model=model, 
                                                            stepsize=num_steps, 
                                                            num_particles=num_particles,
)
        for action_type in ["random", "data"]:
            results_dict[action_type] = \
                compute_uncertainty_distribution(
                    action_type=action_type, action_space=action_space,
                    dataset=dataset, uncertainty_est_fn=uncertainty_est_fn,
                )
    return results_dict

def get_progression_results(
    task: str, policies_to_eval: List[Dict[str, Any]],
) -> Dict[str, Dict[str, np.ndarray]]:
    results_dict = {}
    for name in policies_to_eval:
        filepath = os.path.join("log", task, "tatu_mopo_sde", name)
        res = parse_progress_file(filepath=filepath)
        for k, v in res.items():
            # if k not in results_dict:
                # results_dict[k] = v["mean"]
            if k not in results_dict:
                results_dict[k] = {"mean" : v["mean"], "std" : v["std"]}
            else:
                # results_dict[k] = np.vstack([results_dict[k], v["mean"]])
                results_dict[k]["mean"] = np.vstack([results_dict[k]["mean"], v["mean"]])
                results_dict[k]["std"] = np.vstack([results_dict[k]["std"], v["std"]])
    return results_dict

# def plot_results(results_dict, fig_path, task, colors):
#     plt.figure(figsize=(10, 6))
#     width=0.35
#     result_dict_keys = list(results_dict["random"].keys()) 
#     indices = np.arange(len(result_dict_keys))
#     for action_type, result_dict in results_dict.items():
#         plt.bar(
#             indices-width/2*(1 if action_type == "random" else -1),
#             [result_dict[k]['mean'] for k in result_dict_keys],
#             yerr=[result_dict[k]['std'] for k in result_dict_keys],
#             width=width,
#             label=action_type,
#         )

#     plt.xlabel("Metrics")
#     plt.ylabel("Uncertainty")
#     plt.xticks(indices, result_dict_keys)

#     # plt.yscale('log')
#     plt.legend()
#     plt.title(f"Uncertainty for {task}")
#     plt.savefig(f"{fig_path}.png")

def plot_results(rnd_and_data_res, progression_res, decision_var, fig_path, task, colors):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 16
    plt.figure(figsize=(10, 6))
    progress = progression_res[decision_var]
    # epochs = np.arange(progress.shape[1])
    epochs = np.arange(progress["mean"].shape[1])
    for action_type, result_dict in rnd_and_data_res.items():
        # plot horizontal lines with decision_var
        plt.plot(
            epochs,
            result_dict[decision_var]["mean"]*np.ones_like(epochs),
            label=f"{action_type}",
            color=colors[action_type],
            linestyle='--',
        )
        plt.fill_between(
            epochs,
            result_dict[decision_var]["mean"] - result_dict[decision_var]["std"],
            result_dict[decision_var]["mean"] + result_dict[decision_var]["std"],
            alpha=0.5,
            color=colors[action_type],
        )
    plt.plot(
        epochs,
        np.mean(progress["mean"], axis=0),
        label="progress",
        color=colors["progress"],
    )
    plt.fill_between(
        epochs,
        np.mean(progress["mean"], axis=0) - np.mean(progress["std"], axis=0),
        np.mean(progress["mean"], axis=0) + np.mean(progress["std"], axis=0),
        alpha=0.5,
        color=colors["progress"],
    )

    plt.xlabel("Epochs")
    plt.ylabel(decision_var)
    # plt.title(f"{decision_var} in {task}")
    plt.legend(fontsize=plt.rcParams['font.size']*0.75)
    plt.savefig(f"{fig_path}.png")

# main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="halfcheetah-medium-expert-v2")
    parser.add_argument("--results-dir", type=str, default="uncertainty_progression_results")
    parser.add_argument("--decision-var", type=str, default="diff_density")
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--num-particles", type=int, default=5)
    args= parser.parse_args()

    colors = {
        "random" : "blue",
        "data" : "red",
        "progress": "green",
    }

    models_to_evaluate = {
        "halfcheetah-medium-expert-v2": [
            { 
                "model_name" : "hc_me_final",
                "plot_name" : "NSDE",
                "step" : -2, # The best model
                "policies_to_eval" : [
                    "hc_me_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0914_143928",
                    "hc_me_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0914_143931",
                    "hc_me_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0914_144004",
                    "hc_me_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0914_143937",
                    "hc_me_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0914_143916",
                    ],
            },
        ],
        "halfcheetah-medium-replay-v2": [
            { 
                "model_name" : "hc_mr_final",
                "plot_name" : "NSDE",
                "step" : -2, # The best model
                "policies_to_eval" : [
                    "hc_mr_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0914_143725",
                    "hc_mr_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0914_143740",
                    "hc_mr_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0914_143747",
                    "hc_mr_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0914_143753",
                    "hc_mr_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0914_143758",
                    ],
            },
        ],
        "halfcheetah-medium-v2": [
            { 
                "model_name" : "hc_m_final",
                "plot_name" : "NSDE",
                "step" : -2, # The best model
                "policies_to_eval" : [
                    "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0914_143250",
                    "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0914_143335",
                    "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0914_143415",
                    "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0914_143429",
                    "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0914_143452",
                    ],
            },
        ],
        "hopper-medium-expert-v2": [
            { 
                "model_name" : "hop_me_final",
                "plot_name" : "NSDE",
                "step" : -2, # The best model
                "policies_to_eval" : [
                    "hop_me_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0918_210158",
                    "hop_me_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0918_210310",
                    "hop_me_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0918_210319",
                    "hop_me_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0918_210323",
                    "hop_me_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0918_210330",
                    ],
            },
        ],
        "hopper-medium-replay-v2": [
            { 
                "model_name" : "hop_mr_final",
                "plot_name" : "NSDE",
                "step" : -2, # The best model
                "policies_to_eval" : [
                    "hop_mr_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0918_205953",
                    "hop_mr_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0918_205959",
                    "hop_mr_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0918_210004",
                    "hop_mr_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0918_210007",
                    "hop_mr_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0918_210012",
                    ],
            },
        ],
        "hopper-medium-v2": [
            { 
                "model_name" : "hop_m_final",
                "plot_name" : "NSDE",
                "step" : -2, # The best model
                "policies_to_eval" : [
                    "hop_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0918_205524",
                    "hop_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0918_205623",
                    "hop_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0918_205630",
                    "hop_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0918_205636",
                    "hop_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0918_205641",
                    ],
            },
        ],
        "walker2d-medium-v2": [
            { 
                "model_name" : "wk_m_final",
                "plot_name" : "NSDE",
                "step" : -2, # The best model
                "policies_to_eval" : [
                    "wk_m_final_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0918_210716",
                    "wk_m_final_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0918_210729",
                    "wk_m_final_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0918_210729",
                    "wk_m_final_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0918_210741",
                    "wk_m_final_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0918_210745",
                    ],
            },
        ],
        }
    
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    rnd_data_res = compute_rnd_and_data_results(
        task=args.task, models_to_evaluate=models_to_evaluate[args.task],
        num_steps=args.num_steps, num_particles=args.num_particles,
    )

    progression_res = get_progression_results(
        task=args.task, policies_to_eval=models_to_evaluate[args.task][0]["policies_to_eval"],
    )

    fig_path = f"{args.results_dir}/fig_{args.task}_{args.decision_var}_progress_vs_rnd_vs_data"
    plot_results(rnd_data_res, progression_res, args.decision_var, fig_path, args.task, colors)