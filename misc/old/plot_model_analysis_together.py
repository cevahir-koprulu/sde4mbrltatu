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
from matplotlib.patches import Patch

def plot_results(results, task, fig_path):
    # fontsize to georgia
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 16

    fig, ax = plt.subplots(figsize=(15,6), layout='constrained')
    width = 0.05  # the width of the bars
    tsteps = [1, 10, 15, 20, 25]
    multiplier = 0
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    patterns = ["*", ""]# "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
    legend_elements = []
    add_models_to_legend = True
    for t in tsteps:
        cc = 00
        for task_name, results_dict in results.items():
            pp = 0
            for model_name, result_dict in results_dict.items():
                mean = result_dict["total_error_mean_mean"][t]
                std = result_dict["total_error_mean_std"][t]
                rects = ax.bar(width * multiplier, mean, width, color=colors[cc], hatch=patterns[pp],
                            yerr=std, edgecolor='black', capsize=5)
                if add_models_to_legend:
                    legend_elements.append(Patch(facecolor="none", edgecolor='black', hatch=patterns[pp], label=model_name))
                multiplier += 1
                pp += 1
            add_models_to_legend = False
            if t == tsteps[0]:
                legend_elements.append(Patch(facecolor=colors[cc], label=task_name))
            cc += 1
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Prediction error')
    ax.set_xlabel('Horizon')
    offsets = [4*width + i*width*len(results)*2 + (i-0.5)*width  for i in range(len(tsteps))]
    ax.set_xticks(offsets, tsteps)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_yscale('log')
    ax.legend(handles=legend_elements, markerscale=4, loc='upper left', fontsize=plt.rcParams['font.size']*0.75)

    plt.savefig(f"{fig_path}.png", dpi=1000)

if __name__ == "__main__":
    crossdataset = True
    results_dir = "model_analysis_cross_dataset_results" if crossdataset else "model_analysis_results"
    main_task = "halfcheetah-random-v2" # if crossdataset
    eval_traj_type = 0
    fig_name = "fig_hc_crossdataset" if crossdataset else "fig_hc"
    tasks_and_models = {
        "halfcheetah-medium-expert-v2": [
            { 
                "model_name" : "hc_me_final",
                "plot_name" : "NSDE",
            },
            {
                "model_name" : "critic_num_2_seed_32_0518_155659-halfcheetah_medium_expert_v2_tatu_mopo",
                "plot_name" : "Ens",
            }
        ],
        "halfcheetah-medium-replay-v2": [
            { 
                "model_name" : "hc_mr_final",
                "plot_name" : "NSDE",
            },
            {
                "model_name" : "critic_num_2_seed_32_0518_155620-halfcheetah_medium_replay_v2_tatu_mopo",
                "plot_name" : "Ens",
            }
        ],
        "halfcheetah-medium-v2": [
            { 
                "model_name" : "hc_m_final",
                "plot_name" : "NSDE",
            },
            {
                "model_name" : "critic_num_2_seed_32_0518_155450-halfcheetah_medium_v2_tatu_mopo",
                "plot_name" : "Ens",
            }
        ],
        "halfcheetah-random-v2": [
            { 
                "model_name" : "hc_rand_final",
                "plot_name" : "NSDE",
            },
            {
                "model_name" : "critic_num_2_seed_32_0518_155708-halfcheetah_random_v2_tatu_mopo",
                "plot_name" : "Ens",
            }
        ],
    }
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results = {}
    for task in tasks_and_models:
        if crossdataset:
            if task == main_task:
                continue
            results_path = f"{results_dir}/results_{main_task}_IN_{task}_type={eval_traj_type}.pkl"
        else:
            results_path = f"{results_dir}/results_{task}_type={eval_traj_type}.pkl"
        if os.path.exists(results_path):
            with open(results_path, "rb") as f:
                results_dict = pickle.load(f)
        else:
            raise ValueError(f"Results file not found: {results_path}")
            
        results[task] = results_dict
    
    fig_path = f"{results_dir}/{fig_name}"
    plot_results(results, task, fig_path)