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
from pathlib import Path

def plot_results(results, tsteps, fig_path, fontsize=28):
    # fontsize to georgia
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = fontsize

    tsteps = np.array(tsteps)
    x = np.arange(len(tsteps))

    # Width of each bar group
    bar_width = 0.075 + 0.025*(4-len(results.keys()))
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    patterns = ["*", "", "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", "."]

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(15,6), layout='constrained')
    num_source_tasks = len(results)
    num_models = len(results[list(results.keys())[0]])
    legend_elements = []

    # Plot each category's sequence as a separate set of bars
    for i, source_task in enumerate(results.keys()):
        legend_elements.append(Patch(facecolor=colors[i], label=source_task))
        for j, model in enumerate(results[source_task].keys()):
            if i == len(results.keys()) - 1:
                legend_elements.append(Patch(facecolor="none", edgecolor='black', hatch=patterns[j], label=model))               
            mean = results[source_task][model]["mean_mean"][tsteps-1]
            std = results[source_task][model]["mean_std"][tsteps-1]
            ax.bar(x + (i * num_models + j ) * bar_width, mean, 
                   color=colors[i], hatch=patterns[j], capsize=5, alpha=0.8,
                   edgecolor='black', width=bar_width, label=None, yerr=std)
    # Adding labels, title, and ticks
    ax.set_xlabel('Horizon')
    ax.set_ylabel('Prediction Error')
    ax.set_yscale('log')
    # ax.set_title('Column Plot of Sequences within Categories')
    # ax.set_xticks(x + bar_width * (num_categories - 1) / 2)
    # ax.set_xticklabels([f'Position {i+1}' for i in range(sequence_length)])
    ax.set_xticks(x + bar_width * (num_source_tasks * num_models - 1) / 2)
    ax.set_xticklabels(tsteps, fontsize=fontsize*0.8)
    ax.legend(handles=legend_elements, markerscale=4, loc='upper left', fontsize=fontsize*0.8, ncol=2)
    plt.savefig(f"{fig_path}.pdf", dpi=500)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="halfcheetah")
    parser.add_argument("--models", type=str, nargs="+", default=["nsde_rew", "nsde", "ensemble"])
    parser.add_argument("--cross-dataset", action='store_true', default=False)
    parser.add_argument("--results-dir", type=str, default="model_analysis_results")
    parser.add_argument(f"--horizon", type=int, default=50)
    parser.add_argument(f"--skip", type=int, default=50)
    parser.add_argument("--steps-to-plot", type=int, nargs="+", 
                        default=[1, 10, 15, 20, 25],
                        )    
    parser.add_argument("--fig-name", type=str, default="fig_hc_random")
    args = parser.parse_args()

    if args.cross_dataset:
        task_pairs = [
            ("random-v2", "random-v2"),
            ("random-v2", "medium-v2"),
            ("random-v2", "medium-replay-v2"),
            ("random-v2", "medium-expert-v2"),
        ]
    else:
        task_pairs = [
            # ("random-v2", "random-v2"),
            ("medium-v2", "medium-v2"),
            ("medium-replay-v2", "medium-replay-v2"),
            ("medium-expert-v2", "medium-expert-v2"),
        ]


    # cwd = Path.cwd()
    # results_dir = os.path.join(cwd, args.results_dir)
    results_dir = args.results_dir
    fig_path = f"{results_dir}/figures/{args.fig_name}"

    results = {}
    for target_task, source_task in task_pairs:
        results_path = f"{results_dir}/data/results_{args.env}-{source_task}_in_{args.env}-{target_task}_skip={args.skip}_h={args.horizon}"
        results[f"{args.env}-{source_task}"] = {}
        for model in args.models:
            results_path_ = f"{results_path}_{model}.pkl"
            if os.path.exists(results_path_):
                with open(results_path_, "rb") as f:
                    results_dict = pickle.load(f)
            else:
                raise ValueError(f"Results file not found: {results_path_}")
            results[f"{args.env}-{source_task}"][model] = results_dict
    print(results)
    plot_results(results, args.steps_to_plot, fig_path)