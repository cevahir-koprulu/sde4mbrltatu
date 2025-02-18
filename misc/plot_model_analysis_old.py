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

def plot_results(results, tsteps, fig_path):
    # fontsize to georgia
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 16

    fig, ax = plt.subplots(figsize=(15,6), layout='constrained')
    width = 0.05  # the width of the bars
    multiplier = 0
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    patterns = ["*", "/", "-"]# "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
    legend_elements = []
    add_models_to_legend = True
    for t in tsteps:
        cc = 00
        for task_name, results_dict in results.items():
            pp = 0
            for model_name, result_dict in results_dict.items():
                mean = result_dict["mean_mean"][t-1]
                std = result_dict["mean_std"][t-1]
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
    # offsets = [4*width + i*width*len(results)*2 + (i-0.5)*width  for i in range(len(tsteps))]
    offsets = [i*width*len(results)*len(result_dict)+width*(len(results_dict)//2) for i in range(len(tsteps))]
    ax.set_xticks(offsets, tsteps)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_yscale('log')
    ax.legend(handles=legend_elements, markerscale=4, loc='upper left', fontsize=plt.rcParams['font.size']*0.75)

    plt.savefig(f"{fig_path}.png", dpi=1000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="halfcheetah")
    parser.add_argument("--source-tasks", type=str, nargs="+", 
                        default=["random-v2"],
                        )
    parser.add_argument("--target-tasks", type=str, nargs="+", 
                        default=["random-v2"],
                        # default=["medium-expert-v2", "medium-replay-v2", "medium-v2", "random-v2"],
                        )    
    parser.add_argument("--models", type=str, nargs="+", default=["nsde", "ensemble"])
    parser.add_argument("--cross-dataset", type=bool, default=False)
    parser.add_argument("--results-dir", type=str, default="model_analysis_results")
    parser.add_argument(f"--horizon", type=int, default=50)
    parser.add_argument(f"--skip", type=int, default=50)
    parser.add_argument("--steps-to-plot", type=int, nargs="+", 
                        default=[1, 10, 20, 30, 40, 50],
                        )    
    parser.add_argument("--fig-name", type=str, default="fig_hc_random")
    args = parser.parse_args()

    # task_pairs = [
    #     ("random-v2", "random-v2"),
    #     ("random-v2", "medium-v2"),
    #     ("random-v2", "medium-replay-v2"),
    #     ("random-v2", "medium-expert-v2"),
    #     ("medium-expert-v2", "medium-v2"),
    #     ("medium-expert-v2", "medium-replay-v2"),
    #     ("medium-expert-v2", "medium-expert-v2"),
    # ]

    cwd = Path.cwd()
    results_dir = os.path.join(cwd, args.results_dir)
    fig_path = f"{results_dir}/figures/{args.fig_name}"

    results = {}
    if args.cross_dataset:
        source_task = args.source_tasks[0]
        for target_task in args.target_tasks:
            results[f"{args.env}-{target_task}"] = {}
            results_path = f"{results_dir}/data/results_{args.env}-{source_task}_in_{args.env}-{target_task}_skip={args.skip}_h={args.horizon}"
            for model in args.models:
                results_path_ = f"{results_path}_{model}.pkl"
                if os.path.exists(results_path_):
                    with open(results_path_, "rb") as f:
                        results_dict = pickle.load(f)
                else:
                    raise ValueError(f"Results file not found: {results_path_}")
                results[f"{args.env}-{target_task}"][model] = results_dict
    else:
        for source_task in args.source_tasks:
            results_path = f"{results_dir}/data/results_{args.env}-{source_task}_in_{args.env}-{source_task}_skip={args.skip}_h={args.horizon}"
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