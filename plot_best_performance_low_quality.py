import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from training_info_dict import TRAINING_INFO_DICT
from matplotlib.patches import Patch

def parse_str_for_number(entry):
    first_digit = entry[0] if entry[0].isdigit() else ''
    last_digit = entry[-1] if entry[-1].isdigit() and len(entry)>1 else ''
    return first_digit + entry[1:-1] + last_digit

def parse_progress_file(filepath, num_epochs=1000):
    progress_f = os.path.join(filepath, "progress.txt")
    with open(progress_f, 'r') as f:
        lines = f.readlines()

    normalized_rewards_mean = np.zeros(num_epochs)
    normalized_rewards_std = np.zeros(num_epochs)
    
    line_no = 0
    ep_no = -1
    while line_no < len(lines):
        line_splitted = lines[line_no].split()
        if "Epoch" in line_splitted and "episode_reward_normal:" in line_splitted:
            ep_no += 1
            normalized_rewards_mean[ep_no] = float(line_splitted[3])
            normalized_rewards_std[ep_no] = float(line_splitted[5])
        line_no += 1
    return normalized_rewards_mean, normalized_rewards_std

def get_results_dict(log_dir, tasks, models):
    results = {}
    config_dict = {}
    for task in tasks:
        results[task] = {}
        for model in models:
            print(f"Processing {model}")
            model_log_name = get_model_log_name(model)
            log_dir_ = os.path.join(log_dir, task, model_log_name)
            seeds_dict = TRAINING_INFO_DICT[task][model]["seeds"]
            results[task][model] = None
            config_dict[model] = TRAINING_INFO_DICT[task][model]["config"]
            for seed, result_dir in seeds_dict.items():
                result_path = os.path.join(log_dir_, result_dir)
                print(f"\t{result_dir}")
                mean, std = parse_progress_file(result_path, 1000)
                max_progress = np.array([np.max(mean[:i+1]) for i in range(len(mean))])
                if results[task][model] is None:
                    results[task][model] = max_progress
                else:
                    results[task][model] = np.vstack((results[task][model], max_progress))
            # print(f"\t **** Human normalized score: {np.mean(results[model][:,-1])}+-{np.std(results[model][:,-1])}")
            results[task][model] = results[task][model][:,-1]
            print(results[task][model])
    return results

def plot_results(results_dict, fig_path, fontsize=16, figsize=(10, 4)):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = fontsize

    # Width of each bar group
    bar_width = 0.2
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    models = list(results_dict[list(results_dict.keys())[0]].keys())
    tasks = list(results_dict.keys())
    results_dict_reversed = {}
    for model in models:
        results_dict_reversed[model] = {}
        for task in results_dict.keys():
            results_dict_reversed[model][task] = results_dict[task][model]
    

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    num_tasks = len(tasks)
    num_models = len(models)
    x = np.arange(num_tasks)
    legend_elements = []
    for i, model in enumerate(models):
        mean = [results_dict_reversed[model][task].mean() for task in tasks]
        std = [results_dict_reversed[model][task].std() for task in tasks]
        ax.bar(x + i * bar_width, mean, 
                color=colors[i], hatch="", capsize=5,
                edgecolor='black', width=bar_width, label=None, yerr=std)
        legend_elements.append(Patch(facecolor=colors[i], label=get_model_label(model)))
    # ax.set_xlabel('Low quality datasets')
    ax.set_ylabel('Human normalized score')
    # ax.set_yscale('log')
    ax.grid()
    ax.set_xticks(x + bar_width * (num_models - 1) / 2)
    ax.set_xticklabels(tasks)
    ax.legend(handles=legend_elements, markerscale=4, loc='best', fontsize=plt.rcParams['font.size'])
    plt.savefig(f"{fig_path}.pdf", dpi=500)

def get_model_color(model):
    if model == "mopo":
        return "blue"
    elif model == "tatu_mopo_sde_rew":
        return "orange"
    elif model == "tatu_mopo_sde":
        return "green"
    elif model == "tatu_mopo":
        return "red"
    else:
        raise ValueError(f"Model {model} not recognized")
    
def get_model_log_name(model):
    if model == "mopo":
        return "tatu_mopo"
    elif model == "tatu_mopo_sde_rew":
        return "tatu_mopo_sde"
    else:
        return model

def get_model_label(model):
    if model == "mopo":
        return "MOPO"
    elif model == "tatu_mopo_sde_rew":
        return r"$\mathregular{NUNO}^{\mathregular{R}}$"
    elif model == "tatu_mopo_sde":
        return "NUNO"
    elif model == "tatu_mopo":
        return "TATU+MOPO"
    else:
        return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="log")
    parser.add_argument("--task", type=str, default='d4rl')
    parser.add_argument("--models", type=str, nargs="+", 
                        default=["tatu_mopo_sde", "tatu_mopo_sde_rew", "tatu_mopo", "mopo"])
    parser.add_argument("--fig-name-extra", type=str, default="")
    parser.add_argument("--save-dir", type=str, default="best_performance_results")
    args= parser.parse_args() 

    if args.task == 'd4rl':
        tasks = ["halfcheetah-random-v2", "hopper-random-v2", "walker2d-random-v2"]
    elif args.task == 'neorl':
        tasks = ["HalfCheetah-v3-Low-1000-neorl", "Hopper-v3-Low-1000-neorl", "Walker2d-v3-Low-1000-neorl"]
    else:
        raise ValueError(f"Task {args.task} not recognized")

    results = get_results_dict(args.log_dir, tasks, args.models)
    fig_path = os.path.join(args.save_dir, f"low_quality_{args.task}{args.fig_name_extra}")
    plot_results(results, fig_path)

            
