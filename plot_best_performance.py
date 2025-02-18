import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from training_info_dict import TRAINING_INFO_DICT

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

def get_results_dict(log_dir, task, models):
    results = {}
    config_dict = {}
    for model in models:
        print(f"Processing {model}")
        model_log_name = get_model_log_name(model)
        log_dir_ = os.path.join(log_dir, task, model_log_name)
        seeds_dict = TRAINING_INFO_DICT[task][model]["seeds"]
        results[model] = None
        config_dict[model] = TRAINING_INFO_DICT[task][model]["config"]
        for seed, result_dir in seeds_dict.items():
            result_path = os.path.join(log_dir_, result_dir)
            print(f"\t{result_dir}")
            mean, std = parse_progress_file(result_path, 1000)
            max_progress = np.array([np.max(mean[:i+1]) for i in range(len(mean))])
            if results[model] is None:
                results[model] = max_progress
            else:
                results[model] = np.vstack((results[model], max_progress))
        print(f"\t **** Human normalized score: {np.mean(results[model][:,-1])}+-{np.std(results[model][:,-1])}")
    return results

def plot_results(results, fig_path, fontsize=18, figsize=(6, 4)):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = fontsize

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for i, model in enumerate(results.keys()):
        mean = np.mean(results[model], axis=0)
        std = np.std(results[model], axis=0)
        color = get_model_color(model)
        ax.plot(mean, label=get_model_label(model), color=color)
        ax.fill_between(np.arange(len(mean)), mean-std, mean+std, alpha=0.3, color=color)
    ax.legend()
    plt.xlabel("Number of epochs")
    plt.ylabel("Human normalized score")
    plt.ticklabel_format(axis='x', style='sci', scilimits=(5, 6), useMathText=True)
    plt.grid()
    plt.savefig(f"{fig_path}.pdf", dpi=500)

def get_model_color(model):
    if model == "mopo":
        return "green"
    elif model == "tatu_mopo_sde_rew":
        return "blue"
    elif model == "tatu_mopo_sde":
        return "red"
    elif model == "tatu_mopo":
        return "red"
    elif model == "tatu_mopo_sde_dfd":
        return "blue"
    elif model == "tatu_mopo_sde_disc":
        return "green"
    else:
        raise ValueError(f"Model {model} not recognized")
    
def get_model_log_name(model):
    if model == "mopo":
        return "tatu_mopo"
    elif model == "tatu_mopo_sde_rew":
        return "tatu_mopo_sde"
    elif model == "tatu_mopo_sde_dfd":
        return "tatu_mopo_sde"
    elif model == "tatu_mopo_sde_disc":
        return "tatu_mopo_sde"
    else:
        return model

def get_model_label(model):
    if model == "mopo":
        return "$MOPO$"
    elif model == "tatu_mopo_sde_rew":
        return r"$\mathregular{NUNO}^{\mathregular{R}}$"
    elif model == "tatu_mopo_sde":
        return "NUNO"
    elif model == "tatu_mopo_sde_dfd":
        return r"$\mathregular{NUNO}^{\mathregular{al}}$"
    elif model == "tatu_mopo_sde_disc":
        return r"$\mathregular{NUNO}^{\mathregular{disc}}$"
    elif model == "tatu_mopo":
        return "$TATU+MOPO$"
    else:
        return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="log")
    parser.add_argument("--task", type=str, default="halfcheetah-random-v2")
    parser.add_argument("--models", type=str, nargs="+", 
                        default=["tatu_mopo_sde_rew", "tatu_mopo_sde", "tatu_mopo", "mopo"])
    parser.add_argument("--plot-progression", action='store_true', default=False)
    parser.add_argument("--fig-name-extra", type=str, default="")
    parser.add_argument("--save-dir", type=str, default="best_performance_results")
    args= parser.parse_args() 

    print(f"TASK: {args.task}")
    results = get_results_dict(args.log_dir, args.task, args.models)
    if args.plot_progression:
        fig_path = os.path.join(args.save_dir, f"perf_prog_{args.task}{args.fig_name_extra}")
        plot_results(results, fig_path)