import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def parse_progress_file(filepath, horizon=10, num_epochs=1000):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    normalized_rewards_mean = np.zeros(num_epochs)
    normalized_rewards_std = np.zeros(num_epochs)
    truncation = np.zeros((num_epochs,horizon), dtype=int)
    uncertainty_estimate = np.zeros((num_epochs,horizon))
    mean_rewards = np.zeros((num_epochs,horizon))
    for ep_no in range(num_epochs):
        start_line_no = ep_no*9+1
        normalized_rewards_mean[ep_no] = float(lines[start_line_no+1].split()[3])
        normalized_rewards_std[ep_no] = float(lines[start_line_no+1].split()[5])
        truncation_splitted = lines[start_line_no+4].split()
        uncertainty_estimate_splitted = lines[start_line_no+7].split()
        mean_rewards_splitted = lines[start_line_no+8].split()
        for h in range(horizon):
            if h == 0:
                truncation[ep_no,h] = int(truncation_splitted[4+h])
                uncertainty_estimate[ep_no,h] = float(uncertainty_estimate_splitted[3+h][1:])
                mean_rewards[ep_no,h] = float(mean_rewards_splitted[3+h][1:])
            elif h < horizon-1:
                truncation[ep_no,h] = int(truncation_splitted[4+h])
                uncertainty_estimate[ep_no,h] = float(uncertainty_estimate_splitted[3+h])
                mean_rewards[ep_no,h] = float(mean_rewards_splitted[3+h])
            else:
                truncation[ep_no,h] = int(truncation_splitted[4+h][:-1])
                uncertainty_estimate[ep_no,h] = float(uncertainty_estimate_splitted[3+h][:-1])
                mean_rewards[ep_no,h] = float(mean_rewards_splitted[3+h][:-1])

    return normalized_rewards_mean, normalized_rewards_std, truncation, uncertainty_estimate, mean_rewards

def plot_for_separate(plot_for, results, seeds, model_seeds, iterations):
    for seed in seeds:
        color = seeds[seed]
        label = model_seeds[seed]["label"]
        result = results[seed]["mean"] if plot_for == "human_normalized_score" else results[seed]
        plt.plot(iterations, result, color=color, linewidth=2.0, label=label, marker=".")

def plot_results(log_dir, env_name, models, seeds, settings, plot_for_list, figname_extra):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = settings["fontsize"]

    num_epochs = settings["num_epochs"]
    steps_per_iter = settings["steps_per_iter"]
    fontsize = settings["fontsize"]
    figsize = settings["figsize"]
    bbox_to_anchor = settings["bbox_to_anchor"]
    plot_settings = settings[env_name]
    iterations = np.arange(0, num_epochs, dtype=int)
    iterations_step = iterations*steps_per_iter

    model_seeds = models["seeds"]
    model_config = models["config"]
    results = {
        "human_normalized_score": {},
        "uncertainty": {},
        "truncation": {},
        "mean_reward": {},
    }
    for seed in seeds:
        filepath = os.path.join(log_dir, env_name, "tatu_mopo_sde", model_seeds[seed]["model"], "progress.txt")
        normalized_rewards_mean, normalized_rewards_std, \
            truncation, uncertainty_estimate, mean_rewards = parse_progress_file(filepath, 
                                                                                 horizon=model_config["RL"], 
                                                                                 num_epochs=num_epochs)
        results["human_normalized_score"][seed] = {
            "mean": normalized_rewards_mean,
            "std": normalized_rewards_std,
        }
        results["uncertainty"][seed] = np.mean(uncertainty_estimate, axis=1)
        results["truncation"][seed] = np.sum(truncation,axis=1)/settings["num_rollouts"]
        results["mean_reward"][seed] = np.mean(mean_rewards, axis=1)

    # Print the mean and std of maximum normalized rewards across seeds
    max_human_normalized_scores = []
    for seed in seeds:
        max_human_normalized_scores.append(np.max(results["human_normalized_score"][seed]["mean"]))
    max_human_normalized_scores = np.array(max_human_normalized_scores)
    print(f"Max human normalized scores: {np.mean(max_human_normalized_scores)} ± {np.std(max_human_normalized_scores)}")

    lines = [Line2D([0], [0], color=seeds[seed], linestyle="-", marker="", linewidth=2.0)
            for seed in seeds]
    for plot_for in plot_for_list:
        fig = plt.figure(figsize=figsize)
        plot_for_separate(plot_for, results[plot_for], seeds, model_seeds, iterations_step)
        plt.ticklabel_format(axis='x', style='sci', scilimits=(5, 6), useMathText=True)
        plt.xlabel("Number of environment interactions")
        plt.ylabel(plot_settings[plot_for]["ylabel"])
        plt.xlim([iterations_step[0], iterations_step[-1]])
        plt.ylim(plot_settings[plot_for]["ylim"])
        plt.grid(True)

        lgd = fig.legend(lines, [model_seeds[seed]["label"] for seed in seeds], 
                         ncol=len(seeds), loc="upper center", bbox_to_anchor=bbox_to_anchor,
                        fontsize=fontsize, handlelength=1.0, labelspacing=0., handletextpad=0.5, columnspacing=1.0)

        plot_info = "seeds="
        for seed in seeds:
            plot_info += f"{seed}_" 

        for config in model_config:
            plot_info += f"{config}={model_config[config]}_"

        plot_info += f"prog4{plot_for}"
        # create figures dir if not exists
        if not os.path.exists("figures"):
            os.makedirs("figures")
        figpath = os.path.join("figures", f"{env_name}_{plot_info}{figname_extra}.pdf")
        print(figpath)
        plt.savefig(figpath, dpi=500, bbox_inches='tight', bbox_extra_artists=(lgd,))
        plt.close()


def main():
    log_dir = "/home/ck28372/sde4mbrltatu/log/"
    env_name = "walker2d-random-v2"
    seeds = {
        32: "red",
        62: "blue",
        # 92: "green",
        # 122: "magenta",
        # 152: "cyan",
    }
    figname_extra = ""
    plot_for_list = ["human_normalized_score", "uncertainty", "mean_reward", "truncation"]

    models = {
        "walker2d-random-v2": {
            "config": {
                "RL": 10,
                "CVaR": 0.99,
                "RPC": 0.001,
            },
            "seeds": {
                32: {
                    "label": "seed-32",
                    "model": "wk_rand_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0510_000247",
                },
                62: {
                    "label": "seed-62",
                    "model": "wk_rand_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0510_000254",
                },
            },
        },
        "walker2d-medium-v2": {
            "config": {
                "RL": 10,
                "CVaR": 0.98,
                "RPC": 1,
            },
            "seeds": {
                32: {
                    "label": "seed-32",
                    "model": "wk_m_final_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0510_151713",
                },
                62: {
                    "label": "seed-62",
                    "model": "wk_m_final_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0510_151719",
                },
            },
        },
        "walker2d-medium-replay-v2": {
            "config": {
                "RL": 10,
                "CVaR": 0.95,
                "RPC": 1,
            },
            "seeds": {
                32: {
                    "label": "seed-32",
                    "model": "wk_mr_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0510_151203",
                },
                62: {
                    "label": "seed-62",
                    "model": "wk_mr_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0510_151210",
                },
            },
        },
        "walker2d-medium-expert-v2": {
            "config": {
                "RL": 10,
                "CVaR": 0.98,
                "RPC": 1,
            },
            "seeds": {
                32: {
                    "label": "seed-32",
                    "model": "wk_me_final_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0510_151619",
                },
                62: {
                    "label": "seed-62",
                    "model": "wk_me_final_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0510_151625",
                },
            },
        },

    }

    settings = {
        "fontsize": 20,
        "figsize": (10, 4),
        "bbox_to_anchor": (.5, 1.1),
        "num_epochs": 1000,
        "steps_per_iter": 1000,
        "num_rollouts": 5000,
        "walker2d-random-v2":{
            "human_normalized_score": {
                "ylabel": 'Human Normalized Score',
                "ylim": [0., 30.]
            },
            "uncertainty": {
                "ylabel": 'Uncertainty',
                "ylim": [0.2, 0.4]
            },
            "truncation": {
                "ylabel": 'Truncation Ratio',
                "ylim": [0., 0.075]
            },
            "mean_reward": {
                "ylabel": 'Mean reward',
                "ylim": [-0.1, 1.5]
            },
        },
        "walker2d-medium-v2":{
            "human_normalized_score": {
                "ylabel": 'Human Normalized Score',
                "ylim": [0., 90.]
            },
            "uncertainty": {
                "ylabel": 'Uncertainty',
                "ylim": [0.25, 0.5]
            },
            "truncation": {
                "ylabel": 'Truncation Ratio',
                "ylim": [0., 0.075]
            },
            "mean_reward": {
                "ylabel": 'Mean reward',
                "ylim": [2, 4]
            },
        },
        "walker2d-medium-replay-v2":{
            "human_normalized_score": {
                "ylabel": 'Human Normalized Score',
                "ylim": [0., 100.]
            },
            "uncertainty": {
                "ylabel": 'Uncertainty',
                "ylim": [0.25, 0.5]
            },
            "truncation": {
                "ylabel": 'Truncation Ratio',
                "ylim": [0., 0.075]
            },
            "mean_reward": {
                "ylabel": 'Mean reward',
                "ylim": [1.5, 2.5]
            },
        },
        "walker2d-medium-expert-v2":{
            "human_normalized_score": {
                "ylabel": 'Human Normalized Score',
                "ylim": [0., 115.]
            },
            "uncertainty": {
                "ylabel": 'Uncertainty',
                "ylim": [0.25, 0.5]
            },
            "truncation": {
                "ylabel": 'Truncation Ratio',
                "ylim": [0., 0.075]
            },
            "mean_reward": {
                "ylabel": 'Mean reward',
                "ylim": [3, 4.5]
            },
        },
    }

    plot_results(
        log_dir,
        env_name,
        models[env_name],
        seeds,
        settings,
        plot_for_list,
        figname_extra,
    )

if __name__ == "__main__":
    main()