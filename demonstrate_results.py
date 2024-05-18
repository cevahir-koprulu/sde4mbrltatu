import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def parse_str_for_number(entry):
    first_digit = entry[0] if entry[0].isdigit() else ''
    last_digit = entry[-1] if entry[-1].isdigit() and len(entry)>1 else ''
    return first_digit + entry[1:-1] + last_digit

def parse_progress_file(filepath, horizon=10, num_epochs=1000, has_fake_env_results=False):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    normalized_rewards_mean = np.zeros(num_epochs)
    normalized_rewards_std = np.zeros(num_epochs)
    fake_normalized_rewards_mean = np.zeros(num_epochs)
    fake_normalized_rewards_std = np.zeros(num_epochs)
    truncation = np.zeros((num_epochs,horizon), dtype=int)
    uncertainty_estimate = np.zeros((num_epochs,horizon))
    mean_rewards = np.zeros((num_epochs,horizon))
    line_no = 0
    ep_no = -1
    all_recorded = False
    while line_no < len(lines):
        line_splitted = lines[line_no].split()
        if "Epoch" in line_splitted:
            if "episode_reward_normal:" in line_splitted:
                ep_no += 1
                normalized_rewards_mean[ep_no] = float(line_splitted[3])
                normalized_rewards_std[ep_no] = float(line_splitted[5])
            elif has_fake_env_results and "fake_episode_reward_normal:" in line_splitted:
                fake_normalized_rewards_mean[ep_no] = float(line_splitted[3])
                fake_normalized_rewards_std[ep_no] = float(line_splitted[5])
        elif "halt_num:" in line_splitted:   
            truncation_splitted = line_splitted
            extra_line_no = 1
            while lines[line_no+extra_line_no].split()[0][-1].isdigit():
                truncation_splitted += lines[line_no+extra_line_no].split()
                extra_line_no += 1
            line_no += extra_line_no - 1
        elif "disc:" in line_splitted:
            uncertainty_estimate_splitted = line_splitted
            extra_line_no = 1
            while lines[line_no+extra_line_no].split()[0][-1].isdigit():
                uncertainty_estimate_splitted += lines[line_no+extra_line_no].split()
                extra_line_no += 1
            line_no += extra_line_no - 1
        elif "mean_rew:" in line_splitted:
            mean_rewards_splitted = line_splitted
            extra_line_no = 1
            while lines[line_no+extra_line_no].split()[0][-1].isdigit():
                mean_rewards_splitted += lines[line_no+extra_line_no].split()
                extra_line_no += 1
            line_no += extra_line_no - 1
            all_recorded = True
        line_no += 1
        if all_recorded: 
            trunc_first_digit_idx = 3 if truncation_splitted[3][-1].isdigit() else 4
            for h in range(horizon):
                truncation[ep_no,h] = int(parse_str_for_number(truncation_splitted[trunc_first_digit_idx+h]))
                uncertainty_estimate[ep_no,h] = float(parse_str_for_number(uncertainty_estimate_splitted[3+h]))
                mean_rewards[ep_no,h] = float(parse_str_for_number(mean_rewards_splitted[3+h]))

    return normalized_rewards_mean, normalized_rewards_std, fake_normalized_rewards_mean, fake_normalized_rewards_std, \
          truncation, uncertainty_estimate, mean_rewards

def plot_for_separate(plot_for, results, seeds, model_seeds, iterations):
    for seed in seeds:
        color = seeds[seed]
        label = model_seeds[seed]["label"]
        result = results[seed]["mean"] if plot_for == "human_normalized_score" else results[seed]
        plt.plot(iterations, result, color=color, linewidth=2.0, label=label, marker=".")

def plot_results(log_dir, env_name, models, seeds, settings, plot_for_list, has_fake_env_results, figname_extra):
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
        "fake_normalized_score": {},
        "uncertainty": {},
        "truncation": {},
        "mean_reward": {},
    }
    for seed in seeds:
        filepath = os.path.join(log_dir, env_name, "tatu_mopo_sde", model_seeds[seed]["model"], "progress.txt")
        print(model_seeds[seed]["model"])
        normalized_rewards_mean, normalized_rewards_std, fake_normalized_rewards_mean, fake_normalized_rewards_std, \
            truncation, uncertainty_estimate, mean_rewards = parse_progress_file(filepath, 
                                                                                 horizon=model_config["RL"], 
                                                                                 num_epochs=num_epochs,
                                                                                 has_fake_env_results=has_fake_env_results)
        results["human_normalized_score"][seed] = {
            "mean": normalized_rewards_mean,
            "std": normalized_rewards_std,
        }
        results["fake_normalized_score"][seed] = {
            "mean": fake_normalized_rewards_mean,
            "std": fake_normalized_rewards_std,
        }
        results["uncertainty"][seed] = np.mean(uncertainty_estimate, axis=1)
        results["truncation"][seed] = np.sum(truncation,axis=1)/settings["num_rollouts"]
        results["mean_reward"][seed] = np.mean(mean_rewards, axis=1)

    # Print the mean and std of maximum normalized rewards across seeds
    max_human_normalized_scores = []
    max_human_normalized_scores_idx = {}
    for seed in seeds:
        max_human_normalized_scores_idx[seed] = np.argmax(results["human_normalized_score"][seed]["mean"])
        max_human_normalized_scores.append(results["human_normalized_score"][seed]["mean"][max_human_normalized_scores_idx[seed]])
    max_human_normalized_scores = np.array(max_human_normalized_scores)
    print(f"Max human normalized scores: {np.mean(max_human_normalized_scores)} ± {np.std(max_human_normalized_scores)}")
    if has_fake_env_results:
        max_fake_normalized_scores = []
        for seed in seeds:
            max_fake_normalized_scores.append(results["fake_normalized_score"][seed]["mean"][max_human_normalized_scores_idx[seed]])
        max_fake_normalized_scores = np.array(max_fake_normalized_scores)
        print(f"Max fake normalized scores: {np.mean(max_fake_normalized_scores)} ± {np.std(max_fake_normalized_scores)}")
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
    # env_name = "halfcheetah-random-v2"
    env_name = "HalfCheetah-v3-High-1000-neorl"
    seeds = {
        32: "red",
        62: "blue",
        92: "green",
        122: "magenta",
        # 152: "cyan",
    }
    figname_extra = ""
    plot_for_list = ["human_normalized_score", "uncertainty", "mean_reward", "truncation"]
    has_fake_env_results = False

    models = {
        ########### HALFCHEETAH NeoRL ############
        "HalfCheetah-v3-Low-1000-neorl": {
            "config": {
                "RL": 10,
                "CVaR": 1,
                "RPC": 1,
            },
            "seeds": {
                32: {
                    "label": "seed-32",
                    "model": "hc_l_final_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0517_225453",
                },
                62: {
                    "label": "seed-62",
                    "model": "hc_l_final_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0517_225314",
                },
                92: {
                    "label": "seed-92",
                    "model": "hc_l_final_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0517_225317",
                },
                122: {
                    "label": "seed-122",
                    "model": "hc_l_final_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0517_225316",
                },
            },
        },
        "HalfCheetah-v3-Medium-1000-neorl": {
            "config": {
                "RL": 10,
                "CVaR": 0.99,
                "RPC": 1,
            },
            "seeds": {
                32: {
                    "label": "seed-32",
                    "model": "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0517_014642",
                },
                62: {
                    "label": "seed-62",
                    "model": "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0517_014647",
                },
                92: {
                    "label": "seed-92",
                    "model": "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0517_014659",
                },
                122: {
                    "label": "seed-122",
                    "model": "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0517_014707",
                },
            },
        },
        "HalfCheetah-v3-High-1000-neorl": {
            "config": {
                "RL": 5,
                "CVaR": 0.99,
                "RPC": 1,
            },
            "seeds": {
                32: {
                    "label": "seed-32",
                    "model": "hc_h_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0517_225516",
                },
                62: {
                    "label": "seed-62",
                    "model": "hc_h_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0517_225539",
                },
                92: {
                    "label": "seed-92",
                    "model": "hc_h_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0517_225523",
                },
                122: {
                    "label": "seed-122",
                    "model": "hc_h_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0517_225521",
                },
            },
        },
        ########### HOPPER NeoRL ############
        "Hopper-v3-Low-1000-neorl": {
            "config": {
                "RL": 10,
                "CVaR": 0.99,
                "RPC": 0.001,
            },
            "seeds": {
                32: {
                    "label": "seed-32",
                    "model": "hop_l_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0517_020708",
                },
                62: {
                    "label": "seed-62",
                    "model": "hop_l_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0517_020723",
                },
                92: {
                    "label": "seed-92",
                    "model": "hop_l_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0517_020728",
                },
                122: {
                    "label": "seed-122",
                    "model": "hop_l_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0517_131938",
                },
            },
        },
        "Hopper-v3-Medium-1000-neorl": {
            "config": {
                "RL": 5,
                "CVaR": 0.9,
                "RPC": 0.1,
            },
            "seeds": {
                32: {
                    "label": "seed-32",
                    "model": "hop_m_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0517_131748",
                },
                62: {
                    "label": "seed-62",
                    "model": "hop_m_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0517_131758",
                },
                92: {
                    "label": "seed-92",
                    "model": "hop_m_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0517_131800",
                },
                122: {
                    "label": "seed-122",
                    "model": "hop_m_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0517_131812",
                },
            },
        },
        "Hopper-v3-High-1000-neorl": {
            "config": {
                "RL": 5,
                "CVaR": 0.9,
                "RPC": 0.1,
            },
            "seeds": {
                32: {
                    "label": "seed-32",
                    "model": "hop_h_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0517_224049",
                },
                62: {
                    "label": "seed-62",
                    "model": "hop_h_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0517_224054",
                },
                92: {
                    "label": "seed-92",
                    "model": "hop_h_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0517_224056",
                },
                122: {
                    "label": "seed-122",
                    "model": "hop_h_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0517_224059",
                },
            },
        },
        ########### WALKER2D NeoRL ############
        "Walker2d-v3-Low-1000-neorl": {
            "config": {
                "RL": 5,
                "CVaR": 0.99,
                "RPC": 0.001,
            },
            "seeds": {
                32: {
                    "label": "seed-32",
                    "model": "wk_l_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0517_015600",
                },
                62: {
                    "label": "seed-62",
                    "model": "wk_l_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0517_015637",
                },
                92: {
                    "label": "seed-92",
                    "model": "wk_l_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0517_015643",
                },
                122: {
                    "label": "seed-122",
                    "model": "wk_l_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0517_015652",
                },
            },
        },
        "Walker2d-v3-Medium-1000-neorl": {
            "config": {
                "RL": 5,
                "CVaR": 0.99,
                "RPC": 0.001,
            },
            "seeds": {
                32: {
                    "label": "seed-32",
                    "model": "wk_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0517_021224",
                },
                62: {
                    "label": "seed-62",
                    "model": "wk_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0517_021322",
                },
                92: {
                    "label": "seed-92",
                    "model": "wk_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0517_021328",
                },
                122: {
                    "label": "seed-122",
                    "model": "wk_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0517_021335",
                },
            },
        },
        "Walker2d-v3-High-1000-neorl": {
            "config": {
                "RL": 5,
                "CVaR": 0.99,
                "RPC": 1,
            },
            "seeds": {
                32: {
                    "label": "seed-32",
                    "model": "wk_h_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0517_020135",
                },
                62: {
                    "label": "seed-62",
                    "model": "wk_h_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0517_020137",
                },
                92: {
                    "label": "seed-92",
                    "model": "wk_h_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0517_020139",
                },
                122: {
                    "label": "seed-122",
                    "model": "wk_h_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0517_020142",
                },
            },
        },
        ########### HALFCHEETAH D4RL ############
        "halfcheetah-random-v2": {
            "config": {
                "RL": 20,
                "CVaR": 1,
                "RPC": 0.001,
            },
            "seeds": {
                32: {
                    "label": "seed-32",
                    "model": "hc_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0515_143848",
                },
                62: {
                    "label": "seed-62",
                    "model": "hc_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0515_144109",
                },
                92: {
                    "label": "seed-92",
                    "model": "hc_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0515_144113",
                },
                122: {
                    "label": "seed-122",
                    "model": "hc_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0515_144132",
                },
                152: {
                    "label": "seed-152",
                    "model": "hc_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0515_144145",
                },
            },
        },
        "halfcheetah-medium-v2": {
            "config": {
                "RL": 5,
                "CVaR": 0.99,
                "RPC": 1,
            },
            "seeds": {
                32: {
                    "label": "seed-32",
                    "model": "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0515_164842",
                },
                62: {
                    "label": "seed-62",
                    "model": "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0515_164851",
                },
                92: {
                    "label": "seed-92",
                    "model": "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0515_164857",
                },
                122: {
                    "label": "seed-122",
                    "model": "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0515_164902",
                },
                152: {
                    "label": "seed-152",
                    "model": "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0515_164905",
                },
            },
        },
        "halfcheetah-medium-replay-v2": {
            "config": {
                "RL": 5,
                "CVaR": 0.9,
                "RPC": 1,
            },
            "seeds": {
                32: {
                    "label": "seed-32",
                    "model": "hc_mr_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0515_144719",
                },
                62: {
                    "label": "seed-62",
                    "model": "hc_mr_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0515_144725",
                },
                92: {
                    "label": "seed-92",
                    "model": "hc_mr_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0515_144738",
                },
                122: {
                    "label": "seed-122",
                    "model": "hc_mr_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0515_144741",
                },
                152: {
                    "label": "seed-152",
                    "model": "hc_mr_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0515_144747",
                },
            },
        },
        "halfcheetah-medium-expert-v2": {
            "config": {
                "RL": 10,
                "CVaR": 0.95,
                "RPC": 1,
            },
            "seeds": {
                32: {
                    "label": "seed-32",
                    "model": "hc_me_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0515_165012",
                },
                62: {
                    "label": "seed-62",
                    "model": "hc_me_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0515_165021",
                },
                92: {
                    "label": "seed-92",
                    "model": "hc_me_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0515_165023",
                },
                122: {
                    "label": "seed-122",
                    "model": "hc_me_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0515_165023",
                },
                152: {
                    "label": "seed-152",
                    "model": "hc_me_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0515_165025",
                },
            },
        },
        
        ########### HOPPER D4RL ############
        "hopper-random-v2": {
            "config": {
                "RL": 10,
                "CVaR": 1,
                "RPC": 0.001,
            },
            "seeds": {
                32: {
                    "label": "seed-32",
                    "model": "hop_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0428_152526",
                },
                62: {
                    "label": "seed-62",
                    "model": "hop_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0429_111226",
                },
                92: {
                    "label": "seed-92",
                    "model": "hop_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0511_152510",
                },
                122: {
                    "label": "seed-122",
                    "model": "hop_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0511_152516",
                },
                152: {
                    "label": "seed-152",
                    "model": "hop_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0514_223248",
                },
            },
        },
        "hopper-medium-v2": {
            "config": {
                "RL": 10,
                "CVaR": 0.99,
                "RPC": 1,
            },
            "seeds": {
                32: {
                    "label": "seed-32",
                    "model": "hop_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0512_172225",
                },
                62: {
                    "label": "seed-62",
                    "model": "hop_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0512_172239",
                },
                92: {
                    "label": "seed-92",
                    "model": "hop_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0514_224512",
                },
                122: {
                    "label": "seed-122",
                    "model": "hop_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0514_224528",
                },
                152: {
                    "label": "seed-152",
                    "model": "hop_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0514_224604",
                },
            },
        },
        "hopper-medium-replay-v2": {
            "config": {
                "RL": 10,
                "CVaR": 0.99,
                "RPC": 1,
            },
            "seeds": {
                32: {
                    "label": "seed-32",
                    "model": "hop_mr_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0511_154029",
                },
                62: {
                    "label": "seed-62",
                    "model": "hop_mr_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0511_154035",
                },
                92: {
                    "label": "seed-92",
                    "model": "hop_mr_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0512_182205",
                },
                122: {
                    "label": "seed-122",
                    "model": "hop_mr_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0512_182213",
                },
                152: {
                    "label": "seed-152",
                    "model": "hop_mr_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0514_224951",
                },
            },
        },
        "hopper-medium-expert-v2": {
            "config": {
                "RL": 10,
                "CVaR": 0.99,
                "RPC": 1,
            },
            "seeds": {
                32: {
                    "label": "seed-32",
                    "model": "hop_me_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0511_155127",
                },
                62: {
                    "label": "seed-62",
                    "model": "hop_me_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0511_155136",
                },
                92: {
                    "label": "seed-92",
                    "model": "hop_me_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0512_182707",
                },
                122: {
                    "label": "seed-122",
                    "model": "hop_me_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0512_182708",
                },
                152: {
                    "label": "seed-152",
                    "model": "hop_me_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0514_225328",
                },
            },
        },
        ########### WALKER2D D4RL ############
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
                92: {
                    "label": "seed-92",
                    "model": "wk_rand_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0514_222458",
                },
                122: {
                    "label": "seed-122",
                    "model": "wk_rand_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0514_222518",
                },
                152: {
                    "label": "seed-152",
                    "model": "wk_rand_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0514_222524",
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
                92: {
                    "label": "seed-92",
                    "model": "wk_m_final_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0514_222637",
                },
                122: {
                    "label": "seed-122",
                    "model": "wk_m_final_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0514_222652",
                },
                152: {
                    "label": "seed-152",
                    "model": "wk_m_final_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0514_222658",
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
                92: {
                    "label": "seed-92",
                    "model": "wk_mr_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0514_222821",
                },
                122: {
                    "label": "seed-122",
                    "model": "wk_mr_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0514_222828",
                },
                152: {
                    "label": "seed-152",
                    "model": "wk_mr_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0514_222844",
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
                92: {
                    "label": "seed-92",
                    "model": "wk_me_final_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0514_223003",
                },
                122: {
                    "label": "seed-122",
                    "model": "wk_me_final_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0514_223007",
                },
                152: {
                    "label": "seed-152",
                    "model": "wk_me_final_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0514_223025",
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
        "HalfCheetah-v3-Low-1000-neorl": {
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
                "ylim": [0., 1.5]
            },
        },
        "HalfCheetah-v3-Medium-1000-neorl": {
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
                "ylim": [0., 1.5]
            },
        },
        "HalfCheetah-v3-High-1000-neorl": {
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
                "ylim": [0., 1.5]
            },
        },
        "Hopper-v3-Low-1000-neorl": {
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
                "ylim": [0., 1.5]
            },
        },
        "Hopper-v3-Medium-1000-neorl": {
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
                "ylim": [0., 1.5]
            },
        },
        "Hopper-v3-High-1000-neorl": {
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
                "ylim": [0., 1.5]
            },
        },
        "Walker2d-v3-Low-1000-neorl": {
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
                "ylim": [0., 1.5]
            },
        },
        "Walker2d-v3-Medium-1000-neorl": {
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
                "ylim": [0., 1.5]
            },
        },
        "Walker2d-v3-High-1000-neorl": {
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
                "ylim": [0., 1.5]
            },
        },
        "halfcheetah-random-v2":{
            "human_normalized_score": {
                "ylabel": 'Human Normalized Score',
                "ylim": [0., 35.]
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
        "halfcheetah-medium-v2":{
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
        "halfcheetah-medium-replay-v2":{
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
        "halfcheetah-medium-expert-v2":{
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
        "hopper-random-v2":{
            "human_normalized_score": {
                "ylabel": 'Human Normalized Score',
                "ylim": [0., 35.]
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
        "hopper-medium-v2":{
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
        "hopper-medium-replay-v2":{
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
        "hopper-medium-expert-v2":{
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
        "walker2d-random-v2":{
            "human_normalized_score": {
                "ylabel": 'Human Normalized Score',
                "ylim": [0., 35.]
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
        has_fake_env_results,
        figname_extra,
    )

if __name__ == "__main__":
    main()