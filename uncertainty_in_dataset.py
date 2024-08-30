import argparse
import datetime
import os
import random
import importlib
import math
import gym
import d4rl

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
import tensorflow.compat.v1 as tf

from models.tf_dynamics_models.constructor import construct_model
from models.policy_models import MLP, ActorProb, Critic, DiagGaussian

from policys.policy import SACPolicy,CQLPolicy

from algo import TATU_model_based
from buffer import ReplayBuffer
from trainer import Trainer_modelbsed

from logger import Logger

from nsdes_dynamics.utils_for_d4rl_mujoco import (
    get_formatted_dataset_for_nsde_training,
    get_environment_infos_from_name,
    load_neorl_dataset
)


info_dict = {
    "halfcheetah-random-v2": {
        "tatu_mopo_sde": {  
            "config": {
                "RL": 20,
                "CVaR": 1,
                "RPC": 0.001,
            },
            "seeds": {
                32: "hc_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0520_171344",
                62: "hc_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0520_171425",
                92: "hc_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0520_171432",
                122: "hc_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0520_171434",
                152: "hc_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0520_171451",
            }
        },
        "tatu_mopo":{
            "config": {
                "RL": 5,
                "PC": 2,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0518_155708-halfcheetah_random_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0518_155712-halfcheetah_random_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0518_155715-halfcheetah_random_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0518_155716-halfcheetah_random_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0518_155719-halfcheetah_random_v2_tatu_mopo",
            },
        },
        "mopo":{
            "config": {
                "RL": 5,
                "RPC": 0.5,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0519_152820_pc=1e-06-halfcheetah_random_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0519_152840_pc=1e-06-halfcheetah_random_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0519_152843_pc=1e-06-halfcheetah_random_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0519_152858_pc=1e-06-halfcheetah_random_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0519_152907_pc=1e-06-halfcheetah_random_v2_tatu_mopo",
            },
        },
    },
    "halfcheetah-medium-v2": {
        "tatu_mopo_sde": {
            "config": {
                "RL": 5,
                "CVaR": 0.99,
                "RPC": 1,
            },
            "seeds": {
                32: "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0515_164842",
                62: "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0515_164851",
                92: "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0515_164857",
                122: "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0515_164902",
                152: "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0515_164905",
            },
        },
        "tatu_mopo":{
            "config": {
                "RL": 5,
                "PC": 2,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0518_155450-halfcheetah_medium_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0518_155501-halfcheetah_medium_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0518_155509-halfcheetah_medium_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0518_155514-halfcheetah_medium_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0518_155524-halfcheetah_medium_v2_tatu_mopo",
            },
        },
        "mopo":{
            "config": {
                "RL": 1,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0519_153029_pc=1e-06-halfcheetah_medium_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0519_153032_pc=1e-06-halfcheetah_medium_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0519_153040_pc=1e-06-halfcheetah_medium_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0519_153053_pc=1e-06-halfcheetah_medium_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0519_153056_pc=1e-06-halfcheetah_medium_v2_tatu_mopo",
            },
        },
    },
    "halfcheetah-medium-replay-v2": {
        "tatu_mopo_sde": {
            "config": {
                "RL": 5,
                "CVaR": 0.9,
                "RPC": 1,
            },
            "seeds": {
                32: "hc_mr_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0515_144719",
                62: "hc_mr_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0515_144725",
                92: "hc_mr_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0515_144738",
                122: "hc_mr_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0515_144741",
                152: "hc_mr_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0515_144747",

            },
        },
        "tatu_mopo":{
            "config": {
                "RL": 5,
                "PC": 2,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0518_155620-halfcheetah_medium_replay_v2_tatu_mopo",
                62:  "critic_num_2_seed_62_0518_155545-halfcheetah_medium_replay_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0518_155548-halfcheetah_medium_replay_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0518_155550-halfcheetah_medium_replay_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0518_155555-halfcheetah_medium_replay_v2_tatu_mopo",
            },
        },
        "mopo":{
            "config": {
                "RL": 5,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0519_153143_pc=1e-06-halfcheetah_medium_replay_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0519_153149_pc=1e-06-halfcheetah_medium_replay_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0519_153212_pc=1e-06-halfcheetah_medium_replay_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0519_153216_pc=1e-06-halfcheetah_medium_replay_v2_tatu_mopo",
                152:  "critic_num_2_seed_152_0519_153224_pc=1e-06-halfcheetah_medium_replay_v2_tatu_mopo",
            },
        },
    },
    "halfcheetah-medium-expert-v2": {
        "tatu_mopo_sde": {
            "config": {
                "RL": 10,
                "CVaR": 0.95,
                "RPC": 1,
            },
            "seeds": {
                32: "hc_me_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0515_165012",
                62: "hc_me_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0515_165021",
                92: "hc_me_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0515_165023",
                122: "hc_me_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0515_165023",
                152: "hc_me_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0515_165025",
            },
        },
        "tatu_mopo":{
            "config": {
                "RL": 5,
                "PC": 2,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0518_155659-halfcheetah_medium_expert_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0518_155702-halfcheetah_medium_expert_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0518_155703-halfcheetah_medium_expert_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0518_155708-halfcheetah_medium_expert_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0518_155712-halfcheetah_medium_expert_v2_tatu_mopo",
            },
        },
        "mopo":{
            "config": {
                "RL": 5,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0519_153307_pc=1e-06-halfcheetah_medium_expert_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0519_153330_pc=1e-06-halfcheetah_medium_expert_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0519_153335_pc=1e-06-halfcheetah_medium_expert_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0519_153339_pc=1e-06-halfcheetah_medium_expert_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0519_153343_pc=1e-06-halfcheetah_medium_expert_v2_tatu_mopo",
            },
        },
    },
    }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="tatu_mopo")
    parser.add_argument("--task", type=str, default="halfcheetah-random-v2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument('--auto-alpha', default=True)
    parser.add_argument('--target-entropy', type=int, default=-3)
    parser.add_argument('--alpha-lr', type=float, default=3e-4)

    # Ensemble arguments
    parser.add_argument("--n-ensembles", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--dynamics-model-dir", type=str, default=None)
    parser.add_argument("--dynamics-name", type=str, default=None)

    # Offline MBRL training arguments
    # parser.add_argument("--reward-penalty-coef", type=float, default=0.001)
    # parser.add_argument("--rollout-length", type=int, default=15)
    parser.add_argument("--rollout-batch-size", type=int, default=5000)
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--model-retain-epochs", type=int, default=5)

    parser.add_argument("--real-ratio", type=float, default=0.05)
    # parser.add_argument("--pessimism-coef", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=5.0)

    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--log-freq", type=int, default=1000)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    parser.add_argument("--critic_num", type=int, default=2)
    parser.add_argument("--eval_fake_env", default=False, action='store_true')

    # SDE arguments
    parser.add_argument(
        "--use_diffusion", default=False,
        action='store_true', help="To penalize uncertainty"
    )
    parser.add_argument("--sde_model_id", type=int, default=0)
    parser.add_argument("--cpkt_step", type=int, default=-2)
    parser.add_argument("--sde_num_particles", type=int, default=5)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--jax_gpu_mem_frac", type=str, default='0.5')
    parser.add_argument("--prob_init_obs", type=float, default=0)
    parser.add_argument("--batch_size_trunc_thresh", type=int, default=100)
    parser.add_argument("--num_particles_trunc_thresh", type=int, default=5)
    parser.add_argument("--unc_cvar_coef", type=float, default=0.95)
    parser.add_argument("--threshold_decision_var", type=str, default='diff_density', 
                        choices=['dad_based_diff', 'dad_free_diff', 'diff_density', 'diffusion_value', 'disc'])
    parser.add_argument("--model", type=str, default="")

    args= parser.parse_args()
    
    sde_model_list = {
        # D4RL
        'halfcheetah-random-v2': {
            0: 'hc_rand_final'
        },
        'halfcheetah-medium-v2': {
            0: 'hc_m_final',
        },
        'halfcheetah-medium-replay-v2': {
            0: 'hc_mr_final',
        },
        'halfcheetah-medium-expert-v2': {
            0: 'hc_me_final',
        },
        'hopper-random-v2':{
            0 : 'hop_rand_final',
        },
        'hopper-medium-v2':{
            0 : 'hop_m_final',
        },
        'hopper-medium-replay-v2':{
            0 : 'hop_mr_final',
        },
        'hopper-medium-expert-v2':{
            0: 'hop_me_final',
        },
        'walker2d-random-v2':{
            0 : 'wk_rand_final',
        },
        'walker2d-medium-v2':{
            0 : 'wk_m_final',
        },
        'walker2d-medium-replay-v2':{
            0 : 'wk_mr_final',
        },
        'walker2d-medium-expert-v2':{
            0 : 'wk_me_final',
        },
        # NeoRL
        'HalfCheetah-v3-Low-1000-neorl':{
            0: 'hc_l_final',
        },
        'HalfCheetah-v3-Medium-1000-neorl':{
            0: 'hc_m_final',
        },
        'HalfCheetah-v3-High-1000-neorl':{
            0: 'hc_h_final',
        },
        'Hopper-v3-Low-1000-neorl':{
            0: 'hop_l_final',
        },
        'Hopper-v3-Medium-1000-neorl':{
            0: 'hop_m_final',
        },
        'Hopper-v3-High-1000-neorl':{
            0: 'hop_h_final',
        },
        'Walker2d-v3-Low-1000-neorl':{
            0: 'wk_l_final',
        },
        'Walker2d-v3-Medium-1000-neorl':{
            0: 'wk_m_final',
        },
        'Walker2d-v3-High-1000-neorl':{
            0: 'wk_h_final',
        },
    }

    info = info_dict[args.task][args.algo_name]
    args.algo_name = "tatu_mopo" if "mopo" == args.algo_name else args.algo_name
    args.load_dir = info["seeds"][args.seed]
    args.rollout_length = info["config"]["RL"]
    args.unc_cvar_coef = info["config"].get("CVaR", 1)
    args.reward_penalty_coef = info["config"]["RPC"]
    args.pessimism_coef = info["config"].get("PC", 1)

    # Extract the model name
    model_name = args.model
    if len(model_name) > 0:
        args.sde_model_name = model_name
    else:
        args.sde_model_name = sde_model_list[args.task][args.sde_model_id]

    return args


def fake_step(observations, actions, algo):
    rng = None
    actions = np.array(actions)
    next_observations, rewards, terminals, infos = algo.fake_env.step_eval(
        observations, actions, rng=rng
    )
    return infos["uncertainty"]
        
def main(args):
    # Set numpy print options
    np.set_printoptions(precision=3, floatmode='fixed')
    
    # create env and dataset
    if "neorl" in args.task:
        dataset, env = load_neorl_dataset(args.task, return_env=True)
    else:
        env = gym.make(args.task)
        dataset = d4rl.qlearning_dataset(env)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)

    # Load the initial states in the dataset for later use to full traj simulation
    full_data_set = get_formatted_dataset_for_nsde_training(args.task)
    observ_init_dataset = np.array([_data['y'][0,:] for _data in full_data_set])

    # get environment info
    env_info = get_environment_infos_from_name(args.task)

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tf.set_random_seed(args.seed)

    if args.device != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    env.seed(args.seed)

    # create policy model
    critic_num = args.critic_num
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=[256, 256])
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape)+args.action_dim, hidden_dims=[256, 256])
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape)+args.action_dim, hidden_dims=[256, 256])
    dist = DiagGaussian( 
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True, 
        conditioned_sigma=True
    )

    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)
        
        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)    


    sac_policy = SACPolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        dist=dist,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        device=args.device,
    )
    load_dir = os.path.join(args.logdir, args.task, args.algo_name, args.load_dir)
    sac_policy.load_state_dict(torch.load(os.path.join(load_dir, "best_policy.pth")))

    if not 'sde' in args.algo_name:
        model_dir = os.path.join(load_dir, 'dynamics_model')
        dynamics_model = construct_model(
            obs_dim=np.prod(env.observation_space.shape),
            act_dim=np.prod(env.action_space.shape),
            hidden_dim=200,
            num_networks=args.n_ensembles,
            num_elites=args.n_elites,
            model_type="mlp",
            separate_mean_var=True,
            load_dir=model_dir,
            name="BNN_0",
        )
    elif args.algo_name == "tatu_mopo_sde":
        dynamics_model = {
            'model_name': args.sde_model_name,
            'use_gpu': args.use_gpu,
            'num_particles': args.sde_num_particles,
            'jax_gpu_mem_frac': args.jax_gpu_mem_frac,
            'stepsize': env_info['stepsize'],
            'seed' : args.seed,
            'rollout_batch_size': args.rollout_batch_size,
            "env_name": args.task,
            "ckpt_step": args.cpkt_step,
            "rollout_length": args.rollout_length,
            "batch_size_trunc_thresh": args.batch_size_trunc_thresh,
            "num_particles_trunc_thresh": args.num_particles_trunc_thresh,
            "threshold_decision_var": args.threshold_decision_var,
            "unc_cvar_coef": args.unc_cvar_coef,
        }
    else:
        raise NotImplementedError
    
    ###### BEGIN DUMMY #####
    # create buffer
    offline_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32
    )
    offline_buffer.load_dataset(dataset, observ_init_dataset)
    est_rollout_length = int (args.rollout_length * (1 - args.prob_init_obs) + 
                              args.prob_init_obs * env_info['max_episode_steps'])
    model_buffer = ReplayBuffer(
        buffer_size=10*args.rollout_batch_size* est_rollout_length * args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32
    )
    ###### END DUMMY #####

    # create MOPO algo
    task = args.task.split('-')[0].lower()
    import_path = f"static_fns.{task}"
    if "neorl" in args.task:
        static_fns = importlib.import_module(import_path).StaticFnsNeoRL
    else:
        static_fns = importlib.import_module(import_path).StaticFns
        
    algo = TATU_model_based(
        sac_policy,
        dynamics_model,
        static_fns=static_fns,
        offline_buffer=offline_buffer,
        model_buffer=model_buffer,
        reward_penalty_coef=args.reward_penalty_coef,
        rollout_length=args.rollout_length,
        rollout_batch_size=args.rollout_batch_size,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        pessimism_coef=args.pessimism_coef,
        is_sde = 'sde' in args.algo_name,
        use_diffusion = 'sde' in args.algo_name and args.use_diffusion,
        env_name=task,
        prob_init_obs = args.prob_init_obs,
        max_steps_per_env = env_info['max_episode_steps'],
        unc_cvar_coef=args.unc_cvar_coef,
    )


    from tqdm import tqdm

    chosen_state_features = [8,1]
    noise_std = [0.05, 0.5]
    batch_size = 1000
    num_batches = math.ceil(dataset['observations'].shape[0]/batch_size)
    num_noisy_samples = 5

    # uncertainty = np.zeros((dataset['observations'].shape[0]*num_noisy_samples, 3))
    # for batch_i in tqdm(range(num_batches)):
    #     observations = dataset['observations'][batch_i*batch_size:(batch_i+1)*batch_size]
    #     # Repeat observation for num_noisy_samples times
    #     observations = np.repeat(observations, num_noisy_samples, axis=0)
    #     # Add noise to the chosen state features
    #     for i, feature in enumerate(chosen_state_features):
    #         observations[:, feature] += np.random.normal(0, noise_std[i], observations.shape[0])

    #     # algo.policy.eval()
    #     # actions = algo.policy.sample_action(observations, deterministic=True)

    #     actions = np.repeat(dataset['actions'][batch_i*batch_size:(batch_i+1)*batch_size], num_noisy_samples, axis=0)
        
    #     batch_unc = fake_step(observations, actions, algo)
    #     uncertainty[batch_i*batch_size*num_noisy_samples:(batch_i+1)*batch_size*num_noisy_samples, :2] = observations[:,chosen_state_features]
    #     uncertainty[batch_i*batch_size*num_noisy_samples:(batch_i+1)*batch_size*num_noisy_samples, 2] = 1.0-batch_unc
    #     # input(f"batch {batch_i} done")

    uncertainty = np.zeros((dataset['observations'].shape[0], 3))
    for batch_i in tqdm(range(num_batches)):
        observations = dataset['observations'][batch_i*batch_size:(batch_i+1)*batch_size]
        actions = dataset['actions'][batch_i*batch_size:(batch_i+1)*batch_size]
        
        batch_unc = fake_step(observations, actions, algo)
        uncertainty[batch_i*batch_size:(batch_i+1)*batch_size, :2] = observations[:,chosen_state_features]
        uncertainty[batch_i*batch_size:(batch_i+1)*batch_size, 2] = batch_unc
        # input(f"batch {batch_i} done")

    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(10, 10))
    plt.scatter(uncertainty[:, 0], uncertainty[:, 1], s=1, c=uncertainty[:, 2], cmap='viridis')
    # plt.xlabel('vel_x')
    # plt.ylabel('angle')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('halfcheetah-random-v2_velxVSangle_unc.png', bbox_inches='tight')

    # fake_eval_info = fake_evaluate(obs_inits, algo)
    # # Penalized rewards
    # fake_best_eval_mean_normal = env.get_normalized_score(np.mean(fake_eval_info[f"eval/episode_reward"]))*100
    # fake_std_best_mean_normal = env.get_normalized_score(
    #     np.mean(fake_eval_info[f"eval/episode_reward"])+np.std(fake_eval_info[f"eval/episode_reward"]))*100 - fake_best_eval_mean_normal
    # # Uncertainty
    # fake_best_eval_mean_normal_unc = env.get_normalized_score(np.mean(fake_eval_info[f"eval/episode_uncertainty"]))*100
    # fake_std_best_mean_normal_unc = env.get_normalized_score(
    #     np.mean(fake_eval_info[f"eval/episode_uncertainty"])+np.std(fake_eval_info[f"eval/episode_uncertainty"]))*100 - fake_best_eval_mean_normal_unc
    # # Unpenalized rewards
    # fake_best_eval_mean_normal_unpen = env.get_normalized_score(np.mean(fake_eval_info[f"eval/episode_unpenalized_reward"]))*100
    # fake_std_best_mean_normal_unpen = env.get_normalized_score(
    #     np.mean(fake_eval_info[f"eval/episode_unpenalized_reward"])+np.std(fake_eval_info[f"eval/episode_unpenalized_reward"]))*100 - fake_best_eval_mean_normal_unpen
    
    
    # filename = os.path.join(load_dir, "fake_eval.txt")
    # with open(filename, 'w') as f:
    #     print(f"task: {args.task}, algo: {args.algo_name}, seed: {args.seed}")
    #     print(f"fake_best_eval_mean_normal: {fake_best_eval_mean_normal:.1f} ± {fake_std_best_mean_normal:.1f}", file=f)
    #     print(f"fake_best_eval_mean_normal_unc: {fake_best_eval_mean_normal_unc:.1f} ± {fake_std_best_mean_normal_unc:.1f}", file=f)
    #     print(f"fake_best_eval_mean_normal_unpen: {fake_best_eval_mean_normal_unpen:.1f} ± {fake_std_best_mean_normal_unpen:.1f}", file=f)

if __name__ == "__main__":
    main(get_args())