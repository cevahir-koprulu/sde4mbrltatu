import argparse
import datetime
import os
import random
import importlib

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
from logger import Logger
from trainer import Trainer_modelbsed


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="tatu_mopo_sde")
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

    # ensemble arguments
    parser.add_argument("--n-ensembles", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--dynamics-model-dir", type=str, default=None)
    # offline mbrl training arguments
    parser.add_argument("--reward-penalty-coef", type=float, default=1.0)
    parser.add_argument("--rollout-length", type=int, default=15)
    parser.add_argument("--rollout-batch-size", type=int, default=5000)
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    
    parser.add_argument("--real-ratio", type=float, default=0.05)
    parser.add_argument("--pessimism-coef", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=5.0)

    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--log-freq", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--critic_num", type=int, default=2)

    # SDE arguments
    parser.add_argument("--use_diffusion", default=True, action='store_true', help="To penalize uncertainty")
    parser.add_argument("--sde_model_id", type=int, default=0)
    parser.add_argument("--sde_num_particles", type=int, default=20)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--jax_gpu_mem_frac", type=str, default='0.2')
    parser.add_argument("--dt", type=float, default=0.05)

    args= parser.parse_args()

    sde_model_list = {
        'hopper-random-v2':["~/sde4mbrl/sde4mbrlExamples/d4rl_mujoco/my_models/"+\
            "hopper_random_v2_nn64_hr5_l2type11_nprior5_lr1e-4_siTrue_sde.pkl"],
        'hopper-medium-expert-v2': ["~/sde4mbrl/sde4mbrlExamples/d4rl_mujoco/my_models/"+\
            "hopper_medium_expert_v2_nn64_hr10_l2type1_nprior5_lr1e-4_sde.pkl"],
        'halfcheetah-random-v2': ["~/sde4mbrl/sde4mbrlExamples/d4rl_mujoco/my_models/"+\
                                  "halfcheetah_random_v2_nn64_hr10_l2type7_lr1e-4_siFalse_sde.pkl"]
    }
    args.sde_model_path = sde_model_list[args.task][args.sde_model_id]

    return args


def train(args=get_args()):
    # create env and dataset
    env = gym.make(args.task)
    dataset = d4rl.qlearning_dataset(env)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)

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

    # create policy
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
        device=args.device
    )

    # create dynamics model
    if not 'sde' in args.algo_name:
        dynamics_model = construct_model(
            obs_dim=np.prod(args.obs_shape),
            act_dim=args.action_dim,
            hidden_dim=200,
            num_networks=args.n_ensembles,
            num_elites=args.n_elites,
            model_type="mlp",
            separate_mean_var=True,
            load_dir=args.dynamics_model_dir
        )
    else:
        dynamics_model = {
            'model_path': args.sde_model_path,
            'use_gpu': args.use_gpu,
            'num_particles': args.sde_num_particles,
            'jax_gpu_mem_frac': args.jax_gpu_mem_frac,
            'dt': args.dt,
            'seed' : args.seed,
            'rollout_batch_size': args.rollout_batch_size,
        }

    # create buffer
    offline_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32
    )
    offline_buffer.load_dataset(dataset)
    model_buffer = ReplayBuffer(
        buffer_size=10*args.rollout_batch_size*args.rollout_length*args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32
    )
    
    # create MOPO algo
    task = args.task.split('-')[0]
    import_path = f"static_fns.{task}"
    static_fns = importlib.import_module(import_path).StaticFns
    if args.algo_name == "tatu_mopo" or 'sde' in args.algo_name:
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
        device=args.device
    )
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
    )
    elif args.algo_name == "tatu_combo" or 'sde' in args.algo_name:
        cql_policy = CQLPolicy(
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
            min_q_weight=args.beta,
            )
        algo = TATU_model_based(
            cql_policy,
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
    )
    else:
        raise Exception("Invalid algo name")
        

        
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    if "sde" in args.algo_name:
        log_file = f"diff={args.use_diffusion}_rl={args.rollout_length}_rpc={args.reward_penalty_coef}_pc={args.pessimism_coef}_rr={args.real_ratio}_ep={args.epoch}_rfq={args.rollout_freq}_spe={args.step_per_epoch}_alr={args.actor_lr}_clr={args.critic_lr}_seed={args.seed}_{t0}"
    else:
        log_file = f'critic_num_{critic_num}_seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_{args.algo_name}'
    log_path = os.path.join(args.logdir, args.task, args.algo_name, log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = Logger(writer)

    # create trainer
    trainer = Trainer_modelbsed(
        algo,
        eval_env=env,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        rollout_freq=args.rollout_freq,
        logger=logger,
        log_freq=args.log_freq,
        eval_episodes=args.eval_episodes
    )

    logger.print(f"use diffusion: {args.use_diffusion} rollout_length: {args.rollout_length:d} reward_penalty_coef: {args.reward_penalty_coef:.2f} pessimism_coef: {args.pessimism_coef:.2f} min_q_weight :{args.beta} real_ratio:{args.real_ratio}")
    # pretrain dynamics model on the whole dataset
    trainer.train_dynamics()
    
    # begin train
    trainer.train_policy()
    logger.print(f"env: {args.task} seed: {args.seed} use diffusion: {args.use_diffusion} rollout_length: {args.rollout_length:d} reward_penalty_coef: {args.reward_penalty_coef:.2f} pessimism_coef: {args.pessimism_coef:.2f} min_q_weight :{args.beta} real_ratio:{args.real_ratio}")


if __name__ == "__main__":
    train()