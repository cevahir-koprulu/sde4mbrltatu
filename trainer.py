import time
import os

import numpy as np
import torch
import d4rl
from tqdm import tqdm


class Trainer_modelbsed:
    def __init__(
        self,
        algo,
        eval_env,
        epoch,
        step_per_epoch,
        rollout_freq,
        logger,
        log_freq,
        eval_episodes=10,
        eval_fake_env = False
    ):
        self.algo = algo
        self.eval_env = eval_env

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._rollout_freq = rollout_freq

        self.logger = logger
        self._log_freq = log_freq
        self._eval_episodes = eval_episodes
        self.eval_fake_env = eval_fake_env

    def train_dynamics(self):
        self.algo.learn_dynamics()
        # Save dynamics model
        self.algo.save_dynamics_model(
            save_path=os.path.join(self.logger.writer.get_logdir(), "dynamics_model")
        )

    def train_policy(self):
        start_time = time.time()
        num_timesteps = 0
        # train loop
        best_eval_mean = -1000
        std_best_mean = 0
        for e in range(1, self._epoch + 1):
            self.algo.policy.train()
            with tqdm(total=self._step_per_epoch, desc=f"Epoch #{e}/{self._epoch}") as t:
                while t.n < t.total:
                    rollout_infos = []
                    if num_timesteps % self._rollout_freq == 0:
                        for i in range(10):
                            rollout_info = self.algo.rollout_transitions()
                            rollout_info = { k : np.array(v) for k, v in rollout_info.items() }
                            rollout_infos.append(rollout_info)

                    # update policy by sac
                    loss = self.algo.learn_policy()
                    t.set_postfix(**loss)
                    # logs for rollout
                    if len(rollout_infos) > 0:
                        rollout_infos ={
                            f"Infos/{k}" : np.mean([np.mean(info[k]) for info in rollout_infos]) \
                                for k in rollout_infos[0].keys() \
                                    if k not in ['max_disc', 'threshold']
                        }
                    else:
                        rollout_infos = {}
                    # log
                    if num_timesteps % self._log_freq == 0:
                        for k, v in loss.items():
                            self.logger.record(k, v, num_timesteps, printed=False)
                        for k, v in rollout_infos.items():
                            self.logger.record(k, v, num_timesteps, printed=False)
                    num_timesteps += 1
                    t.update(1)

            # evaluate uncertainty
            eval_unc_stats = self._evaluate_uncertainty()
            self.log_eval_unc_stats(eval_unc_stats, e, num_timesteps)

            # evaluate current policy
            eval_info, obs_inits = self._evaluate()
            ep_reward_mean, ep_reward_std = self.log_eval_info(eval_info, e, num_timesteps)

            # if self.eval_fake_env:
            #     fake_eval_info = self._fake_evaluate(obs_inits)
            #     fake_ep_reward_mean, fake_ep_reward_std = \
            #         self.log_eval_info(fake_eval_info, e, num_timesteps, pref='fake_')

            for k, v in rollout_info.items():
                self.logger.print(f"Rollout Info: {k}: {v}")

            if ep_reward_mean > best_eval_mean:
                best_eval_mean = ep_reward_mean
                std_best_mean = ep_reward_std
                best_obs_inits = obs_inits
                # if self.eval_fake_env:
                #     fake_best_eval_mean = fake_ep_reward_mean
                #     fake_std_best_mean = fake_ep_reward_std
                torch.save(self.algo.policy.state_dict(), os.path.join(self.logger.writer.get_logdir(), "best_policy.pth"))      
            # save policy
            torch.save(self.algo.policy.state_dict(), os.path.join(self.logger.writer.get_logdir(), "policy.pth"))
            
        self.print_end_info(ep_reward_mean, ep_reward_std, best_eval_mean, std_best_mean)
        # if self.eval_fake_env:
            # self.print_end_info(
            #     fake_ep_reward_mean, fake_ep_reward_std, 
            #     fake_best_eval_mean, fake_std_best_mean,
            #     pref='fake_'
            # )

        if self.eval_fake_env:
            self.algo.policy.load_state_dict(torch.load(os.path.join(self.logger.writer.get_logdir(), "best_policy.pth")))
            fake_eval_info = self._fake_evaluate(best_obs_inits)
            # Penalized rewards
            fake_best_eval_mean_normal = self.eval_env.get_normalized_score(np.mean(fake_eval_info[f"eval/episode_reward"]))*100
            fake_std_best_mean_normal = self.eval_env.get_normalized_score(
                np.mean(fake_eval_info[f"eval/episode_reward"])+np.std(fake_eval_info[f"eval/episode_reward"]))*100 - fake_best_eval_mean_normal
            # Uncertainty
            fake_best_eval_mean_normal_unc = self.eval_env.get_normalized_score(np.mean(fake_eval_info[f"eval/episode_uncertainty"]))*100
            fake_std_best_mean_normal_unc = self.eval_env.get_normalized_score(
                np.mean(fake_eval_info[f"eval/episode_uncertainty"])+np.std(fake_eval_info[f"eval/episode_uncertainty"]))*100 - fake_best_eval_mean_normal_unc
            # Unpenalized rewards
            fake_best_eval_mean_normal_unpen = self.eval_env.get_normalized_score(np.mean(fake_eval_info[f"eval/episode_unpenalized_reward"]))*100
            fake_std_best_mean_normal_unpen = self.eval_env.get_normalized_score(
                np.mean(fake_eval_info[f"eval/episode_unpenalized_reward"])+np.std(fake_eval_info[f"eval/episode_unpenalized_reward"]))*100 - fake_best_eval_mean_normal_unpen
            self.logger.print(f"fake_best_eval_mean_normal: {fake_best_eval_mean_normal:.1f} ± {fake_std_best_mean_normal:.1f}")
            self.logger.print(f"fake_best_eval_mean_normal_unc: {fake_best_eval_mean_normal_unc:.1f} ± {fake_std_best_mean_normal_unc:.1f}")
            self.logger.print(f"fake_best_eval_mean_normal_unpen: {fake_best_eval_mean_normal_unpen:.1f} ± {fake_std_best_mean_normal_unpen:.1f}")

        self.logger.print("total time: {:.3f}s".format(time.time() - start_time))
        # self.logger.print("number of critics: {:d}".format(self.algo.policy.critic_num))

    def print_end_info(self, ep_reward_mean, ep_reward_std, best_eval_mean, std_best_mean, pref=''):
        self.logger.print(f"{pref}last_eval_mean: {ep_reward_mean:.3f} ± {ep_reward_std:.3f}")
        last_eval_mean_normal = self.eval_env.get_normalized_score(ep_reward_mean)*100
        last_eval_std_normal = self.eval_env.get_normalized_score(ep_reward_mean+ep_reward_std)*100 - last_eval_mean_normal
        self.logger.print(f"{pref}last_eval_mean_normal: {last_eval_mean_normal:.1f} ± {last_eval_std_normal:.1f}")
        
        self.logger.print(f"{pref}best_eval_mean: {best_eval_mean:.3f} ± {std_best_mean:.3f}")
        best_eval_mean_normal = self.eval_env.get_normalized_score(best_eval_mean)*100
        std_best_mean_normal = self.eval_env.get_normalized_score(best_eval_mean + std_best_mean)*100 - best_eval_mean_normal
        self.logger.print(f"{pref}best_eval_mean_normal: {best_eval_mean_normal:.1f} ± {std_best_mean_normal:.1f}")

    def log_eval_info(self, eval_info, e, num_timesteps, pref=''):
        eval_name = "eval"
        ep_reward_mean, ep_reward_std = np.mean(eval_info[f"{eval_name}/episode_reward"]), np.std(eval_info[f"{eval_name}/episode_reward"])
        ep_length_mean, ep_length_std = np.mean(eval_info[f"{eval_name}/episode_length"]), np.std(eval_info[f"{eval_name}/episode_length"])
        ep_reward_max = np.max(eval_info[f"{eval_name}/episode_reward"])
        ep_reward_min = np.min(eval_info[f"{eval_name}/episode_reward"])

        # Normalized score mean and std
        eval_name = f"{pref}eval"
        ep_reward_mean_normal = self.eval_env.get_normalized_score(ep_reward_mean)*100
        ep_reward_std_normal = self.eval_env.get_normalized_score(ep_reward_mean+ep_reward_std)*100 - ep_reward_mean_normal
        ep_reward_max_normal = self.eval_env.get_normalized_score(ep_reward_max)*100
        ep_reward_min_normal = self.eval_env.get_normalized_score(ep_reward_min)*100
        self.logger.record(f"{eval_name}/episode_reward", ep_reward_mean, num_timesteps, printed=False)
        self.logger.record(f"{eval_name}/episode_length", ep_length_mean, num_timesteps, printed=False)
        self.logger.record(f"{eval_name}/episode_reward_normal", ep_reward_mean_normal, num_timesteps, printed=False)
        self.logger.record(f"{eval_name}/max_episode_reward_normal", ep_reward_max_normal, num_timesteps, printed=False)
        self.logger.record(f"{eval_name}/min_episode_reward_normal", ep_reward_min_normal, num_timesteps, printed=False)
        self.logger.print(f"Epoch #{e}: {pref}episode_reward: {ep_reward_mean:.3f} ± {ep_reward_std:.3f}, {pref}episode_length: {ep_length_mean:.3f} ± {ep_length_std:.3f}")
        self.logger.print(f"Epoch #{e}: {pref}episode_reward_normal: {ep_reward_mean_normal:.1f} ± {ep_reward_std_normal:.1f}")
        return ep_reward_mean, ep_reward_std
    
    def log_eval_unc_stats(self, eval_unc_stats, e, num_timesteps):
        for k, v in eval_unc_stats.items():
            for k_i, v_i in v.items():
                self.logger.record(f"eval_unc_stats/{k}_{k_i}", v_i, num_timesteps, printed=False)
            self.logger.print(f"Epoch #{e}: eval_unc_stats/{k}: {eval_unc_stats[k]['mean']:.3f} ± {eval_unc_stats[k]['std']:.3f}")

    def _evaluate(self):
        self.algo.policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0
        obs_inits = [obs]

        while num_episodes < self._eval_episodes:
            action = self.algo.policy.sample_action(obs, deterministic=True)
            next_obs, reward, terminal, _ = self.eval_env.step(action)
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()
                obs_inits.append(obs)
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }, obs_inits[:self._eval_episodes]

    def _evaluate_uncertainty(self):
        return self.algo.compute_uncertainty_distribution()

    def _fake_evaluate(self, obs_inits):
        self.algo.policy.eval()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0
        episode_uncertainty = 0
        episode_unpenalized_reward = 0
        rng = None
        for obs_init in obs_inits:
            obs = np.array(obs_init)
            # _iter_num = 0
            for _iter_num in range(self.algo.max_steps_per_env):
                try:
                    action = self.algo.policy.sample_action(obs, deterministic=True)
                except:
                    pass
                action = np.array(action)
                next_obs, reward, terminal, infos = self.algo.fake_env.step_eval(
                    obs, action, rng=rng
                )
                rng = infos['next_rng']
                uncertainty = infos['uncertainty']
                unpenalized_reward = infos['unpenalized_reward']
                episode_reward += reward
                episode_length += 1
                episode_uncertainty += uncertainty
                episode_unpenalized_reward += unpenalized_reward
                obs = next_obs
                if terminal or _iter_num == self.algo.max_steps_per_env-1:
                    eval_ep_info_buffer.append(
                        {   "episode_reward": episode_reward,
                            "episode_length": episode_length,
                            "episode_uncertainty": episode_uncertainty,
                            "episode_unpenalized_reward": episode_unpenalized_reward
                         }
                    )
                    num_episodes +=1
                    episode_reward, episode_length = 0, 0
                    episode_uncertainty = 0
                    episode_unpenalized_reward = 0
                    break
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer],
            "eval/episode_uncertainty": [ep_info["episode_uncertainty"] for ep_info in eval_ep_info_buffer],
            "eval/episode_unpenalized_reward": [ep_info["episode_unpenalized_reward"] for ep_info in eval_ep_info_buffer],
        }
        
        

class Trainer_modelfree:
    def __init__(
        self,
        algo,
        eval_env,
        epoch,
        step_per_epoch,
        rollout_freq,
        logger,
        log_freq,
        eval_episodes=10
    ):
        self.algo = algo
        self.eval_env = eval_env

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._rollout_freq = rollout_freq

        self.logger = logger
        self._log_freq = log_freq
        self._eval_episodes = eval_episodes

    def train_policy(self):
        start_time = time.time()
        num_timesteps = 0
        # train loop
        best_eval_mean = -1000
        std_best_mean = 0
        for e in range(1, self._epoch + 1):
            self.algo.policy.train()
            with tqdm(total=self._step_per_epoch, desc=f"Epoch #{e}/{self._epoch}") as t:
                while t.n < t.total:
                    loss = self.algo.learn_policy()

                    t.set_postfix(**loss)
                    # log
                    if num_timesteps % self._log_freq == 0:
                        for k, v in loss.items():
                            self.logger.record(k, v, num_timesteps, printed=False)
                    num_timesteps += 1
                    t.update(1)
            
            # evaluate current policy
            eval_info = self._evaluate()
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            # Normalized score mean and std
            ep_reward_mean_normal = self.eval_env.get_normalized_score(ep_reward_mean)*100
            ep_reward_std_normal = self.eval_env.get_normalized_score(ep_reward_std)*100
            self.logger.record("eval/episode_reward", ep_reward_mean, num_timesteps, printed=False)
            self.logger.record("eval/episode_length", ep_length_mean, num_timesteps, printed=False)
            self.logger.record("eval/episode_reward_normal", ep_reward_mean_normal, num_timesteps, printed=False)
            self.logger.print(f"Epoch #{e}: episode_reward: {ep_reward_mean:.3f} ± {ep_reward_std:.3f}, episode_length: {ep_length_mean:.3f} ± {ep_length_std:.3f}")
            self.logger.print(f"Epoch #{e}: episode_reward_normal: {ep_reward_mean_normal:.1f} ± {ep_reward_std_normal:.1f}")
            

            if ep_reward_mean > best_eval_mean:
                best_eval_mean = ep_reward_mean
                std_best_mean = ep_reward_std
            # save policy
            # try:
            #     torch.save(self.algo.policy.state_dict(), os.path.join(self.logger.writer.get_logdir(), "policy.pth"))
            # except:
            #     pass

        self.logger.print(f"last_eval_mean: {ep_reward_mean:.3f} ± {ep_reward_std:.3f}")
        last_eval_mean_normal = self.eval_env.get_normalized_score(ep_reward_mean)*100
        last_eval_std_normal = self.eval_env.get_normalized_score(ep_reward_std)*100       
        self.logger.print(f"last_eval_mean_normal: {last_eval_mean_normal:.1f} ± {last_eval_std_normal:.1f}")    

        self.logger.print(f"best_eval_mean: {best_eval_mean:.3f} ± {std_best_mean:.3f}")
        best_eval_mean_normal = d4rl.get_normalized_score(self.eval_env.unwrapped.spec.id,best_eval_mean)*100
        std_best_mean_normal = d4rl.get_normalized_score(self.eval_env.unwrapped.spec.id,std_best_mean)*100
        self.logger.print(f"best_eval_mean_normal: {best_eval_mean_normal:.1f} ± {std_best_mean_normal:.1f}")

        self.logger.print("total time: {:.3f}s".format(time.time() - start_time))
        # self.logger.print("number of critics: {:d}".format(self.algo.policy.critic_num))

    def _evaluate(self):
        self.algo.policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < self._eval_episodes:
            action = self.algo.policy.sample_action(obs, deterministic=True)
            next_obs, reward, terminal, _ = self.eval_env.step(action)
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }

