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
                        # self.logger.print(f'rollout_info: {rollout_info}')
                        # if num_timesteps % self._log_freq == 0:
                        #     print('max_disc:',rollout_info['max_disc'])
                        #     print('threshold:',rollout_info['threshold'])
                        #     print('halt_num:',rollout_info['halt_num'])
                        #     print('halt_ratio:',rollout_info['halt_ratio'])
                        #     print('Trunc Value:',rollout_info['disc'])
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
            
            # evaluate current policy
            eval_info = self._evaluate()
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            ep_reward_max = np.max(eval_info["eval/episode_reward"])
            ep_reward_min = np.min(eval_info["eval/episode_reward"])
            # Normalized score mean and std
            ep_reward_mean_normal = self.eval_env.get_normalized_score(ep_reward_mean)*100
            ep_reward_std_normal = self.eval_env.get_normalized_score(ep_reward_mean+ep_reward_std)*100 - ep_reward_mean_normal
            ep_reward_max_normal = self.eval_env.get_normalized_score(ep_reward_max)*100
            ep_reward_min_normal = self.eval_env.get_normalized_score(ep_reward_min)*100
            self.logger.record("eval/episode_reward", ep_reward_mean, num_timesteps, printed=False)
            self.logger.record("eval/episode_length", ep_length_mean, num_timesteps, printed=False)
            self.logger.record("eval/episode_reward_normal", ep_reward_mean_normal, num_timesteps, printed=False)
            self.logger.record("eval/max_episode_reward_normal", ep_reward_max_normal, num_timesteps, printed=False)
            self.logger.record("eval/min_episode_reward_normal", ep_reward_min_normal, num_timesteps, printed=False)
            self.logger.print(f"Epoch #{e}: episode_reward: {ep_reward_mean:.3f} ± {ep_reward_std:.3f}, episode_length: {ep_length_mean:.3f} ± {ep_length_std:.3f}")
            self.logger.print(f"Epoch #{e}: episode_reward_normal: {ep_reward_mean_normal:.1f} ± {ep_reward_std_normal:.1f}")
            for k, v in rollout_info.items():
                self.logger.print(f"Rollout Info: {k}: {v}")

            if ep_reward_mean > best_eval_mean:
                best_eval_mean = ep_reward_mean
                std_best_mean = ep_reward_std
            # save policy
            torch.save(self.algo.policy.state_dict(), os.path.join(self.logger.writer.get_logdir(), "policy.pth"))

        self.logger.print(f"last_eval_mean: {ep_reward_mean:.3f} ± {ep_reward_std:.3f}")
        last_eval_mean_normal = self.eval_env.get_normalized_score(ep_reward_mean)*100
        last_eval_std_normal = self.eval_env.get_normalized_score(ep_reward_mean+ep_reward_std)*100 - last_eval_mean_normal    
        self.logger.print(f"last_eval_mean_normal: {last_eval_mean_normal:.1f} ± {last_eval_std_normal:.1f}")    

        self.logger.print(f"best_eval_mean: {best_eval_mean:.3f} ± {std_best_mean:.3f}")
        best_eval_mean_normal = self.eval_env.get_normalized_score(best_eval_mean)*100
        std_best_mean_normal = self.eval_env.get_normalized_score(best_eval_mean + std_best_mean)*100 - best_eval_mean_normal
        # best_eval_mean_normal = d4rl.get_normalized_score(self.eval_env.unwrapped.spec.id,best_eval_mean)*100
        # std_best_mean_normal = d4rl.get_normalized_score(self.eval_env.unwrapped.spec.id,std_best_mean)*100
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

