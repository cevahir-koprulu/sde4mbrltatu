import os
import numpy as np

from collections import defaultdict

from models.tf_dynamics_models.fake_env import FakeEnv, FakeEnv_tatu, FakeEnv_SDE_Trunc
from models.tf_dynamics_models.constructor import format_samples_for_training


def update_stats_dict(stats_dict, data):
    stats_dict["max"].append(np.max(data))
    stats_dict["mean"].append(np.mean(data))
    stats_dict["min"].append(np.min(data))
    stats_dict["std"].append(np.std(data))

class MOPO():
    def __init__(
        self,
        policy,
        dynamics_model,
        static_fns,
        offline_buffer,
        model_buffer,
        reward_penalty_coef,
        rollout_length,
        rollout_batch_size,
        batch_size,
        real_ratio,
    ):

        self.policy = policy
        self.dynamics_model = dynamics_model
        self.static_fns = static_fns
        self.fake_env = FakeEnv(
            self.dynamics_model,
            self.static_fns,
            penalty_coeff=reward_penalty_coef,
            penalty_learned_var=True
        )
        self.offline_buffer = offline_buffer
        self.model_buffer = model_buffer
        self._reward_penalty_coef = reward_penalty_coef
        self._rollout_length = rollout_length
        self._rollout_batch_size = rollout_batch_size
        self._batch_size = batch_size
        self._real_ratio = real_ratio

    def _sample_initial_transitions(self):
        return self.offline_buffer.sample(self._rollout_batch_size)

    def rollout_transitions(self):
        init_transitions = self._sample_initial_transitions()
        # rollout
        observations = init_transitions["observations"]
        for _ in range(self._rollout_length):
            actions = self.policy.sample_action(observations)
            next_observations, rewards, terminals, infos = self.fake_env.step(observations, actions)
            self.model_buffer.add_batch(observations, next_observations, actions, rewards, terminals)
        
            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]

    def learn_dynamics(self):
        data = self.offline_buffer.sample_all()
        train_inputs, train_outputs = format_samples_for_training(data)
        max_epochs = 1 if self.dynamics_model.model_loaded else None
        loss = self.dynamics_model.train(
            train_inputs,
            train_outputs,
            batch_size=self._batch_size,
            max_epochs=max_epochs,
            holdout_ratio=0.2
        )
        return loss

    def learn_policy(self):
        real_sample_size = int(self._batch_size * self._real_ratio)
        fake_sample_size = self._batch_size - real_sample_size
        real_batch = self.offline_buffer.sample(batch_size=real_sample_size)
        fake_batch = self.model_buffer.sample(batch_size=fake_sample_size)
        data = {
            "observations": np.concatenate([real_batch["observations"], fake_batch["observations"]], axis=0),
            "actions": np.concatenate([real_batch["actions"], fake_batch["actions"]], axis=0),
            "next_observations": np.concatenate([real_batch["next_observations"], fake_batch["next_observations"]], axis=0),
            "terminals": np.concatenate([real_batch["terminals"], fake_batch["terminals"]], axis=0),
            "rewards": np.concatenate([real_batch["rewards"], fake_batch["rewards"]], axis=0)
        }
        loss = self.policy.learn(data)
        return loss
    
    def save_dynamics_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.dynamics_model.save(save_path, timestep=0)

class TATU_model_based():
    def __init__(
        self,
        policy,
        dynamics_model,
        static_fns,
        offline_buffer,
        model_buffer,
        reward_penalty_coef,
        rollout_length,
        rollout_batch_size,
        batch_size,
        real_ratio,
        pessimism_coef,
        is_sde = False,
        use_diffusion=True,
        env_name="",
        prob_init_obs = 0,
        max_steps_per_env = 1000,
    ):

        self.policy = policy
        self.dynamics_model = dynamics_model
        self.static_fns = static_fns
        self.is_sde = is_sde
        if not self.is_sde:
            self.fake_env = FakeEnv_tatu(
                self.dynamics_model,
                self.static_fns,
                penalty_coeff=reward_penalty_coef,
                penalty_learned_var=True
            )
        else:
            self.fake_env = FakeEnv_SDE_Trunc(
                self.dynamics_model,
                self.static_fns,
                penalty_coeff=reward_penalty_coef,
                penalty_learned_var=True,
                use_diffusion=use_diffusion,
                env_name=env_name,
            )
        self.offline_buffer = offline_buffer
        self.model_buffer = model_buffer
        self._reward_penalty_coef = reward_penalty_coef
        self._rollout_length = rollout_length
        self._rollout_batch_size = rollout_batch_size
        self._batch_size = batch_size
        self._real_ratio = real_ratio
        self._max_disc = None
        self._pessimism_coef = pessimism_coef
        self.prob_init_obs = prob_init_obs
        self.max_steps_per_env = max_steps_per_env


    def compute_max_disc(self):
        all_offline_data = self.offline_buffer.sample_all()

        observations = all_offline_data["observations"]
        actions = all_offline_data["actions"]

        max_disc = -100
        mini_batch = 1000
        slice_num = len(observations)// mini_batch
        alone_num = len(observations)% mini_batch
        for i in range(slice_num):
            obs = observations[i*mini_batch:(i+1)*mini_batch,:]
            act = actions[i*mini_batch:(i+1)*mini_batch,:]
            mini_max_disc = np.max(self.fake_env.compute_disc(obs,act))
            if mini_max_disc > max_disc:
                max_disc = mini_max_disc
        if alone_num != 0:
            obs = observations[slice_num*mini_batch:,:]
            act = actions[slice_num*mini_batch:,:]
            mini_max_disc = np.max(self.fake_env.compute_disc(obs,act))
            if mini_max_disc > max_disc:
                max_disc = mini_max_disc

        return max_disc

    def _sample_initial_transitions(self):
        return self.offline_buffer.sample(self._rollout_batch_size)

    def rollout_transitions(self):
        # Now should we start from the initial observation and rollout the trajectory till the end
        _rollout_length = self._rollout_length
        if self.prob_init_obs > 0:
            arr_to_pick = [self._rollout_length, self.max_steps_per_env]
            _rollout_length = np.random.choice(arr_to_pick, p=[1-self.prob_init_obs, self.prob_init_obs])
            print("####### Rolling out for {} steps #######".format(_rollout_length) )
        
        if _rollout_length == self.max_steps_per_env:
            init_transitions = self.offline_buffer.sample_initial_observations(self._rollout_batch_size)
            observations = init_transitions
        else:
            init_transitions = self._sample_initial_transitions()
            observations = init_transitions["observations"]

        if self._max_disc == None:
            self._max_disc = self.compute_max_disc()
            print('---> max_disc :  ', self._max_disc)

        threshold = 1/self._pessimism_coef *self._max_disc
        # print('max_disc',self._max_disc)
        # print('threshold',threshold)
        # rollout
        cumul_error = np.zeros(len(observations))
        halt_info ={"max_disc":self._max_disc,
                    "threshold":threshold,
                    "halt_num":[],
                    "halt_ratio":[],
                    "stop_ratio":[]}

        reward_stats = defaultdict(list)
        penalty_stats = defaultdict(list)
        error_stats = defaultdict(list)

        # for _ in range(self._rollout_length):
        for _nrolls in range(_rollout_length):
            actions = self.policy.sample_action(observations)
            prev_error = np.copy(cumul_error)
            next_observations, rewards, terminals, next_cumul_error, infos = self.fake_env.step(observations, actions, cumul_error, threshold)
            update_stats_dict(reward_stats, rewards)
            update_stats_dict(penalty_stats, infos["penalty"])
            update_stats_dict(error_stats, next_cumul_error-prev_error)

            self.model_buffer.add_batch(observations, next_observations, actions, rewards, terminals)
        
            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                print("####### Halted at {} steps #######".format(_nrolls))
                break
            cumul_error = next_cumul_error[nonterm_mask]
            observations = next_observations[nonterm_mask]
            stop_ratio = len(observations) / len(next_observations)
            halt_info["halt_num"].append(infos["halt_num"])
            halt_info["halt_ratio"].append(infos["halt_ratio"])
            halt_info["stop_ratio"].append(stop_ratio)
        
        print("Reward Stats:\n", reward_stats)
        print("Penalty Stats:\n", penalty_stats)
        # print("Error Stats:\n", error_stats)
        print("Halt Info:\n", halt_info)
        del reward_stats, penalty_stats, error_stats
        
        return halt_info

    def learn_dynamics(self):
        if self.is_sde:
            return 0.0
        data = self.offline_buffer.sample_all()
        train_inputs, train_outputs = format_samples_for_training(data)
        max_epochs = 1 if self._real_ratio == 1 else None
        loss = self.dynamics_model.train(
            train_inputs,
            train_outputs,
            batch_size=self._batch_size,
            max_epochs=max_epochs,
            holdout_ratio=0.2
        )
        return loss

    def learn_policy(self):
        real_sample_size = int(self._batch_size * self._real_ratio)
        fake_sample_size = self._batch_size - real_sample_size
        real_batch = self.offline_buffer.sample(batch_size=real_sample_size)
        fake_batch = self.model_buffer.sample(batch_size=fake_sample_size)
        data = {
            "observations": np.concatenate([real_batch["observations"], fake_batch["observations"]], axis=0),
            "actions": np.concatenate([real_batch["actions"], fake_batch["actions"]], axis=0),
            "next_observations": np.concatenate([real_batch["next_observations"], fake_batch["next_observations"]], axis=0),
            "terminals": np.concatenate([real_batch["terminals"], fake_batch["terminals"]], axis=0),
            "rewards": np.concatenate([real_batch["rewards"], fake_batch["rewards"]], axis=0)
        }
        loss = self.policy.learn(data)
        return loss
    
    def save_dynamics_model(self, save_path):
        if self.is_sde:
            return
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.dynamics_model.save(save_path, timestep=0)

class TATU_Model_free():
    def __init__(
        self,
        policy,
        static_fns,
        offline_buffer,
        model_buffer,
        batch_size,
        real_ratio,
    ):

        self.policy = policy
        self.static_fns = static_fns
        self.offline_buffer = offline_buffer
        self.model_buffer = model_buffer
        self._batch_size = batch_size
        self._real_ratio = real_ratio


    def learn_policy(self):
        real_sample_size = int(self._batch_size * self._real_ratio)
        fake_sample_size = self._batch_size - real_sample_size
        real_batch = self.offline_buffer.sample(batch_size=real_sample_size)
        fake_batch = self.model_buffer.sample(batch_size=fake_sample_size)
        data = {
            "observations": np.concatenate([real_batch["observations"], fake_batch["observations"]], axis=0),
            "actions": np.concatenate([real_batch["actions"], fake_batch["actions"]], axis=0),
            "next_observations": np.concatenate([real_batch["next_observations"], fake_batch["next_observations"]], axis=0),
            "terminals": np.concatenate([real_batch["terminals"], fake_batch["terminals"]], axis=0),
            "rewards": np.concatenate([real_batch["rewards"], fake_batch["rewards"]], axis=0)
        }
        loss = self.policy.learn(data)
        return loss

class TATU_bcq():
    def __init__(
        self,
        policy,
        static_fns,
        offline_buffer,
        model_buffer,
        batch_size,
        real_ratio,
    ):

        self.policy = policy
        self.static_fns = static_fns
        self.offline_buffer = offline_buffer
        self.model_buffer = model_buffer
        self._batch_size = batch_size
        self._real_ratio = real_ratio


    def learn_policy(self, iterations=1):
        real_sample_size = int(self._batch_size * self._real_ratio)
        fake_sample_size = self._batch_size - real_sample_size
        for it in range(iterations):
            real_batch = self.offline_buffer.sample(batch_size=real_sample_size)
            fake_batch = self.model_buffer.sample(batch_size=fake_sample_size)
            data = {
                "observations": np.concatenate([real_batch["observations"], fake_batch["observations"]], axis=0),
                "actions": np.concatenate([real_batch["actions"], fake_batch["actions"]], axis=0),
                "next_observations": np.concatenate([real_batch["next_observations"], fake_batch["next_observations"]], axis=0),
                "terminals": np.concatenate([real_batch["terminals"], fake_batch["terminals"]], axis=0),
                "rewards": np.concatenate([real_batch["rewards"], fake_batch["rewards"]], axis=0)
            }
            loss = self.policy.learn(data)
        return loss

