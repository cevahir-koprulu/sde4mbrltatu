import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done =  np.isfinite(next_obs).all(axis=-1) \
                    * np.abs(next_obs[:,1:] < 100).all(axis=-1) \
                    * (height > .7) \
                    * (np.abs(angle) < .2)

        done = ~not_done
        done = done[:,None]
        # print(done.shape)
        return done
    
    @staticmethod
    def single_step_reward(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        alive_bonus = 1.0
        # reward = obs[...,5]
        reward = (obs[...,5] + next_obs[...,5])/2.0 # When the agent is alive, reward is its x-velocity
        # print('Reward shape', reward.shape)
        reward += alive_bonus # * StaticFns.termination_fn(obs, act, next_obs).ravel()
        reward -= 1e-3 * np.square(act).sum(axis=-1)
        return reward

class StaticFnsNeoRL:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        z = next_obs[:, 1:2]
        angle = next_obs[:, 2:3]
        state = next_obs[:, 3:]

        min_state, max_state = (-100.0, 100.0)
        min_z, max_z = (0.7, float('inf'))
        min_angle, max_angle = (-0.2, 0.2)

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state), axis=-1, keepdims=True)
        healthy_z = np.logical_and(min_z < z, z < max_z)
        healthy_angle = np.logical_and(min_angle < angle, angle < max_angle)

        is_healthy = np.logical_and(np.logical_and(healthy_state, healthy_z), healthy_angle)

        done = np.logical_not(is_healthy).reshape(-1, 1)
        return done
    
    @staticmethod
    def single_step_reward(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        
        o = obs
        a = act
        o_ = next_obs

        x_position_before = o[:, 0]
        x_position_after = o_[:, 0]
        dt = 0.008
        _forward_reward_weight = 1.0
        x_velocity = (x_position_after - x_position_before) / dt

        forward_reward = _forward_reward_weight * x_velocity
        healthy_reward = 1.0

        rewards = forward_reward + healthy_reward
        costs = (a ** 2).sum(axis=-1)

        reward = rewards - 1e-3 * costs

        # (batch_size, )
        return reward