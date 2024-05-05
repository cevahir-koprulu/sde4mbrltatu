import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done =  (height > 0.8) \
                    * (height < 2.0) \
                    * (angle > -1.0) \
                    * (angle < 1.0)
        done = ~not_done
        done = done[:,None]
        return done

    @staticmethod
    def single_step_reward(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        alive_bonus = 1.0
        reward = (obs[...,8] + next_obs[...,8])/2.0 # When the agent is alive, reward is its x-velocity
        reward += alive_bonus # *StaticFns.termination_fn(obs, act, next_obs)
        reward -= 1e-3 * np.square(act).sum(axis=-1)
        return reward

class StaticFnsNeoRL:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        min_z, max_z = (0.8, 2.0)
        min_angle, max_angle = (-1.0, 1.0)
        min_state, max_state = (-100.0, 100.0)
        
        z = next_obs[:, 1:2]
        angle = next_obs[:, 2:3]
        state = next_obs[:, 3:]
        
        healthy_state = np.all(np.logical_and(min_state < state, state < max_state), axis=-1, keepdims=True)
        healthy_z = np.logical_and(min_z < z, z < max_z)
        healthy_angle = np.logical_and(min_angle < angle, angle < max_angle)
        is_healthy = np.logical_and(np.logical_and(healthy_state, healthy_z), healthy_angle)
        done = np.logical_not(is_healthy).reshape(-1, 1)
        return done

    @staticmethod
    def single_step_reward(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        action = act
        obs_next = next_obs

        singel_reward = False
        if len(obs.shape) == 1:
            singel_reward = True
            obs = obs.reshape(1, -1)
        if len(action.shape) == 1:
            action = action.reshape(1, -1)
        if len(obs_next.shape) == 1:
            obs_next = obs_next.reshape(1, -1)

        timestep = 0.002
        frame_skip = 4
        dt = timestep * frame_skip
        x_velocity = (obs_next[:, 0] - obs[:, 0]) / dt
        forward_reward = x_velocity

        #healthy_z = [0.8 < i < 2.0 for i in obs[:,1]]
        #healthy_angle = [-1 < i < 1 for i in obs[:,2]]
        #is_healthy = healthy_z and healthy_angle
        healthy_reward = 1 #is_healthy

        rewards = forward_reward + healthy_reward
        costs = 1e-3 * (action ** 2).sum(axis=-1)
        reward = rewards - costs

        if singel_reward:
            reward = reward[0].item()
        else:
            reward = reward.reshape(-1, 1)
        return reward