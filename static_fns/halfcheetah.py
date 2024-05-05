import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        done = np.array([False]).repeat(len(obs))
        done = done[:,None]
        return done

    @staticmethod
    def recompute_reward_fn(obs, act, next_obs, rew):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        new_rew = -(rew + 0.1 * np.sum(np.square(act))) - 0.1 * np.sum(np.square(act))
        return new_rew

    @staticmethod
    def single_step_reward(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        reward_ctrl = -0.1 * np.square(act).sum(axis=-1)
        reward_run = (obs[...,8] + next_obs[...,8])/2.0 # When the agent is alive, reward is its x-velocity
        reward = reward_ctrl + reward_run
        return reward

class StaticFnsNeoRL:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        done = np.array([False]).repeat(len(obs))
        done = done[:,None]
        return done

    @staticmethod
    def single_step_reward(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
    
        action = act
        obs_next = next_obs
        singel_reward = False
        if len(obs.shape) == 1:
            obs = obs.reshape(1,-1)
            singel_reward = True
        if len(action.shape) == 1:
            action = action.reshape(1,-1)
        if len(obs_next.shape) == 1:
            obs_next = obs_next.reshape(1,-1)
        
        
        forward_reward_weight = 1.0 
        ctrl_cost_weight = 0.1
        dt = 0.05
        array_type = np
        ctrl_cost = ctrl_cost_weight * array_type.sum(array_type.square(action),axis=1)
        
        x_position_before = obs[:,0]
        x_position_after = obs_next[:,0]
        x_velocity = ((x_position_after - x_position_before) / dt)

        forward_reward = forward_reward_weight * x_velocity
        
        reward = forward_reward - ctrl_cost
        
        if singel_reward:
            reward = reward[0].item()
        else:
            reward = reward.reshape(-1,1)
        return reward