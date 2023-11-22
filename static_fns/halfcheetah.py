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