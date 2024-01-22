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