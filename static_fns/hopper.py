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
        return done
    
    @staticmethod
    def single_step_reward(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        alive_bonus = 1.0
        # reward = obs[...,5]
        reward = (obs[...,5] + next_obs[...,5])/2.0 # When the agent is alive, reward is its x-velocity
        reward += alive_bonus*StaticFns.termination_fn(obs, act, next_obs)
        reward -= 1e-3 * np.square(act).sum(axis=-1)
        return reward
