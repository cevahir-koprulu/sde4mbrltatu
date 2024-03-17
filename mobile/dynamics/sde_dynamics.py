import numpy as np
import torch
# import torch.nn as nn

from typing import Callable, List, Tuple, Dict, Optional
from functools import partial


class SDEDynamics:
    def __init__(
        self,
        model_path: str,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        reward_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        dt: float = 0.008,
        sde_use_gpu: bool = True,
        sde_jax_gpu_mem_frac: float = 0.2,
        sde_num_particles: int = 10,
        num_particles_lcb: int = 10,
        sde_mean_sample: bool = False,
        rollout_batch_size: int = 50000,
        seed_init: int = 10,
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric", # Or diffusion
        model_name: str = "hopper",
    ) -> None:
        self.terminal_fn = terminal_fn
        self.reward_fn = reward_fn
        self.sde_use_gpu = sde_use_gpu
        self.sde_jax_gpu_mem_frac = sde_jax_gpu_mem_frac
        self.sde_num_particles = sde_num_particles
        self.num_particles_lcb = num_particles_lcb
        self.sde_mean_sample = sde_mean_sample
        self.dt = dt
        self._penalty_coef = penalty_coef
        self._uncertainty_mode = uncertainty_mode
        self.model_name = model_name
        self.init_model(model_path, rollout_batch_size, seed_init) 
    
    def init_model(self, model_path: str, rollout_batch_size: int, seed_init: int) -> None:
        """ Initialize the sde model and define the functions for predicting and so on
        """
        if "hopper" in self.model_name:
            from models.sde_models.hopper_sde import load_predictor_function
        elif "halfcheetah" in self.model_name:
            from models.sde_models.halfcheetah_sde import load_predictor_function
        elif "walker" in self.model_name:
            from models.sde_models.walker_sde import load_predictor_function
        else:
            raise NotImplementedError("env_name {} not implemented".format(self.env_name))
        
        import jax
        import os

        if self.sde_use_gpu:
            os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(self.sde_jax_gpu_mem_frac)
            backend = 'gpu'
        else:
            backend = 'cpu'

        # Random number generator
        self.next_rng = jax.random.PRNGKey(seed_init)

        # Find the file
        model_path = os.path.expanduser(model_path)

        # Load the SDE model from the file
        my_step_sde = load_predictor_function(model_path, prior_dist=False, nonoise=False,
                                        modified_params ={
                                            'horizon' : 1, 
                                            'num_particles' : self.sde_num_particles,
                                            'stepsize': self.dt,
                                            }, 
                                        return_control=False, 
                                        return_time_steps=False)
        
        step_lcb_compute = load_predictor_function(model_path, prior_dist=False, nonoise=False,
                                        modified_params ={
                                            'horizon' : 1, 
                                            'num_particles' : self.sde_num_particles * self.num_particles_lcb,
                                            'stepsize': self.dt,
                                            }, 
                                        return_control=False, 
                                        return_time_steps=False)

        def _my_pred_fn(x, u, _rng, __step_fn=my_step_sde):
            """ Return the predicted next state
            """
            assert len(x.shape) == len(u.shape) == 2
            print(x.shape, u.shape)
            next_rng, _rng = jax.random.split(_rng, 2)
            _rng = jax.random.split(_rng, x.shape[0])
            pred_state = jax.vmap(__step_fn, in_axes=(0, 0, 0))(x, u, _rng)[:, :, 1, :]
            # transpose the output to be [num_particles, batch_size, obs_dim]
            pred_state = jax.lax.transpose(pred_state, (1, 0, 2))
            return pred_state, next_rng
        self._predict = jax.jit(_my_pred_fn, backend=backend)

        # AUgmenting the predict function to handle the case where the batch size is less than rollout_batch_size
        def augmented_pred_fn(x, u , rng):
            batch_x = x.shape[0]
            last_x = x[-1]
            last_u = u[-1]
            if batch_x < rollout_batch_size:
                last_x = np.repeat(last_x[None], rollout_batch_size-batch_x, axis=0)
                last_u = np.repeat(last_u[None], rollout_batch_size-batch_x, axis=0)
                x = np.concatenate((x, last_x), axis=0)
                u = np.concatenate((u, last_u), axis=0)
            pred_state, next_rng = self._predict(x, u, rng)
            if batch_x < rollout_batch_size:
                pred_state = pred_state[:,:batch_x,:]
            return np.array(pred_state), next_rng
        
        # self.predict = jax.jit(_my_pred_fn, backend=backend)
        self.predict = augmented_pred_fn
        self.predict_for_lcb = jax.jit(lambda x, u, rng: _my_pred_fn(x, u, rng, step_lcb_compute), backend=backend)


    @ torch.no_grad()
    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        "imagine single forward step"
        # obs: [batch_size, obs_dim]
        # action: [batch_size, action_dim]
        ensemble_samples,  self.next_rng = self.predict(obs, action, self.next_rng)

        # Pick a single model from the ensemble
        if self.sde_mean_sample:
            samples = np.mean(ensemble_samples, axis=0)
        else:
            num_models, batch_size, _ = ensemble_samples.shape
            model_inds = np.random.choice(num_models, size=batch_size)
            batch_inds = np.arange(0, batch_size)
            samples = ensemble_samples[model_inds, batch_inds]

        # Compute the std
        std = np.std(ensemble_samples, axis=0)

        # Compute the reward
        next_obs = samples
        reward = self.reward_fn(obs, action, next_obs)
        reward = np.expand_dims(reward, 1)
        terminal = self.terminal_fn(obs, action, next_obs)

        info = {}
        info["raw_reward"] = reward

        if self._penalty_coef:
            if self._uncertainty_mode == "aleatoric":
                penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
            else:
                raise NotImplementedError
            penalty = np.expand_dims(penalty, 1).astype(np.float32)
            assert penalty.shape == reward.shape
            reward = reward - self._penalty_coef * penalty
            info["penalty"] = penalty
        
        return next_obs, reward, terminal, info
    
    @ torch.no_grad()
    def predict_next_obs(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        # First convert the obs and action to numpy for jax computation
        _obs_device = obs.device
        obs = obs.cpu().numpy()
        action = action.cpu().numpy()
        # Now predict
        ensemble_samples, self.next_rng = self.predict_for_lcb(obs, action, self.next_rng)
        ensemble_samples = np.array(ensemble_samples[None])
        assert ensemble_samples.ndim == 4
        # Now convert back to torch
        ensemble_samples = torch.from_numpy(ensemble_samples).to(_obs_device)
        return ensemble_samples
