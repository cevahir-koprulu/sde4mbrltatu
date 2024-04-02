import numpy as np
import torch

from typing import Callable, Tuple, Dict


class SDEDynamics:
    def __init__(
        self,
        model_name: str,
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
        task_name: str = "hopper-medium-expert-v2",
        ckpt_step: int = -2
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
        self.task_name = task_name
        self.ckpt_step = ckpt_step
        self.init_model(model_name, rollout_batch_size, seed_init) 
    
    def init_model(
        self,
        model_name: str,
        rollout_batch_size: int,
        seed_init: int
    ) -> None:
        """ Initialize the sde model and define the functions for predicting and so on
        """
        # Set the GPU memory fraction used by JAX
        import os
        jax_gpu_mem_frac = self.model.get('jax_gpu_mem_frac', "-1")
        _jax_gpu_mem_frac = float(jax_gpu_mem_frac)
        if _jax_gpu_mem_frac > 0:
            os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(jax_gpu_mem_frac)

        # Import jax after setting the GPU memory fraction
        import jax
        # Import the function to load the SDE model sampler
        from nsdes_dynamics.load_learned_nsdes import \
            load_system_sampler_from_model_name

        # Load the sampler -> This function also returns the diffusion term
        horizon_pred = 1
        sampler_fn = load_system_sampler_from_model_name(
            self.task_name,
            model_name = model_name,
            stepsize = self.dt,
            horizon = horizon_pred,
            num_particles = self.sde_num_particles,
            step = self.ckpt_step,
            integration_method=None
        )
        sampler_fn_lcb = load_system_sampler_from_model_name(
            self.task_name,
            model_name = model_name,
            stepsize = self.dt,
            horizon = horizon_pred,
            num_particles = self.sde_num_particles * self.num_particles_lcb,
            step = self.ckpt_step,
            integration_method=None
        )

        # Backend for compilation
        use_gpu = self.sde_use_gpu
        init_seed = seed_init
        backend = 'cpu' if not use_gpu else 'gpu'
        # Random number generator to propagate through the calls of predict
        self.next_rng = jax.random.PRNGKey(init_seed)

        # Define the prediction function
        def _my_pred_fn(
            x : jax.numpy.ndarray,
            u : jax.numpy.ndarray,
            rng : jax.random.PRNGKey,
            _sampler_fn = sampler_fn,
        ):
            """
            Return the predicted next state and the diffusion values
            at the current state.
            
            Returns:
                pred_states : jax.numpy.ndarray [num_particles, batch_size, obs_dim]
                The predicted states of the system.
                diffusion_value : jax.numpy.ndarray [num_particles, batch_size, obs_dim]
                The diffusion values at the current state and control.
            """
            # Some checks
            assert len(x.shape) == len(u.shape) == 2

            # Split the rng to match the input dimensions
            next_rng, rng = jax.random.split(rng, 2)
            rng = jax.random.split(rng, x.shape[0])

            # Expand the dimension of u with horizon_pred
            u = jax.numpy.expand_dims(u, axis=1)
            u = jax.numpy.repeat(u, horizon_pred, axis=1)

            # Define the vmap sampler function
            _temp_sampler =  lambda _x, _u, _rng: \
                _sampler_fn(state=_x, control=_u, rng_key=_rng)
            pred_states, _ = \
                jax.vmap(_temp_sampler)(x, u, rng)
            
            # Extract the next state and diffusion from the output trajectory
            pred_states = pred_states[:, :, 1, :]
            # transpose the output to be [num_particles, batch_size, obs_dim]
            pred_states = jax.lax.transpose(pred_states, (1, 0, 2))

            return pred_states, next_rng

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
        self.predict_for_lcb = \
            jax.jit(lambda x, u, rng: \
                _my_pred_fn(x, u, rng, sampler_fn_lcb), backend=backend)

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
