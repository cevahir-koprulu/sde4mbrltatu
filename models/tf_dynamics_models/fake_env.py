import numpy as np
import tensorflow.compat.v1 as tf


class FakeEnv:
    def __init__(self, model, config,
                penalty_coeff=0.,
                penalty_learned_var=False,
                penalty_learned_var_random=False):
        self.model = model
        self.config = config
        self.penalty_coeff = penalty_coeff
        self.penalty_learned_var = penalty_learned_var
        self.penalty_learned_var_random = penalty_learned_var_random

    '''
        x : [ batch_size, obs_dim + 1 ]
        means : [ num_models, batch_size, obs_dim + 1 ]
        vars : [ num_models, batch_size, obs_dim + 1 ]
    '''
    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1/2 * (k * np.log(2*np.pi) + np.log(variances).sum(-1) + \
            (np.power(x-means, 2)/variances).sum(-1))
        
        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means,0).mean(-1)

        return log_prob, stds

    def step(self, obs, act, deterministic=False):
        assert len(obs.shape) == len(act.shape)
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = \
            self.model.predict(inputs, factored=True)
        ensemble_model_means[:,:,1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + \
                np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        if not deterministic:
            #### choose one model from ensemble
            _, batch_size, _ = ensemble_model_means.shape
            model_inds = self.model.random_inds(batch_size)
            batch_inds = np.arange(0, batch_size)
            samples = ensemble_samples[model_inds, batch_inds]
            model_means = ensemble_model_means[model_inds, batch_inds]
            model_stds = ensemble_model_stds[model_inds, batch_inds]
            ####
        else:
            samples = np.mean(ensemble_samples, axis=0)
            model_means = np.mean(ensemble_model_means, axis=0)
            model_stds = np.mean(ensemble_model_stds, axis=0)

        log_prob, dev = self._get_logprob(
            samples, ensemble_model_means, ensemble_model_vars
        )

        rewards, next_obs = samples[:,:1], samples[:,1:]
        terminals = self.config.termination_fn(obs, act, next_obs)
        rewards = np.expand_dims(
            self.config.single_step_reward(obs, act, next_obs), 1
        )

        batch_size = model_means.shape[0]
        return_means = np.concatenate(
            (model_means[:,:1], terminals, model_means[:,1:]),
            axis=-1
        )
        return_stds = np.concatenate(
            (model_stds[:,:1], np.zeros((batch_size,1)), model_stds[:,1:]),
            axis=-1
        )

        if self.penalty_coeff != 0:
            if not self.penalty_learned_var:
                ensemble_means_obs = ensemble_model_means[:,:,1:]
                diffs = ensemble_means_obs - np.mean(ensemble_means_obs, axis=0)
                normalize_diffs = False
                if normalize_diffs:
                    obs_dim = next_obs.shape[1]
                    obs_sigma = self.model.scaler.cached_sigma[0,:obs_dim]
                    diffs = diffs / obs_sigma
                dists = np.linalg.norm(diffs, axis=2)   # distance in obs space
                penalty = np.max(dists, axis=0)         # max distances over models
            else:
                penalty = np.amax(np.linalg.norm(ensemble_model_stds, axis=2), axis=0)

            penalty = np.expand_dims(penalty, 1)
            assert penalty.shape == rewards.shape
            unpenalized_rewards = rewards
            penalized_rewards = rewards - self.penalty_coeff * penalty
        else:
            penalty = None
            unpenalized_rewards = rewards
            penalized_rewards = rewards

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            unpenalized_rewards = unpenalized_rewards[0]
            penalized_rewards = penalized_rewards[0]
            terminals = terminals[0]

        info = {'mean': return_means, 'std': return_stds,
                'log_prob': log_prob, 'dev': dev,
                'unpenalized_rewards': unpenalized_rewards,
                'penalty': penalty, 'penalized_rewards': penalized_rewards
        }
        return next_obs, penalized_rewards, terminals, info

    ## for debugging computation graph
    def step_ph(self, obs_ph, act_ph, deterministic=False):
        assert len(obs_ph.shape) == len(act_ph.shape)

        inputs = tf.concat([obs_ph, act_ph], axis=1)
        # inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = \
            self.model.create_prediction_tensors(inputs, factored=True)
        # ensemble_model_means, ensemble_model_vars = \
        #   self.model.predict(inputs, factored=True)
        ensemble_model_means = \
            tf.concat(
                [ensemble_model_means[:,:,0:1],
                 ensemble_model_means[:,:,1:] + obs_ph[None]
                ],
                axis=-1
            )
        # ensemble_model_means[:,:,1:] += obs_ph
        ensemble_model_stds = tf.sqrt(ensemble_model_vars)
        # ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            # ensemble_samples = ensemble_model_means +\
            #   np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds
            ensemble_samples = ensemble_model_means + \
                tf.random.normal(tf.shape(ensemble_model_means)) * ensemble_model_stds

        samples = ensemble_samples[0]

        rewards, next_obs = samples[:,:1], samples[:,1:]
        terminals = self.config.termination_ph_fn(obs_ph, act_ph, next_obs)
        info = {}

        return next_obs, rewards, terminals, info

    def close(self):
        pass



class FakeEnv_tatu:
    def __init__(self, model, config,
                penalty_coeff=0.,
                penalty_learned_var=False,
                penalty_learned_var_random=False):
        self.model = model
        self.config = config
        self.penalty_coeff = penalty_coeff
        self.penalty_learned_var = penalty_learned_var
        self.penalty_learned_var_random = penalty_learned_var_random

    '''
        x : [ batch_size, obs_dim + 1 ]
        means : [ num_models, batch_size, obs_dim + 1 ]
        vars : [ num_models, batch_size, obs_dim + 1 ]
    '''
    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1/2 * (k * np.log(2*np.pi) + np.log(variances).sum(-1) + \
            (np.power(x-means, 2)/variances).sum(-1))
        
        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means,0).mean(-1)

        return log_prob, stds

    def step(self, obs, act, cumul_error, threshold, deterministic=False):
        assert len(obs.shape) == len(act.shape)
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = \
            self.model.predict(inputs, factored=True)
        ensemble_model_means[:,:,1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + \
                np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        if not deterministic:
            #### choose one model from ensemble
            _, batch_size, _ = ensemble_model_means.shape
            model_inds = self.model.random_inds(batch_size)
            batch_inds = np.arange(0, batch_size)
            samples = ensemble_samples[model_inds, batch_inds]
            model_means = ensemble_model_means[model_inds, batch_inds]
            model_stds = ensemble_model_stds[model_inds, batch_inds]
            ####
        else:
            samples = np.mean(ensemble_samples, axis=0)
            model_means = np.mean(ensemble_model_means, axis=0)
            model_stds = np.mean(ensemble_model_stds, axis=0)

        log_prob, dev = self._get_logprob(
            samples, ensemble_model_means, ensemble_model_vars
        )

        rewards, next_obs = samples[:,:1], samples[:,1:]
        # rewards = self.config.single_step_reward(obs, act, next_obs).reshape(-1,1)
        terminals = self.config.termination_fn(obs, act, next_obs)

        batch_size = model_means.shape[0]
        return_means = np.concatenate(
            (model_means[:,:1], terminals, model_means[:,1:]), axis=-1
        )
        return_stds = np.concatenate(
            (model_stds[:,:1], np.zeros((batch_size,1)), model_stds[:,1:]),
            axis=-1
        )

        if self.penalty_coeff != 0:
            
            ensemble_means_obs = ensemble_model_means[:,:,1:]
            # average predictions over models
            mean_obs_means = np.mean(ensemble_means_obs, axis=0)
            diffs = ensemble_means_obs - mean_obs_means
            normalize_diffs = False
            if normalize_diffs:
                obs_dim = next_obs.shape[1]
                obs_sigma = self.model.scaler.cached_sigma[0,:obs_dim]
                diffs = diffs / obs_sigma
            dists = np.linalg.norm(diffs, axis=2)   # distance in obs space
            
            disc = np.max(dists, axis=0)
            if not self.penalty_learned_var:
                # max distances over models
                penalty = np.max(dists, axis=0)
            else:
                penalty = np.amax(
                    np.linalg.norm(ensemble_model_stds, axis=2), axis=0
                )

            penalty = np.expand_dims(penalty, 1)
            assert penalty.shape == rewards.shape
            unpenalized_rewards = rewards
            penalized_rewards = rewards - self.penalty_coeff * penalty
        else:
            penalty = None
            unpenalized_rewards = rewards
            penalized_rewards = rewards

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            unpenalized_rewards = unpenalized_rewards[0]
            penalized_rewards = penalized_rewards[0]
            terminals = terminals[0]

        ##  compute cumulative error and terminate some trajectories      
        cumul_error += disc
        unknown = np.where(cumul_error > threshold)
        terminals[unknown] = [True]  
        halt_num = len(unknown[0])
        halt_ratio = len(unknown[0])/len(obs)
        # print('halt_num:',halt_num)
        # print('halt_ratio:',halt_ratio)

        info = {'mean': return_means, 'std': return_stds,
                'log_prob': log_prob, 'dev': dev,
                'unpenalized_rewards': unpenalized_rewards,
                'penalty': penalty, 'penalized_rewards': penalized_rewards,
                'halt_num':halt_num,'halt_ratio':halt_ratio
            }
        return next_obs, penalized_rewards, terminals, cumul_error, info

    ## for debugging computation graph
    def step_ph(self, obs_ph, act_ph, deterministic=False):
        assert len(obs_ph.shape) == len(act_ph.shape)

        inputs = tf.concat([obs_ph, act_ph], axis=1)
        # inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = \
            self.model.create_prediction_tensors(inputs, factored=True)
        # ensemble_model_means, ensemble_model_vars = \
        #   self.model.predict(inputs, factored=True)
        ensemble_model_means = tf.concat(
            [ensemble_model_means[:,:,0:1], 
             ensemble_model_means[:,:,1:] + obs_ph[None]
            ], axis=-1
        )
        # ensemble_model_means[:,:,1:] += obs_ph
        ensemble_model_stds = tf.sqrt(ensemble_model_vars)
        # ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            # ensemble_samples = ensemble_model_means + \
            #       np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds
            ensemble_samples = ensemble_model_means + \
                tf.random.normal(tf.shape(ensemble_model_means)) * ensemble_model_stds

        samples = ensemble_samples[0]

        rewards, next_obs = samples[:,:1], samples[:,1:]
        terminals = self.config.termination_ph_fn(obs_ph, act_ph, next_obs)
        info = {}

        return next_obs, rewards, terminals, info

    def close(self):
        pass

    def compute_disc(self,obs,act):
        assert len(obs.shape) == len(act.shape)

        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = \
            self.model.predict(inputs, factored=True)
        ensemble_model_means[:,:,1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        ensemble_means_obs = ensemble_model_means[:,:,1:]
        # average predictions over models
        mean_obs_means = np.mean(ensemble_means_obs, axis=0)
        diffs = ensemble_means_obs - mean_obs_means
        # normalize_diffs = False
        # if normalize_diffs:
        #     obs_dim = next_obs.shape[1]
        #     obs_sigma = self.model.scaler.cached_sigma[0,:obs_dim]
        #     diffs = diffs / obs_sigma
        
        # distance in obs space
        dists = np.linalg.norm(diffs, axis=2)
        disc = np.max(dists, axis=0)
        return disc


class FakeEnv_SDE_Trunc:
    """ Fake environment for SDE models with truncation.
    """
    def __init__(
        self,
        model,
        config,
        penalty_coeff=0.,
        penalty_learned_var=False,
        penalty_learned_var_random=False,
        use_diffusion=False
    ):
        self.model = model
        self.config = config
        self.penalty_coeff = penalty_coeff
        self.penalty_learned_var = penalty_learned_var
        self.penalty_learned_var_random = penalty_learned_var_random
        self.use_diffusion = use_diffusion
        self.env_name = self.model["env_name"]
        self.init_model()
    
    def init_model(self,):
        """
        Load the SDE model and define the prediction function.
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
        sampler_fn, self.sde_model = load_system_sampler_from_model_name(
            self.env_name,
            model_name = self.model['model_name'],
            stepsize = self.model['stepsize'],
            horizon = horizon_pred,
            num_particles = self.model['num_particles'],
            step = self.model.get('ckpt_step', -2),
            integration_method=None,
            return_sde_model= True
        )

        # Backend for compilation
        use_gpu = self.model.get('use_gpu', False)
        init_seed = self.model['seed']
        rollout_batch_size = self.model['rollout_batch_size']
        backend = 'cpu' if not use_gpu else 'gpu'
        
        print('backend:',backend)
        print('rollout_batch_size:',rollout_batch_size)
        print('init_seed:',init_seed)
        print('use_gpu:',use_gpu)
        print('model_name:',self.model['model_name'])
        print('stepsize:',self.model['stepsize'])
        print('horizon:',1)
        print('num_particles:',self.model['num_particles'])
        print('ckpt_step:',self.model.get('ckpt_step', -2))
        print('integration_method:',None)

        # Random number generator to propagate through the calls of predict
        self.next_rng = jax.random.PRNGKey(init_seed)

        # Define the prediction function
        def _my_pred_fn(
            x : jax.numpy.ndarray,
            u : jax.numpy.ndarray,
            rng : jax.random.PRNGKey
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
                sampler_fn(state=_x, control=_u, rng_key=_rng)
            pred_states, pred_feats = \
                jax.vmap(_temp_sampler)(x, u, rng)

            # Get the diffusion term
            name_diff = 'diffusion_value' # diffusion_value
            if name_diff not in pred_feats:
                pred_feats[name_diff] = jax.numpy.zeros_like(pred_states)
            diffusion_value = pred_feats[name_diff]

            # Extract the next state and diffusion from the output trajectory
            pred_states = pred_states[:, :, 1, :]
            diffusion_value = jax.numpy.abs(diffusion_value[:, :, -1, :])

            # transpose the output to be [num_particles, batch_size, obs_dim]
            pred_states = jax.lax.transpose(pred_states, (1, 0, 2))
            diffusion_value = jax.lax.transpose(diffusion_value, (1, 0, 2))

            return pred_states, diffusion_value, next_rng

        # Jit the prediction function
        self._predict = jax.jit(_my_pred_fn, backend=backend)

        def augmented_pred_fn(x, u , rng):
            """ Wrapper around prediction function to varying 
            batch sizes.
            """
            batch_x = x.shape[0]
            last_x = x[-1]
            last_u = u[-1]
            # Complte x, u to have a batch size of rollout_batch_size
            # so that to avoid recompilation
            if batch_x < rollout_batch_size:
                last_x = np.repeat(last_x[None], rollout_batch_size-batch_x, axis=0)
                last_u = np.repeat(last_u[None], rollout_batch_size-batch_x, axis=0)
                x = np.concatenate((x, last_x), axis=0)
                u = np.concatenate((u, last_u), axis=0)
            # Do the prediction
            pred_state, diff_value, next_rng = self._predict(x, u, rng)
            pred_state = np.array(pred_state)
            diff_value = np.array(diff_value)
            # Trim the output to the original batch size
            if batch_x < rollout_batch_size:
                pred_state = pred_state[:,:batch_x,:]
                diff_value = diff_value[:,:batch_x,:]
            
            return pred_state, diff_value, next_rng

        self.predict = augmented_pred_fn   

    def _get_logprob(self, x, means, variances):
        """ This function is not used but defined for consistency.
        """
        k = x.shape[-1]
        ## [ num_networks, batch_size ]
        log_prob = -1/2 * (k * np.log(2*np.pi) + \
            (np.power(x-means, 2)/1.0).sum(-1))
        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)
        ## [ batch_size ]
        log_prob = np.log(prob)
        stds = np.std(means,0).mean(-1)
        return log_prob, stds

    def step(self, obs, act, cumul_error, threshold, deterministic=False):
        """
        Perform the step function given a batch of observations and actions.
        """
        assert len(obs.shape) == len(act.shape)
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        # Predict the next state and diffusion values
        # and store the next random number generator
        predicted_particles, predicted_diffusion, self.next_rng = \
            self.predict(obs, act, self.next_rng)

        if not deterministic:
            #### Choose one particle randomly
            num_particles, batch_size, _ = predicted_particles.shape
            particle_inds = np.random.choice(num_particles, size=batch_size)
            batch_inds = np.arange(0, batch_size)
            chosen_particles = predicted_particles[particle_inds, batch_inds]
        else:
            ### Choose the mean of predicted particles
            chosen_particles = np.mean(predicted_particles, axis=0)

        # Mean and standard deviation of the predicted particles
        model_means = np.mean(predicted_particles, axis=0)
        model_stds = np.std(predicted_particles, axis=0)

        # Log probability of the chosen particles
        log_prob, dev = self._get_logprob(
            chosen_particles, predicted_particles, None
        )

        # Define the next observation and compute the rewards
        next_obs = chosen_particles
        rewards = self.config.single_step_reward(obs, act, next_obs).reshape(-1,1)
        # Sanity check of the shape of rewards
        # it should always be [batch_size, 1]
        if len(rewards.shape) > 2 :
            rewards_mean = np.mean(rewards, axis=0)
            rewards_std = np.std(rewards, axis=0)
        else:
            rewards_mean = rewards
            rewards_std = np.zeros_like(rewards_mean)

        # Check if the episode should be terminated
        terminals = self.config.termination_fn(obs, act, next_obs)

        # STore the return means and stds
        batch_size = model_means.shape[0]
        return_means = \
            np.concatenate((rewards_mean, terminals, model_means), axis=-1)
        return_stds = \
            np.concatenate((rewards_std, np.zeros((batch_size,1)), model_stds), axis=-1)

        # MOPO style reward-based penalty of the uncertainty
        if self.penalty_coeff != 0:
            if not self.use_diffusion: # No use of diffusion
                diffs = predicted_particles - np.mean(predicted_particles, axis=0)
                # distance in obs space
                dists = np.linalg.norm(diffs, axis=2)
                disc = np.max(dists, axis=0)
                if not self.penalty_learned_var:
                    # max discrepancy between particles
                    penalty = np.max(dists, axis=0)
                else:
                    # norm of std of particles -> correlation to distance above
                    penalty = np.linalg.norm(
                        np.std(predicted_particles, axis=0),
                        axis=-1
                    )
            else: # We use the diffusion as penalty
                # diffusion as penalty
                penalty = np.linalg.norm(predicted_diffusion, axis=-1)
                # penalty = np.max(penalty, axis=0)
                penalty = np.mean(penalty, axis=0)
                disc_sde = penalty

            # Make the penalty a column vector
            penalty = np.expand_dims(penalty, 1)
            assert penalty.shape == rewards.shape
            unpenalized_rewards = rewards

            # Enforce the penalty on the rewards
            penalized_rewards = rewards - self.penalty_coeff * penalty
        else:
            penalty = 0
            unpenalized_rewards = rewards
            penalized_rewards = rewards
            disc = 0
            disc_sde = 0

        if return_single: # IN case no batch
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            unpenalized_rewards = unpenalized_rewards[0]
            penalized_rewards = penalized_rewards[0]
            terminals = terminals[0]

        ##  compute cumulative error and terminate some trajectories      
        if not self.use_diffusion:
            cumul_error += disc
        else:
            cumul_error += disc_sde

        # Check if the cumulative error is above the threshold
        unknown = np.where(cumul_error > threshold)
        terminals[unknown] = [True]  
        halt_num = len(unknown[0])
        halt_ratio = len(unknown[0])/len(obs)

        info = {'mean': return_means, 'std': return_stds,
                'log_prob': log_prob, 'dev': dev,
                'unpenalized_rewards': unpenalized_rewards,
                'penalty': penalty, 'penalized_rewards': penalized_rewards,
                'halt_num':halt_num,'halt_ratio':halt_ratio
        }
        return next_obs, penalized_rewards, terminals, cumul_error, info

    def close(self):
        pass

    def compute_disc(self,obs,act):
        """
        Compute the maximum discrepancy given a batch of obs and act.
        The discrepancy is computed as the maximum distance between
        particles in the predicted trajectory or the maximum diffusion
        value.
        """
        assert len(obs.shape) == len(act.shape)
        assert obs.ndim > 1 and act.ndim > 1

        # Predict the next state and diffusion values
        predicted_particles, predicted_diffusion, self.next_rng = \
            self._predict(obs, act, self.next_rng)

        # Convert the predictions to numpy arrays
        predicted_particles = np.array(predicted_particles)
        predicted_diffusion = np.array(predicted_diffusion)
        if self.use_diffusion:
            disc = np.mean(np.linalg.norm(predicted_diffusion, axis=-1), axis=0)
            # Let's multiply by the horizon so that the cut threshold
            # can depend on accumulated error
            disc = disc * self.model['rollout_length']
        else:
            # Let's multiply by the horizon so that the cut threshold
            # can depend on accumulated error
            diffs = predicted_particles - np.mean(predicted_particles, axis=0)
            dists = np.linalg.norm(diffs, axis=2) * \
                self.model['rollout_length']
            disc = np.max(dists, axis=0)
            # disc = np.ones_like(disc) * 34

        return disc
