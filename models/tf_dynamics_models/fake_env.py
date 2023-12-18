import numpy as np
import tensorflow.compat.v1 as tf
import pdb

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
        log_prob = -1/2 * (k * np.log(2*np.pi) + np.log(variances).sum(-1) + (np.power(x-means, 2)/variances).sum(-1))
        
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
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        ensemble_model_means[:,:,1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        if not deterministic:
            #### choose one model from ensemble
            num_models, batch_size, _ = ensemble_model_means.shape
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

        log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        rewards, next_obs = samples[:,:1], samples[:,1:]
        terminals = self.config.termination_fn(obs, act, next_obs)
        rewards = np.expand_dims(self.config.single_step_reward(obs, act, next_obs), 1)

        batch_size = model_means.shape[0]
        return_means = np.concatenate((model_means[:,:1], terminals, model_means[:,1:]), axis=-1)
        return_stds = np.concatenate((model_stds[:,:1], np.zeros((batch_size,1)), model_stds[:,1:]), axis=-1)

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

        info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev,
                'unpenalized_rewards': unpenalized_rewards, 'penalty': penalty, 'penalized_rewards': penalized_rewards}
        return next_obs, penalized_rewards, terminals, info

    ## for debugging computation graph
    def step_ph(self, obs_ph, act_ph, deterministic=False):
        assert len(obs_ph.shape) == len(act_ph.shape)

        inputs = tf.concat([obs_ph, act_ph], axis=1)
        # inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.create_prediction_tensors(inputs, factored=True)
        # ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        ensemble_model_means = tf.concat([ensemble_model_means[:,:,0:1], ensemble_model_means[:,:,1:] + obs_ph[None]], axis=-1)
        # ensemble_model_means[:,:,1:] += obs_ph
        ensemble_model_stds = tf.sqrt(ensemble_model_vars)
        # ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            # ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds
            ensemble_samples = ensemble_model_means + tf.random.normal(tf.shape(ensemble_model_means)) * ensemble_model_stds

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
        log_prob = -1/2 * (k * np.log(2*np.pi) + np.log(variances).sum(-1) + (np.power(x-means, 2)/variances).sum(-1))
        
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
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        ensemble_model_means[:,:,1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        if not deterministic:
            #### choose one model from ensemble
            num_models, batch_size, _ = ensemble_model_means.shape
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

        log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        rewards, next_obs = samples[:,:1], samples[:,1:]
        terminals = self.config.termination_fn(obs, act, next_obs)

        batch_size = model_means.shape[0]
        return_means = np.concatenate((model_means[:,:1], terminals, model_means[:,1:]), axis=-1)
        return_stds = np.concatenate((model_stds[:,:1], np.zeros((batch_size,1)), model_stds[:,1:]), axis=-1)

        if self.penalty_coeff != 0:
            
            ensemble_means_obs = ensemble_model_means[:,:,1:]
            mean_obs_means = np.mean(ensemble_means_obs, axis=0)     # average predictions over models
            diffs = ensemble_means_obs - mean_obs_means
            normalize_diffs = False
            if normalize_diffs:
                obs_dim = next_obs.shape[1]
                obs_sigma = self.model.scaler.cached_sigma[0,:obs_dim]
                diffs = diffs / obs_sigma
            dists = np.linalg.norm(diffs, axis=2)   # distance in obs space
            
            disc = np.max(dists, axis=0)
            if not self.penalty_learned_var:
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

        ##  compute cumulative error and terminate some trajectories      
        cumul_error += disc
        unknown = np.where(cumul_error > threshold)
        terminals[unknown] = [True]  
        halt_num = len(unknown[0])
        halt_ratio = len(unknown[0])/len(obs)
        # print('halt_num:',halt_num)
        # print('halt_ratio:',halt_ratio)

        info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev,
                'unpenalized_rewards': unpenalized_rewards, 'penalty': penalty, 'penalized_rewards': penalized_rewards,'halt_num':halt_num,'halt_ratio':halt_ratio}
        return next_obs, penalized_rewards, terminals, cumul_error, info

    ## for debugging computation graph
    def step_ph(self, obs_ph, act_ph, deterministic=False):
        assert len(obs_ph.shape) == len(act_ph.shape)

        inputs = tf.concat([obs_ph, act_ph], axis=1)
        # inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.create_prediction_tensors(inputs, factored=True)
        # ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        ensemble_model_means = tf.concat([ensemble_model_means[:,:,0:1], ensemble_model_means[:,:,1:] + obs_ph[None]], axis=-1)
        # ensemble_model_means[:,:,1:] += obs_ph
        ensemble_model_stds = tf.sqrt(ensemble_model_vars)
        # ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            # ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds
            ensemble_samples = ensemble_model_means + tf.random.normal(tf.shape(ensemble_model_means)) * ensemble_model_stds

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
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        ensemble_model_means[:,:,1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        ensemble_means_obs = ensemble_model_means[:,:,1:]
        mean_obs_means = np.mean(ensemble_means_obs, axis=0)     # average predictions over models
        diffs = ensemble_means_obs - mean_obs_means
        # normalize_diffs = False
        # if normalize_diffs:
        #     obs_dim = next_obs.shape[1]
        #     obs_sigma = self.model.scaler.cached_sigma[0,:obs_dim]
        #     diffs = diffs / obs_sigma
        dists = np.linalg.norm(diffs, axis=2)   # distance in obs space
        
        disc = np.max(dists, axis=0)

        return disc


class FakeEnv_SDE_Trunc:
    def __init__(self, model, config,
                penalty_coeff=0.,
                penalty_learned_var=False,
                penalty_learned_var_random=False,
                use_diffusion=True,
                env_name="",
                ):
        self.model = model
        self.config = config
        self.penalty_coeff = penalty_coeff
        self.penalty_learned_var = penalty_learned_var
        self.penalty_learned_var_random = penalty_learned_var_random
        self.use_diffusion = use_diffusion
        self.env_name = env_name
        self.init_model()
    
    def init_model(self,):
        """ Load the SDE and all the required terms
        """
        if self.env_name == "hopper":
            from models.sde_models.hopper_sde import load_predictor_function, load_learned_diffusion
        elif self.env_name == "halfcheetah":
            from models.sde_models.halfcheetah_sde import load_predictor_function, load_learned_diffusion
        # elif self.env_name == "walker2d":
        #     from sde4mbrlExamples.d4rl_mujoco.walker2d_sde import load_predictor_function, load_learned_diffusion
        # elif self.env_name == "ant":
        #     from sde4mbrlExamples.d4rl_mujoco.ant_sde import load_predictor_function, load_learned_diffusion
        else:
            raise NotImplementedError("env_name {} not implemented".format(self.env_name))
        import jax
        import os
        jax_gpu_mem_frac = self.model['jax_gpu_mem_frac']
        model_path = self.model['model_path']
        use_gpu = self.model['use_gpu']
        num_particles = self.model['num_particles']
        tau = self.model['dt']
        init_seed = self.model['seed']
        rollout_batch_size = self.model['rollout_batch_size']
        if use_gpu:
            # This doesn't work as Jax has been already imported and initialized
            # os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(jax_gpu_mem_frac)
            backend = 'gpu'
        else:
            backend = 'cpu'

        self.next_rng = jax.random.PRNGKey(init_seed)
        # Find the file
        model_path = os.path.expanduser(model_path)
        # Load the SDE model from the file
        my_sde = load_predictor_function(model_path, prior_dist=False, nonoise=False,
                                        modified_params ={
                                            'horizon' : 1, 
                                            'num_particles' : num_particles,
                                            'stepsize': tau,
                                            }, 
                                        return_control=False, 
                                        return_time_steps=False)

        # Define the prediction function
        def _my_pred_fn(x, u, _rng):
            """ Return the predicted next state
            """
            assert len(x.shape) == len(u.shape) == 2
            print(x.shape, u.shape)
            next_rng, _rng = jax.random.split(_rng, 2)
            _rng = jax.random.split(_rng, x.shape[0])
            pred_state = jax.vmap(my_sde, in_axes=(0, 0, 0))(x, u, _rng)[:, :, 1, :]
            # transpose the output to be [num_particles, batch_size, obs_dim]
            pred_state = jax.lax.transpose(pred_state, (1, 0, 2))
            # Com
            return pred_state, next_rng
        self._predict = jax.jit(_my_pred_fn, backend=backend)
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

        if self.use_diffusion:
            my_sde_diffusion = load_learned_diffusion(model_path)
            # Define the diffusion function
            def _my_diff_fn(x, u, _rng):
                """ Return the predicted next state
                """
                assert len(x.shape) == len(u.shape) == 2
                # next_rng, _rng = jax.random.split(_rng, 2)
                # _rng = jax.random.split(_rng, x.shape[0])
                diff = jax.vmap(my_sde_diffusion)(x, u)
                return diff, _rng
            self._diffusion = jax.jit(_my_diff_fn, backend=backend)
            def augmented_diff_fn(x, u , rng):
                batch_x = x.shape[0]
                last_x = x[-1]
                last_u = u[-1]
                if batch_x < rollout_batch_size:
                    last_x = np.repeat(last_x[None], rollout_batch_size-batch_x, axis=0)
                    last_u = np.repeat(last_u[None], rollout_batch_size-batch_x, axis=0)
                    x = np.concatenate((x, last_x), axis=0)
                    u = np.concatenate((u, last_u), axis=0)
                diff, next_rng = self._diffusion(x, u, rng)
                if batch_x < rollout_batch_size:
                    diff = diff[:batch_x,:]
                return np.array(diff), next_rng
            self.diffusion = augmented_diff_fn        

    '''
        x : [ batch_size, obs_dim + 1 ]
        means : [ num_models, batch_size, obs_dim + 1 ]
        vars : [ num_models, batch_size, obs_dim + 1 ]
    '''
    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        # log_prob = -1/2 * (k * np.log(2*np.pi) + np.log(variances).sum(-1) + (np.power(x-means, 2)/variances).sum(-1))
        log_prob = -1/2 * (k * np.log(2*np.pi) + (np.power(x-means, 2)/1.0).sum(-1))
        
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

        predicted_particles, self.next_rng = self.predict(obs, act, self.next_rng)

        if not deterministic:
            #### Choose one particle randomly
            num_particles, batch_size, _ = predicted_particles.shape
            particle_inds = np.random.choice(num_particles, size=batch_size)
            batch_inds = np.arange(0, batch_size)
            chosen_particles = predicted_particles[particle_inds, batch_inds]
        else:
            ### Choose the mean of predicted particles
            chosen_particles = np.mean(predicted_particles, axis=0)
        # print("chosen_particles", chosen_particles )
        model_means = np.mean(predicted_particles, axis=0)
        model_stds = np.std(predicted_particles, axis=0)

        log_prob, dev = self._get_logprob(chosen_particles, predicted_particles, None)

        next_obs = chosen_particles
        rewards = np.expand_dims(self.config.single_step_reward(obs, act, next_obs), 1)
        if len(rewards.shape) > 2 :
            rewards_mean = np.mean(rewards, axis=0)
            rewards_std = np.std(rewards, axis=0)
        else:
            rewards_mean = rewards
            rewards_std = np.zeros_like(rewards_mean)
        terminals = self.config.termination_fn(obs, act, next_obs)

        batch_size = model_means.shape[0]
        return_means = np.concatenate((rewards_mean, terminals, model_means), axis=-1)
        return_stds = np.concatenate((rewards_std, np.zeros((batch_size,1)), model_stds), axis=-1)

        if self.penalty_coeff != 0:
            
            diffs = predicted_particles - np.mean(predicted_particles, axis=0)
            dists = np.linalg.norm(diffs, axis=2)   # distance in obs space
            
            disc = np.max(dists, axis=0)
            if not self.use_diffusion:
                if not self.penalty_learned_var:
                    # max discrepancy between particles
                    penalty = np.max(dists, axis=0)      
                else:
                    # norm of std of particles
                    penalty = np.linalg.norm(np.std(predicted_particles, axis=0), axis=-1)
            else:
                # diffusion as penalty
                predicted_diffusion, self.next_rng = self.diffusion(obs, act, self.next_rng)
                predicted_diffusion = np.array(predicted_diffusion)
                penalty = np.sum(predicted_diffusion, axis=-1)
                # penalty = np.sum(predicted_diffusion, axis=-1)/threshold
                # penalty = np.clip(np.sum(predicted_diffusion, axis=-1), a_min=-np.inf, a_max=1.0)
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
        if not self.use_diffusion:
            cumul_error += disc
        else:
            cumul_error += np.sum(predicted_diffusion, axis=-1)
        unknown = np.where(cumul_error > threshold)
        terminals[unknown] = [True]  
        halt_num = len(unknown[0])
        halt_ratio = len(unknown[0])/len(obs)

        info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev,
                'unpenalized_rewards': unpenalized_rewards, 'penalty': penalty, 'penalized_rewards': penalized_rewards,'halt_num':halt_num,'halt_ratio':halt_ratio}
        return next_obs, penalized_rewards, terminals, cumul_error, info

    # ## for debugging computation graph
    # def step_ph(self, obs_ph, act_ph, deterministic=False):
    #     assert len(obs_ph.shape) == len(act_ph.shape)

    #     inputs = tf.concat([obs_ph, act_ph], axis=1)
    #     # inputs = np.concatenate((obs, act), axis=-1)
    #     ensemble_model_means, ensemble_model_vars = self.model.create_prediction_tensors(inputs, factored=True)
    #     # ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
    #     ensemble_model_means = tf.concat([ensemble_model_means[:,:,0:1], ensemble_model_means[:,:,1:] + obs_ph[None]], axis=-1)
    #     # ensemble_model_means[:,:,1:] += obs_ph
    #     ensemble_model_stds = tf.sqrt(ensemble_model_vars)
    #     # ensemble_model_stds = np.sqrt(ensemble_model_vars)

    #     if deterministic:
    #         ensemble_samples = ensemble_model_means
    #     else:
    #         # ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds
    #         ensemble_samples = ensemble_model_means + tf.random.normal(tf.shape(ensemble_model_means)) * ensemble_model_stds

    #     samples = ensemble_samples[0]

    #     rewards, next_obs = samples[:,:1], samples[:,1:]
    #     terminals = self.config.termination_ph_fn(obs_ph, act_ph, next_obs)
    #     info = {}

    #     return next_obs, rewards, terminals, info

    def close(self):
        pass

    def compute_disc(self,obs,act):
        assert len(obs.shape) == len(act.shape)

        # inputs = np.concatenate((obs, act), axis=-1)
        if self.use_diffusion:
            predicted_diffusion, self.next_rng = self.diffusion(obs, act, self.next_rng)
            ensemble_diffs = np.array(predicted_diffusion)
            disc = np.max(np.sum(predicted_diffusion, axis=-1))
        else:
            predicted_particles, self.next_rng = self.predict(obs, act, self.next_rng)
            diffs = predicted_particles - np.mean(predicted_particles, axis=0)
            dists = np.linalg.norm(diffs, axis=2)   # distance in obs space            
            disc = np.max(dists, axis=0)
        return disc