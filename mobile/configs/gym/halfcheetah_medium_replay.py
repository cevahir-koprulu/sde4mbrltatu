from copy import deepcopy
from mobile.configs.gym.default import default_args


halfcheetah_medium_replay_args = deepcopy(default_args)
halfcheetah_medium_replay_args["rollout_length"] = 5
halfcheetah_medium_replay_args["penalty_coef"] = 0.5

# halfcheetah_medium_replay_args["penalty_coef"] = 2
# halfcheetah_medium_replay_args["rollout_length"] = 5
halfcheetah_medium_replay_args["dt"] = 0.05
halfcheetah_medium_replay_args['sde_num_particles'] = 5
halfcheetah_medium_replay_args['sde_mean_sample'] = False
halfcheetah_medium_replay_args['sde_model_name'] = 'medium_replay_hc_vf4_hr-10_dt-0.010_sde.pkl'