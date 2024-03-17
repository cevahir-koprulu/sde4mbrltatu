from copy import deepcopy
from mobile.configs.gym.default import default_args


hopper_medium_args = deepcopy(default_args)
hopper_medium_args["rollout_length"] = 5
hopper_medium_args["penalty_coef"] = 1.5
hopper_medium_args["auto_alpha"] = False


hopper_medium_args["penalty_coef"] = 5.0
hopper_medium_args["rollout_length"] = 15
hopper_medium_args["dt"] = 0.008
hopper_medium_args['sde_num_particles'] = 5
hopper_medium_args['sde_mean_sample'] = False
hopper_medium_args['sde_model_name'] = 'medium_hop_v13_hr-20_dt-0.002_sde.pkl'