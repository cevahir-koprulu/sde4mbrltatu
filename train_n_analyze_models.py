import os
import numpy as np

import copy
import pickle

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt


def train_model(cfg_name):
    """ Train the neural SDE model of the Keisuke car
        The configuration file must be located inside keisuke/ and should be a yaml file.
        The configuration file contain entry for model_output name, dataset and so on. Refer to the keisuke/model_n_optimizer_cfg.yaml for an example.
        The directory is made such that the models and dataset are all stored inside the keisuke/ directory.
    """
    if 'halfcheetah' in cfg_name:
        from models.sde_models.halfcheetah_sde import train_sde
        train_sde(cfg_name)
    elif 'hopper' in cfg_name:
        from models.sde_models.hopper_sde import train_sde
        train_sde(cfg_name)
    elif 'walker' in cfg_name:
        from models.sde_models.walker_sde import train_sde
        train_sde(cfg_name)


def analyze_model(dataset, model_names, hr, num_extra_steps, num_sample, num_traj = 1, use_train=False, seed=10, plot_xevol=False):
    """ Do the analysis of some of the existing models
    """
    # Load the dataset
    current_dir = os.path.expanduser(os.path.dirname(os.path.abspath(__file__)))
    data_dir = current_dir + '/models/sde_models/training_dataset/' + dataset + '_dataset.pkl'
    if not os.path.exists(data_dir):
        raise ValueError("The dataset {} does not exist".format(data_dir))
    # Open the file and load the data
    with open(data_dir, 'rb') as f:
        data = pickle.load(f)

    # Extract the data
    data = data['test_data'] if not use_train else data['train_data']

    # Print the number of trajectories
    print("Number of trajectories: {}".format(len(data)))
    lentraj = [s['y'].shape[0] for s in data]
    # print ('Total number of datapoints: ', sum(lentraj))
    # print('Number of datapints per trajectories: ', lentraj)

    # Pick randomly a trajectory
    np.random.seed(seed)

    # Load the model, predictor function and useful constants for plotting
    if 'halfcheetah' in dataset:
        from models.sde_models.halfcheetah_sde import load_predictor_function, OBS_NAMES, CONTROL_NAMES, TIMESTEP_ENV
    elif 'hopper' in dataset:
        from models.sde_models.hopper_sde import load_predictor_function, OBS_NAMES, CONTROL_NAMES, TIMESTEP_ENV
    elif 'walker' in dataset:
        from models.sde_models.walker_sde import load_predictor_function, OBS_NAMES, CONTROL_NAMES, TIMESTEP_ENV
    else:
        raise ValueError("The dataset {} is not supported".format(dataset))

    # We need to eliminate the trajectories that are too short according to the horizon and the number of extra steps
    valid_idx_traj = np.array([ _i for _i, _len in enumerate(lentraj) if _len >= num_extra_steps*hr])
    traj_idx = np.random.choice(len(valid_idx_traj), size=num_traj, replace=False) # Get the index of a random trajectory
    
    traj_idx = valid_idx_traj[traj_idx] # Get the actual index of the trajectory
    curr_traj_y_list, curr_traj_u_list = [], []
    for _idx in traj_idx:
        curr_traj_y_list.append(np.concatenate((data[_idx]['y'], data[_idx]['y'][-1:]), axis=0))
        curr_traj_u_list.append(data[_idx]['u'][:curr_traj_y_list[-1].shape[0]])
    # curr_traj_y = np.concatenate((data[traj_idx]['y'], data[traj_idx]['y'][-1:]), axis=0)
    # curr_traj_u = data[traj_idx]['u'][:curr_traj_y.shape[0]]

    # Time evolution of the trajectory
    # traj_time_evol = np.array([TIMESTEP_ENV * i for i in range(curr_traj_y_list[-1].shape[0])])
    curr_traj_time_list = [np.array([TIMESTEP_ENV * i for i in range(_y.shape[0])]) for _y in curr_traj_y_list]

    # Names of the states and controls
    state_names = OBS_NAMES
    control_names = CONTROL_NAMES

    # We do a two column plot
    num_states = len(state_names)
    num_controls = len(control_names)

    # maximum plots per column
    MAX_PLOTS_PER_COL = 5
    PER_CELL_FIG_SIZE = (4, 4)

    # number of rows and columns
    total_axis = num_states + num_controls
    num_cols = min(MAX_PLOTS_PER_COL, total_axis)
    num_rows = (total_axis // num_cols) + (1 if total_axis % num_cols != 0 else 0)
    TOTAL_FIG_SIZE = (num_cols * PER_CELL_FIG_SIZE[0], num_rows * PER_CELL_FIG_SIZE[1])

    # Define the figures
    fig_state, fig_err, fig_tot_err = None, None, None
    # Pick the colors
    gt_color = '#000000'
    # Pick color for each model -> LImited number to plot
    model_colors = ['#ff0000', '#00ff00', '#0000ff', '#ff00ff', '#00ffff']
    line_width = 2
    
    # Iterate through the models
    print('INFO: Analyzing the following models: ', model_names)
    for model_idx, model_name in enumerate(model_names):

        # Define up the sampling configuration
        sampling_cfg = {
            'num_particles' : num_sample,
            'horizon' : hr,
            'stepsize' : num_extra_steps * TIMESTEP_ENV,
        }

        # Load the model, the predictor function, and perform the analysis
        model_fn, t_model = create_model(model_name, sampling_cfg, load_predictor_function)
        base_model_fn_jit = jax.jit(model_fn)
        err_res = []
        for curr_traj_y, curr_traj_u, traj_time_evol in zip(curr_traj_y_list, curr_traj_u_list, curr_traj_time_list):
            _xres, _tres, _err_res = n_steps_analysis(curr_traj_y, curr_traj_u, base_model_fn_jit, t_model, TIMESTEP_ENV, traj_time_evol)
            err_res.extend(_err_res)

        curr_color = model_colors[model_idx]
        _model_name = model_name.split('_sde.pkl')[0] if '_sde.pkl' in model_name else model_name

        # # Figures to plot the results
        if plot_xevol:
            if fig_state is None:
                fig_state, axs_state = plt.subplots(num_rows, num_cols, figsize=TOTAL_FIG_SIZE, sharex=True)
                axs_state = axs_state.flatten()

            axs = axs_state
            for i in range(num_states):

                # Plot the ground truth
                if model_idx == 0:
                    axs[i].plot(traj_time_evol, curr_traj_y[:,i], color=gt_color, label='Ground truth', linewidth=line_width)
                
                # Extract the current color and model name
                curr_color = model_colors[model_idx]
                _model_name = model_name.split('_sde.pkl')[0] if '_sde.pkl' in model_name else model_name
                # print('Xres: ', _xres.shape, _xres.shape)

                # Plot the results for the current model
                for k in range(_xres.shape[0]):
                    axs[i].plot(_tres, _xres[k,:,i], color=curr_color, label=_model_name if k == 0 else None)
                
                # Set the labels
                if model_idx == len(model_names)-1:
                    axs[i].set_xlabel('Time (s)')
                    axs[i].set_ylabel(state_names[i])
                    axs[i].grid(True)
                    if i == 0:
                        axs[i].legend()

            # Plot the controls
            if model_idx == 0:
                for i in range(num_controls):
                    axs[i + num_states].plot(traj_time_evol[:-1], curr_traj_u[:,i], color=gt_color, label='Ground truth')
                    axs[i + num_states].set_xlabel('Time (s)')
                    axs[i + num_states].set_ylabel(control_names[i])
                    axs[i + num_states].grid(True)
        
        # Now ler's plot the average error over the prediction horizon for each state
        _error_full = err_res
        _err_array = _error_full[0]
        full_state_error = np.array([_v[:,:num_states] for _v in _error_full])
        # metric_fn = getattr(np, metric_fun)
        # full_state_error = metric_fn(full_state_error, axis=0)
        full_state_error_mean = np.mean(full_state_error, axis=0)
        full_state_error_min = np.min(full_state_error, axis=0)
        full_state_error_max = np.max(full_state_error, axis=0)

        if fig_err is None:
            fig_err, axs_err = [], []
            for _i in range(1):
                _fig, _axs = plt.subplots(num_rows,num_cols,figsize=TOTAL_FIG_SIZE,sharex=True)
                _axs = _axs.flatten()
                fig_err.append(_fig)
                axs_err.append(_axs)
        
        # for _fig, _axs in zip(fig_err, axs_err):
        _fig, axs = fig_err[0], axs_err[0]
        for i in range(num_states):
            state_error = full_state_error_mean[:,i]
            axs[i].plot(_err_array[:,-1], state_error, color=curr_color, label=_model_name)
            axs[i].fill_between(_err_array[:,-1], full_state_error_min[:,i], full_state_error_max[:,i], facecolor=curr_color, alpha=0.25, edgecolor='k')
            if model_idx == len(model_names)-1:
                axs[i].set_xlabel('Time (s)')
                axs[i].set_ylabel('Error in ' + state_names[i])
                axs[i].grid(True)
                if i == 0:
                    axs[i].legend()

        # Now let's plot the total error in norm and standard deviation
        norm_error = np.array([_v[:,num_states] for _v in _error_full])
        norm_error_mean = np.mean(norm_error, axis=0)
        norm_error_min = np.min(norm_error, axis=0)
        norm_error_max = np.max(norm_error, axis=0)

        # norm_error = metric_fn(norm_error, axis=0)

        if fig_tot_err is None:
            fig_tot_err, axs_tot_err = plt.subplots(1,2,figsize=(10,5),sharex=True)
            axs_tot_err = axs_tot_err.flatten()

        # Now we show the norm and standard deviation of the error
        # _err_array = _error_full[0]
        axs_tot_err[0].plot(_err_array[:,-1], norm_error_mean, color=curr_color, label=_model_name)
        axs_tot_err[0].fill_between(_err_array[:,-1], norm_error_min, norm_error_max, facecolor=curr_color, alpha=0.25, edgecolor='k')
        axs_tot_err[0].set_xlabel('Time (s)')
        axs_tot_err[0].set_ylabel('Norm of the error')
        axs_tot_err[0].grid(True)

        std_error = np.array([_v[:,num_states+1] for _v in _error_full])
        std_error_mean = np.mean(std_error, axis=0)
        std_error_min = np.min(std_error, axis=0)
        std_error_max = np.max(std_error, axis=0)
        # Now we show the norm and standard deviation of the error
        axs_tot_err[1].plot(_err_array[:,-1], std_error_mean, color=curr_color, label=_model_name)
        axs_tot_err[1].fill_between(_err_array[:,-1], std_error_min, std_error_max, facecolor=curr_color, alpha=0.25)
        axs_tot_err[1].set_xlabel('Time (s)')
        axs_tot_err[1].set_ylabel('Standard deviation of the error')
        axs_tot_err[1].grid(True)
    
    # plt.show()
    plt.savefig('analysis.png')


def create_model(model_name, sampling_cfg, load_fn):
    """ Load model sampler and time evolution function
    """
    sampling_cfg = copy.deepcopy(sampling_cfg)
    bae_model_fn, t_model = load_fn(model_name, 
                                modified_params = sampling_cfg, 
                                return_time_steps=True,
                                return_control=True,)
    return bae_model_fn, t_model


def n_steps_analysis(xtraj, utraj, jit_sampling_fn, time_evol, data_stepsize, traj_time_evol):
    """Compute the time evolution of the mean and variance of the SDE at each time step

    Args:
        xtraj (TYPE): The trajectory of the states
        utraj (TYPE): The trajectory of the inputs
        jit_sampling_fn (TYPE): The sampling function return an array of size (num_particles, horizon, state_dim)
        time_evol (TYPE): The time evolution of the sampling technique

    Returns:
        TYPE: The multi-sampled state evolution
        TYPE: The time step evolution for plotting
    """
    sampler_horizon = len(time_evol) - 1
    dt_sampler = time_evol[1] - time_evol[0]

    # Check if dt_sampler and data_stepsize are close enough
    if abs(dt_sampler - data_stepsize) < 1e-5:
        quot = 1
    else:
        assert dt_sampler > data_stepsize-1e-5, "The time step of the sampling function must be larger than the data step size"
        assert abs(dt_sampler % data_stepsize) <= 1e-6, "The time step of the sampling function must be a multiple of the data step size"
        quot = dt_sampler / data_stepsize
    # print(time_evol)

    # print(dt_sampler, data_stepsize, dt_sampler % sampler_horizon, sampler_horizon % dt_sampler)
    # assert dt_sampler > data_stepsize-1e-6, "The time step of the sampling function must be larger than the data step size"
    # assert abs(dt_sampler % data_stepsize) <= 1e-6, "The time step of the sampling function must be a multiple of the data step size"
    quot = dt_sampler / data_stepsize
    # Take the closest integer to quot
    num_steps2data  = int(quot + 0.5)
    # Compute the actual horizon for splitting the trajectories
    traj_horizon = num_steps2data * sampler_horizon
    utraj = utraj[:xtraj.shape[0]-1] # Remove the input rows if it is same or more than the state
    # Split the trajectory into chunks of size num_steps2data
    total_traj_size = (utraj.shape[0] // (traj_horizon)) * traj_horizon
    # print('INFO: ', quot, num_steps2data, traj_horizon, total_traj_size, dt_sampler)
    # print('INFO: ', xtraj.shape, utraj.shape, total_traj_size, traj_horizon, num_steps2data, sampler_horizon)

    # print('INFO: ', quot, num_steps2data, traj_horizon, total_traj_size, dt_sampler)
    # DOwngrade the xevol to its first 4 states
    _xevol = xtraj[:total_traj_size+1]
    _xevol_cut = _xevol[::num_steps2data]
    _xevol_cut = np.array([_xevol_cut[i:i+sampler_horizon+1] for i in range(0, _xevol_cut.shape[0]-sampler_horizon, sampler_horizon)])
    uevol = utraj[:total_traj_size]
    uevol = uevol.reshape(-1, sampler_horizon, num_steps2data, uevol.shape[-1])
    xevol = _xevol[::traj_horizon]
    # print(xevol.shape, _xevol_cut.shape)
    assert _xevol_cut.shape[0]+1 == xevol.shape[0], "The number of trajectories must be the same for the states and inputs"
    # Reshape the time evolution
    m_tevol = traj_time_evol[:total_traj_size+1][::traj_horizon]

    # print('INFO: ', m_tevol.shape, xevol.shape, uevol.shape)
    # print(xevol.shape)
    # print(uevol.shape)
    # assert xevol.shape[0] == uevol.shape[0], "The number of trajectories must be the same for the states and inputs"
    # Initial random number generator
    rng = jax.random.PRNGKey(20)
    rng, s_rng = jax.random.split(rng)
    xres = []
    tres = []
    err_analysis = []
    for i in range(uevol.shape[0]):
        rng, s_rng = jax.random.split(rng)
        # _curr_u = np.mean(uevol[i], axis=-2)
        _curr_u = uevol[i,:,0,:]
        _curr_x = xevol[i]
        _xpred, _ = jit_sampling_fn(_curr_x, _curr_u, s_rng) # (num_particles, horizon+1, state_dim)
        _xpred = np.array(_xpred)
        _tevol = m_tevol[i] + time_evol
        # Let's compute the error with respect to the groundtruth
        _mean_xevol = np.mean(_xpred, axis=0)
        # print('Mean xevol: ', _mean_xevol.shape, _xevol_cut[i].shape)
        _error_xevol = np.abs(_mean_xevol - _xevol_cut[i])
        _norm_xevol = np.linalg.norm(_error_xevol, axis=-1)
        std_xevol = np.sum(np.std(_xpred, axis=0), axis=-1)
        # print('STD dev: ', jnp.sum(std_xevol))
        # COmpute the cumulative error
        _cum_error_xevol = np.cumsum(_error_xevol, axis=0) / np.arange(1, _error_xevol.shape[0]+1)[:,None]
        _cum_norm_xevol = np.cumsum(_norm_xevol, axis=0) / np.arange(1, _norm_xevol.shape[0]+1)
        cum_std_xevol = np.cumsum(std_xevol, axis=0) / np.arange(1, std_xevol.shape[0]+1)
        # COncatenate these 3 results
        _res_analysis = np.concatenate([_cum_error_xevol, _cum_norm_xevol[:,None], cum_std_xevol[:,None], _tevol[:,None]], axis=-1)
        if i < xevol.shape[0]-1:
            _xpred = _xpred[:,:-1,:]
            _tevol = _tevol[:-1]
        xres.append(_xpred)
        tres.append(_tevol)
        err_analysis.append(_res_analysis)
    # Merge the results along the horizon axis
    xres = np.concatenate(xres, axis=1)
    _tres = np.concatenate(tres, axis=0)
    # print(xres.shape, _tres.shape)
    return xres, _tres, err_analysis

if __name__ == '__main__':
    import argparse
    # python train_n_analyze_models.py --train --cfg_train halfcheetah.yaml
    # python train_n_analyze_models.py --analyze --dataset halfcheetah-random-v2 --hr 100 --ntraj 5 --use_train --seed 
    # Argument parser
    parser = argparse.ArgumentParser(description='Script for training and analyzing the neural SDE models')
    parser.add_argument('--cfg_train', type=str, default= 'halfcheetah.yaml', help='Name of the yaml training configuration file')
    parser.add_argument('--train', default=False, action='store_true', help='Flag to train the model')
    parser.add_argument('--analyze', default=False, action='store_true', help='Flag to analyze the model')
    parser.add_argument('--dataset', type=str, default='halfcheetah-random-v2', help='Name of the dataset')
    parser.add_argument('--model_names', type=str, nargs='+', default=['random_hc_vf4_hr-10_dt-0.010_sde.pkl',], help='Name of the model')
    parser.add_argument('--hr', type=int, default=100, help='Horizon for the sampling')
    parser.add_argument('--num_extra_steps', type=int, default=1, help='The number of extra step wrt data step size for the sampling')
    parser.add_argument('--nsample', type=int, default=100, help='Number of samples for the sampling')
    parser.add_argument('--ntraj', type=int, default=1, help='Number of trajectories to show')
    parser.add_argument('--seed', type=int, default=10, help='Seed for the random number generator')
    parser.add_argument('--use_train', default=False, action='store_true', help='Flag to use the train dataset')
    parser.add_argument('--plot_xevol', default=False, action='store_true', help='Flag to plot the evolution of the states')
    # parser.add_argument('--metrics_fn', type=str,  default='max',  help='Metrics function to use for the analysis')

    # Parse the arguments
    args = parser.parse_args()

    if args.train:
        train_model(args.cfg_train)

    if args.analyze:
        # JAX_PLATFORM_NAME=cpu python train_n_analyze_models.py --use_mean --analyze --dataset halfcheetah-random-v2 --hr 30 --use_train --seed 55 --model_names hc_rand_v2_dsc0.3_hr-5_dt-0.05_sde.pkl hc_rand_v2_dsc0.1_hr-5_dt-0.05_sde.pkl
        analyze_model(args.dataset, args.model_names, args.hr, args.num_extra_steps, args.nsample, args.ntraj, args.use_train, args.seed, args.plot_xevol)