import os
import urllib.request
import h5py
import numpy as np
from tqdm import tqdm

DATASET_URLS = {
    'halfcheetah-random-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_random.hdf5',
    'halfcheetah-medium-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_medium.hdf5',
    'halfcheetah-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_expert.hdf5',
    'halfcheetah-medium-replay-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_mixed.hdf5',
    'halfcheetah-medium-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_medium_expert.hdf5',
    'walker2d-random-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_random.hdf5',
    'walker2d-medium-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_medium.hdf5',
    'walker2d-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_expert.hdf5',
    'walker2d-medium-replay-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker_mixed.hdf5',
    'walker2d-medium-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_medium_expert.hdf5',
    'hopper-random-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_random.hdf5',
    'hopper-medium-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_medium.hdf5',
    'hopper-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_expert.hdf5',
    'hopper-medium-replay-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_mixed.hdf5',
    'hopper-medium-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_medium_expert.hdf5',
    'ant-random-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_random.hdf5',
    'ant-medium-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_medium.hdf5',
    'ant-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_expert.hdf5',
    'ant-medium-replay-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_mixed.hdf5',
    'ant-medium-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_medium_expert.hdf5',
    'ant-random-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_random_expert.hdf5',
}


REF_MIN_SCORE = {
    'halfcheetah-random-v0' : -280.178953 ,
    'halfcheetah-medium-v0' : -280.178953 ,
    'halfcheetah-expert-v0' : -280.178953 ,
    'halfcheetah-medium-replay-v0' : -280.178953 ,
    'halfcheetah-medium-expert-v0' : -280.178953 ,
    'walker2d-random-v0' : 1.629008 ,
    'walker2d-medium-v0' : 1.629008 ,
    'walker2d-expert-v0' : 1.629008 ,
    'walker2d-medium-replay-v0' : 1.629008 ,
    'walker2d-medium-expert-v0' : 1.629008 ,
    'hopper-random-v0' : -20.272305 ,
    'hopper-medium-v0' : -20.272305 ,
    'hopper-expert-v0' : -20.272305 ,
    'hopper-medium-replay-v0' : -20.272305 ,
    'hopper-medium-expert-v0' : -20.272305 ,
    'ant-random-v0' : -325.6,
    'ant-medium-v0' : -325.6,
    'ant-expert-v0' : -325.6,
    'ant-medium-replay-v0' : -325.6,
    'ant-medium-expert-v0' : -325.6,
}

REF_MAX_SCORE = {
    'halfcheetah-random-v0' : 12135.0 ,
    'halfcheetah-medium-v0' : 12135.0 ,
    'halfcheetah-expert-v0' : 12135.0 ,
    'halfcheetah-medium-replay-v0' : 12135.0 ,
    'halfcheetah-medium-expert-v0' : 12135.0 ,
    'walker2d-random-v0' : 4592.3 ,
    'walker2d-medium-v0' : 4592.3 ,
    'walker2d-expert-v0' : 4592.3 ,
    'walker2d-medium-replay-v0' : 4592.3 ,
    'walker2d-medium-expert-v0' : 4592.3 ,
    'hopper-random-v0' : 3234.3 ,
    'hopper-medium-v0' : 3234.3 ,
    'hopper-expert-v0' : 3234.3 ,
    'hopper-medium-replay-v0' : 3234.3 ,
    'hopper-medium-expert-v0' : 3234.3 ,
    'ant-random-v0' : 3879.7,
    'ant-medium-v0' : 3879.7,
    'ant-expert-v0' : 3879.7,
    'ant-medium-replay-v0' : 3879.7,
    'ant-medium-expert-v0' : 3879.7,
}


#Gym-MuJoCo V1/V2 envs
for env in ['halfcheetah', 'hopper', 'walker2d', 'ant']:
    for dset in ['random', 'medium', 'expert', 'medium-replay', 'full-replay', 'medium-expert']:
        #v1 envs
        dset_name = env+'_'+dset.replace('-', '_')+'-v1'
        env_name = dset_name.replace('_', '-')
        DATASET_URLS[env_name] = 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v1/%s.hdf5' % dset_name
        REF_MIN_SCORE[env_name] = REF_MIN_SCORE[env+'-random-v0']
        REF_MAX_SCORE[env_name] = REF_MAX_SCORE[env+'-random-v0']

        #v2 envs
        dset_name = env+'_'+dset.replace('-', '_')+'-v2'
        env_name = dset_name.replace('_', '-')
        DATASET_URLS[env_name] = 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/%s.hdf5' % dset_name
        REF_MIN_SCORE[env_name] = REF_MIN_SCORE[env+'-random-v0']
        REF_MAX_SCORE[env_name] = REF_MAX_SCORE[env+'-random-v0']


def set_dataset_path(path):
    global DATASET_PATH
    DATASET_PATH = path
    os.makedirs(path, exist_ok=True)


set_dataset_path(os.environ.get('D4RL_DATASET_DIR', os.path.expanduser('~/.d4rl/datasets')))


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def filepath_from_url(dataset_url):
    _, dataset_name = os.path.split(dataset_url)
    dataset_filepath = os.path.join(DATASET_PATH, dataset_name)
    return dataset_filepath


def download_dataset_from_url(dataset_url):
    dataset_filepath = filepath_from_url(dataset_url)
    if not os.path.exists(dataset_filepath):
        print('Downloading dataset:', dataset_url, 'to', dataset_filepath)
        urllib.request.urlretrieve(dataset_url, dataset_filepath)
    if not os.path.exists(dataset_filepath):
        raise IOError("Failed to download dataset from %s" % dataset_url)
    return dataset_filepath

def get_dataset(env_name):
    dataset_url = DATASET_URLS[env_name]
    h5path = download_dataset_from_url(dataset_url)

    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]
    # Run a few quick sanity checks
    for key in ['observations', 'actions', 'rewards', 'terminals', 'timeouts']:
        assert key in data_dict, 'Dataset is missing key %s' % key
    N_samples = data_dict['observations'].shape[0]
    if data_dict['rewards'].shape == (N_samples, 1):
        data_dict['rewards'] = data_dict['rewards'][:, 0]
    assert data_dict['rewards'].shape == (N_samples,), 'Reward has wrong shape: %s' % (
        str(data_dict['rewards'].shape))
    if data_dict['terminals'].shape == (N_samples, 1):
        data_dict['terminals'] = data_dict['terminals'][:, 0]
    assert data_dict['terminals'].shape == (N_samples,), 'Terminals has wrong shape: %s' % (
        str(data_dict['rewards'].shape))
    
    return data_dict

def get_q_learning_dataset(env_name):
    dataset_url = DATASET_URLS[env_name]
    h5path = download_dataset_from_url(dataset_url)

    dataset = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                dataset[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                dataset[k] = dataset_file[k][()]

    _max_episode_steps = 1000
    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in tqdm(range(N-1), desc="Formatting data"):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == _max_episode_steps - 1)
        if final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append([reward])
        done_.append([done_bool])
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }


def get_formatted_dataset_for_nsde_training(env_name, min_traj_len=0):
    data_dict = get_dataset(env_name)
    episode_ends = np.argwhere((data_dict['timeouts']==1) + (data_dict['terminals']==1)).reshape(-1)
    t_final_prev = -1
    full_data = []
    ep_len = np.concatenate(([episode_ends[0]], episode_ends[1:] - episode_ends[:-1]), axis=0)
    print(f"Ratio of short trajectories: {ep_len[ep_len<=min_traj_len].shape[0]/ep_len.shape[0]*100}")
    for t_final in episode_ends:
        if (t_final - t_final_prev) <= min_traj_len:
            t_final_prev = t_final
            continue
        full_data.append({'y': data_dict['observations'][t_final_prev+1:t_final+1],
                          'u': data_dict['actions'][t_final_prev+1:t_final+1]})
        t_final_prev = t_final
    return full_data


def get_skrl_memory_version_of_dataset(env_name):
    tensors_names = ["states", "actions", "rewards", "next_states", "terminated"]
    data_keys = ["observations", "actions", "rewards", "next_observations", "terminals"]
    tensor_name_to_data_key = {tensor_name: data_key for tensor_name, data_key in zip(tensors_names, data_keys)}
    data_dict = get_q_learning_dataset(env_name)
    return {name: np.array(data_dict[tensor_name_to_data_key[name]]) for name in tensors_names}

def get_states_from_dataset(env_name, only_initial_states=False):
    data_dict = get_dataset(env_name)
    if only_initial_states:
        episode_ends = np.argwhere((data_dict['timeouts']==1) + (data_dict['terminals']==1)).reshape(-1)
        t_final_prev = -1
        full_data = []
        for t_final in episode_ends:
            full_data.append(data_dict['observations'][t_final_prev+1])
            t_final_prev = t_final
        return np.array(full_data)        
    return data_dict['observations']