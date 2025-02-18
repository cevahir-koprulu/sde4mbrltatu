import numpy as np
from matplotlib import pyplot as plt
from nsdes_dynamics.utils_for_d4rl_mujoco import get_dataset 


dataset = get_dataset('halfcheetah-random-v2')
obs = dataset['observations']
figure = plt.figure(figsize=(10, 10))
plt.scatter(obs[:, 8], obs[:, 1], color='black', s=1)
# plt.xlabel('vel_x')
# plt.ylabel('angle')
plt.xticks([])
plt.yticks([])
plt.savefig('halfcheetah-random-v2_velxVSangle.png', bbox_inches='tight')
