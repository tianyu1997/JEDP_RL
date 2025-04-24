import numpy as np
import random
import torch
import gymnasium as gym

learning_rate  = 0.00001
gamma           = 0.98
lmbda           = 0.9
eps_clip        = 0.15
K_epoch         = 10
rollout_len    = 3
buffer_size    = 10
minibatch_size = 8
save_interval  = 1000
env_obs_dim = 6
time_length = 5
print_interval = 100
loss_threshold = 0.00005
seed = 42
epoisodes = 100000
critic_coef = 1
max_length = 300
distance_threshold = 0.15
confidence_threshold = 0.98
n_steps = 2048
batch_size = 64

def get_input(s):
    s_d = s['desired_goal']-s['achieved_goal']
    input = np.concatenate([s['achieved_goal'], s_d])
    return input

def set_seed(seed):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if hasattr(gym.spaces, 'seed'):
        gym.spaces.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False