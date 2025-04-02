import numpy as np

learning_rate  = 0.0001
gamma           = 0.9
lmbda           = 0.9
eps_clip        = 0.15
K_epoch         = 10
rollout_len    = 3
buffer_size    = 10
minibatch_size = 32
save_interval  = 1000
env_obs_dim = 6
time_length = 5
print_interval = 20
loss_threshold = 0.00005

def get_input(s):
    s_d = s['desired_goal']-s['achieved_goal']
    input = np.concatenate([s['achieved_goal'], s_d])
    return input
