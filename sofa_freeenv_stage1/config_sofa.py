import numpy as np
import random
import torch
import gymnasium as gym

input_dim = 9
output_dim = 9
lr = 0.0003
scheduler_factor = 0.1
scheduler_patience = 10
hidden_size = 128
clip_value = 0.5
gamma           = 0.9
lmbda           = 0.9
eps_clip        = 0.15
K_epoch         = 10
rollout_len    = 3
buffer_size    = 10
minibatch_size = 128
save_interval  = 77 # the number of episode to reopen SOFA for reloading
# save_interval  = 1000
# env_obs_dim = 6
time_length = 5
print_interval = 20
# loss_threshold = 0.00005
loss_threshold = 0.1
seed = 42
epoisodes = save_interval 
confidence_loss_coef = 0.01
epochs = 10

predictior_checkpoint_path = 'checkpoints/jacobian_predictor.pth'
model_path = "jacobian_predictor.pth"

attempt=1 # the environment attempt number for different setting
logno = 4
wbid=f"{logno}04082025" # logno & start date

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


def get_env(name="endo"):
    if name == "endo":
        import gymnasium as gym
        from gymnasium.envs.registration import register

        register(
            id='endogym-v0',
            entry_point='JEDP_RL_sofaenv_selfdefinetrainingpipeline_test.supervise_env:SoftRobotEnv',
        )
        # available environments
        import gymnasium as gym
        all_envs = gym.envs.registry
        env_names = list(all_envs.keys())

        # Print each environment name
        for env in env_names:
            print(env)

        env = gym.make('endogym-v0', nocontactpts=1600)  # Make sure 'endogym-v0' is registered
        env.reset(seed=seed)
    # if not hasattr(env, 'get_observation') or not hasattr(env, 'get_jacobian') or not hasattr(env, 'set_action'):
    #     raise AttributeError("The environment must have 'get_observation', 'get_jacobian' and 'set_action' methods.")
    return env

def get_random_action(random_range=0.1):
    action = np.random.uniform(-random_range, random_range, 3)
    return action