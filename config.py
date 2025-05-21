import numpy as np
import gymnasium as gym
import random
import torch
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet

#### Configuration for the Jacobian predictor
seed = 42
lr = 0.0003
input_dim = 13
output_dim = 21
model_path = "jacobian_predictor.pth"
scheduler_factor = 0.1
scheduler_patience = 10
hidden_size = 128
clip_value = 0.5
epochs = 100000
batch_size=8
confidence_k = 1/30
confidence_bias = 0.1
confidence_loss_coef = 0.01
log_interval = 100
save_interval = 10000
predictior_checkpoint_path = 'checkpoints/jacobian_predictor.pth'

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_random_action(random_range=0.1):
    action = np.random.uniform(-random_range, random_range, 7)
    return action

def get_env(name="panda"):
    if name == "panda":
        env = Panda(
            sim=PyBullet(render_mode="rgb_array", renderer="Tiny"),
            block_gripper=False,
            base_position=None,
            control_type="joints",
        )
    if not hasattr(env, 'get_observation') or not hasattr(env, 'get_jacobian') or not hasattr(env, 'set_action'):
        raise AttributeError("The environment must have 'get_observation', 'get_jacobian' and 'set_action' methods.")
    return env


#### Configuration for the Jacobian explorer
max_length = 50
n_steps = 2048
n_envs = 32
predict_erroe_threshold=(1.5)*1e-3 #认为预测成功的阈值
predictor_path = f'jacobian_predictor_0.pth'  # Path to the pre-trained explorer model
explorer_obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(128 * 3,), dtype=np.float32)
explorer_action_space = gym.spaces.Box(low=-0.05, high=0.05, shape=(7,), dtype=np.float32)
explorer_net_arch = dict(pi=[256, 128, 64, 32], vf=[256, 128, 64, 32])
iterations = 10
total_timesteps = 1e7
reward_advantage_coef = -100
reward_norm_coef = -0.01
norm_bias = 0.03
done_reward = 10
load_explorer = False
explorer_checkpoint_path = 'checkpoints/explorer.pth'

#### Configuration for the active Jacobian explorer
aje_max_length = 50
aje_n_steps = 2048
aje_n_envs = 32
distance_threshold = 0.1 # Distance threshold for reaching the goal
explorer_path = f'explorer_model_0'  # Path to the pre-trained explorer model
aje_net_arch = dict(pi=[256, 128, 64, 32], vf=[256, 128, 64, 32])
aje_iterations = 10
aje_total_timesteps = 1e7
confidence_threshold = 0.9
aje_obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(128 * 3,), dtype=np.float32)
aje_action_space = gym.spaces.Box(low=-0.05, high=0.05, shape=(7,), dtype=np.float32)

def get_goal(name="panda"):
    if name == "panda":
        goal_range=0.3
        goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        goal = gym.Env.np_random.uniform(goal_range_low, goal_range_high)
        return goal
    
reward_distance_coef = 10
reward_length_coef = -0.001
reward_done_coef = 10