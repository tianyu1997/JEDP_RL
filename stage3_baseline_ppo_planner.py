import numpy as np
import torch
import gymnasium as gym
import wandb
from config import *
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance
from collections import deque
from stable_baselines3.stable_baselines3 import PPO as SB3PPO
from stable_baselines3.stable_baselines3.common.env_util import make_vec_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO_Plan_Env(gym.Env):
    def __init__(self):
        """
        Initialize the exploration environment.
        Args:
            model_path (str): Path to the pre-trained Jacobian predictor model.
        """
        super().__init__()
        self.max_length = aje_max_length
        self.device = device
        self.distance_threshold = distance_threshold
    
        self.observation_space = aje_obs_space
        self.action_space = aje_action_space
        self.env = get_env()
        self.reset()

    def reset(self, **kwargs):
        """
        Reset the environment.
        Returns:
            np.ndarray: Initial observation.
        """
        self.length = 0
        self.env.reset()
        self.old_ee = self.env.get_observation()
        self.goal = get_goal()
        self.action = get_random_action(0.1)
        self.env.set_action(self.action)
        self.new_ee = self.env.get_observation()
        obs, _ = self.get_obs_and_reward()
        
        return obs, {}
    
    
    def get_obs_and_reward(self):
        """
        Compute the reward based on the difference between actual and predicted Jacobians.
        Returns:
            float: Reward value.
        """
        old_d = distance(self.old_ee, self.goal)
        new_d = distance(self.new_ee, self.goal)
        reward = reward_distance_coef*(old_d - new_d) + reward_length_coef *self.length + reward_done_coef*np.array(new_d < self.distance_threshold, dtype=np.float32)
        obs = np.concatenate([self.old_ee, self.action, self.new_ee])
        return obs, reward

    def step(self, action):
        """
        Take a step in the environment.
        Args:
            action (np.ndarray): Action to take.
        Returns:
            tuple: Observation, reward, done, truncated, and info.
        """
        self.action = action
        self.old_ee = self.env.get_observation()
        self.env.set_action(self.action)
        self.new_ee = self.env.get_observation()
        d = distance(self.new_ee, self.goal)
        done = d < self.distance_threshold
        obs, reward = self.get_obs_and_reward()
        # print(f"reward: {reward}")
        self.length += 1
        if self.length > self.max_length:
            done = True
        return obs, reward, done, False, {}

def main():
    """
    Main function to train the PPO agent in the exploration environment.
    """
    set_seed(seed)  # Set random seed
    env = make_vec_env(lambda: PPO_Plan_Env(),  n_envs=aje_n_envs)  # Vectorized environment for Stable-Baselines3
    policy_kwargs = dict(net_arch=aje_net_arch)
    # Initialize PPO agent
    model = SB3PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_explorer_tensorboard/", n_steps=n_steps, batch_size=batch_size)
    if load_ppo:
        model.load(ppo_checkpoint_path)  # Load the pre-trained model if available
    for i in range(aje_iterations):
        # Train the agent for a short period
        model.learn(total_timesteps=aje_total_timesteps)
        # Save the model periodically
        model.save(f"checkpoints/ppo_model_{i}")
    

if __name__ == "__main__":
    main()