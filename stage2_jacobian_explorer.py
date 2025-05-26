from JEDP_RL_sofaenv_selfdefinetrainingpipeline.stage1_jacobian_predictor import JacobianPredictor
import numpy as np
import torch
import gymnasium as gym
import wandb
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet
from torch.distributions import Normal
from collections import deque
from stable_baselines3.stable_baselines3 import PPO as SB3PPO
from stable_baselines3.stable_baselines3.common.env_util import make_vec_env
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Explore_Env(gym.Env):
    def __init__(self, model_path, env=None, predict_erroe_threshold=predict_erroe_threshold, max_length=max_length):
        """
        Initialize the exploration environment.
        Args:
            model_path (str): Path to the pre-trained Jacobian predictor model.
        """
        super().__init__()
        self.predict_erroe_threshold = predict_erroe_threshold
        self.max_length = max_length
        self.device = device
        self.model = JacobianPredictor(input_dim=input_dim, output_dim=output_dim)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        self.observation_space = explorer_obs_space
        self.action_space = explorer_action_space
        if env is None:
            self.env = get_env()
        else:
            self.env = env

    def reset(self, **kwargs):
        """
        Reset the environment.
        Returns:
            np.ndarray: Initial observation.
        """
        self.old_predict_loss = 0
        self.model.reset()
        self.env.reset()
        self.old_obs = self.env.get_observation()
        self.action = get_random_action(0.05)
        self.env.set_action(self.action)
        self.new_obs = self.env.get_observation()
        obs, _, self.old_predict_loss = self.get_obs_and_reward()
        self.length = 0
        return obs, {}

    
    def get_obs_and_reward(self):
        """
        Compute the reward based on the difference between actual and predicted Jacobians.
        Returns:
            float: Reward value.
        """
        input_tensor = torch.tensor(
            np.concatenate([self.old_ee, self.action, self.new_ee]), dtype=torch.float32
        ).to(self.device).flatten().unsqueeze(0)
        actual_jacobian = self.env.get_jacobian()
        actual_jacobian = torch.flatten(torch.tensor(np.array(actual_jacobian), dtype=torch.float32)).to(self.device)
        predict_jacobian, _ = self.model(input_tensor)
        predict_jacobian = predict_jacobian.squeeze()
        predict_loss = torch.nn.functional.mse_loss(predict_jacobian, actual_jacobian).item()
        predict_advantage = predict_loss - self.old_predict_loss
        norm_loss = max(0, np.linalg.norm(self.action) - norm_bias)
        reward = reward_advantage_coef * predict_advantage + reward_norm_coef * norm_loss
        return self.model.state, reward, predict_loss

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
        
        obs, reward, self.old_predict_loss = self.get_obs_and_reward()
        # print(f"reward: {reward}")
        self.length += 1
        done = False
        # print(f"predict_loss: {self.old_predict_loss}")
        if self.old_predict_loss < self.predict_erroe_threshold:
            done = True
            reward += done_reward
        if self.length > self.max_length:
            done = True
        return obs, reward, done, False, {}


def main():
    """
    Main function to train the PPO agent in the exploration environment.
    """
    set_seed(seed)  # Set random seed
    env = make_vec_env(lambda: Explore_Env(predictor_path, predict_erroe_threshold=predict_erroe_threshold), n_envs=n_envs)  # Vectorized environment for Stable-Baselines3
    policy_kwargs = dict(net_arch=explorer_net_arch)
    # Initialize PPO agent
    model = SB3PPO("MlpPolicy", env, verbose=1,policy_kwargs=policy_kwargs, tensorboard_log="./ppo_explorer_tensorboard/", n_steps=n_steps, batch_size=batch_size)
    if load_explorer:
        model.load(explorer_checkpoint_path)  # Load the pre-trained model if available
    for i in range(iterations):
        # Train the agent for a short period
        model.learn(total_timesteps=total_timesteps)
        # Save the model periodically
        model.save(f"checkpoints/explorer_model{i}")
    

if __name__ == "__main__":
    main()