from stage1_jacobian_predictor import JacobianPredictor
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

class Aje_Plan_Env(gym.Env):
    def __init__(self, predictor_path, explorer_path, confidence_threshold=confidence_threshold):
        """
        Initialize the exploration environment.
        Args:
            model_path (str): Path to the pre-trained Jacobian predictor model.
        """
        super().__init__()
        self.max_length = aje_max_length
        self.device = device
        self.distance_threshold = distance_threshold
        self.confidence_threshold = confidence_threshold
        self.predictor = JacobianPredictor(input_dim=input_dim, output_dim=output_dim)
        self.predictor.load_state_dict(torch.load(predictor_path, map_location=self.device))
        self.predictor.eval()
        self.predictor.to(self.device)
        self.explorer = SB3PPO.load(explorer_path, device=self.device)
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
        self.predictor.reset()
        self.env.reset()
        self.old_obs = self.env.get_observation()
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
        confidence = 0
        while confidence < self.confidence_threshold:
            input_tensor = torch.tensor(
                np.concatenate([self.old_ee, self.action, self.new_ee]), dtype=torch.float32
            ).to(self.device).flatten().unsqueeze(0)
            actual_jacobian = self.env.get_jacobian()
            actual_jacobian = torch.flatten(torch.tensor(np.array(actual_jacobian), dtype=torch.float32)).to(self.device)
            predict_jacobian, confidence = self.predictor(input_tensor)
            explore_action = self.explorer.predict(self.predictor.state, deterministic=True)
            self.old_ee = self.env.get_ee_position()
            self.action = explore_action[0]
            self.env.set_action(self.action)
            self.env.sim.step()
            self.new_ee = self.env.get_ee_position()
        return self.predictor.state, reward

    def step(self, action):
        """
        Take a step in the environment.
        Args:
            action (np.ndarray): Action to take.
        Returns:
            tuple: Observation, reward, done, truncated, and info.
        """
        self.action = action
        self.old_ee = self.env.get_ee_position()
        self.env.set_action(self.action)

        self.new_ee = self.env.get_ee_position()
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
    # index = 2  # Index for the model
    env = make_vec_env(lambda: Aje_Plan_Env(predictor_path, explorer_path),  n_envs=aje_n_envs)  # Vectorized environment for Stable-Baselines3
    policy_kwargs = dict(net_arch=aje_net_arch)
    # Initialize PPO agent
    model = SB3PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_explorer_tensorboard/", n_steps=aje_n_steps, batch_size=batch_size)
    # model.load(f"aje_model_{0}")  # Load the pre-trained model if available
    for i in range(aje_iterations):
        # Train the agent for a short period
        model.learn(total_timesteps=aje_total_timesteps)
        # Save the model periodically
        model.save(f"checkpoints/aje_model_{i}")
    

if __name__ == "__main__":
    main()