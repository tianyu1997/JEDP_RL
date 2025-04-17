from predict_jacobian import JacobianPredictor
import numpy as np
import torch
import gymnasium as gym
import wandb
from config import *
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet
from t_ppo import PPO
from torch.distributions import Normal
from collections import deque
from stable_baselines3.stable_baselines3 import PPO as SB3PPO
from stable_baselines3.stable_baselines3.common.env_util import make_vec_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Explore_Env(gym.Env):
    def __init__(self, model_path, reward_threshold=-7e-3, max_length=50):
        """
        Initialize the exploration environment.
        Args:
            model_path (str): Path to the pre-trained Jacobian predictor model.
        """
        super().__init__()
        self.reward_threshold = reward_threshold
        self.max_length = max_length
        self.device = device
        self.model = JacobianPredictor(input_dim=13, output_dim=21)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(128 * 3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(7,), dtype=np.float32)
        self.robot = Panda(
            sim=PyBullet(render_mode="rgb_array", renderer="Tiny"),
            block_gripper=False,
            base_position=None,
            control_type="joints",
        )

    def reset(self, **kwargs):
        """
        Reset the environment.
        Returns:
            np.ndarray: Initial observation.
        """
        self.model.reset()
        self.robot.reset()
        self.old_ee = self.robot.get_ee_position()
        self.action = np.random.uniform(-0.1, 0.1, 7)
        self.robot.set_action(self.action)
        self.robot.sim.step()
        self.new_ee = self.robot.get_ee_position()
        obs, _ = self.get_obs_and_reward()
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
        actual_jacobian = self.robot.get_jacobian()
        actual_jacobian = torch.flatten(torch.tensor(np.array(actual_jacobian), dtype=torch.float32)).to(self.device)
        predict_jacobian, _ = self.model(input_tensor)
        predict_jacobian = predict_jacobian.squeeze()
        return self.model.state, -torch.nn.functional.mse_loss(predict_jacobian, actual_jacobian).item()

    def step(self, action):
        """
        Take a step in the environment.
        Args:
            action (np.ndarray): Action to take.
        Returns:
            tuple: Observation, reward, done, truncated, and info.
        """
        self.action = action
        self.old_ee = self.robot.get_ee_position()
        self.robot.set_action(self.action)
        self.robot.sim.step()
        self.new_ee = self.robot.get_ee_position()
        
        obs, reward = self.get_obs_and_reward()
        # print(f"reward: {reward}")
        self.length += 1
        done = False
        if reward > self.reward_threshold:
            done = True
            reward += 1
        if self.length > self.max_length:
            done = True
        return obs, reward, done, False, {}

    def seed(self, seed=None):
        """
        Set the random seed for the environment.
        Args:
            seed (int): Seed value.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.robot.sim.seed(seed)

def main():
    """
    Main function to train the PPO agent in the exploration environment.
    """
    set_seed(seed)  # Set random seed
    index = 1  # Index for the model
    model_path = f'jacobian_predictor_{index}.pth'
    env = make_vec_env(lambda: Explore_Env(model_path, reward_threshold=(8-index)*-1e-3), n_envs=32)  # Vectorized environment for Stable-Baselines3

    # Initialize PPO agent
    model = SB3PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_explorer_tensorboard/", n_steps=128)
    model.load(f"ppo_explorer_model{index-1}")  # Load the pre-trained model if available
    for i in range(10):
        # Train the agent for a short period
        model.learn(total_timesteps=1e6)
        # Save the model periodically
        model.save(f"checkpoints/ppo_explorer_model{i}")
    

if __name__ == "__main__":
    main()


