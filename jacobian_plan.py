from predict_jacobian import JacobianPredictor
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
    def __init__(self, predictor_path, goal_range=0.3, ):
        """
        Initialize the exploration environment.
        Args:
            model_path (str): Path to the pre-trained Jacobian predictor model.
        """
        super().__init__()
        self.max_length = max_length
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        self.predictor = JacobianPredictor(input_dim=13, output_dim=21)
        self.predictor.load_state_dict(torch.load(predictor_path, map_location=self.device))
        self.predictor.eval()
        self.predictor.to(self.device)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(128 * 3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(7,), dtype=np.float32)
        self.robot = Panda(
            sim=PyBullet(render_mode="rgb_array", renderer="Tiny"),
            block_gripper=False,
            base_position=None,
            control_type="joints",
        )
        self.reset()

    def reset(self, **kwargs):
        """
        Reset the environment.
        Returns:
            np.ndarray: Initial observation.
        """
        self.predictor.reset()
        self.robot.reset()
        self.old_ee = self.robot.get_ee_position()
        self.goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
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
        old_d = distance(self.old_ee, self.goal)
        new_d = distance(self.new_ee, self.goal)
        reward = 10*(old_d - new_d) + 10*np.array(new_d < self.distance_threshold, dtype=np.float32)
        
        input_tensor = torch.tensor(
            np.concatenate([self.old_ee, self.action, self.new_ee]), dtype=torch.float32
        ).to(self.device).flatten().unsqueeze(0)
        actual_jacobian = self.robot.get_jacobian()
        actual_jacobian = torch.flatten(torch.tensor(np.array(actual_jacobian), dtype=torch.float32)).to(self.device)
        predict_jacobian, confidence = self.predictor(input_tensor)
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
        self.old_ee = self.robot.get_ee_position()
        self.robot.set_action(self.action)
        self.robot.sim.step()
        self.new_ee = self.robot.get_ee_position()
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
    predictor_path = f'jacobian_predictor_0.pth'
      # Path to the pre-trained explorer model
    env = make_vec_env(lambda: Aje_Plan_Env(predictor_path),  n_envs=32)  # Vectorized environment for Stable-Baselines3

    # Initialize PPO agent
    model = SB3PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_explorer_tensorboard/", n_steps=n_steps, batch_size=batch_size)
    # model.load(f"aje_model_{0}")  # Load the pre-trained model if available
    for i in range(10):
        # Train the agent for a short period
        model.learn(total_timesteps=1e6)
        # Save the model periodically
        model.save(f"checkpoints/jacobian_model_{i}")
    

if __name__ == "__main__":
    main()