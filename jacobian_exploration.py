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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Explore_Env(gym.Env):
    def __init__(self, model_path):
        """
        Initialize the exploration environment.
        Args:
            model_path (str): Path to the pre-trained Jacobian predictor model.
        """
        super().__init__()
        self.device = device
        self.model = JacobianPredictor(input_dim=13, output_dim=21)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(128 * 3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        self.robot = Panda(
            sim=PyBullet(render_mode="rgb_array", renderer="Tiny"),
            block_gripper=False,
            base_position=None,
            control_type="joints",
        )

    def reset(self):
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
        return obs

    
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
        done = reward > -5e3
        return obs, reward, done, False, {}

def main():
    """
    Main function to train the PPO agent in the exploration environment.
    """
    set_seed(seed)  # Set random seed
    name = f'ppo_1_{minibatch_size}_{seed}_{critic_coef}'
    wandb.init(project="Explorer", name=name)  # Initialize wandb project
    model_path = 'jacobian_predictor_epoch_100000.pth'
    env = Explore_Env(model_path)
    model = PPO(env.observation_space.shape[0], env.action_space.shape[0], action_scale=[0.05, 0.05])
    
    score, score_count = 0.0, 0
    rollout = []
    for n_epi in range(epoisodes):
        s = env.reset()
        count, done = 0, False

        while count < 100 and not done:
            for t in range(rollout_len):
                mu, std = model.pi(s)
                dist = Normal(mu, std)
                a = dist.sample()
                log_prob = dist.log_prob(a)

                a = a.detach().cpu().numpy()
                s_prime, r, done, _, _ = env.step(a)

                rollout.append((s, a, r, s_prime, log_prob, done))
                if len(rollout) == rollout_len:
                    model.put_data(rollout)
                    rollout = []
                score += r
                score_count += 1
                count += 1

            model.train_net('actor')

        if n_epi % print_interval == 0 and n_epi != 0:
            avg_score = score / score_count
            print(f"# of episode: {n_epi}, avg score: {avg_score:.5f}, optimization step: {model.optimization_step}")
            wandb.log({"episode": n_epi, "avg_score": avg_score, "optimization_step": model.optimization_step})
            score, score_count = 0.0, 0

        if n_epi % save_interval == 0 and n_epi != 0:
            checkpoint_path = f"checkpoints/{name}_checkpoint_{n_epi}.pth"
            torch.save(model.state_dict(), checkpoint_path)

    final_checkpoint_path = f"checkpoints/{name}_checkpoint_{n_epi}.pth"
    torch.save(model.state_dict(), final_checkpoint_path)
    env.close()

if __name__ == '__main__':
    main()