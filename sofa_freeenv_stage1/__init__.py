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