import  numpy as np
import math
import tyro
import gymnasium as gym
from safe_control_gym.envs.gym_game.ReachAvoidGame import ReachAvoidGameEnv, ReachAvoidEasierGame
from safe_control_gym.envs.gym_game.DubinGame import DubinReachAvoidEasierGame
from safe_control_gym.envs.gym_game.RARLGame import RARLGameEnv

from stable_baselines3.common.env_checker import check_env


# Map boundaries
map = ([-0.99, 0.99], [-0.99, 0.99])  # The map boundaries
# Obstacles and target areas
obstacles = [
    ([-0.1, 0.1], [-1.0, -0.3]),  # First obstacle
    ([-0.1, 0.1], [0.3, 0.6])     # Second obstacle
]
target = ([0.6, 0.8], [0.1, 0.3])
des={'goal0': [0.6, 0.8, 0.1, 0.3]}


env = DubinReachAvoidEasierGame()
# env = ReachAvoidEasierGame()
print(env.observation_space)
print(env.action_space)
print(env.state)

