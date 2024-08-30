import  numpy as np
import math
import tyro
import gymnasium as gym
from safe_control_gym.envs.gym_game.ReachAvoidGame import ReachAvoidGameEnv, ReachAvoidEasierGame
from safe_control_gym.envs.gym_game.DubinGame import DubinReachAvoidEasierGame
from safe_control_gym.envs.gym_game.RARLGame import RARLGameEnv
from safe_control_gym.envs.gym_game.DubinRARLGame import DubinRARLGameEnv

from stable_baselines3.common.env_checker import check_env


# Map boundaries
map = ([-0.99, 0.99], [-0.99, 0.99])  # The map boundaries
# Obstacles and target areas
obstacles = {'obs1': [100, 100, 100, 100]}
target = ([0.6, 0.8], [0.1, 0.3])
des={'goal0': [0.6, 0.8, 0.1, 0.3]}


# env = DubinReachAvoidEasierGame()
# env = ReachAvoidEasierGame()
env = RARLGameEnv()
# env = DubinRARLGameEnv()
obs = env.reset()

# print(env.observation_space)
# print(env.action_space)
# print(env.state.shape)
print(env.adversary_disturbance)
# print(obs.shape)
# print(env.PYB_FREQ)
# print(env.CTRL_FREQ)
# print(env.frequency)

# for i in range(10):
#     obs, info = env.reset()
#     print(env.state)

# initial_attacker=np.array([[-0.7, 0.5, -1.0]])  # Hanyang: shape (1, 3)
# initial_defender=np.array([[0.7, -0.5, 1.00]])  # Hanyang: shape (1, 3)

# # current_state = np.vstack((initial_attacker, initial_defender))
# current_state = np.concatenate((initial_attacker, initial_defender))
# print(current_state.shape)
