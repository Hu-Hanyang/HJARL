import  numpy as np
import math
import tyro
import gymnasium as gym
from safe_control_gym.envs.gym_game.ReachAvoidGame import ReachAvoidGameEnv, ReachAvoidEasierGame
from safe_control_gym.envs.gym_game.DubinGame import DubinReachAvoidEasierGame
from safe_control_gym.envs.gym_game.RARLGame import RARLGameEnv
from safe_control_gym.envs.gym_game.DubinRARLGame import DubinRARLGameEnv
from odp.Grid import Grid

from stable_baselines3.common.env_checker import check_env


# Step 0 initilize the map 
map = {'map': [-1., 1., -1., 1.]}  # Hanyang: rectangele [xmin, xmax, ymin, ymax]
des = {'goal0': [0.6, 0.8, 0.1, 0.3]}  # Hanyang: rectangele [xmin, xmax, ymin, ymax]
obstacles = {'obs1': [100, 100, 100, 100]}  # Hanyang: rectangele [xmin, xmax, ymin, ymax]
# Step 1 load the value function, initilize the grids
# value1vs0 = np.load('safe_control_gym/envs/gym_game/values/1vs0Dubin_easier.npy')
value1vs1 = np.load('safe_control_gym/envs/gym_game/values/1vs1Dubin_easier.npy')
# grid1vs0 = Grid(np.array([-1.1, -1.1, -math.pi]), np.array([1.1, 1.1, math.pi]), 3, np.array([100, 100, 200]), [2])
grid1vs1 = Grid(np.array([-1.1, -1.1, -math.pi, -1.1, -1.1, -math.pi]), np.array([1.1, 1.1, math.pi, 1.1, 1.1, math.pi]), 
                        6, np.array([28, 28, 28, 28, 28, 28]), [2, 5])

attackers = np.array([[-0.67, -0.67, 1.57], [-0.77, -0.78, 1.57], [-0.77, -0.72, 1.57], [-0.77, 0.02, 0.02], 
                              [-0.64, -0.7, 1.56], [0.05, -0.74, 1.55], [-0.61, 0.71, -np.pi/2], [-0.02, 0.76, 0.02]])  # Hanyang: shape (8, 3)
defenders = np.array([[0.72, 0.73, -np.pi/2], [0.7, 0.79, -np.pi/2], [0.68, -0.72, 1.57], [0.68, -0.72, 1.57],
                              [-0.69, 0.71, 0.02], [0.64, 0.75, -np.pi/2], [-0.64, -0.67, 1.57], [-0.67, -0.10, -np.pi/2]])


num_experiments = 8

for i in range(num_experiments):
    joint_slice = grid1vs1.get_index(np.concatenate((attackers[i], defenders[i])))
    print(f"The {i+1}th initial value is {value1vs1[joint_slice]}. \n")
    


# env = DubinReachAvoidEasierGame()
# env = ReachAvoidGameEnv()
# env = ReachAvoidEasierGame()
# env = RARLGameEnv()
# env = DubinRARLGameEnv()
# obs = env.reset()

# print(env.observation_space)
# print(env.action_space)
# print(env.state.shape)
# print(env.adversary_disturbance)
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
