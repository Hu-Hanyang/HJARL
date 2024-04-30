import os
from safe_control_gym.envs.gym_control.cartpole import CartPole
# from safe_control_gym.envs.gym_control.cartpole_distb import CartPoleHJDistbEnv

"""
Original CartPole Disturbances Setting:
disturbances:
    observation:
    - disturbance_func: white_noise
        std: 4.0
    action:
    - disturbance_func: white_noise
        std: 4.0
"""

env = CartPole(disturbances={'observation': [{'disturbance_func':'white_noise', 'std': 0.04}], 
                             'action': [{'disturbance_func':'white_noise', 'std': 4.0}]})
obs = env.reset()
# print(obs)
print(f"The observation space is {env.observation_space}. \n")
# print(env.TASK)  # Task.STABILIZATION
# print(f"The DISTURBANCES is {env.DISTURBANCES}. \n")
# print(env.adversary_disturbance)
print(f"The RANDOMIZED_INERTIAL_PROP is {env.RANDOMIZED_INERTIAL_PROP}. \n")
# print(env.RANDOMIZED_INIT)
# print(env.INFO_IN_RESET)
print(f"The cost used is {env.COST}. \n")


# env = CartPoleHJDistbEnv()
# # env = CartPoleHJDistbEnv(disturbances={'observation': [{'disturbance_func':'white_noise', 'std': 0.04}]})
# obs = env.reset()
# print(obs)
# # print(f"The DISTURBANCES is {env.DISTURBANCES}. \n")
# # print(f"The RANDOMIZED_INERTIAL_PROP is {env.RANDOMIZED_INERTIAL_PROP}. \n")
# # print(f"The observation space is {env.observation_space}. \n")
# # print(f"The action space is {env.action_space}. \n")
# print(f"The current state is {env.state}. \n")