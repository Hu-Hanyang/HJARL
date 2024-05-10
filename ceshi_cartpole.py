import os
from safe_control_gym.envs.gym_control.cartpole_distb import CartPoleDistbEnv, CartPoleFixedDistb, CartPoleBoltzDistb
from safe_control_gym.envs.gym_control.cartpole import CartPole
# from safe_control_gym.envs.gym_control.cartpole_distb import CartPoleHJDistbEnv
import numpy as np

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

# def transfer(distb_level):
#     index = int(distb_level * 10)
#     allowable_distb_levels = np.arange(0.0, 2.1, 0.1)
#     return allowable_distb_levels[index]


# allowable_distb_levels = np.arange(0.0, 2.1, 0.1)
# a = transfer(0.3)
# print(f"********* The a={a} is in the allowable distb levels is {a in allowable_distb_levels}. ********* \n")

# # print(f"********* The allowable_distb_levels is {allowable_distb_levels}. ********* \n")
# # print(f"********* The 0.3 is in the  is allowable_distb_levels: {0.3 in allowable_distb_levels}. ********* \n")
# print(np.round(0.3, 2))
env = CartPoleFixedDistb()
print(f"********* The self.distb_level is {env.distb_level}. ********* \n")
obs = env.reset()
print(f"********* The self.disturbances is {env.disturbances}. ********* \n")
print(f"********* The self.adversary_disturbance is {env.adversary_disturbance}. ********* \n")  
print(f"********* The task is {env.TASK }. ********* \n")
print(f"********* The self.constraints is {env.constraints}. ********* \n")
# print(f"The initial position is {env.state[0:3]}. \n")
# print(f"The obs is {env.observation_space}")
# print(f"The action is {env.action_space}")
print(f"********** The shape of the observation space is {env.observation_space.shape}.********** \n")
print(f"********** The disturbance type is {env.distb_type}.********** \n")
# print(f"********** The disturbance level is {env.distb_level}. ********** \n")
print(f"********** The DISTURBANCE_MODES is {env.DISTURBANCE_MODES}. ********** \n")
print(f"********** The enable reset distribution is {env.RANDOMIZED_INIT}. ********** \n")
print(f"********** The self.DISTURBANCES is {env.DISTURBANCES}. ********** \n")


# env = CartPoleHJDistbEnv()
# # env = CartPoleHJDistbEnv(disturbances={'observation': [{'disturbance_func':'white_noise', 'std': 0.04}]})
# obs = env.reset()
# print(obs)
# # print(f"The DISTURBANCES is {env.DISTURBANCES}. \n")
# # print(f"The RANDOMIZED_INERTIAL_PROP is {env.RANDOMIZED_INERTIAL_PROP}. \n")
# # print(f"The observation space is {env.observation_space}. \n")
# # print(f"The action space is {env.action_space}. \n")
# print(f"The current state is {env.state}. \n")