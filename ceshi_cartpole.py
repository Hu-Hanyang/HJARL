import os
from safe_control_gym.envs.gym_control.cartpole_distb import CartPoleDistbEnv, CartPoleFixedDistb, CartPoleBoltzDistb, CartPoleRandomDistb, CartPoleNullDistb
# from safe_control_gym.envs.gym_control.cartpole import CartPole
from safe_control_gym.envs.gym_control.cartpole_v0 import CartPole
# from safe_control_gym.envs.gym_control.cartpole_distb import CartPoleHJDistbEnv
import numpy as np
import imageio

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

def transfer(distb_level):
    index = int(distb_level * 10)
    allowable_distb_levels = np.arange(0.0, 2.1, 0.1)
    return allowable_distb_levels[index]


# # Function to create GIF
# def create_gif(image_list, filename, duration=0.1):
#     images = []
#     for img in image_list:
#         images.append(img.astype(np.uint8))  # Convert to uint8 for imageio
#     imageio.mimsave(f'{filename}', images, duration=duration)



# allowable_distb_levels = np.arange(0.0, 2.1, 0.1)
# a = 0.3
# print(f"********* The a={a} is in the allowable distb levels is {a in allowable_distb_levels}. ********* \n")

# # print(f"********* The allowable_distb_levels is {allowable_distb_levels}. ********* \n")
# # print(f"********* The 0.3 is in the  is allowable_distb_levels: {0.3 in allowable_distb_levels}. ********* \n")
# print(np.round(0.3, 2))

# env = CartPole(randomized_init=True, seed=42)
# env = CartPoleFixedDistb()
# env = CartPoleRandomDistb()
# env = CartPoleBoltzDistb()
env = CartPoleNullDistb()
obs = env.reset()
print(f"********* The init_obs is {obs}. ********* \n")
steps = 10
force = 0.1

for _ in range(steps):
    obs, rew, done, info = env.step(force)
    # print(f"********* The obs is {obs}. ********* \n")
    # print(f"********* The rew is {rew}. ********* \n")
    # print(f"********* The done is {done}. ********* \n")
    # print(f"********* The info is {info}. ********* \n")
print(f"********* The obs is {obs}. ********* \n")
print(f"********* The rew is {rew}. ********* \n")
    
    
# obs, rew, done, info = env.step(0.0)
# print(f"********* The obs is {obs}. ********* \n")
# print(f"********* The rew is {rew}. ********* \n")
# print(f"********* The done is {done}. ********* \n")
# print(f"********* The info is {info}. ********* \n")

# print(f"********* The self.at_reset is {env.at_reset}. ********* \n")
# print(f"********* The env NAME is {env.NAME}. ********* \n")

# print(f"********* The obs is {obs.shape}. ********* \n")
# passive_disturb = 'dynamics' in env.disturbances
# print(f"********* The passive_disturb is {passive_disturb}. ********* \n")

# adv_disturb = env.adversary_disturbance == 'dynamics'
# print(f"********* The adv_disturb is {adv_disturb}. ********* \n")

# print(f"********* The self.disturbances is {env.disturbances}. ********* \n")
# print(f"********* The self.adversary_disturbance is {env.adversary_disturbance}. ********* \n")  
# print(f"********* The task is {env.TASK }. ********* \n")
# print(f"********* The self.constraints is {env.constraints}. ********* \n")
# # print(f"The initial position is {env.state[0:3]}. \n")
# print(f"The obs is {env.observation_space}")
# print(f"The action is {env.action_space}")
# # print(f"********** The shape of the observation space is {env.observation_space.shape}.********** \n")
# print(f"********** The DISTURBANCE_MODES is {env.DISTURBANCE_MODES}. ********** \n")
# print(f"********** The enable reset distribution is {env.RANDOMIZED_INIT}. ********** \n")
# print(f"********** The self.DISTURBANCES is {env.DISTURBANCES}. ********** \n")
# print(f"********** The self.COST is {env.COST}. ********** \n")
# print(f"********** The self.out_of_bounds is {env.out_of_bounds}. ********** \n")
# print(f"********** The disturbance type is {env.distb_type}.********** \n")
# print(f"********** The disturbance level is {env.distb_level}. ********** \n")


# env = CartPoleHJDistbEnv()
# # env = CartPoleHJDistbEnv(disturbances={'observation': [{'disturbance_func':'white_noise', 'std': 0.04}]})
# obs = env.reset()
# print(obs)
# # print(f"The DISTURBANCES is {env.DISTURBANCES}. \n")
# # print(f"The RANDOMIZED_INERTIAL_PROP is {env.RANDOMIZED_INERTIAL_PROP}. \n")
# # print(f"The observation space is {env.observation_space}. \n")
# # print(f"The action space is {env.action_space}. \n")
# print(f"The current state is {env.state}. \n")

# # random distb generation test
# low = np.array([-2.0])
# high = np.array([+2.0])
# hj_distb_force = np.random.uniform(low, high)
# print(f"The hj_distb_force is {hj_distb_force}. \n")
# print(f"The hj_distb_force shape is {hj_distb_force.shape}. \n")

# # Generate gifs to check
# num_gifs = 1
# frames = [[] for _ in range(num_gifs)]
# num=0
# while num < num_gifs:
#     terminated, truncated = False, False
#     rewards = 0.0
#     steps = 0
#     max_steps=50
#     init_obs = env.reset()
#     print(f"The init_obs shape is {init_obs.shape}")
#     print(f"The initial position is {init_obs[0:3]}")
#     frames[num].append(env.render())  # the return frame is np.reshape(rgb, (h, w, 4))
    
#     for _ in range(max_steps):
#         if _ == 0:
#             obs = init_obs

#         # Select control
#         # manual control
#         # motor = -0.78
#         # action = np.array([motor, motor, motor, motor])  # shape: (4, )
#         action = 0.00
#         # random control
#         # action = env.action_space.sample()

#         # # load the trained model
#         # ac, trained_env, env_distb = utils.load_actor_critic_and_env_from_disk(ckpt)
#         # ac.eval()
#         # obs = torch.as_tensor(obs, dtype=torch.float32)
#         # action, *_ = ac(obs)

#         obs, reward, done, info = env.step(action)
#         # print(f"The shape of the obs in the output of the env.step is {obs.shape}")
#         # print(f"The current reward of the step{_} is {reward} and this leads to {terminated} and {truncated}")
#         # print(f"The current penalty of the step{_} is {info['current_penalty']} and the current distance is {info['current_dist']}")
#         frames[num].append(env.render())
#         rewards += reward
#         steps += 1
        
#         if done or steps>=max_steps:
#             print(f"[INFO] Test {num} is done with rewards = {rewards} and {steps} steps.")
#             create_gif(frames[num], f'{num}-{env.NAME}-{env.distb_level}distb_level-motor{motor}-obs_noise{env.RANDOMIZED_INIT}-{steps}steps.gif', duration=0.1)
#             # print(f"The final position is {obs[0:3]}.")
#             num += 1
#             break
# env.close()