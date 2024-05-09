from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_distb import QuadrotorDistb, QuadrotorFixedDistb, QuadrotorBoltzDistb
import numpy as np
import imageio


# Function to create GIF
def create_gif(image_list, filename, duration=0.1):
    images = []
    for img in image_list:
        images.append(img.astype(np.uint8))  # Convert to uint8 for imageio
    imageio.mimsave(f'{filename}', images, duration=duration)



# env = CartPoleHJDistbEnv()
# print(env.reset())

# env = QuadrotorFixedDistb()
env = QuadrotorBoltzDistb()
env.reset()
print(f"********* The self.disturbances is {env.disturbances}. ********* \n")
print(f"********* The self.adversary_disturbance is {env.adversary_disturbance}. ********* \n")  
print(f"********* The task is {env.TASK }. ********* \n")
print(f"********* The self.PHYSICS is {env.PHYSICS}. ********* \n")
print(f"********* The self.constraints is {env.constraints}. ********* \n")
# print(f"The initial position is {env.state[0:3]}. \n")
# print(f"The obs is {env.observation_space}")
# print(f"The action is {env.action_space}")
print(f"********** The shape of the observation space is {env.observation_space.shape}.********** \n")
print(f"********** The disturbance type is {env.distb_type}.********** \n")
# print(f"********** The disturbance level is {env.distb_level}. ********** \n")
print(f"********** The DISTURBANCE_MODES is {env.DISTURBANCE_MODES}. ********** \n")
print(f"********** The self.DISTURBANCES is {env.DISTURBANCES}. ********** \n")
print(f"********** The enable reset distribution is {env.RANDOMIZED_INIT}. ********** \n")

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
#         motor = -0.78
#         action = np.array([motor, motor, motor, motor])  # shape: (4, )
        
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



# env = Quadrotor()
# env.reset()
# # print(f"The observation space is {env.observation_space}.")
# print(f"The shape of the observation space is {env.observation_space.shape}")
# # print(f"The action is {env.action_space}")
# print(f"The shape of action space is {env.action_space.shape}. \n")

# env1 = QuadrotorBoltzDistb()
# env1.reset()
# print(f"The shape of the distb env is {env1.observation_space.shape}")
# print(f"The shape of the distb env is {env1.action_space.shape}")