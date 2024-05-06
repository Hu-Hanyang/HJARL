from safe_control_gym.envs.gym_control.cartpole_distb import CartPoleHJDistbEnv
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_distb import QuadrotorDistb, QuadrotorFixedDistb, QuadrotorBoltzDistb
import numpy as np
from gymnasium import spaces


# env = CartPoleHJDistbEnv()
# print(env.reset())

# env = QuadrotorBoltzDistb()
# env.reset()
# print(f"The obs is {env.observation_space}")
# print(f"The action is {env.action_space}")


lo = -np.inf
hi = np.inf

obs_lower_bound = np.array([lo,lo,0, lo,lo,lo,lo, lo,lo,lo, lo,lo,lo] )
obs_upper_bound = np.array([hi,hi,hi, hi,hi,hi,hi, hi,hi,hi, hi,hi,hi] )
#### Add action buffer to observation space ################
act_lo = -1
act_hi = +1
obs_lower_bound = np.hstack([obs_lower_bound, np.array([act_lo,act_lo,act_lo,act_lo])])
obs_upper_bound = np.hstack([obs_upper_bound, np.array([act_hi,act_hi,act_hi,act_hi])])

state_space = spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

action_dim = 4
act_lower_bound = np.array(-1*np.ones(action_dim))
act_upper_bound = np.array(+1*np.ones(action_dim))
# Hanyang: define the action space for 6D quadrotor
action_space = spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

print(f"The action space is {action_space}")
print(f"The state space is {state_space}")
