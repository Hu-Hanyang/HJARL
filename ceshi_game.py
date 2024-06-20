import  numpy as np
import tyro
import gymnasium as gym
from safe_control_gym.envs.gym_game.ReachAvoidGame import ReachAvoidGameEnv, ReachAvoidTestGame
from safe_control_gym.experiments.train_game import Args



def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        # if capture_video and idx == 0:
        #     # env = gym.make(env_id, render_mode="rgb_array")
        #     env = ReachAvoidGameEnv()
        #     env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # else:
        #     # env = gym.make(env_id)
        env = ReachAvoidTestGame()
        # env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk

args = tyro.cli(Args)
# env setup
# envs = gym.vector.SyncVectorEnv(
#     [make_env(args.env_id, i, args.capture_video, "ceshi_game", args.gamma) for i in range(1)])

# env = gym.vector.SyncVectorEnv([make_env(args.env_id, 0, args.capture_video, "ceshi_game", args.gamma)])
env = ReachAvoidTestGame()

# obs, info = env.reset()
# print(f"The obs is {obs}. \n")

# initial_attacker = np.array([[-0.1, 0.0]])
# initial_defender = np.array([[0.0, 0.0]])
# env = ReachAvoidGameEnv(initial_attacker=initial_attacker, initial_defender=initial_defender, random_init=False)
# print(f"The initial state is {env.}. \n")
# obs, info = env.reset()
print(f"The state space of the env is {env.observation_space}. \n")
print(f"The action space of the env is {env.action_space}. \n")
# print(f"The {envs.state}")

# print(f"The obs is {obs} and the shape of the obs is {obs.shape}. \n")
# print(f"The state of the attacker is {env.state[0]} and the state of the defender is {env.state[1]}. \n")

# print(f"The state space of the env is {env.observation_space}. \n")
# print(f"The action space of the env is {env.action_space}. \n")
# print(f"The obs is {obs}. \n")
# print(f"The state of the attacker is {env.state[0]} and the state of the defender is {env.state[1]}. \n")

# for i in range(10):
#     obs = env.reset()
#     print(f"The initial player seed is {env.initial_players_seed}. \n")
#     print(f"The obs is {obs}. \n")


# action = np.array([-0.1, 0.0])

# for i in range(10):
#     obs, reward, terminated, truncated, info = env.step(action)
#     print(f"The obs is {obs}. \n")
#     print(f"The reward is {reward}. \n")
#     # print(f"The state of the attacker is {env.state[0]} and the state of the defender is {env.state[1]}. \n")
#     if terminated or truncated:
#         print(f"The game is terminated: {terminated} and the game is truncated: {truncated} at the step {i}. \n")
#         break

# current_attackers = np.array([[0.0, 0.0]])
# current_defenders = np.array([[3.0, 4.0]])
# distance = np.linalg.norm(current_attackers[0] - current_defenders[0])
# print(f"The distance between the attacker and the defender is {distance}. \n")

# obstacles = {'obs1': [-0.1, 0.1, -1.0, -0.3], 'obs2': [-0.1, 0.1, 0.3, 0.6]}

# def _check_area(state, area):
#         """Check if the state is inside the area.

#         Parameters:
#             state (np.ndarray): the state to check
#             area (dict): the area dictionary to be checked.
        
#         Returns:
#             bool: True if the state is inside the area, False otherwise.
#         """
#         x, y = state  # Unpack the state assuming it's a 2D coordinate

#         for bounds in area.values():
#             x_lower, x_upper, y_lower, y_upper = bounds
#             if x_lower <= x <= x_upper and y_lower <= y <= y_upper:
#                 return True

#         return False

# print(f"The defender is in the obstacle area: {_check_area(current_defenders[0], obstacles)}. \n")
# reward = 0.0
# reward += -1.0 if _check_area(current_defenders[0], obstacles) else 0.0