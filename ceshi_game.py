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
# env = ReachAvoidTestGame()

# obs, info = env.reset()
# print(f"The obs is {obs}. \n")

# initial_attacker = np.array([[-0.1, 0.0]])
# initial_defender = np.array([[0.0, 0.0]])
# env = ReachAvoidGameEnv(initial_attacker=initial_attacker, initial_defender=initial_defender, random_init=False)
# print(f"The initial state is {env.}. \n")
# obs, info = env.reset()
# print(f"The state space of the env is {env.observation_space}. \n")
# print(f"The action space of the env is {env.action_space}. \n")
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



 # Map boundaries
min_val, max_val = -0.9, 0.9

# Obstacles and target areas
obstacles = [
([-0.1, 0.1], [-1.0, -0.3]),  # First obstacle
([-0.1, 0.1], [0.3, 0.6])     # Second obstacle
]
target = ([0.6, 0.8], [0.1, 0.3])

def is_valid_position(pos):
    x, y = pos
    # Check boundaries
    if not (min_val <= x <= max_val and min_val <= y <= max_val):
        return False
    # Check obstacles
    for (ox, oy) in obstacles:
        if ox[0] <= x <= ox[1] and oy[0] <= y <= oy[1]:
            return False
    # Check target
    if target[0][0] <= x <= target[0][1] and target[1][0] <= y <= target[1][1]:
        return False
    return True

def generate_position(current_seed):
    np.random.seed(current_seed)
    while True:
        pos = np.round(np.random.uniform(min_val, max_val, 2), 1)
        if is_valid_position(pos):
            return pos

def generate_neighborpoint(position, distance, radius, seed):
    """
    Generate a random point within a circle whose center is a specified distance away from a given position.

    Parameters:
    position (tuple): The (x, y) coordinates of the initial position.
    distance (float): The distance from the initial position to the center of the circle.
    radius (float): The radius of the circle.
    seed (int): The random seed.

    Returns:
    tuple: A random (x, y) point within the specified circle.
    """
    np.random.seed(seed)
    while True:
        # Randomly choose an angle to place the circle's center
        angle = np.random.uniform(0, 2 * np.pi)
        
        # Determine the center of the circle
        center_x = position[0] + distance * np.cos(angle)
        center_y = position[1] + distance * np.sin(angle)
        
        # Generate a random point within the circle
        point_angle = np.random.uniform(0, 2 * np.pi)
        point_radius = np.sqrt(np.random.uniform(0, 1)) * radius
        point_x = center_x + point_radius * np.cos(point_angle)
        point_y = center_y + point_radius * np.sin(point_angle)

        if is_valid_position((point_x, point_y)):
            return (point_x, point_y)

# attacker = np.array([[0.0, 0.0]])
attacker = np.round(np.random.uniform(min_val, max_val, 2), 1)
print(f"The attacker is {attacker}. \n")
defender = generate_neighborpoint(attacker, 0.5, 0.1, 0)
print(f"The defender is {defender}. \n")
print(f"The distance between the attacker and the defender is {np.linalg.norm(attacker - defender)}. \n")