import  numpy as np
import tyro
import gymnasium as gym
from safe_control_gym.envs.gym_game.ReachAvoidGame import ReachAvoidGameEnv, ReachAvoidEasierGame

from stable_baselines3.common.env_checker import check_env


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

obstacles = {'obs1': [-0.1, 0.1, -1.0, -0.3], 'obs2': [-0.1, 0.1, 0.3, 0.6]}
new_obstacles = {'obs1': [100, 100, 100, 100]}

def _check_area(state, area):
        """Check if the state is inside the area.

        Parameters:
            state (np.ndarray): the state to check
            area (dict): the area dictionary to be checked.
        
        Returns:
            bool: True if the state is inside the area, False otherwise.
        """
        x, y = state  # Unpack the state assuming it's a 2D coordinate

        for bounds in area.values():
            x_lower, x_upper, y_lower, y_upper = bounds
            if x_lower <= x <= x_upper and y_lower <= y <= y_upper:
                return True

        return False

# state = np.array([[0.0, 0.5]])
# # print(f"The state is in the obstacle area: {_check_area(state, obstacles)}. \n")
# # print(f"The state is in the new obstacle area: {_check_area(state, new_obstacles)}. \n")
# print(f"The shape of the state is {state.shape}. \n")
# print(f"The shape of the state[0] is {state[0].shape}. \n")
# print(f"The state[0] to list is {state[0].tolist()}. \n")  

# print(f"The defender is in the obstacle area: {_check_area(current_defenders[0], obstacles)}. \n")
# reward = 0.0
# reward += -1.0 if _check_area(current_defenders[0], obstacles) else 0.0

# env = ReachAvoidEasierGame()
# print(f"The obstacles of the ReachAvoidEasierGame is {env.obstacles}. \n")


# # env = ReachAvoidGameEnv(init_type='random')
# # Map boundaries
# map = ([-0.99, 0.99], [-0.99, 0.99])  # The map boundaries
# # Obstacles and target areas
# obstacles = [
#     ([-0.1, 0.1], [-1.0, -0.3]),  # First obstacle
#     ([-0.1, 0.1], [0.3, 0.6])     # Second obstacle
# ]
# target = ([0.6, 0.8], [0.1, 0.3])

# def _is_valid_attacker(pos):
#     x, y = pos
#     # Check map boundaries
#     if not (map[0][0] <= x <= map[0][1] and map[1][0] <= y <= map[1][1]):
#         return False
#     # Check obstacles
#     for (ox, oy) in obstacles:
#         if ox[0] <= x <= ox[1] and oy[0] <= y <= oy[1]:
#             return False
#     # Check target
#     if target[0][0] <= x <= target[0][1] and target[1][0] <= y <= target[1][1]:
#         return False
#     return True
        
# def _is_valid_defender(defender_pos, attacker_pos):
#     x, y = defender_pos
#     # Check map boundaries
#     if not (map[0][0] <= x <= map[0][1] and map[1][0] <= y <= map[1][1]):
#         return False
#     # Check obstacles
#     for (ox, oy) in obstacles:
#         if ox[0] <= x <= ox[1] and oy[0] <= y <= oy[1]:
#             return False
#     # Check the relative distance
#     if np.linalg.norm(defender_pos - attacker_pos) <= 0.10:
#         return False
#     return True

# def _generate_attacker_pos():
#     """Generate a random position for the attacker.
    
#     Returns:
#         attacker_pos (tuple): the attacker position.
#     """
#     while True:
#         attacker_x = np.random.uniform(map[0][0], map[0][1])
#         attacker_y = np.random.uniform(map[1][0], map[1][1])
#         attacker_pos = np.round((attacker_x, attacker_y), 1)
#         if _is_valid_attacker(attacker_pos):
#             break
#     return attacker_pos


# def _generate_random_positions(current_seed, init_player_call_counter):
#     """Generate a random position for the attacker and defender.

#     Args:
#         current_seed (int): the random seed.
#         self.init_player_call_counter (int): the init_player function call counter.
    
#     Returns:
#         attacker_pos (tuple): the attacker position.
#         defender_pos (tuple): the defender position.
#     """
#     np.random.seed(current_seed)
#     # Generate the attacker position
#     attacker_pos = _generate_attacker_pos()
#     # Generate the defender position
#     while True:
#         defender_x = np.random.uniform(map[0][0], map[0][1])
#         defender_y = np.random.uniform(map[1][0], map[1][1])
#         defender_pos = np.round((defender_x, defender_y), 1)
#         if _is_valid_defender(defender_pos, attacker_pos):
#             break
    
#     return attacker_pos, defender_pos


# def _generate_distance_points(current_seed, init_player_call_counter):
#     """Generate the attacker and defender positions based on the relative distance.

#     Args:
#         current_seed (int): the random seed.
#         init_player_call_counter (int): the call counter.

#     Returns:
#         list: the attacker and defender positions.
#     """
#     np.random.seed(current_seed)
#     # Generate the attacker position
#     attacker_pos = _generate_attacker_pos()
    
#     # Determine the distance based on the call counter
#     stage = -1
#     if init_player_call_counter < 3000:  # [0.10, 0.20]
#         distance_init = 0.15
#         r_init = 0.05
#         stage = 0
#     elif init_player_call_counter < 6000:  # [0.20, 0.50]
#         distance_init = 0.35
#         r_init = 0.15
#         stage = 1
#     elif init_player_call_counter < 10000:  # [0.50, 1.00]
#         distance_init = 0.75
#         r_init = 0.25
#         stage = 2
#     elif init_player_call_counter < 15000:  # [1.00, 2.00]
#         distance_init = 1.50
#         r_init = 0.50
#         stage = 3
#     elif init_player_call_counter < 20000:  # [2.00, 2.80]
#         distance_init = 2.40
#         r_init = 0.40
#         stage = 4
#     else:  # [0.10, 2.80]
#         distance_init = 1.45
#         r_init = 1.35
#         stage = 5
#     # Generate the defender position
#     defender_pos = _generate_neighborpoint(attacker_pos, distance_init, r_init)

#     return attacker_pos, defender_pos, stage

# def _generate_neighborpoint(attacker_pos, distance, radius):
#     """
#     Generate a random point within a circle whose center is a specified distance away from a given position.

#     Parameters:
#     attacker_pos (list): The (x, y) coordinates of the initial position.
#     distance (float): The distance from the initial position to the center of the circle.
#     radius (float): The radius of the circle.

#     Returns:
#     defender_pos (list): A random (x, y) point whose relative distance between the input position is .
#     """

#     while True:
#         # Randomly choose an angle to place the circle's center
#         angle = np.random.uniform(0, 2 * np.pi)
        
#         # Determine the center of the circle
#         center_x = attacker_pos[0] + distance * np.cos(angle)
#         center_y = attacker_pos[1] + distance * np.sin(angle)
        
#         # Generate a random point within the circle
#         point_angle = np.random.uniform(0, 2 * np.pi)
#         point_radius = np.sqrt(np.random.uniform(0, 1)) * radius
#         defender_x = center_x + point_radius * np.cos(point_angle)
#         defender_y = center_y + point_radius * np.sin(point_angle)

#         # In case all generated points are outside of the map
#         x_min, x_max, y_min, y_max = map[0][0], map[0][1], map[1][0], map[1][1]
#         defender_x = max(min(defender_x, x_max), x_min)
#         defender_y = max(min(defender_y, y_max), y_min)
#         defender_pos = np.array([defender_x, defender_y])

        
#         if _is_valid_defender(defender_pos, attacker_pos):
#             break

#     return defender_pos

# def _generate_difficulty_points(current_seed, init_player_call_counter):
#     """Generate attacker and defender initial positions based on the difficulty level.
#     difficulty_level 0: there is no obstacle between attacker and defender and the relative distance is [0.10, 0.50];
#     difficulty_level 1: there is no obstacle between attacker and defender and the relative distance is [0.50, 1.50];
#     difficulty_level 2: there is no obstacle between attacker and defender and the relative distance is [1.50, 2.80];
#     difficulty_level 3: there is an obstacle between attacker and defender;

#     Args:
#         difficulty_level (int): the difficulty level of the game, designed by the relative position of obstacles and target areas.
#         seed (int): the initialization random seed.

#     Returns:
#         attacker_pos (list): the initial position of the attacker.
#         defender_pos (list): the initial position of the defender.
#     """
#     np.random.seed(current_seed)
#     # Generate the attacker position
#     difficulty_level = -1
#     if init_player_call_counter < 3000:  # difficulty_level 0, # [0.10, 0.50]
#         difficulty_level = 0
#         distance = 0.30
#         r = 0.20
#         attacker_pos = _generate_attacker_pos()
#         defender_pos = _generate_neighborpoint(attacker_pos, distance, r)
#     elif init_player_call_counter < 8000:  # difficulty_level 1, # [0.50, 1.50]
#         difficulty_level = 1
#         distance = 1.00
#         r = 0.50
#         attacker_pos = _generate_attacker_pos()
#         defender_pos = _generate_neighborpoint(attacker_pos, distance, r)
#     elif init_player_call_counter < 15000:  # difficulty_level 2, # [1.50, 2.80]
#         difficulty_level = 2
#         distance = 2.15
#         r = 0.65
#         attacker_pos = _generate_attacker_pos()
#         defender_pos = _generate_neighborpoint(attacker_pos, distance, r)
#     else:
#         difficulty_level = 3
#         attacker_pos, defender_pos = _generate_obstacle_neighborpoints()
    
#     return attacker_pos, defender_pos, difficulty_level

# def _generate_obstacle_neighborpoints():
#     """Generate the attacker and defender positions near the obstacles.
    
#     Returns:
#         attacker_pos (tuple): the attacker position.
#         defender_pos (tuple): the defender position
#     """
#     # Sample y position from the obstacles
#     y_positions = [obstacles[0][1], obstacles[1][1]]
#     attacker_y = np.random.uniform(*y_positions[np.random.choice(len(y_positions))])
#     defender_y = np.random.uniform(*y_positions[np.random.choice(len(y_positions))])
#     # Sample x position for attacker and defender
#     attacker_x = np.random.uniform(-0.99, -0.15)
#     defender_x = np.random.uniform(0.15, 0.99)
    
#     attacker_pos = np.array([attacker_x, attacker_y])
#     defender_pos = np.array([defender_x, defender_y])

#     return attacker_pos, defender_pos


# # attacker_pos, defender_pos = _generate_random_positions(2, 2)
# # print(f"The attacker position is {attacker_pos} and the defender position is {defender_pos}. \n")

# # attacker_pos, defender_pos, stage = _generate_distance_points(2, 3)
# # print(f"The attacker position is {type(attacker_pos)}, and its sahpe is {attacker_pos.shape}. \n")
# # print(f"The defender position is {type(defender_pos)}, and its sahpe is {defender_pos.shape}. \n")
# # # print(f"The attacker position is {attacker_pos} and the defender position is {defender_pos}. \n")

# # attacker_pos, defender_pos, difficulty_level = _generate_difficulty_points(2, 3)
# attacker_pos, defender_pos, difficulty_level = _generate_difficulty_points(2, 16000)
# print(f"The attacker position is {type(attacker_pos)}. \n")
# print(f"The defender position is {type(defender_pos)}. \n")
# print(f"The attacker position is {attacker_pos} and the defender position is {defender_pos}. \n")
# print(f"========== The relative distance is {np.linalg.norm(attacker_pos - defender_pos):.2f} in BaseGame.py. ========== \n ")


x_values = np.linspace(-1, 1, 100)
print(f"The x_values is {x_values}. \n")
print(f"The shape of the x_values is {x_values.shape}. \n")

x_range = np.arange(-0.95, 1.0, 0.05)
print(f"The x_range is {x_range}. \n")
print(f"The shape of the x_range is {x_range.shape}. \n")