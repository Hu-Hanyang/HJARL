import os
import tyro
from typing import Callable
import numpy as np
from safe_control_gym.experiments.train_game_cleanrl import Args, layer_init
import gymnasium as gym
import torch
import torch.nn as nn
from datetime import datetime
from torch.distributions.normal import Normal
from safe_control_gym.utils.plotting import animation, current_status_check, record_video
from safe_control_gym.envs.gym_game.ReachAvoidGame import ReachAvoidGameTest

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, set_random_seed
from stable_baselines3.common.monitor import Monitor




map = {'map': [-1., 1., -1., 1.]}  # Hanyang: rectangele [xmin, xmax, ymin, ymax]
des = {'goal0': [0.6, 0.8, 0.1, 0.3]}  # Hanyang: rectangele [xmin, xmax, ymin, ymax]
obstacles = {'obs1': [-0.1, 0.1, -1.0, -0.3], 'obs2': [-0.1, 0.1, 0.3, 1.0]} 


def check_area(state, area):
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


def getAttackersStatus(attackers, defenders, last_status):
        """Returns the current status of all attackers.

        Returns
            ndarray, shape (num_attackers,)

        """
        num_attackers = attackers.shape[0]
        num_defenders = defenders.shape[0]
        new_status = np.zeros(num_attackers)
        current_attacker_state = attackers
        current_defender_state = defenders

        for num in range(num_attackers):
            if last_status[num]:  # attacker has arrived or been captured
                new_status[num] = last_status[num]
            else: # attacker is free last time
                # check if the attacker arrive at the des this time
                if check_area(current_attacker_state[num], des):
                    new_status[num] = 1
                # # check if the attacker gets stuck in the obstacles this time (it won't usually)
                # elif self._check_area(current_attacker_state[num], self.obstacles):
                #     new_status[num] = -1
                #     break
                else:
                    # check if the attacker is captured
                    for j in range(num_defenders):
                        if np.linalg.norm(current_attacker_state[num] - current_defender_state[j]) <= 0.1:
                            print(f"================ The {num} attacker is captured. ================ \n")
                            new_status[num] = -1
                            break

            return new_status


# def make_env(env_id, idx, capture_video, run_name, gamma):
#     def thunk():
#         # if capture_video and idx == 0:
#         #     # env = gym.make(env_id, render_mode="rgb_array")
#         #     env = ReachAvoidTestGame()
#         #     # env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         # else:
#         #     # env = gym.make(env_id)
#         env = ReachAvoidTestGame()
#         # env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
#         # env = gym.wrappers.RecordEpisodeStatistics(env)
#         # env = gym.wrappers.ClipAction(env)
#         # env = gym.wrappers.NormalizeObservation(env)
#         # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
#         # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
#         # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
#         return env

#     return thunk


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
            nn.Tanh(),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            # action = probs.sample()
            action = action_mean.detach()  # Hanyang: test deterministic policy
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    

def test_sb3():
    # Set up env hyperparameters.
    n_env = 8
    env_seed = 2024
    # Setp up algorithm hyperparameters.
    total_timesteps = 1e7
    batch_size = 64
    n_epochs = 15
    n_steps = 2048
    seed = 0
    target_kl = 0.01

    # Load the trained model
    trained_model = os.path.join('training_results', "game/sb3/", f'seed_{env_seed}', f'{total_timesteps}steps/', 'final_model.zip')
    assert os.path.exists(trained_model), f"[ERROR] The trained model {trained_model} does not exist, please check the loading path or train one first."
    model = PPO.load(trained_model)
    
    # Create the environment.
    initial_attackers = np.array([[-0.5, 0.8]])
    initial_defenders = np.array([[0.3, -0.3]])
    
    envs = ReachAvoidGameTest(initial_attacker=initial_attackers, initial_defender=initial_defenders, random_init=False)
    # print(f"The state space of the env is {envs.observation_space}. \n")  # Box(-1.0, 1.0, (1, 4)
    # print(f"The action space of the env is {envs.action_space}. \n")  # Box(-1.0, 1.0, (1, 2)

    # num = 0
    # while num in range(eval_episodes):
    step = 0
    attackers_status = []
    attackers_status.append(np.zeros(1))
    attackers_traj, defenders_traj = [], []

    obs, _ = envs.reset()
    # print(obs.shape)  # (4,)
    initial_obs = obs.copy()
    print(f"========== The initial state is {initial_obs} in the test_game. ========== \n")
    attackers_traj.append(np.array([obs[:2]]))
    defenders_traj.append(np.array([obs[2:]]))
    episodic_returns = []

    for act in range(int(10*200)):
        actions, _ = model.predict(obs, deterministic=False)
        
        # actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        # print(f"Step {act}: the action is {actions}. \n")
        next_obs, reward, terminated, truncated, infos = envs.step(actions)
        step += 1
        print(f"Step {step}: the reward is {reward}. \n")
        attackers_traj.append(np.array([next_obs[:2]]))
        defenders_traj.append(np.array([next_obs[2:]]))
        # print(f"Step {step}: the relative distance is {np.linalg.norm(next_obs[:, :2] - next_obs[:, 2:])}. \n")
        # print(f"Step {step}: the current position of the attacker is {next_obs[:2]}. \n")
        attackers_status.append(getAttackersStatus(np.array([next_obs[:2]]), np.array([next_obs[2:]]), attackers_status[-1]))

        if terminated or truncated:
            break
        else:
            obs = next_obs
    # print(f"================ The {num} game is over at the {step} step ({step / 200} seconds. ================ \n")
    print(f"================ The game is over at the {step} step ({step / 200} seconds. ================ \n")
    current_status_check(attackers_status[-1], step)
    animation(attackers_traj, defenders_traj, attackers_status)
    # record_video(attackers_traj, defenders_traj, attackers_status, filename=f'1vs1_{datetime.now().strftime("%Y.%m.%d_%H:%M")}.mp4', fps=10)



if __name__ == "__main__":

    test_sb3()
    
    # # Check the final result
    # current_status_check(envs.current_attackers_status, step=None)
    # # Visualize the game
    # animation(envs.attackers_traj, envs.defenders_traj, envs.attackers_status)
    