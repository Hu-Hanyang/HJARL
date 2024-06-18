import os
import tyro
from typing import Callable
import numpy as np
from safe_control_gym.experiments.train_game import Args, layer_init
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from safe_control_gym.utils.plotting import animation, current_status_check
from safe_control_gym.envs.gym_game.ReachAvoidGame import ReachAvoidTestGame


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


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        # if capture_video and idx == 0:
        #     # env = gym.make(env_id, render_mode="rgb_array")
        #     env = ReachAvoidTestGame()
        #     # env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
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
    

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int, #TODO: Hanyang: need to change this to only one episode, that is to say, one game.
    save_path: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
    gamma: float = 0.99,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, save_path, gamma)])
    # print(f"The state space of the env is {envs.observation_space}. \n")
    # print(f"The action space of the env is {envs.action_space}. \n")
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    # num = 0
    # while num in range(eval_episodes):
    step = 0
    attackers_status = []
    attackers_status.append(np.zeros(1))
    attackers_traj, defenders_traj = [], []

    obs, _ = envs.reset()
    print(f"========== The initial state is {obs} in the test_game. ========== \n")
    attackers_traj.append(obs[:, :2])
    defenders_traj.append(obs[:, 2:])
    episodic_returns = []

    for _ in range(int(10*200)):
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        # print(f"Step {_} action is {actions}. \n")
        next_obs, reward, terminated, truncated, infos = envs.step(actions.cpu().numpy())
        step += 1
        attackers_traj.append(next_obs[:, :2])
        defenders_traj.append(next_obs[:, 2:])
        attackers_status.append(getAttackersStatus(next_obs[:, :2], next_obs[:, 2:], attackers_status[-1]))

        if terminated[0] or truncated[0]:
            break
        else:
            obs = next_obs
    # print(f"================ The {num} game is over at the {step} step ({step / 200} seconds. ================ \n")
    print(f"================ The game is over at the {step} step ({step / 200} seconds. ================ \n")
    current_status_check(attackers_status[-1], step)
    animation(attackers_traj, defenders_traj, attackers_status)
        
    return episodic_returns, envs


if __name__ == "__main__":
    # Load the trained model
    args = tyro.cli(Args)
    args.seed = 2024
    args.total_timesteps = 1e7
    args.exp_name = "train_game.cleanrl_model"
    run_name = os.path.join('training_results/' + 'game/ppo/' +f'{args.seed}/' + f'{args.total_timesteps}' )
    model_path = f"{run_name}/{args.exp_name}"
    assert os.path.exists(model_path), f"Model path {model_path} does not exist."
    
    episodic_returns, envs = evaluate(
        model_path,
        make_env,
        "reach_avoid",
        eval_episodes=1,
        save_path=f"eval",
        Model=Agent,
        device="cpu",
        capture_video=False,
    )
    
    # # Check the final result
    # current_status_check(envs.current_attackers_status, step=None)
    # # Visualize the game
    # animation(envs.attackers_traj, envs.defenders_traj, envs.attackers_status)
    