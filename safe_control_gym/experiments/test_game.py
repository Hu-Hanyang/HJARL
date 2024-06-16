import os
import tyro
from typing import Callable
from safe_control_gym.experiments.train_game import Agent, make_env, Args
import gymnasium as gym
import torch
from safe_control_gym.utils.plotting import animation, current_status_check


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int, #TODO: Hanyang: need to change this to only one episode, that is to say, one game.
    save_path: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    gamma: float = 0.99,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, save_path, gamma)])
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    print(f"========== The initial state is {obs}. ========== \n")
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns, envs


if __name__ == "__main__":
    # Load the trained model
    args = tyro.cli(Args)
    args.seed = 2024
    args.total_timesteps = 1e7
    args.exp_name = "train_game.cleanrl_model"
    run_name = os.path.join('training_results/' + 'game/ppo/' +f'{args.seed}/' + f'{args.total_timesteps}' )
    model_path = f"{run_name}/{args.exp_name}.cleanrl_model"
    assert os.path.exists(model_path), f"Model path {model_path} does not exist."
    
    episodic_returns, envs = evaluate(
        model_path,
        make_env,
        "reach_avoid",
        eval_episodes=10,
        save_path=f"eval",
        Model=Agent,
        device="cpu",
        capture_video=False,
    )
    
    # Check the final result
    current_status_check(envs.current_attackers_status, step=None)
    # Visualize the game
    animation(envs.attackers_traj, envs.defenders_traj, envs.attackers_status)
    