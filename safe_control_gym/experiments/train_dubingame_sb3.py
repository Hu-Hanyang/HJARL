'''Template training/plotting/testing script.'''

import os
import time
import torch
import argparse

from safe_control_gym.envs.gym_game.DubinGame import DubinReachAvoidEasierGame

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


class SaveEveryXStepsCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super(SaveEveryXStepsCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path+'/checkpoints/'

    def _init_callback(self):
        # Create the save directory if it does not exist
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            # Save the model
            model_path = os.path.join(self.save_path, f'model_{self.n_calls}_steps.zip')
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved at {self.n_calls} steps")
        return True
    


def train_game(init_type, total_steps):
    # Set up env hyperparameters.
    n_env = 2
    env_seed = 42
    # Setp up algorithm hyperparameters.
    total_timesteps = total_steps
    batch_size = 64
    n_epochs = 15
    n_steps = 2048  #TODO: test this to 300 also
    policy_seed = 42
    target_kl = 0.01
    # Set up saving directory.
    env = DubinReachAvoidEasierGame()
    assert env.init_type == init_type, f"init_type is not matched. The env.init_type is {env.init_type}, but the input is {init_type}."
    
    filename = os.path.join('training_results', f"dubin_game/sb3/{init_type}_init/", f'seed_{env_seed}', f'{total_timesteps}steps')

    # Create the environment.
    train_env = make_vec_env(DubinReachAvoidEasierGame, 
                             n_envs=n_env, 
                             seed=env_seed)
    print(f"==============The environment is ready.============== \n")
    # Create the model.
    model = PPO('MlpPolicy',
            train_env,
            batch_size=batch_size,
            n_epochs=n_epochs,
            n_steps=n_steps,
            seed=policy_seed,
            target_kl=target_kl, 
            tensorboard_log=filename+'/tb/',
            verbose=1)
    
    print(f"==============The model is ready.============== \n")
    # Create the callback list.
    save_freq = 1000000
    save_path = filename
    save_callback = SaveEveryXStepsCallback(save_freq, save_path, verbose=1)

    #### Train the model #######################################
    print("Start training")
    start_time = time.perf_counter()
    model.learn(total_timesteps=int(total_timesteps), callback=save_callback)

    #### Save the model ########################################
    model.save(filename+'/final_model.zip')
    print(f"========== The model is saved at {filename}. ========== \n")
    duration = time.perf_counter() - start_time
    print(f"========== The time of training is {duration//3600}hours-{(duration%3600)//60}minutes-{(duration%3600)%60}seconds. ========== \n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    # parser.add_argument('--optimality',           default="1vs0",        type=str,           help='The initilaization method (default: random)', metavar='')
    parser.add_argument('--init_type',           default="random",        type=str,           help='The initilaization method (default: random)', metavar='')
    parser.add_argument('--total_steps',         default=1e7,             type=float,         help='The total training steps (default: 2e7)', metavar='')
    
    args = parser.parse_args()
    
    train_game(init_type=args.init_type, total_steps=args.total_steps)
    # python safe_control_gym/experiments/train_dubingame_sb3.py --init_type random --total_steps 1e7
    # python safe_control_gym/experiments/train_dubingame_sb3.py --init_type random --total_steps 2e7

