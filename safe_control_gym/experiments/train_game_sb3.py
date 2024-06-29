'''Template training/plotting/testing script.'''

import os
import shutil
from functools import partial

import munch
import yaml
import time

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.plotting import plot_from_logs
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs, set_device_from_config, set_seed_from_config

from safe_control_gym.envs.gym_game.ReachAvoidGame import ReachAvoidGameEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, set_random_seed
from stable_baselines3.common.monitor import Monitor



def train_game():
    # Set up env hyperparameters.
    n_env = 8
    env_seed = 2024
    # Setp up algorithm hyperparameters.
    total_timesteps = 2e7
    batch_size = 64
    n_epochs = 15
    n_steps = 2048
    seed = 0
    target_kl = 0.01
    # Set up saving directory.
    filename = os.path.join('training_results', "game/sb3/", f'seed_{env_seed}', f'{total_timesteps}steps')

    # Create the environment.
    train_env = make_vec_env(ReachAvoidGameEnv, 
                             n_envs=n_env, 
                             seed=env_seed)
    print(f"==============The environment is ready.============== \n")
    # Create the model.
    model = PPO('MlpPolicy',
            train_env,
            batch_size=batch_size,
            n_epochs=n_epochs,
            n_steps=n_steps,
            seed=seed,
            target_kl=target_kl, 
            tensorboard_log=filename+'/tb/',
            verbose=1)

    #### Train the model #######################################
    print("Start training")
    start_time = time.perf_counter()
    model.learn(total_timesteps=int(total_timesteps))

    #### Save the model ########################################
    model.save(filename+'/final_model.zip')
    print(filename)
    duration = time.perf_counter() - start_time
    print(f"The time of training is {duration//3600}hours-{(duration%3600)//60}minutes-{(duration%3600)%60}seconds. \n")


if __name__ == '__main__':
    train_game()
