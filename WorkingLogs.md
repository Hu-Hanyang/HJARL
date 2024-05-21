# Working Logs

## Training 
The training file lies in `safe_control_gym/experiments/train_rl_controller.py`. When run this file, some necessary arguments need to be added: 
1. `--task` the task environment, now we support two main envs: `cartpole` and `quadrotor_distb`, details can be found in the following table;
2. `--algo` the training algorithm, now we support `ppo`, `rarl`, and `rap`;
3. `--use_gpu` the device used in training, should be `True` for using GPU;
4. `--seed` the training random seed used in training.

For instance, the training command for the env `cartpole_boltz` with algo `ppo` is:
`python safe_control_gym/experiments/train_rl_controller.py --task cartpole_distb --algo ppo --use_gpu True --seed 42`. 

python safe_control_gym/experiments/train_rl_controller.py --task cartpole_fixed --algo ppo --use_gpu True --seed 42

python safe_control_gym/experiments/train_rl_controller.py --task quadrotor_boltz --algo ppo --use_gpu True --seed 42

python safe_control_gym/experiments/train_rl_controller.py --task quadrotor_boltz --algo ppo --use_gpu True --seed 40026

python safe_control_gym/experiments/train_rl_controller.py --task quadrotor_null --algo rarl --use_gpu True --seed 42

When training env with 'fixed' disturbances, add another argument `--distb_level 1.0` and revise the distb_level in the corresponding env.

python safe_control_gym/experiments/train_rl_controller.py --task cartpole_fixed --algo ppo --distb_level 1.0 --use_gpu True --seed 42


Table: Details of the envs
| Env Name  | Description | 
| --------- | ----------- |
| cartpole  | The cartpole env with no disturbance |
| cartpole_boltz | The cartpole env with Boltzman distributed disturbance |
| cartpole_fixed | The cartpole env with constant HJ disturbance |
| quadrotor_null| The quadrotor env with no disturbance |
| quadrotor_boltz | The quadrotor env with Boltzman distributed disturbance |
| quadrotor_fixed | The quadrotor env with constant HJ disturbance |

## Test
### Need to make the ppo.yaml file consistent with the training file before test
The test file lies in `safe_control_gym/experiments/test_rl_controller.py`. When run this file, some necessary arguments need to be added: 
1. `--trained_task` the trained env we want to load the model from;
2. `--algo` the trained algorithm we want to load the model used, now we support `ppo` and `rarl`;
3. `--task` the test environment, now we support `cartpole`, `cartpole_distb` and `quadrotor_distb`(not sure);
4. `--seed` the training random seed used in training.

For instance, we want to test the trained algo `rarl` in the env `cartpole` in the test env `cartpole_distb`, the test command is:
`python safe_control_gym/experiments/test_rl_controller.py --trained_task cartpole --algo rarl --task cartpole_distb --seed 42`. 

When test env with 'fixed' disturbances, add another two arguments `--trained_distb_level 1.0` and `--test_distb_level 1.0` to specify the test model and test env. Also remember to revise the distb_level in the corresponding env.

python safe_control_gym/experiments/test_rl_controller.py --trained_task cartpole_boltz --algo ppo --task cartpole_fixed --seed 42

python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_boltz --algo ppo --task quadrotor_fixed --seed 42

python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_boltz --algo ppo --task quadrotor_null --seed 42

python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_boltz --algo ppo --task quadrotor_fixed --test_distb_level 1.0 --seed 42

python safe_control_gym/experiments/test_rl_controller.py --trained_task cartpole --algo rap --task cartpole_fixed --seed 42 

python safe_control_gym/experiments/test_rl_controller.py --task cartpole_fixed --algo ppo --test_distb_level 1.5 --trained_distb_level 1.0 --use_gpu True --seed 42

## Env Info


