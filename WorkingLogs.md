# Working Logs

## Environment and Algorithm Info
Path: `safe_control_gym/envs`
### CartPole and its derivatives
Relative path: `/gym_control`
| Env Name  | Description | Characteristic | 
| --------- | ----------- |  ----------- |
| cartpole  | The original cartpole env with no disturbance | Baseline for rap and rarl | 
| cartpole_boltz | The cartpole env with Boltzman distributed disturbance | Combined with ppo |
| cartpole_fixed | The cartpole env with constant HJ disturbance | Need manually tune the distb_level, combined with ppo | 
| cartpole_random | The cartpole env with bounded random disturbance | Need manually set bounds, combined with ppo | 

### Quadrotor and its derivatives
Relative path: `/gym_pybullet_drones`
| Env Name  | Description | Characteristic | 
| --------- | ----------- |  ----------- |
| quadrotor_null| The quadrotor env with no disturbance | Baseline for rap and rarl |
| quadrotor_boltz | The quadrotor env with Boltzman distributed disturbance | Combined with ppo | 
| quadrotor_fixed | The quadrotor env with constant HJ disturbance | Need manually tune the distb_level, combined with ppo |
| quadrotor_random | The quadrotor env with bounded random disturbance | Need manually set bounds, combined with ppo | 


## Training
File path:  `safe_control_gym/experiments/train_rl_controller.py`. 
Some necessary arguments need to be added: 
1. `--task` the task environment, now we support two main envs: `cartpole` and `quadrotor_distb`, details can be found in the following table;
2. `--algo` the training algorithm, now we support `ppo`, `rarl`, and `rap`;
3. `--use_gpu` the device used in training, should be `True` for using GPU;
4. `--seed` the training random seed used in training;
5. `--distb_level 1.0` when training env with 'fixed' disturbances, add another argument `--distb_level 1.0` and revise the distb_level in the corresponding env.

**Working procedures**
1. Select the `env` and `algo` to be trained, if it's a fixed env, then choose the `distb_level` in the `env`;
2. Configure the `algo.yaml` file (path: `safe_control_gym/controllers/algo/algo.yaml`) corresponding to the selected `env` (copy the `algo_env.yaml` content to the `algo.yaml` file);
3. Choose other necessary arguments. 

### CartPole and its derivatives
| env | algo  | algo_env.yaml | commands | else
| --------- | ----------- |  ----------- | ----------- | ----------- |
| `cartpole`| `ppo` | `ppo_cartpole.yaml` | `python safe_control_gym/experiments/train_rl_controller.py --task cartpole --algo ppo --use_gpu True --seed 42` | Baseline0 |
| `cartpole_boltz`| `ppo` | `ppo_cartpole.yaml` | `python safe_control_gym/experiments/train_rl_controller.py --task cartpole_boltz --algo ppo --use_gpu True --seed 42` | Our proposed method |
| `cartpole_fixed`| `ppo` | `ppo_cartpole.yaml` | `python safe_control_gym/experiments/train_rl_controller.py --task cartpole_fixed --algo ppo --use_gpu True --seed 42 --distb_level 1.0` | Baseline3, take care of the distb_level |
| `cartpole_random`| `ppo` | `ppo_cartpole.yaml` | `python safe_control_gym/experiments/train_rl_controller.py --task cartpole_random --algo ppo --use_gpu True --seed 42` | Baseline4 (not trained)|
| `cartpole`| `rarl` | `rarl_cartpole.yaml` | `python safe_control_gym/experiments/train_rl_controller.py --task cartpole --algo rarl --use_gpu True --seed 42` | Baseline1 |
| `cartpole`| `rap` | `rap_cartpole.yaml` | `python safe_control_gym/experiments/train_rl_controller.py --task cartpole --algo rap --use_gpu True --seed 42` | Baseline2 |


python safe_control_gym/experiments/train_rl_controller.py --task cartpole --algo ppo --use_gpu True --seed 42
python safe_control_gym/experiments/train_rl_controller.py --task cartpole_null --algo ppo --use_gpu True --seed 42
python safe_control_gym/experiments/train_rl_controller.py --task cartpole_null --algo rarl --use_gpu True --seed 42

### Quadrotor and its derivatives
| env | algo  | algo_env.yaml | commands | else
| --------- | ----------- |  ----------- | ----------- | ----------- |
| `quadrotor_null`| `ppo` | `ppo_quadrotor.yaml` | `python safe_control_gym/experiments/train_rl_controller.py --task quadrotor_null --algo ppo --use_gpu True --seed 42` | Baseline0 |
| `quadrotor_null`| `rarl` | `rarl_quadrotor.yaml` | `python safe_control_gym/experiments/train_rl_controller.py --task quadrotor_null --algo rarl --use_gpu True --seed 42` | Baseline1 |
| `quadrotor_null`| `rap` | `rap_quadrotor.yaml` | `python safe_control_gym/experiments/train_rl_controller.py --task quadrotor_null --algo rap --use_gpu True --seed 42` | Baseline2 |
| `quadrotor_boltz`| `ppo` | `ppo_quadrotor.yaml` | `python safe_control_gym/experiments/train_rl_controller.py --task quadrotor_boltz --algo ppo --use_gpu True --seed 42` | Our proposed method |
| `quadrotor_fixed`| `ppo` | `ppo_quadrotor.yaml` | `python safe_control_gym/experiments/train_rl_controller.py --task quadrotor_fixed --algo ppo --use_gpu True --seed 42 --distb_level 1.0` | Baseline4, take care of the distb_level |
| `quadrotor_random`| `ppo` | `ppo_quadrotor.yaml` | `python safe_control_gym/experiments/train_rl_controller.py --task quadrotor_random --algo ppo --use_gpu True --seed 42` | Baseline5 (not trained)|


## Test
File path:  `safe_control_gym/experiments/test_rl_controller.py` and `safe_control_gym/experiments/test_hj_controller.py`.
Some necessary arguments need to be added: 
1. `--trained_task` the trained env we want to load the model from;
2. `--algo` the trained algorithm we want to load the model used, now we support `ppo` and `rarl`;
3. `--task` the test environment, now we support `cartpole`, `cartpole_distb` and `quadrotor_distb`(not sure);
4. `--seed` the training random seed used in training;
4. `--render` whether to make videos or gifs, or just log the performances;
5. `--trained_distb_level 1.0` and `--test_distb_level 1.0` when use trained or test the env with 'fixed' disturbances, make sure to revise the distb_level in the test env. 

**Working procedures**
For rl_based controllers:
1. Select the `-- trained_task` (trained env) and `-- algo` used during training, if it's a fixed env, then choose the `-- trained_distb_level` in the trained env (select the model to be tested);
2. Configure the `algo.yaml` file (path: `safe_control_gym/controllers/algo/algo.yaml`) corresponding to the selected trained env (copy the `algo_env.yaml` content to the `algo.yaml` file);
3. Select the `-- task` (test env), if it's a fixed env, then choose the `-- test_distb_level` in the test env, 
4. Choose other necessary arguments. 

For instance, we want to test the trained algo `rarl` in the env `cartpole` in the test env `cartpole_distb`, the test command is:
`python safe_control_gym/experiments/test_rl_controller.py --trained_task cartpole --algo rarl --task cartpole_distb --seed 42`. 

When test env with 'fixed' disturbances, add another two arguments `--trained_distb_level 1.0` and `--test_distb_level 1.0` to specify the test model and test env. Also remember to revise the distb_level in the corresponding env.

python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_boltz --algo ppo --task quadrotor_random --seed 42

python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_null --algo rap --task quadrotor_fixed --test_distb_level 1.0 --seed 42

python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_null --algo rarl --task quadrotor_random --seed 42

python safe_control_gym/experiments/test_hj_controller.py --algo hj --task cartpole_fixed --test_distb_level 1.5 --seed 42 --render

python safe_control_gym/experiments/test_hj_controller.py --algo hj --task cartpole_random --seed 42

python safe_control_gym/experiments/test_rl_controller.py --trained_task cartpole_boltz --algo ppo --task cartpole_random --seed 42

python safe_control_gym/experiments/test_rl_controller.py --trained_task cartpole --algo rap --task cartpole_random --seed 42

python safe_control_gym/experiments/test_rl_controller.py --trained_task cartpole_boltz --algo ppo --task cartpole_fixed --test_distb_level 1.5 --seed 42

python safe_control_gym/experiments/test_rl_controller.py --trained_task cartpole_boltz --algo ppo --task cartpole_random --seed 42

python safe_control_gym/experiments/test_rl_controller.py --trained_task cartpole_null --algo ppo --task cartpole_null --seed 42


## Env Info


