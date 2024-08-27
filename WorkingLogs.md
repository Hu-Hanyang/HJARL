# Working Logs

## Environment and Algorithm Info
Path: `safe_control_gym/envs`

### 1. Quadrotor and its derivatives
Relative path: `/gym_pybullet_drones`
| Env Name  | Description | Characteristic | 
| --------- | ----------- |  ----------- |
| quadrotor_null| The quadrotor env with no disturbance | Baseline for rap and rarl |
| quadrotor_boltz | The quadrotor env with Boltzman distributed disturbance | Combined with ppo | 
| quadrotor_fixed | The quadrotor env with constant HJ disturbance | Need manually tune the distb_level, combined with ppo |
| quadrotor_random | The quadrotor env with bounded random disturbance | Need manually set bounds, combined with ppo | 

The observation is (17,) = [pos (3: xyz), quat (4), vel (3), ang_v (3), last_clipped_action (4)]

The action is (4,) = pwm for each motor (pwm = 30000 + np.clip(action, -1, +1) * 30000, where action is the output of the policy network)

### 2.  One vs. one reach-avoid game
Relative path:`/gym_game`
| Env Name  | Description | Characteristic | 
| --------- | ----------- |  ----------- |
| easier_game | A 1 vs. 1 reach-avoid game using SIG dynamics without obstacles | Our proposed method |
| rarl_game | The quadrotor env with Boltzman distributed disturbance | Combined with ppo | 
| rap_game | The quadrotor env with constant HJ disturbance | Need manually tune the distb_level, combined with ppo |
| dubin_game | A 1 vs. 1 readch-avoid game using DubinCar3D dynamics without obstacles | Our proposed method | 



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
3. Configure the `seed` and other hyperparameters in the corresponding env class, for example `QuadrotorNullDistb`;
4. Choose other necessary arguments. 


### Quadrotor and its derivatives
| env | algo  | algo_env.yaml | commands | else
| --------- | ----------- |  ----------- | ----------- | ----------- |
| `quadrotor_null`| `ppo` | `ppo_quadrotor.yaml` | `python safe_control_gym/experiments/train_rl_controller.py --task quadrotor_null --algo ppo --use_gpu True --seed 42` | Baseline0 |
| `quadrotor_null`| `rarl` | `rarl_quadrotor.yaml` | `python safe_control_gym/experiments/train_rl_controller.py --task quadrotor_null --algo rarl --use_gpu True --seed 42` | Baseline1 |
| `quadrotor_null`| `rap` | `rap_quadrotor.yaml` | `python safe_control_gym/experiments/train_rl_controller.py --task quadrotor_null --algo rap --use_gpu True --seed 42` | Baseline2 |
| `quadrotor_boltz`| `ppo` | `ppo_quadrotor.yaml` | `python safe_control_gym/experiments/train_rl_controller.py --task quadrotor_boltz --algo ppo --use_gpu True --seed 42` | Our proposed method |
| `quadrotor_fixed`| `ppo` | `ppo_quadrotor.yaml` | `python safe_control_gym/experiments/train_rl_controller.py --task quadrotor_fixed --algo ppo --use_gpu True --seed 42 --distb_level 1.0` | Baseline4, take care of the distb_level |
| `quadrotor_random`| `ppo` | `ppo_quadrotor.yaml` | `python safe_control_gym/experiments/train_rl_controller.py --task quadrotor_random --algo ppo --use_gpu True --seed 42` | Baseline5 (not trained)|
python safe_control_gym/experiments/train_rl_controller.py --task quadrotor_boltz --algo ppo --use_gpu True --seed 2024
python safe_control_gym/experiments/train_rl_controller.py --task quadrotor_null --algo ppo --use_gpu True --seed 1000

### 1 vs. 1 reach-avoid game and its derivatives
| env | algo  | algo_env.yaml | commands | else
| --------- | ----------- |  ----------- | ----------- | ----------- |
| `easier_game`| `ppo` | `none` | `python safe_control_gym/experiments/train_easiergame_sb3.py  --optimality 1vs0 --init_type random --total_steps 1e7` | Our proposed method |
| `rarl_game`| `rarl` | `rarl_easiergame.yaml` | `python safe_control_gym/experiments/train_rl_controller.py --task rarl_game --algo rarl --use_gpu True --seed 42` | Baseline0 |
| `rap_game`| `rap` | `rap_easiergame.yaml` | `python safe_control_gym/experiments/train_rl_controller.py --algo rap --task rarl_game --seed 2024 --use_gpu ` | Baseline1 |
| `dubin_game`| `ppo` | `none` | `python safe_control_gym/experiments/.py ` | Real-world experiment |
| `dubin_rarl_game`| `rarl` | `none` | `python safe_control_gym/experiments/train_rl_controller.py --task dubin_rarl_game --algo rarl --use_gpu True --seed 42 ` | Real-world experiment baseline1 |
| `dubin_rarl_game`| `rap` | `none` | `python safe_control_gym/experiments/train_rl_controller.py --task dubin_rarl_game --algo rap --use_gpu True --seed 42 ` | Real-world experiment baseline1 |


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

### test in the quadrotor_random
python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_boltz --algo ppo --task quadrotor_random  --seed 2024
<!-- python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_random --algo ppo --task quadrotor_random  --seed 42 -->
python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_null --algo ppo --task quadrotor_random  --seed 42
python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_null --algo rarl --task quadrotor_random  --seed 42
python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_null --algo rap --task quadrotor_random  --seed 42


### test in the quadrotor_fixed
python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_boltz --algo ppo --task quadrotor_fixed --test_distb_level 1.0 --seed 2024
python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_null --algo ppo --task quadrotor_fixed --test_distb_level 1.0  --seed 42
python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_null --algo rarl --task quadrotor_fixed --test_distb_level 1.0  --seed 42
python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_null --algo rap --task quadrotor_fixed  --test_distb_level 1.0 --seed 42

python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_boltz --algo ppo --task quadrotor_fixed --test_distb_level 0.8 --seed 42
python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_null --algo ppo --task quadrotor_fixed --test_distb_level 0.8  --seed 42
python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_null --algo rarl --task quadrotor_fixed --test_distb_level 0.8  --seed 42
python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_null --algo rap --task quadrotor_fixed  --test_distb_level 0.8 --seed 42

### test in the quadrotor_wind
python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_boltz --algo ppo --task quadrotor_wind --seed 42
python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_null --algo ppo --task quadrotor_wind --seed 42
python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_null --algo rarl --task quadrotor_wind --seed 42
python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_null --algo rap --task quadrotor_wind --seed 42

### test in the quadrotor_random_hj
python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_boltz --algo ppo --task quadrotor_randomhj --seed 2024
python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_null --algo ppo --task quadrotor_randomhj --seed 42
python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_null --algo rarl --task quadrotor_randomhj  --seed 42
python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_null --algo rap --task quadrotor_randomhj  --seed 42


## Env Info
python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_boltz --algo ppo --task quadrotor_random --seed 42  --render



