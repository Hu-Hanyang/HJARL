# Working Logs
## Environment Building
### Features of Envs
**BenchmarkEnv(gym.Env, ABC)**
1. Basic attributes:
    NAME
    URDF_PATH
    DISTURBANCE_MODES=None  # dict
    TASK_INFO

2. \_\_init\_\_():
    task: Task.STABILIZATION, Task.TRACKING
    cost: Cost.RL_REWARD
    pyb_freq
    ctrl_freq
    episode_len_sec
    init_state
    randomized_init  # bool
    Domain randomization related (False)
    Constraints related (False)
    Disturbances related
        disturbances=None

    self.DISTURBANCES = disturbances
    self.adversary_disturbance = adversary_disturbance
    self.adversary_disturbance_offset = adversary_disturbance_offset
    self.adversary_disturbance_scale = adversary_disturbance_scale
    self._setup_disturbances()
        self.disturbances = {}

3. Methods:
    def seed(self, seed)
    def set_adversary_control(self, action)  # Sets disturbance by an adversary controller, called before (each) step().
    def _setup_disturbances(self)
    def before_reset(self, seed)  # Pre-processing before calling `.reset()`, a housekeeping function
    def after_reset(self, obs, info)  # Post-processing after calling `.reset()`
    def _preprocess_control(self, action)
    def before_step(self, action)  # Pre-processing before calling `.step()`
    def after_step(self, obs, rew, done, info)
    def _generate_trajectory

4. Disturbances related (adversarial is not considered here):
    DISTURBANCE_MODES = None  # class instance
    self.DISTURBANCES = disturbances # initialization method, dictionary
    def _setup_disturbances()  # 
    

CartPole(BenchmarkEnv)
1. Basic attributes:
    NAME = 'cartpole'
    URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'cartpole_template.urdf')
    DISTURBANCE_MODES = {'observation': {'dim': 4}, 'action': {'dim': 1}, 'dynamics': {'dim': 2}}
    TASK_INFO = {
        'stabilization_goal': [0],
        'stabilization_goal_tolerance': 0.05,
        'trajectory_type': 'circle',
        'num_cycles': 1,
        'trajectory_plane': 'zx',
        'trajectory_position_offset': [0, 0],
        'trajectory_scale': 0.2
    }

2. \_\_init\_\_():
    self.PYB_CLIENT
    self.RENDER_HEIGHT
    self.RENDER_WIDTH
    self.INIT_X, self.INIT_X_DOT, self.INIT_THETA, self.INIT_THETA_DOT
    self._setup_symbolic  # symbolic dyanmics initialization

3. Normal working flow:
    3.1 obs, info = env.reset(seed=None)  # Mandatory to call at least once after \_\_init\_\_().
    3.2 obs, rew, done, info = env.step(action)
        super().before_reset(seed=seed)
        PyBullet simulation reset
        # Choose randomized or deterministic inertial properties (one domain randomization application), create new urdf file, and change to new PyBullet dynamics. Randomize initial state, override to new random initial states
        force = super().before_step(action)
        self._advance_simulation(force)  # Apply the commanded forces and adversarial actions to the cartpole. The PyBullet simulation is stepped PYB_FREQ/CTRL_FREQ times. Calculate the adversarial disturbances here.
        # Update the state from the PyBullet
        self.state
        # Standard Gym return.
        obs = self._get_observation()
        rew = self._get_reward()
        done = self._get_done()
        info = self._get_info()
        obs, rew, done, info = super().after_step(obs, rew, done, info)


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

python safe_control_gym/experiments/train_rl_controller.py --task cartpole_fixed --algo ppo --distb_level 1.0 --use_gpu True --seed 42


Table: Details of the envs
| Env Name  | Description | 
| --------- | ----------- |
| cartpole  | The cartpole env with no disturbance |
| cartpole_distb | The cartpole env with disturbance |
| quadrotor_distb | The quadrotor env with disturbance |

## Test
### Need to make the ppo.yaml file consistent with the training file before test
The test file lies in `safe_control_gym/experiments/test_rl_controller.py`. When run this file, some necessary arguments need to be added: 
1. `--trained_task` the trained env we want to load the model from;
2. `--algo` the trained algorithm we want to load the model used, now we support `ppo` and `rarl`;
3. `--task` the test environment, now we support `cartpole`, `cartpole_distb` and `quadrotor_distb`(not sure);
4. `--seed` the training random seed used in training.

For instance, we want to test the trained algo `rarl` in the env `cartpole` in the test env `cartpole_distb`, the test command is:
`python safe_control_gym/experiments/test_rl_controller.py --trained_task cartpole --algo rarl --task cartpole_distb --seed 42`. 

python safe_control_gym/experiments/test_rl_controller.py --trained_task cartpole_boltz --algo ppo --task cartpole_fixed --seed 42

python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_boltz --algo ppo --task quadrotor_fixed --seed 42

python safe_control_gym/experiments/test_rl_controller.py --trained_task quadrotor_boltz --algo ppo --task quadrotor_null --seed 42

python safe_control_gym/experiments/test_rl_controller.py --trained_task cartpole --algo rap --task cartpole_fixed --seed 42 

python safe_control_gym/experiments/test_rl_controller.py --task cartpole_fixed --algo ppo --test_distb_level 1.5 --trained_distb_level 1.0 --use_gpu True --seed 42

## Env Info


