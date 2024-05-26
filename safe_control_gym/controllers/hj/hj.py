'''Hamilton-Jacobi control class for Crazyflies.
#TODO: Not finished yet!!!
Based on work conducted at UTIAS' DSL by SiQi Zhou and James Xu.
'''

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.hj.hj_utils import distur_gener_cartpole
from safe_control_gym.envs.benchmark_env import Task


class HJ(BaseController):
    '''Hamilton-Jacobi controller.'''

    def __init__(
            self,
            env_func,
            # Model args.
            distb_level: float = 0.0,  # Hanyang: the disturbance level of the value function used
            **kwargs):
        '''Creates task and controller.

        Args:
            env_func (Callable): Function to instantiate task/environment.
            q_lqr (list): Diagonals of state cost weight.
            r_lqr (list): Diagonals of input/action cost weight.
            discrete_dynamics (bool): If to use discrete or continuous dynamics.
        '''

        super().__init__(env_func, **kwargs)

        self.env = env_func()
        # Controller params.
        self.distb_level = distb_level

    def reset(self):
        '''Prepares for evaluation.'''
        self.env.reset()

    def close(self):
        '''Cleans up resources.'''
        self.env.close()

    def select_action(self, obs, info=None):
        '''Determine the action to take at the current timestep.

        Args:
            obs (ndarray): The observation at this timestep.
            info (dict): The info at this timestep.

        Returns:
            action (ndarray): The action chosen by the controller.
        '''

        step = self.extract_step(info)
        
        hj_ctrl_force, _ = distur_gener_cartpole(obs, self.distb_level)
        assert self.env.TASK == Task.STABILIZATION, "The task should be stabilization."

        return hj_ctrl_force
