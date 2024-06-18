'''Base environment class module for the reach-avoid game.

'''

import numpy as np

from odp.Grid import Grid
from safe_control_gym.envs.gym_game.utilities import find_sign_change1vs0, spa_deriv, find_sign_change1vs1
from safe_control_gym.envs.gym_game.BaseRLGame import BaseRLGameEnv
from safe_control_gym.envs.gym_game.BaseGame import Dynamics


class ReachAvoidGameEnv(BaseRLGameEnv):
    """Multi-agent reach-avoid games class for SingleIntegrator dynamics."""

    ################################################################################
    
    def __init__(self,
                 num_attackers: int=1,
                 num_defenders: int=1,
                 attackers_dynamics=Dynamics.SIG,  
                 defenders_dynamics=Dynamics.FSIG,
                 initial_attacker: np.ndarray=None,  # shape (num_atackers, state_dim), np.array([[-0.4, -0.8]])
                 initial_defender: np.ndarray=None,  # shape (num_defenders, state_dim), np.array([[0.3, -0.8]])
                 ctrl_freq: int = 200,
                 seed = 42,
                 random_init = True,
                 uMode="min", 
                 dMode="max",
                 output_folder='results',
                 game_length_sec=10,
                 map={'map': [-1.0, 1.0, -1.0, 1.0]},  # Hanyang: rectangele [xmin, xmax, ymin, ymax]
                 des={'goal0': [0.6, 0.8, 0.1, 0.3]},  # Hanyang: rectangele [xmin, xmax, ymin, ymax]
                 obstacles: dict = {'obs1': [-0.1, 0.1, -1.0, -0.3], 'obs2': [-0.1, 0.1, 0.3, 1.0]}  # Hanyang: rectangele [xmin, xmax, ymin, ymax]
                 ):
        """Initialization of a generic aviary environment.

        Parameters
        ----------
        num_attackers : int, optional
            The number of attackers in the environment.
        num_defenders : int, optional
            The number of defenders in the environment.
        initial_attacker : np.ndarray, optional
            The initial states of the attackers.
        initial_defender : np.ndarray, optional
            The initial states of the defenders.
        attacker_physics : Physics instance
            A dictionary contains the dynamics of the attackers.
        defender_physics : Physics instance
            A dictionary contains the dynamics of the defenders.
        ctrl_freq : int, optional
            The control frequency of the environment.
        seed : int, optional
        random_init: bool, optional
        uMode : str, optional
            The mode of the attacker, default is "min".
        dMode : str, optional
            The mode of the defender, default is "max".
        output_folder : str, optional
            The folder where to save logs.
        game_length_sec=20 : int, optional
            The maximum length of the game in seconds.
        map : dict, optional
            The map of the environment, default is rectangle.
        des : dict, optional
            The goal in the environment, default is a rectangle.
        obstacles : dict, optional
            The obstacles in the environment, default is rectangle.

        """
           
        super().__init__(num_attackers=num_attackers, num_defenders=num_defenders, 
                         attackers_dynamics=attackers_dynamics, defenders_dynamics=defenders_dynamics, 
                         initial_attacker=initial_attacker, initial_defender=initial_defender, 
                         ctrl_freq=ctrl_freq, seed=seed, random_init=random_init, output_folder=output_folder
                         )
        
        assert map is not None, "Map must be provided in the game."
        assert des is not None, "Destination must be provided in the game."
        
        self.map = map
        self.des = des
        self.obstacles = obstacles
        self.GAME_LENGTH_SEC = game_length_sec
        self.uMode = uMode
        self.dMode = dMode
        # Load necessary values for the attacker control
        self.grid1vs0 = Grid(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 2, np.array([100, 100])) 
        # self.grid1vs1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([45, 45, 45, 45]))
        # self.value1vs1 = np.load('safe_control_gym/envs/gym_game/values/1vs1Attacker.npy')
        self.value1vs0 = np.load('safe_control_gym/envs/gym_game/values/1vs0Attacker.npy')

    
    def step(self, action):
        """Advances the environment by one simulation step.

        Parameters
        ----------
        action : ndarray | (dim_action, )
            The input action for the defender.

        Returns
        -------
        ndarray | dict[..]
            The step's observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        float | dict[..]
            The step's reward value(s), check the specific implementation of `_computeReward()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is over, check the specific implementation of `_computeTerminated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is truncated, check the specific implementation of `_computeTruncated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is trunacted, always false.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """
        
        #### Step the simulation using the desired physics update ##        
        attackers_action = self._computeAttackerActions()  # ndarray, shape (num_defenders, dim_action)
        clipped_action = np.clip(action.copy(), -1.0, +1.0)  # Hanyang: clip the action to [-1, 1]
        defenders_action = clipped_action.reshape(self.NUM_DEFENDERS, 2)  # ndarray, shape (num_defenders, dim_action)
        self.attackers.step(attackers_action)
        self.defenders.step(defenders_action)
        #### Update and all players' information #####
        self._updateAndLog()
        #### Prepare the return values #############################
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        
        #### Advance the step counter ##############################
        self.step_counter += 1
        #### Log the actions taken by the attackers and defenders ################
        self.attackers_actions.append(attackers_action)
        self.defenders_actions.append(defenders_action)
        
        return obs, reward, terminated, truncated, info
    
    
    def _computeAttackerActions(self):
        """Computes the current actions of the attackers.

        Must be implemented in a subclass.

        """
        current_attacker_state = self.attackers._get_state().copy()
        control_attackers = np.zeros((self.NUM_ATTACKERS, 2))
        for i in range(self.NUM_ATTACKERS):
            neg2pos, pos2neg = find_sign_change1vs0(self.grid1vs0, self.value1vs0, current_attacker_state[i])
            if len(neg2pos):
                control_attackers[i] = self.attacker_control_1vs0(self.grid1vs0, self.value1vs0, current_attacker_state[i], neg2pos)
            else:
                control_attackers[i] = (0.0, 0.0)
                
        return control_attackers
    
    
    def _getAttackersStatus(self):
        """Returns the current status of all attackers.

        Returns
            ndarray, shape (num_attackers,)

        """
        new_status = np.zeros(self.NUM_ATTACKERS)
        if self.step_counter == 0:  # Befire the first step
            return new_status
        else:       
            last_status = self.attackers_status[-1]
            current_attacker_state = self.attackers._get_state()
            current_defender_state = self.defenders._get_state()

            for num in range(self.NUM_ATTACKERS):
                if last_status[num]:  # attacker has arrived or been captured
                    new_status[num] = last_status[num]
                else: # attacker is free last time
                    # check if the attacker arrive at the des this time
                    if self._check_area(current_attacker_state[num], self.des):
                        new_status[num] = 1
                    # # check if the attacker gets stuck in the obstacles this time (it won't usually)
                    # elif self._check_area(current_attacker_state[num], self.obstacles):
                    #     new_status[num] = -1
                    #     break
                    else:
                        # check if the attacker is captured
                        for j in range(self.NUM_DEFENDERS):
                            if np.linalg.norm(current_attacker_state[num] - current_defender_state[j]) <= 0.1:
                                new_status[num] = -1
                                break

            return new_status
        

    def _check_area(self, state, area):
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
    

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_PLAYERS*dim, ), concatenate the attackers' and defenders' observations.

        """
        obs = self.state.flatten()

        return obs
    
    
    def _computeReward(self):
        """Computes the current reward value.

        Once the attacker is captured: +100
        Once the attacker arrived at the goal: -100
        The defender hits the obstacle: -100
        One step and nothing happens: maybe use the distance between the attacker and the defender as a sign?
        In status, 0 stands for free, -1 stands for captured, 1 stands for arrived

        Returns
        -------
        float
            The reward.

        """
        last_attacker_status = self.attackers_status[-2]
        current_attacker_status = self.attackers_status[-1]
        reward = 0.0
        # check the attacker status
        for num in range(self.NUM_ATTACKERS):
            reward += (current_attacker_status[num] - last_attacker_status[num]) * (-200)
        # check the defender status
        current_defender_state = self.defenders._get_state().copy()
        reward += -200 if self._check_area(current_defender_state[0], self.obstacles) else 0.0
        # check the relative distance difference or relative distance
        current_attacker_state = self.attackers._get_state().copy()
        current_relative_distance = np.linalg.norm(current_attacker_state[0] - current_defender_state[0])
        last_relative_distance = np.linalg.norm(self.attackers_traj[-2][0] - self.defenders_traj[-2][0])
        # reward += (current_relative_distance - last_relative_distance) * -1.0 / (2*np.sqrt(2))
        reward += -current_relative_distance
        
        return reward

    
    def _computeTerminated(self):
        """Computes the current done value.
        done = True if all attackers have arrived or been captured.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        # defender hits the obstacle or the attacker is captured or the attacker has arrived or the attacker hits the obstacle
        # check the attacker status
        current_attacker_status = self.attackers_status[-1]
        attacker_done = np.all((current_attacker_status == 1) | (current_attacker_status == -1))
        # check the defender status: hit the obstacle, or the attacker is captured
        current_defender_state = self.defenders._get_state().copy()
        defender_done = self._check_area(current_defender_state[0], self.obstacles)
        # final done
        done = True if attacker_done or defender_done else False
        
        return done
        
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        if self.step_counter/self.CTRL_FREQ > self.GAME_LENGTH_SEC:
            return True
        else:
            return False

    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        info = {}
        info['current_steps'] = self.step_counter
        info['current_attackers_status'] = self.attackers_status[-1]
        
        return info 
    

    def _computeAttackerActions(self):
        """Computes the current actions of the attackers.

        """
        control_attackers = np.zeros((self.NUM_ATTACKERS, 2))
        current_attacker_state = self.attackers._get_state().copy()
        for i in range(self.NUM_ATTACKERS):
            neg2pos, pos2neg = find_sign_change1vs0(self.grid1vs0, self.value1vs0, current_attacker_state[i])
            if len(neg2pos):
                control_attackers[i] = self.attacker_control_1vs0(self.grid1vs0, self.value1vs0, current_attacker_state[i], neg2pos)
            else:
                control_attackers[i] = (0.0, 0.0)

        return control_attackers
    

    def attacker_control_1vs0(self, grid1vs0, value1vs0, attacker, neg2pos):
        """Return a list of 2-dimensional control inputs of one defender based on the value function
        
        Args:
        grid1vs0 (class): the corresponding Grid instance
        value1vs0 (ndarray): 1v1 HJ reachability value function with only final slice
        attacker (ndarray, (dim,)): the current state of one attacker
        neg2pos (list): the positions of the value function that change from negative to positive
        """
        current_value = grid1vs0.get_value(value1vs0[..., 0], list(attacker))
        if current_value > 0:
            value1vs0 = value1vs0 - current_value
        v = value1vs0[..., neg2pos] # Minh: v = value1v0[..., neg2pos[0]]
        spat_deriv_vector = spa_deriv(grid1vs0.get_index(attacker), v, grid1vs0)
        opt_a1, opt_a2 = self.optCtrl_1vs0(spat_deriv_vector)

        return (opt_a1, opt_a2)
    

    def attacker_control_1vs1(self, grid1vs1, value1vs1, current_state, neg2pos):
        """Return a list of 2-dimensional control inputs of one defender based on the value function
        
        Args:
        grid1vs1 (class): the corresponding Grid instance
        value1vs1 (ndarray): 1v1 HJ reachability value function with only final slice
        current_state (ndarray, (dim,)): the current state of one attacker + one defender
        neg2pos (list): the positions of the value function that change from negative to positive
        """
        current_value = grid1vs1.get_value(value1vs1[..., 0], list(current_state))
        if current_value > 0:
            value1vs1 = value1vs1 - current_value
        v = value1vs1[..., neg2pos]
        spat_deriv_vector = spa_deriv(grid1vs1.get_index(current_state), v, grid1vs1)
        opt_a1, opt_a2 = self.optCtrl_1vs1(spat_deriv_vector)

        return (opt_a1, opt_a2)
    
    
    def optCtrl_1vs1(self, spat_deriv):
        """Computes the optimal control (disturbance) for the attacker in a 1 vs. 1 game.
        
        Parameters:
            spat_deriv (tuple): spatial derivative in all dimensions
        
        Returns:
            tuple: a tuple of optimal control of the defender (disturbances)
        """
        opt_u1 = self.attackers.uMax
        opt_u2 = self.attackers.uMax
        deriv1 = spat_deriv[0]
        deriv2 = spat_deriv[1]
        crtl_len = np.sqrt(deriv1*deriv1 + deriv2*deriv2)
        if self.uMode == "min":
            if crtl_len == 0:
                opt_u1 = 0.0
                opt_u2 = 0.0
            else:
                opt_u1 = - self.attackers.speed * deriv1 / crtl_len
                opt_u2 = - self.attackers.speed * deriv2 / crtl_len
        else:
            if crtl_len == 0:
                opt_u1 = 0.0
                opt_u2 = 0.0
            else:
                opt_u1 = self.defenders.speed * deriv1 / crtl_len
                opt_u2 = self.defenders.speed * deriv2 / crtl_len

        return (opt_u1, opt_u2)


    def optCtrl_1vs0(self, spat_deriv):
        """Computes the optimal control (disturbance) for the attacker in a 1 vs. 0 game.
        
        Parameters:
            spat_deriv (tuple): spatial derivative in all dimensions
        
        Returns:
            tuple: a tuple of optimal control of the defender (disturbances)
        """
        opt_a1 = self.attackers.uMax
        opt_a2 = self.attackers.uMax
        deriv1 = spat_deriv[0]
        deriv2 = spat_deriv[1]
        ctrl_len = np.sqrt(deriv1*deriv1 + deriv2*deriv2)
        if self.uMode == "min":
            if ctrl_len == 0:
                opt_a1 = 0.0
                opt_a2 = 0.0
            else:
                opt_a1 = - self.attackers.speed * deriv1 / ctrl_len
                opt_a2 = - self.attackers.speed * deriv2 / ctrl_len
        else:
            if ctrl_len == 0:
                opt_a1 = 0.0
                opt_a2 = 0.0
            else:
                opt_a1 = self.attackers.speed * deriv1 / ctrl_len
                opt_a2 = self.attackers.speed * deriv2 / ctrl_len

        return (opt_a1, opt_a2)



class ReachAvoidTestGame(ReachAvoidGameEnv):
    NAME = 'reach_avoid_test'
    def __init__(self, *args,  **kwargs):  # distb_level=1.0, randomization_reset=False,
        # Set disturbance_type to 'fixed' regardless of the input
        kwargs['random_init'] = False
        kwargs['initial_attacker'] = np.array([[-0.4, -0.8]])
        kwargs['initial_defender'] = np.array([[0.0, 0.0]])
        kwargs['seed'] = 42
        super().__init__(*args, **kwargs)