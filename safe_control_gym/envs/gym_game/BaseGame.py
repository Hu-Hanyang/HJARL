'''Base environment class module for the reach-avoid game.

This module also contains enumerations for cost functions, tasks, disturbances, and quadrotor types.
'''

import numpy as np
import gymnasium as gym
from safe_control_gym.envs.gym_game.utilities import make_agents


class Dynamics:
    """Physics implementations enumeration class."""

    SIG = {'id': 'sig', 'action_dim': 2, 'state_dim': 2, 'speed': 1.0}           # Base single integrator dynamics
    FSIG = {'id': 'fsig', 'action_dim': 2, 'state_dim': 2, 'speed': 1.5}         # Faster single integrator dynamics with feedback
    
class BaseGameEnv(gym.Env):
    """Base class for the multi-agent reach-avoid game Gym environments."""
    
    def __init__(self,
                 num_attackers: int=1,
                 num_defenders: int=1,
                 attackers_dynamics=Dynamics.SIG,  
                 defenders_dynamics=Dynamics.FSIG,
                 initial_attacker: np.ndarray=None,  # shape (num_atackers, state_dim)
                 initial_defender: np.ndarray=None,  # shape (num_defenders, state_dim)
                 ctrl_freq: int = 200,
                 seed: int = None,
                 output_folder='results',
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
        output_folder : str, optional
            The folder where to save logs.

        """
        #### Constants #############################################
        self.CTRL_FREQ = ctrl_freq
        self.SIM_TIMESTEP = 1. / self.CTRL_FREQ  # 0.005s
        self.seed = seed
        self.initial_players_seed = seed
        #### Parameters ############################################
        self.NUM_ATTACKERS = num_attackers
        self.NUM_DEFENDERS = num_defenders
        self.NUM_PLAYERS = self.NUM_ATTACKERS + self.NUM_DEFENDERS
        #### Options ###############################################
        self.ATTACKER_PHYSICS = attackers_dynamics
        self.DEFENDER_PHYSICS = defenders_dynamics
        self.OUTPUT_FOLDER = output_folder
        #### Input initial states ####################################
        self.init_attackers = initial_attacker
        self.init_defenders = initial_defender
        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and all players' information #####
        self._updateAndLog()
    
    ################################################################################

    def _housekeeping(self):
        """Housekeeping function.

        Initialize all loggers, counters, and variables that need to be reset at the beginning of each episode
        in the `reset()` function.

        """
        if self.init_attackers is None and self.init_defenders is None:
            self.init_attackers, self.init_defenders = self.initial_players()            
        #### Set attackers and defenders ##########################
        self.attackers = make_agents(self.ATTACKER_PHYSICS, self.NUM_ATTACKERS, self.init_attackers, self.CTRL_FREQ)
        self.defenders = make_agents(self.DEFENDER_PHYSICS, self.NUM_DEFENDERS, self.init_defenders, self.CTRL_FREQ)
        #### Initialize/reset counters, players' trajectories and attackers status ###
        self.step_counter = 0
        self.attackers_traj = []
        self.defenders_traj = []
        self.attackers_status = []  # 0 stands for free, -1 stands for captured, 1 stands for arrived 
        self.attackers_actions = []
        self.defenders_actions = []

    ################################################################################

    def _updateAndLog(self):
        """Update and log all players' information after inialization, reset(), or step.

        """
        # Update the state
        self.state = np.vstack([self.attackers._get_state().copy(), self.defenders._get_state().copy()])
        # Log the state and trajectory information
        self.attackers_traj.append(self.attackers._get_state().copy())
        self.defenders_traj.append(self.defenders._get_state().copy())
        self.attackers_status.append(self._getAttackersStatus().copy())
    
    ################################################################################
    
    def initial_players(self):
        '''Set the initial positions for all players.
        
        Returns:
            attackers (np.ndarray): the initial positions of the attackers
            defenders (np.ndarray): the initial positions of the defenders
        '''
        np.random.seed(self.initial_players_seed)
    
        # Map boundaries
        min_val, max_val = -1.0, 1.0
        
        # Obstacles and target areas
        obstacles = [
            ([-0.1, 0.1], [-1.0, -0.3]),  # First obstacle
            ([-0.1, 0.1], [0.3, 0.6])     # Second obstacle
        ]
        target = ([0.6, 0.8], [0.1, 0.3])
        
        def is_valid_position(pos):
            x, y = pos
            # Check boundaries
            if not (min_val <= x <= max_val and min_val <= y <= max_val):
                return False
            # Check obstacles
            for (ox, oy) in obstacles:
                if ox[0] <= x <= ox[1] and oy[0] <= y <= oy[1]:
                    return False
            # Check target
            if target[0][0] <= x <= target[0][1] and target[1][0] <= y <= target[1][1]:
                return False
            return True
        
        def generate_position(current_seed):
            np.random.seed(current_seed)
            while True:
                pos = np.round(np.random.uniform(min_val, max_val, 2), 1)
                if is_valid_position(pos):
                    return pos
        
        def distance(pos1, pos2):
            return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
        
        attacker_seed = self.initial_players_seed
        defender_seed = self.initial_players_seed + 1
        
        while True:
            attacker_pos = generate_position(attacker_seed)
            defender_pos = generate_position(defender_seed)
            
            if distance(attacker_pos, defender_pos) > 1.0:
                break
            defender_seed += 1  # Change the seed for the defender until a valid position is found
        
        self.initial_players_seed += 1
        
        return np.array([attacker_pos]), np.array([defender_pos])

    ################################################################################
    
    def reset(self, seed : int = None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict[..], optional
            Additinonal options, unused

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """        
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and all players' information #####
        self._updateAndLog()
        #### Prepare the observation #############################
        obs = self._computeObs()
        
        return obs
    
    ################################################################################

    def step(self,action):
        #TODO: Hanyang: change the action only for the defender
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
        defenders_action = action.copy().reshape(self.NUM_DEFENDERS, 2)  # ndarray, shape (num_defenders, dim_action)
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
    
    ################################################################################

    def _getAttackersStatus(self):
        """Returns the current status of all attackers.
        -------
        Must be implemented in a subclass.

        """
        raise NotImplementedError
    
    ################################################################################
    
    def _actionSpace(self):
        """Returns the action space of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError
           
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError
    
    ################################################################################
    
    def _computeObs(self):
        """Returns the current observation of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError
    
    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Must be implemented in a subclass.

        Parameters
        ----------
        clipped_action : ndarray | dict[..]
            The input clipped_action for one or more drones.

        """
        raise NotImplementedError

    ################################################################################

    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError
    
    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeAttackerActions(self):
        """Computes the current actions of the attackers.

        Must be implemented in a subclass.

        """
        raise NotImplementedError