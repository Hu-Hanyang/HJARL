from safe_control_gym.envs.gym_game.ReachAvoidGame import ReachAvoidGameEnv


env = ReachAvoidGameEnv()
obs = env.reset()
print(f"The state space of the env is {env.observation_space}. \n")
print(f"The action space of the env is {env.action_space}. \n")
print(f"The obs is {obs} and the shape of the obs is {obs.shape}. \n")
print(f"The state of the attacker is {env.state[0]} and the state of the defender is {env.state[1]}. \n")