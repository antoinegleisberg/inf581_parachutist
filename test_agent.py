from env import ParachutistEnv
from wind import Wind, constant_wind, perlin_noise_wind, linear_wind
from dqn_agent import DQN
from ddpg_agent import DDPG
from agent_baseline import RandomAgent, LeftAgent, RightAgent

"""
a code to test an agent in the environment
the agent must have an act method that takes an observation and returns an action"""

# PARAMETERS
# ------------------ Params ------------------#
CONTINUOUS = False
START_CLOSED = True
WIND = Wind(perlin_noise_wind)
EPISODES = 200
# --------------------------------------------#

env = ParachutistEnv()
env.parachutist.is_continuous = CONTINUOUS
env.parachutist.wind = WIND
env.parachutist.params.start_closed = START_CLOSED


# AGENT WE WANT TO TEST
agent = DQN(env)
print("Agent loaded")
print("Training agent...")
agent.train(episodes=EPISODES, env=env)
