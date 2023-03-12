import pygame

from env import ParachutistEnv, Action
from wind import Wind, constant_wind, perlin_noise_wind, linear_wind
from reinforce_agent import ReinforceAgent
from dqn_agent import DQN
from ddpg_agent import DDPG
from agent_baseline import RandomAgent, LeftAgent, RightAgent

"""
a code to test an agent in the environment
the agent must have an act method that takes an observation and returns an action"""

# PARAMETERS
#------------------ Params ------------------#
CONTINUOUS = False 
START_CLOSED = True
WIND=Wind(perlin_noise_wind)
EPISODES=200
#--------------------------------------------#

env = ParachutistEnv()
env.parachutist.is_continuous = CONTINUOUS
env.parachutist.wind = WIND
env.parachutist.params.start_closed=START_CLOSED



# AGENT WE WANT TO TEST
agent =ReinforceAgent(env)
print("Agent loaded")
print("Training agent...")
agent.train(episodes=EPISODES, env=env)


# ask in terminal if you want to test the agent
test = input("Do you want to test the agent? (y/n) ")


# LIVE TEST OF THE AGENT
def test_agent(agent, env: ParachutistEnv):
    pygame.init()
    observation = env.reset()
    done = False
   

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        act = agent.act(observation)
        action = Action.from_int(act)

        # apply each action for 100 frames

        for _ in range(50):
            observation, reward, done, info = env.step(action)
            env.render()
            if done:
                break

    pygame.quit()


test_agent(agent, env)
