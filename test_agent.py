import pygame
from dataclasses import dataclass, field
from typing import List, Tuple, Mapping
import numpy as np
from enum import Enum
from env import *
from agent_baseline import *
from dqn_agent import *

"""
a code to test an agent in the environment
the agent must have an act method that takes an observation and returns an action"""

#PARAMETERS
env=ParachutistEnv()
env.parachutist.wind=[2,0]



#AGENT WE WANT TO TEST 
agent = DQN(env)
agent.train(episodes=1,env=env)

#LIVE TEST OF THE AGENT
def test_agent(agent, env: ParachutistEnv):
    pygame.init()
    wind=env.parachutist.wind
    observation = env.reset()
    done = False
    env.parachutist.wind=wind

    while not done:
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
        act=agent.act(observation)
        action=Action.from_int(act)


        print(action)
        # apply each action for 100 frames

        for _ in range(50):
            observation, reward, done, info = env.step(action)
            env.render()
            if done:
                break

    pygame.quit()

test_agent(agent,env)
