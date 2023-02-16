import pygame
from dataclasses import dataclass, field
from typing import List, Tuple, Mapping
import numpy as np
from enum import Enum
from env import *
from agent_baseline import *


#PARAMETERS
env=ParachutistEnv()
vent=[10,0]



#AGENT WE WANT TO TEST 
agent = LeftAgent(env)

#LIVE TEST OF THE AGENT
def test_agent(agent, env: ParachutistEnv, vent=[0,0]):
    pygame.init()
    observation = env.reset()
    done = False
    env.parachutist.wind=vent

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

#test_agent(agent,env, vent)
