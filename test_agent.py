import pygame
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Mapping
import numpy as np
from enum import Enum
from env import *


env=ParachutistEnv()
env.wind=[10,0]
pygame.init()

observation = env.reset()
done = False

#create an agent with random policy on pygame
class Agent(ABC):
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    @abstractmethod
    def act(self, observation):
        pass

class RandomAgent(Agent):
    def act(self, observation):
        return np.random.choice(self.action_space)
    
agent = RandomAgent(env)

while not done:
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
    act=np.random.choice(env.action_space)# Random action
    action=Action.from_int(act)


    print(action)
    # apply each action for 100 frames

    for _ in range(50):
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            break

pygame.quit()


