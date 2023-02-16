from abc import ABC, abstractmethod
import numpy as np



class Agent(ABC):
    def __init__(self, env):
        self.action_space = env.action_space

    @abstractmethod
    def act(self, observation):
        pass


class RandomAgent(Agent):
    def act(self, observation):
        return np.random.choice(self.action_space)
    
class LeftAgent(Agent):
    def act(self, observation):
        return 1
    
class RightAgent(Agent):
    def act(self, observation):
        return 2
    
