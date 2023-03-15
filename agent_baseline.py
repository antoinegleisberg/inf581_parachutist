from abc import ABC, abstractmethod
import numpy as np


class Agent(ABC):
    def __init__(self, env):
        self.action_space = env.action_space

    @abstractmethod
    def act(self, observation):
        pass

    @abstractmethod
    def train(self, env, episodes):
        pass


class RandomAgent(Agent):
    def act(self, observation):
        return np.random.choice(self.action_space)

    def train(self, env, episodes):
        pass


class LeftAgent(Agent):
    def act(self, observation):
        return 1

    def train(self, env, episodes):
        pass


class RightAgent(Agent):
    def act(self, observation):
        return 2

    def train(self, env, episodes):
        pass
