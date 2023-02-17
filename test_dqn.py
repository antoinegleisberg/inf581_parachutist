import numpy as np
from env import *
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import deque
import random
import torch
import copy
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from dqn_agent import *
from wind import *

"""
a code to test the DQN agent
-> run python test_dqn.py
"""


EPISODES = 1
sync_freq = 10



env=ParachutistEnv()
action_space=env.action_space
agent=DQN(env)


best_reward = -1000
average_reward = 0
episode_number = []
average_reward_number = []

j=0
for i in tqdm(range(1, EPISODES+1)):
    state = env.reset()
    env.parachutist.reset()
    env.parachutist.wind=Wind(constant_wind)

    score = 0
    while True:
        j+=1
        action = agent.act(state)
        print(j)

        # play action for 10 frames so that the agent can't change its action in a milli second
        for _ in range(10):
            state_, reward, done, info = env.step(Action.from_int(action))
            state = torch.tensor(state).float()
            state_ = torch.tensor(state_).float()
            exp = (state, action, reward, state_, done)
            agent.replay.add(exp)
            agent.learn()

            state = state_
            score += reward

            if j % sync_freq == 0:
                agent.network2.load_state_dict(agent.network.state_dict())
            if done:
                break

        if done:
            if score > best_reward:
                best_reward = score
            average_reward += score 
            if i%5==0:
                print("Episode {} Average Reward {} Best Reward {} Last Reward {} Epsilon {}".format(i, average_reward/i, best_reward, score, agent.returning_epsilon()))
            print(state)
            break
            
            
    episode_number.append(i)
    average_reward_number.append(average_reward/i)
            

plt.plot(episode_number, average_reward_number)
plt.show()