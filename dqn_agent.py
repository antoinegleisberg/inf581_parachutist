import numpy as np
from env import Env, Action
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import deque
import random
import torch
import copy
from tqdm.notebook import tqdm
from agent_baseline import Agent

"""DQN Agent
"""


LR = 0.0001
MEM_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.95
EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.999
EXPLORATION_MIN = 0.001


class Network(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        self.input_shape = 6
        self.action_shape = 4
        self.action_space = env.action_space

        self.fc1 = nn.Linear(self.input_shape, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.action_shape)

        self.optimizer = optim.Adam(self.parameters(), lr=LR)
        self.loss = nn.MSELoss()
        # self.to(DEVICE)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class ReplayBuffer:
    def __init__(self):
        self.memory = deque(maxlen=MEM_SIZE)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self):
        minibatch = random.sample(self.memory, BATCH_SIZE)

        state1_batch = torch.stack([s1 for (s1, a, r, s2, d) in minibatch])
        action_batch = torch.tensor([a for (s1, a, r, s2, d) in minibatch])
        reward_batch = torch.tensor([r for (s1, a, r, s2, d) in minibatch])
        state2_batch = torch.stack([s2 for (s1, a, r, s2, d) in minibatch])
        done_batch = torch.tensor([d for (s1, a, r, s2, d) in minibatch])

        return (state1_batch, action_batch, reward_batch, state2_batch, done_batch)


# create a deep DQN agent
class DQN(Agent):
    def __init__(self, env):
        self.replay = ReplayBuffer()
        self.exploration_rate = EXPLORATION_MAX
        self.network = Network(env)
        self.network2 = copy.deepcopy(self.network)  # A
        self.network2.load_state_dict(self.network.state_dict())
        self.action_space = env.action_space

    def act(self, observation):
        if random.random() < self.exploration_rate:
            return np.random.choice(self.action_space)

        # Convert observation to PyTorch Tensor
        state = torch.tensor(observation).float().detach()
        # state = state.to(DEVICE)
        state = state.unsqueeze(0)

        # Get Q(s,.)
        q_values = self.network(state)

        # Choose the action to play
        action = q_values.argmax().item()

        return action

    def learn(self):
        if len(self.replay.memory) < BATCH_SIZE:
            return

        # Sample minibatch s1, a1, r1, s1', done_1, ... , sn, an, rn, sn', done_n
        state1_batch, action_batch, reward_batch, state2_batch, done_batch = self.replay.sample()

        # Compute Q values (call self.network and apply the squeeze method on the result)
        q_values = self.network(state1_batch).squeeze()

        with torch.no_grad():
            # Compute next Q values (call self.network and apply the squeeze method on the result)
            next_q_values = self.network2(state2_batch).squeeze()

        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

        batch_indices = torch.tensor(batch_indices, dtype=torch.long)
        action_batch = torch.tensor(action_batch, dtype=torch.long)

        predicted_value_of_now = q_values[batch_indices, action_batch]
        predicted_value_of_future = next_q_values.max(dim=1)[0]

        # Compute the q_target
        # boolean to float conversion

        done_batch = done_batch.float()
        q_target = reward_batch + GAMMA * predicted_value_of_future * (1 - done_batch.numpy())
        q_target = torch.tensor(q_target).float()

        # Compute the loss (c.f. self.network.loss())
        loss = self.network.loss(q_target, predicted_value_of_now)

        # Complute 𝛁Q
        self.network.optimizer.zero_grad()

        loss.backward()
        self.network.optimizer.step()

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def returning_epsilon(self):
        return self.exploration_rate

    def train(self, env: Env, episodes=50, sync_freq=10):
        vent = env.parachutist.wind

        best_reward = -1000
        average_reward = 0
        j = 0
        for i in tqdm(range(1, episodes + 1)):
            state = env.reset()
            env.parachutist.reset()
            env.parachutist.wind = vent

            score = 0
            while True:
                j += 1
                action = self.act(state)

                # play action for 10 frames so that the agent can't change its action in a milli second
                for _ in range(10):
                    state_, reward, done, info = env.step(Action.from_int(action))
                    state = torch.tensor(state).float()
                    state_ = torch.tensor(state_).float()
                    exp = (state, action, reward, state_, done)
                    self.replay.add(exp)
                    self.learn()

                    state = state_
                    score += reward

                    if j % sync_freq == 0:
                        self.network2.load_state_dict(self.network.state_dict())
                    if done:
                        break

                if done:
                    if score > best_reward:
                        best_reward = score
                    average_reward += score
                    if i % 5 == 0:
                        print(
                            "Episode {} Average Reward {} Best Reward {} Last Reward {}".format(
                                i, average_reward / i, best_reward, score
                            )
                        )
                    print(state)
                    break
