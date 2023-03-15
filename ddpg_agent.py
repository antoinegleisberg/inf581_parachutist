# Source https://github.com/vy007vikas/PyTorch-ActorCriticRL

from __future__ import division
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from agent_baseline import Agent
from env import Env

EPS = 0.003
BATCH_SIZE = 128
LEARNING_RATE = 0.001
GAMMA = 0.95
TAU = 0.001
S_DIM = 6
A_DIM = 2
A_MAX = 1
MAX_BUFFER = 1000000
NOISE_DECAY = 1


def soft_update(target, source, tau):
    """
    Copies the parameters from source network (x) to target network (y) using the below update
    y = TAU*x + (1 - TAU)*y
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    """
    Copies the parameters from source network to target network
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# Based on:
# http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


class MemoryBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.len = 0

    def sample(self, count):
        """
        samples a random batch from the replay memory buffer
        :param count: batch size
        :return: batch (numpy array)
        """
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)

        s_arr = np.float32([arr[0] for arr in batch])
        a_arr = np.float32([arr[1] for arr in batch])
        r_arr = np.float32([arr[2] for arr in batch])
        s1_arr = np.float32([arr[3] for arr in batch])

        return s_arr, a_arr, r_arr, s1_arr

    def len(self):
        return self.len

    def add(self, s, a, r, s1):
        """
        adds a particular transaction in the memory buffer
        :param s: current state
        :param a: action taken
        :param r: reward received
        :param s1: next state
        :return:
        """
        transition = (s, a, r, s1)
        self.len += 1
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1.0 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        param state_dim: Dimension of input state (int)
        param action_dim: Dimension of input action (int)
        """
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fcs1 = nn.Linear(state_dim, 128)
        self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
        self.fcs2 = nn.Linear(128, 64)
        self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())

        self.fca1 = nn.Linear(action_dim, 64)
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

        self.fc2 = nn.Linear(128, 64)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(64, 1)
        self.fc3.weight.data.uniform_(-EPS, EPS)

    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        s1 = F.relu(self.fcs1(state))
        s2 = F.relu(self.fcs2(s1))
        a1 = F.relu(self.fca1(action))
        x = torch.cat((s2, a1), dim=1)

        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_lim):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of output action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :return:
        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(128, 64)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(64, action_dim)
        self.fc3.weight.data.uniform_(-EPS, EPS)

    def forward(self, state):
        """
        returns policy function Pi(s) obtained from actor network
        this function is a gaussian prob distribution for all actions
        with mean lying in (-1,1) and sigma lying in (0,1)
        The sampled action can , then later be rescaled
        :param state: Input state (Torch Variable : [n,state_dim] )
        :return: Output action (Torch Variable: [n,action_dim] )
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # activation function to limit action in [-1,1]
        x = F.sigmoid(x)
        # print("inside", self.fc3(x))
        action = (1 + F.tanh(self.fc3(x))) / 2
        return action


class DDPG(Agent):
    def __init__(self, env):
        """
        param env: Environment
        """
        ram = MemoryBuffer(MAX_BUFFER)
        self.state_dim = S_DIM
        self.action_dim = A_DIM
        self.action_lim = A_MAX
        self.ram = ram
        self.noise_scale = 1.0
        self.noise_decay = NOISE_DECAY
        self.iter = 0
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_dim)

        self.actor = Actor(self.state_dim, self.action_dim, self.action_lim)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_lim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), LEARNING_RATE)

        self.critic = Critic(self.state_dim, self.action_dim)
        self.target_critic = Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), LEARNING_RATE
        )

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

    def act(self, state):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = Variable(torch.from_numpy(state))
        action = self.actor.forward(state).detach()
        new_action = (
            action.data.numpy() + self.noise.sample() * self.noise_scale
        )  # + np.random.randn(A_DIM) * 0.5
        self.noise_scale *= self.noise_decay
        return new_action

    def optimize(self):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        s1, a1, r1, s2 = self.ram.sample(BATCH_SIZE)

        s1 = Variable(torch.from_numpy(s1))
        a1 = Variable(torch.from_numpy(a1))
        r1 = Variable(torch.from_numpy(r1))
        s2 = Variable(torch.from_numpy(s2))

        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        a2 = self.target_actor.forward(s2).detach()
        next_val = torch.squeeze(self.target_critic.forward(s2, a2).detach())
        # y_exp = r + gamma*Q'( s2, pi'(s2))
        y_expected = r1 + GAMMA * next_val
        # y_pred = Q( s1, a1)
        y_predicted = torch.squeeze(self.critic.forward(s1, a1))
        # compute critic loss, and update the critic
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # ---------------------- optimize actor ----------------------
        pred_a1 = self.actor.forward(s1)
        loss_actor = -1 * torch.sum(self.critic.forward(s1, pred_a1))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        soft_update(self.target_actor, self.actor, TAU)
        soft_update(self.target_critic, self.critic, TAU)

    def train(self, env: Env, episodes=50):
        print(f"Training agent {self.__class__.__name__}")
        success = 0
        plot_rewards = []
        plot_success = []
        for eps in range(1, episodes + 1):
            s = np.float32(env.reset())
            score = 0
            while True:
                action = self.act(s)
                for i in range(30):
                    env.render()

                    s2, r, done, info = env.step(action)
                    s2 = np.float32(s2)
                    self.ram.add(s, action, r, s2)
                    s = s2
                    score += r
                    self.optimize()
                    if done:
                        break
                if done:
                    if r == 100:
                        success += 1
                    break
            plot_rewards.append(score)
            plot_success.append(success / eps)
            print("Episode : ", eps, " Reward : ", score)
        episode_number = np.arange(1, episodes + 1)
        plt.plot(episode_number, plot_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward")
        plt.title("DDPG Reward during training")
        plt.show()
        plt.plot(episode_number, plot_success)
        plt.xlabel("Episode")
        plt.ylabel("Success rate since beginning")
        plt.title("DDPG Average success rate during training")
        plt.show()
