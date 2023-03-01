import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.distributions import Categorical
from matplotlib import pyplot as plt

from env import ParachutistEnv, Action, Env
from agent_baseline import Agent

"""Reinforce Agent
Inspired by:

https://github.com/coldhenry/RL-REINFORCE-Pytorch/blob/main/REINFORCE_continuous.py


"""

env = ParachutistEnv()
env.parachutist.verbose = False


# for reproducibility
# env.seed(1)
# torch.manual_seed(1)

gamma = 0.99
batch_size = 500


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        num_hidden = 24
        num_hidden2=12
        self.l1 = nn.Linear(6, num_hidden)
        self.l2= nn.Linear(num_hidden,num_hidden2)
        self.l3 = nn.Linear(num_hidden2, 4)

    def forward(self, x):
        # fully connected model
        model = torch.nn.Sequential(self.l1, nn.ReLU(), self.l2,nn.ReLU(),self.l3, nn.Softmax(dim=-1))
        return model(x)


class ReinforceAgent(Agent):
    def __init__(self, env: Env):
        super().__init__(env)
        self.policy = Policy()
        self.policy.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.action_space = env.action_space

    def predict(self, state):

        action_pb = self.policy(Variable(state))
        dist = Categorical(action_pb)
        action = dist.sample()
        log_pb = dist.log_prob(action)

        return action, action_pb, log_pb

    def discounted_reward(self, rewards, gamma=0.9):

        r = []
        for t in range(1, len(rewards) + 1):
            for t_ in range(t, len(rewards) + 1):
                r.append(torch.pow(torch.tensor(gamma), (t_ - t)) * rewards[t_ - t])
        r = np.sum(r)
        return r

    def act(self, state):
        action, _, _ = self.predict(torch.FloatTensor(state))
        return action.numpy()

    def train(self, env: Env, episodes=100):

        plot_reward = []
        plot_success = []
        success = 0

        for eps in range(1,episodes+1):
            total_rewards = 0
            batch_count = 0
            traj_count = 0

            states, rewards = [], []
            s_curr = env.reset()
            env.parachutist.reset()
            done = False

            log_sum = 0

            while True:#batch_count < batch_size:

                # update count

                action, _, log_pb = self.predict(torch.FloatTensor(s_curr))
                print(Action.from_int(action.numpy()))
                for i in range (30):
                    #batch_count += 1

                    log_sum += log_pb
                    s_next, reward, done, _ = env.step(Action.from_int(action.numpy()))
                    s_curr = s_next

                    states.append(s_next)
                    rewards.append(reward)
                    if done:
                        break

                if done: #or batch_count >= batch_size:

                    s_curr = env.reset()
                    env.parachutist.reset()

                    traj_count += 1

                    # discounted reward of a trajectory
                    batch_log_pb = log_sum
                    batch_reward = self.discounted_reward(rewards, gamma)

                    total_rewards += sum(rewards)

                    states, rewards = [], []

                    log_sum = 0
                    done = False
                    break
       

            if reward==100:
                success+=1
            loss = batch_reward * batch_log_pb
            loss = -loss / traj_count

            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()

            mean_reward = total_rewards / traj_count
            print(
                "Episode: {} / Avg. last {}: {:.2f}".format(
                    eps,
                    batch_size,
                    mean_reward,
                )
            )
          
            plot_reward.append(mean_reward)
            plot_success.append(success/eps)


        t = np.arange(1, episodes+1, 1)
      
        plt.plot(t, plot_reward)
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.title('REINFORCE Reward during training')
        plt.show()

        plt.plot(t, plot_success)
        plt.xlabel('Episode')
        plt.ylabel('Success rate since beginning')
        plt.title('REINFORCE Average success rate during training')
        plt.show()
