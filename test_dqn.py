import torch
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

from env import ParachutistEnv, Action
from wind import Wind, constant_wind
from dqn_agent import DQN

"""
a code to test the DQN agent
-> run python test_dqn.py
"""


EPISODES = 200
sync_freq = 10


env = ParachutistEnv()
action_space = env.action_space
agent = DQN(env)


best_reward = -1000
episode_number = []
average_reward_number = []

j=0
success=0
plot_success=[]
for i in tqdm(range(1, EPISODES+1)):
    state = env.reset()
    env.parachutist.reset()
    env.parachutist.wind=Wind(constant_wind)
    average_reward = 0

    score = 0
    while True:
        j += 1
        action = agent.act(state)
        print(j)

        # play action for 10 frames so that the agent can't change its action in a milli second
        for _ in range(30):
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
            if reward==100:
                success+=1
            if i%5==0:
                print("Episode {} Average Reward {} Best Reward {} Last Reward {} Epsilon {}".format(i, average_reward/i, best_reward, score, agent.returning_epsilon()))
            print(state)
            break

    episode_number.append(i)
    average_reward_number.append(average_reward)
    plot_success.append(success/i)
    #save average reward
    np.save('_dqn_average_reward1', average_reward_number)
            

plt.plot(episode_number, average_reward_number)
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.title('DQN Reward during training')
plt.show()

plt.plot(episode_number, plot_success)
plt.xlabel('Episode')
plt.ylabel('Success rate since beginning')
plt.title('DQN Average success rate during training')
plt.show()
