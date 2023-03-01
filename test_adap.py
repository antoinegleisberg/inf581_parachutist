from dqn_agent import *
from reinforce_agent import *

"""
a code to test an agent in the environment
the agent must have an act method that takes an observation and returns an action"""

#PARAMETERS
episodes=5#nb of test episodes

env=ParachutistEnv()
env.parachutist.verbose=False
env.parachutist.wind=Wind(constant_wind)

new_env=ParachutistEnv()
new_env.parachutist.verbose=False
new_env.parachutist.wind=Wind(constant_wind)



#AGENT WE WANT TO TEST 
agent = ReinforceAgent(env)
agent.train(env,episodes=400)


success=0
average_reward = 0
best_reward = -1000
vent=new_env.parachutist.wind
for eps in tqdm(range(1, episodes+1)):

    state = new_env.reset()
    new_env.parachutist.reset()
    new_env.parachutist.wind=vent
    
    score = 0
    while True:
        action = agent.act(state)

        # play action for 10 frames so that the agent can't change its action in a milli second
        for _ in range(30):
            state, reward, done, info = new_env.step(Action.from_int(action))
            state = torch.tensor(state).float()
           

            score += reward

            if done:
                break

        if done:
            average_reward += score
            if score > best_reward:
                best_reward = score
            if reward==100:
                success+=1
           

            if eps%5==0:
                print("Episode {}  Best Reward {} Last Reward {}".format(eps,  best_reward, score))
            print(state)
            break

print("Average reward: ", average_reward/episodes)
print("Success rate: ", success/episodes)
print("Best reward: ", best_reward)