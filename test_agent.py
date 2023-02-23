import pygame
from env import ParachutistEnv, Action
from wind import Wind, constant_wind
from reinforce_agent import ReinforceAgent

"""
a code to test an agent in the environment
the agent must have an act method that takes an observation and returns an action"""

# PARAMETERS
env = ParachutistEnv()
env.parachutist.verbose = False
env.parachutist.wind = Wind(constant_wind)


# AGENT WE WANT TO TEST
agent = ReinforceAgent(env)
agent.train(episodes=20, env=env)

# ask in terminal if you want to test the agent
test = input("Do you want to test the agent? (y/n) ")


# LIVE TEST OF THE AGENT
def test_agent(agent, env: ParachutistEnv):
    pygame.init()
    wind = env.parachutist.wind
    observation = env.reset()
    done = False
    env.parachutist.wind = wind

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        act = agent.act(observation)
        action = Action.from_int(act)

        # apply each action for 100 frames

        for _ in range(50):
            observation, reward, done, info = env.step(action)
            env.render()
            if done:
                break

    pygame.quit()


test_agent(agent, env)
