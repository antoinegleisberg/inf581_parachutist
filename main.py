import pygame
from env import ParachutistEnv, Action
from wind import Wind, linear_wind


if __name__ == "__main__":
    pygame.init()
    env = ParachutistEnv()
    env.parachutist.wind = Wind(linear_wind)
    # Set the wind of the environment:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            # detect if user presses arrow keys
            input = [0, 0]
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    input[0] = 1
                if event.key == pygame.K_RIGHT:
                    input[1] = 1

        action = Action.from_tuple(input)
        state, reward, done, dic = env.step(action)
        print(reward)
        # stop if done
        if done:
            print("DONE")
            raise Exception()

        env.render()
