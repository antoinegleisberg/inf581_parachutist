import pygame
import numpy as np
from env import *



if __name__ == "__main__":
    pygame.init()
    env = ParachutistEnv()
    # Set the wind of the environment:
    env.parachutist.wind=np.array([4.,0.])
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
        state, reward, done, dic=env.step(action)
        #stop if done
        if done:
            print('DONE')
            raise Exception()
        

       

        env.render()
