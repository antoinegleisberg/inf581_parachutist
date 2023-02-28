import pygame
from env import ParachutistEnv


def check_pygame_exit(event: pygame.event.Event) -> None:
    if event.type == pygame.QUIT:
        pygame.quit()
        exit()


def get_input(event) -> None:
    input = [0, 0]
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_LEFT:
            input[0] = 1
        if event.key == pygame.K_RIGHT:
            input[1] = 1
        if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
            input[0] = 1
            input[1] = 1
    return input


CONTINUOUS = True


if __name__ == "__main__":
    pygame.init()
    env = ParachutistEnv()
    input = [0, 0]  # used for continuous environment only
    while True:
        for event in pygame.event.get():
            # detect if user closes window
            check_pygame_exit(event)

            # detect if user presses arrow keys
            if not CONTINUOUS:
                input = get_input(event)
            else:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        input[0] += 0.1
                        input[1] -= 0.1
                    if event.key == pygame.K_RIGHT:
                        input[0] -= 0.1
                        input[1] += 0.1
                    if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                        input[0] = 1
                        input[1] = 1

        state, reward, done, dic = env.step(input)

        if done:
            print("DONE")
            break

        env.render()

    pygame.quit()
