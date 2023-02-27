import pygame
from env import ParachutistEnv, Action


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


if __name__ == "__main__":
    pygame.init()
    env = ParachutistEnv()
    while True:
        for event in pygame.event.get():
            # detect if user closes window
            check_pygame_exit(event)

            # detect if user presses arrow keys
            input = get_input(event)

        action = Action.from_tuple(input)
        state, reward, done, dic = env.step(action)

        if done:
            print("DONE")
            break

        env.render()

    pygame.quit()
