import pygame
from env import ParachutistEnv
from wind import Wind, constant_wind, perlin_noise_wind, linear_wind

# ------------------ Params ------------------#
CONTINUOUS = False
START_CLOSED = True
WIND = Wind(constant_wind)
# --------------------------------------------#


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
    env.parachutist.wind = WIND
    env.parachutist.is_continuous = CONTINUOUS
    input = [0, 0]  # used for continuous environment only
    pygame.key.set_repeat(500, 500)
    score = 0
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
                        input[0] += 0.3
                        input[1] -= 0.3
                    if event.key == pygame.K_RIGHT:
                        input[0] -= 0.3
                        input[1] += 0.3
                    if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                        input[0] = 1
                        input[1] = 1

        state, reward, done, dic = env.step(input)
        score += reward

        if done:
            print("DONE")
            break

        env.render()

    pygame.quit()
    print("Score: ", score)
