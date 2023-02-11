import pygame
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
from enum import Enum

Vec = np.ndarray


class Action(Enum):
    NONE = 0
    LEFT = 1
    RIGHT = 2
    BOTH = 3

    @classmethod
    def from_tuple(cls, L: Tuple[int, int]) -> "Action":
        if L[0] == 1 and L[1] == 1:
            return Action.BOTH
        if L[0] == 1:
            return Action.LEFT
        if L[1] == 1:
            return Action.RIGHT
        return Action.NONE


class Env(ABC):
    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def step(self, action):
        raise NotImplementedError()

    @abstractmethod
    def render(self):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()


@dataclass
class Parachutist:
    """Used to represent the parachutist."""

    # Relative coordinates of the elements of the parachutist.
    parachute: List[Vec] = field(default_factory=list, init=False)
    default_parachute: List[Vec] = field(default_factory=list, init=False)
    left_pulled_parachute: List[Vec] = field(default_factory=list, init=False)
    right_pulled_parachute: List[Vec] = field(default_factory=list, init=False)
    both_pulled_parachute: List[Vec] = field(default_factory=list, init=False)
    body: List[Vec] = field(default_factory=list, init=False)
    strings: List[Tuple[Vec, Vec]] = field(default_factory=list, init=False)

    rotation: float = field(default=0, init=False)
    position: Vec = field(default=np.array([0.0, 0.0]), init=False)
    velocity: Vec = field(default=np.array([0.0, 0.0]), init=False)

    time_step: float = field(default=0.1, init=False)

    wind: Vec = field(default=np.array([0, 0]), init=False)

    def __post_init__(self):
        self.left_pulled_parachute = [
            np.array([-40, -45]),
            np.array([-25, -60]),
            np.array([25, -60]),
            np.array([45, -50]),
        ]

        self.right_pulled_parachute = [
            np.array([-45, -50]),
            np.array([-25, -60]),
            np.array([25, -60]),
            np.array([40, -45]),
        ]

        self.default_parachute = [
            np.array([-45, -50]),
            np.array([-25, -60]),
            np.array([25, -60]),
            np.array([45, -50]),
        ]

        self.both_pulled_parachute = [
            np.array([-40, -45]),
            np.array([-25, -60]),
            np.array([25, -60]),
            np.array([40, -45]),
        ]

        self.parachute = self.default_parachute
        self.body = [np.array([-5, -10]), np.array([5, -10]), np.array([5, 10]), np.array([-5, 10])]
        self.strings = [(np.array([0, 0]), x) for x in self.parachute]

    def pull(self, action: Action):
        """Pulls the parachute."""
        left = action == Action.LEFT or action == Action.BOTH
        right = action == Action.RIGHT or action == Action.BOTH

        if left and right:
            self.parachute = self.both_pulled_parachute
        elif left:
            self.parachute = self.left_pulled_parachute
        elif right:
            self.parachute = self.right_pulled_parachute
        else:
            self.parachute = self.default_parachute

        self.strings = [(np.array([0, 0]), x) for x in self.parachute]

    def apply_forces(self):
        """Applies forces to the parachutist."""
        gravity = np.array([0, 9.81])

        air_volumic_mass = 1.292
        C = 0.47
        drag = 0.5 * air_volumic_mass * C * np.linalg.norm(self.velocity) ** 2

        left_wing = self.parachute[1] - self.parachute[0]
        left_drag = drag * np.array([left_wing[1], -left_wing[0]]) / np.linalg.norm(left_wing)

        right_wing = self.parachute[3] - self.parachute[2]
        right_drag = drag * np.array([right_wing[1], -right_wing[0]]) / np.linalg.norm(right_wing)

        center_wing = self.parachute[2] - self.parachute[1]
        center_drag = drag * np.array([center_wing[1], -center_wing[0]]) / np.linalg.norm(center_wing)

        if np.linalg.norm(self.velocity) > 10:
            pass
            print("Too fast !")
            self.velocity = self.velocity / np.linalg.norm(self.velocity)
            self.velocity = np.array([0.0, 0.0])
            return
        self.velocity += (gravity + center_drag + left_drag + right_drag) * self.time_step
        self.position += self.velocity * self.time_step

    def draw(self, screen: pygame.Surface):
        """Draws the parachutist on the screen."""
        offset = self.position + np.array([400, 300])
        pygame.draw.lines(screen, (255, 255, 255), False, [coord + offset for coord in self.parachute], 10)
        pygame.draw.ellipse(screen, (255, 255, 255), pygame.Rect((self.body + offset)[0], (10, 20)))
        for string in self.strings:
            pygame.draw.line(screen, (255, 255, 255), string[0] + offset, string[1] + offset, 2)


class ParachutistEnv(Env):
    def __init__(self):
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        self.parachutist = Parachutist()

    def reset(self):
        pass

    def step(self, action: Action):
        self.parachutist.pull(action)
        self.parachutist.apply_forces()

    def render(self):
        self.screen.fill((0, 0, 0))
        self.parachutist.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pass


if __name__ == "__main__":
    pygame.init()
    env = ParachutistEnv()
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
        env.step(action)

        env.render()
