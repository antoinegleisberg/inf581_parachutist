import pygame
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Mapping
import numpy as np
from enum import Enum
from wind import Wind

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

    @classmethod
    def from_int(cls, i: int) -> "Action":
        if i == 1:
            return Action.LEFT
        if i == 2:
            return Action.RIGHT
        if i == 3:
            return Action.BOTH
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

    teta: float = field(default=0, init=False)
    teta_dot: float = field(default=0, init=False)
    position: Vec = field(default=np.array([0.0, 0.0]), init=False)
    velocity: Vec = field(default=np.array([0.0, 0.0]), init=False)
    mass: float = field(default=2, init=False)

    time_step: float = field(default=0.1, init=False)
    time: float = field(default=0, init=False)

    # wind is a function that takes x and y and t and returns a vector
    # define name function in wind.py
    wind: Wind = field(default=Wind(), init=False)

    verbose: bool = True  # printing info

    max_speed: float = 40

    def reset(self):
        """Resets the parachutist to its initial state."""
        self.teta = 0
        self.teta_dot = 0
        self.position = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])
        self.parachute = self.default_parachute
        self.strings = [(np.array([0, 0]), x) for x in self.parachute]

    def __post_init__(self):
        self.left_pulled_parachute = [
            np.array([-40, -25]),
            np.array([-5, -60]),
            np.array([5, -60]),
            np.array([45, -65]),
        ]

        self.right_pulled_parachute = [
            np.array([-45, -65]),
            np.array([-5, -60]),
            np.array([5, -60]),
            np.array([40, -25]),
        ]

        self.default_parachute = [
            np.array([-45, -55]),
            np.array([-5, -60]),
            np.array([5, -60]),
            np.array([45, -55]),
        ]

        self.both_pulled_parachute = [  # CAN PULLED BOTH SIDES ???
            np.array([-40, -25]),
            np.array([-5, -60]),
            np.array([5, -60]),
            np.array([40, -25]),
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
        """Applies forces to the parachutist.
        Source: https://www.physagreg.fr/mecanique/m12/M12-chute-libre-frottements.pdf
        """
        gravity = np.array([0, 9.81])

        air_volumic_mass = 1.292
        C = 0.47

        # 1st wing
        left_wing = self.parachute[1] - self.parachute[0]
        left_normal = np.array([left_wing[1], -left_wing[0]]) / np.linalg.norm(left_wing)
        left_wing_vel = np.dot(self.velocity - self.wind.get_wind(self.position[0], self.position[1], self.time), left_normal)
        left_drag: float = 0.5 * air_volumic_mass * C * np.linalg.norm(left_wing_vel) ** 2
        left_drag = -left_drag * left_normal * np.sign(left_wing_vel)

        # 2nd wing
        right_wing = self.parachute[3] - self.parachute[2]
        right_normal = np.array([right_wing[1], -right_wing[0]]) / np.linalg.norm(right_wing)
        right_wing_vel = np.dot(self.velocity - self.wind.get_wind(self.position[0], self.position[1], self.time), right_normal)
        right_drag: float = 0.5 * air_volumic_mass * C * np.linalg.norm(right_wing_vel) ** 2
        right_drag = -right_drag * right_normal * np.sign(right_wing_vel)

        # center wing
        center_wing = self.parachute[2] - self.parachute[1]
        center_normal = np.array([center_wing[1], -center_wing[0]]) / np.linalg.norm(center_wing)
        center_wing_vel = np.dot(self.velocity - self.wind.get_wind(self.position[0], self.position[1], self.time), center_normal)
        center_drag: float = 0.5 * air_volumic_mass * C * np.linalg.norm(center_wing_vel) ** 2
        center_drag = center_drag * center_normal

        self.velocity += (gravity + (center_drag + left_drag + right_drag) / self.mass) * self.time_step
        self.position += self.velocity * self.time_step
        if self.verbose:
            print("velocity", self.velocity)
            print("position", self.position)

    def apply_momentum(self):  # with respect to axis going through middle top of the parachute (0,-60)

        air_volumic_mass = 1.292
        C = 0.47
        gravity = np.array([0, 9.81])

        # moment of gravity
        lever_length = np.linalg.norm((self.parachute[2] + self.parachute[1]) / 2)
        r = lever_length * np.array([np.sin(self.teta), np.cos(self.teta)])
        # print dim of teta_offset
        gravity_moment = -self.mass * gravity[1] * lever_length * np.sin(self.teta)

        # 1st wing
        left_wing = self.parachute[1] - self.parachute[0]
        left_normal = np.array([left_wing[1], -left_wing[0]]) / np.linalg.norm(left_wing)
        left_wing_vel = np.dot(self.velocity - self.wind.get_wind(self.position[0], self.position[1], self.time), left_normal)
        left_drag: float = 0.5 * air_volumic_mass * C * np.linalg.norm(left_wing_vel) ** 2
        left_drag = -left_drag * left_normal * np.sign(left_wing_vel)

        # moment of force
        left_r = (self.parachute[1] + self.parachute[0]) / 2 - (self.parachute[2] + self.parachute[1]) / 2
        left_moment = -np.cross(left_r, left_drag)

        # 2nd wing
        right_wing = self.parachute[3] - self.parachute[2]
        right_normal = np.array([right_wing[1], -right_wing[0]]) / np.linalg.norm(right_wing)
        right_wing_vel = np.dot(self.velocity - self.wind.get_wind(self.position[0], self.position[1], self.time), right_normal)
        right_drag: float = 0.5 * air_volumic_mass * C * np.linalg.norm(right_wing_vel) ** 2
        right_drag = -right_drag * right_normal * np.sign(right_wing_vel)

        # moment of force
        right_r = (self.parachute[3] + self.parachute[2]) / 2 - (self.parachute[2] + self.parachute[1]) / 2
        right_moment = -np.cross(right_r, right_drag)

        # center wing~
        """
        no moment of force since force is perpendicular to the axis of rotation
        """

        inertia = 0.5 * self.mass * np.linalg.norm(r) ** 2
        drag_moment = left_moment + right_moment
        self.teta_dot += (
            (gravity_moment + drag_moment) / inertia
        ) * self.time_step - 0.05 * self.teta_dot * self.time_step
        self.teta += self.teta_dot * self.time_step

    def draw(self, screen: pygame.Surface):
        """Draws the parachutist on the screen."""

        # take angle teta into account: teta_offset on the body
        string_length = np.linalg.norm((self.parachute[2] + self.parachute[1]) / 2)
        teta_offset = np.array([string_length * np.sin(self.teta), -string_length * (1 - np.cos(self.teta))])
        offset = self.position + np.array([400, 300])
        pygame.draw.lines(screen, (255, 255, 255), False, [coord + offset for coord in self.parachute], 10)

        pygame.draw.ellipse(screen, (255, 255, 255), pygame.Rect((self.body + teta_offset + offset)[0], (10, 20)))
        for string in self.strings:
            pygame.draw.line(screen, (255, 255, 255), string[0] + offset + teta_offset, string[1] + offset, 2)

        # draw island in the middle
        pygame.draw.ellipse(screen, (0, 255, 0), pygame.Rect((340, 540), (100, 10)))


class ParachutistEnv(Env):
    def __init__(self, pygame_used=True):

        self.pygame_used = pygame_used
        if pygame_used:
            self.screen = pygame.display.set_mode((800, 600))
            self.clock = pygame.time.Clock()
        self.parachutist = Parachutist()

        self.island_pos = np.array([0, 234])
        self.side_x_pos = np.array([-344, 344])
        self.action_space = [0, 1, 2, 3]

        self.stepnumber = 0
        self.game_over = False
        self.prev_shaping = None

    def reset(self):
        verbose = self.parachutist.verbose

        if self.pygame_used:
            self.screen = pygame.display.set_mode((800, 600))
            self.clock = pygame.time.Clock()
        self.parachutist = Parachutist()

        self.parachutist.verbose = verbose

        self.stepnumber = 0
        self.parachutist.time = 0

        self.game_over = False
        self.prev_shaping = None


        state = np.array(
            [
                self.parachutist.position[0],
                self.parachutist.position[1],
                self.parachutist.teta,
                self.parachutist.teta_dot,
                self.parachutist.velocity[0],
                self.parachutist.velocity[1],
            ]
        )

        return state

    def step(self, action: Action) -> Tuple[Parachutist, float, bool, Mapping]:
        """
        @params:
            - action: The action to perform.

        @returns:
            - parachutist: The parachutist, representing the state of the environment.
            - reward: The reward of the action.
            - done: Whether the episode is over.
            - truncated: Whether the episode is truncated.
            - info: Additional information.
        """
        self.parachutist.pull(action)
        self.parachutist.apply_forces()
        self.parachutist.apply_momentum()

        # State variables for reward
        state = np.array(
            [
                self.parachutist.position[0],
                self.parachutist.position[1],
                self.parachutist.teta,
                self.parachutist.teta_dot,
                self.parachutist.velocity[0],
                self.parachutist.velocity[1],
            ]
        )

        # REWARD -------------------------------------------------------------------------------------------------------
        x_distance = abs(self.island_pos[0] - self.parachutist.position[0])
        y_distance = abs(self.island_pos[1] - self.parachutist.position[1])
        # state variables for reward
        distance = np.linalg.norm((3 * x_distance, y_distance))  # weight x position more
        speed = np.linalg.norm(self.parachutist.velocity)
        groundcontact = self.parachutist.position[1] > self.island_pos[1]
        brokenleg = ((np.abs(self.parachutist.teta) > np.pi / 6 or speed > 10)) and groundcontact
        water = groundcontact and (
            self.parachutist.position[0] < self.island_pos[0] - 50
            or self.parachutist.position[0] > self.island_pos[0] + 50
        )
        outside = self.parachutist.position[0] < self.side_x_pos[0] or self.parachutist.position[0] > self.side_x_pos[1]
        landed = groundcontact and not brokenleg and not water

        done = False

        reward = 0

        if outside or brokenleg or water and self.parachutist.verbose:
            print("outside", outside)
            print("brokenleg", brokenleg)
            print("water", water)
            self.game_over = True

        if self.parachutist.verbose:
            print("distance", distance)
        shaping = -speed - distance - self.parachutist.teta_dot - self.parachutist.teta
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        if self.game_over:
            done = True
            reward = -100
        elif landed:
            done = True
            reward = 100
            if self.parachutist.verbose:
                print("landed")

        # END OF STEP --------------------------------------------------------------------------------------------------

        self.stepnumber += 1
        self.parachutist.time = self.stepnumber * self.parachutist.time_step
        if self.parachutist.verbose:
            print("reward", reward)

        return state, reward, done, {}

    def render(self):
        self.screen.fill((0, 0, 0))
        self.parachutist.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pass
