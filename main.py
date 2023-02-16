import pygame
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Mapping
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

    teta: float = field(default=0, init=False)
    teta_dot: float = field(default=0, init=False)
    position: Vec = field(default=np.array([0.0, 0.0]), init=False)
    velocity: Vec = field(default=np.array([0.0, 0.0]), init=False)
    mass: float = field(default=1, init=False)

    time_step: float = field(default=0.1, init=False)

    wind: Vec = field(default=np.array([0, 0]), init=False)
    
    max_speed: float = 40

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

        self.both_pulled_parachute = [#CAN PULLED BOTH SIDES ???
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

        #1st wing
        left_wing = self.parachute[1] - self.parachute[0]
        left_normal = np.array([left_wing[1], -left_wing[0]]) / np.linalg.norm(left_wing)
        left_wing_vel=np.dot(self.velocity+self.wind, left_normal)
        left_drag: float = 0.5 * air_volumic_mass * C * np.linalg.norm(left_wing_vel) ** 2      
        left_drag = left_drag * left_normal

        #2nd wing
        right_wing = self.parachute[3] - self.parachute[2]
        right_normal = np.array([right_wing[1], -right_wing[0]]) / np.linalg.norm(right_wing)
        right_wing_vel=np.dot(self.velocity+self.wind, right_normal)
        right_drag: float = 0.5 * air_volumic_mass * C * np.linalg.norm(right_wing_vel) ** 2
        right_drag = right_drag * right_normal

        #center wing
        center_wing = self.parachute[2] - self.parachute[1]
        center_normal = np.array([center_wing[1], -center_wing[0]]) / np.linalg.norm(center_wing)
        center_wing_vel=np.dot(self.velocity+self.wind, center_normal)
        center_drag: float = 0.5 * air_volumic_mass * C * np.linalg.norm(center_wing_vel) ** 2
        center_drag = center_drag * center_normal

        if np.linalg.norm(self.velocity) > self.max_speed:
            pass
            print("Too fast !")
            print("high velocity", self.velocity)
            # self.velocity = self.velocity / np.linalg.norm(self.velocity)
            # self.velocity = np.array([0.0, 0.0])
            # return
            raise Exception()


        self.velocity += (gravity + (center_drag + left_drag + right_drag) / self.mass) * self.time_step
        self.position += self.velocity * self.time_step
        print("velocity", self.velocity)
        print("position", self.position)
    
    def apply_momentum(self):# with respect to axis going through middle top of the parachute (0,-60)

        air_volumic_mass = 1.292
        C = 0.47
        gravity = np.array([0, 9.81])

        #moment of gravity
        l=np.linalg.norm((self.parachute[2] + self.parachute[1])/2)
        r=l*np.array([np.sin(self.teta),np.cos(self.teta)])
        #print dim of teta_offset
        gravity_moment = -self.mass * gravity[1] * l*np.sin(self.teta)

        #1st wing
        left_wing = self.parachute[1] - self.parachute[0]
        left_normal = np.array([left_wing[1], -left_wing[0]]) / np.linalg.norm(left_wing)
        left_wing_vel=np.dot(self.velocity+self.wind, left_normal)
        left_drag: float = 0.5 * air_volumic_mass * C * np.linalg.norm(left_wing_vel) ** 2      
        left_drag = left_drag * left_normal

        #moment of force
        left_r=(self.parachute[1] + self.parachute[0])/2 - (self.parachute[2] + self.parachute[1])/2
        left_moment = -np.cross(left_r, left_drag)
      


        #2nd wing
        right_wing = self.parachute[3] - self.parachute[2]
        right_normal = np.array([right_wing[1], -right_wing[0]]) / np.linalg.norm(right_wing)
        right_wing_vel=np.dot(self.velocity+self.wind, right_normal)
        right_drag: float = 0.5 * air_volumic_mass * C * np.linalg.norm(right_wing_vel) ** 2
        right_drag = right_drag * right_normal

        #moment of force
        right_r=(self.parachute[3] + self.parachute[2])/2 -(self.parachute[2] + self.parachute[1])/2
        right_moment = -np.cross(right_r, right_drag)
    

        #center wing~
        """
        no moment of force since force is perpendicular to the axis of rotation
        """
        if np.linalg.norm(self.teta) > np.pi/4:
            
            print("Too much angle !")
            print("high teta", self.teta)
            raise Exception()
        
        

        inertia=0.5*self.mass*np.linalg.norm(r)**2
        drag_moment=left_moment+right_moment
        self.teta_dot +=   ((gravity_moment+drag_moment  ) / inertia)*self.time_step - 0.05*self.teta_dot*self.time_step
        self.teta += self.teta_dot * self.time_step
    
    
    def draw(self, screen: pygame.Surface):
        """Draws the parachutist on the screen."""

        #take angle teta into account: teta_offset on the body
        l=np.linalg.norm((self.parachute[2] + self.parachute[1])/2)
        teta_offset = np.array([l*np.sin(self.teta), -l*(1-np.cos(self.teta))])
        offset = self.position + np.array([400, 300])
        pygame.draw.lines(screen, (255, 255, 255), False, [coord + offset  for coord in self.parachute], 10)
     
        pygame.draw.ellipse(screen, (255, 255, 255), pygame.Rect((self.body+teta_offset + offset)[0], (10, 20)))
        for string in self.strings:
            pygame.draw.line(screen, (255, 255, 255), string[0] + offset + teta_offset, string[1] + offset, 2)

        #draw island in the middle
        pygame.draw.ellipse(screen, (0, 255, 0), pygame.Rect((340, 540), (100, 10)))


class ParachutistEnv(Env):
    

    def __init__(self):
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        self.parachutist = Parachutist()
        self.stepnumber = 0
        self.game_over = False
        self.island_pos=np.array([0,234])
        self.side_x_pos=np.array([-344,344])

    def reset(self):
        pass

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

        #State variables for reward
        state=np.array([self.parachutist.position[0],self.parachutist.position[1],
        self.parachutist.teta,self.parachutist.teta_dot,
        self.parachutist.velocity[0],self.parachutist.velocity[1]])

        

        
        # REWARD -------------------------------------------------------------------------------------------------------
        x_distance=abs(self.island_pos[0]-self.parachutist.position[0])
        y_distance=abs(self.island_pos[1]-self.parachutist.position[1])
        # state variables for reward
        distance = np.linalg.norm((3 * x_distance, y_distance))  # weight x position more
        speed = np.linalg.norm(self.parachutist.velocity)
        groundcontact = self.parachutist.position[1] >self.island_pos[1]
        brokenleg = (
            (np.abs(self.parachutist.teta) > np.pi / 6 or speed > 10)
        ) and groundcontact
        water= (groundcontact and (self.parachutist.position[0] < self.island_pos[0]-100 or self.parachutist.position[0] > self.island_pos[0]+100))
        outside = (self.parachutist.position[0] < self.side_x_pos[0] or self.parachutist.position[0] > self.side_x_pos[1])
        landed = (groundcontact and not brokenleg and not water)
        
        done = False

        reward =0

        if outside or brokenleg or water:
            print("outside", outside)
            print("brokenleg", brokenleg)
            print("water", water)
            self.game_over = True

        if self.game_over:
            done = True
        else:
            # reward shaping
                 
            if landed:
                print("landed")
                reward = 100
                done = True

        if done:
            reward += - 2 * (speed + distance + np.abs(self.parachutist.teta) + np.abs(speed))


        # END OF STEP -------------------------------------------------------------------------------------------------------

        self.stepnumber += 1
        print('reward',reward)

        return state, reward, done, {}
        

    def render(self):
        self.screen.fill((0, 0, 0))
        self.parachutist.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pass

@dataclass
class Agent:
    
    def constant_policy (state,args): return Action.NONE
    
    args=[]
    policy=constant_policy
    name="Unnamed agent"


class DiscreteAgent(Agent):
    """
    Create an Agent that use a discrete representation of the state space to implement algorithms for limited state space size (such as QLearning)

    Args:
        N_space_grid: number of division of a space axis (total number of chunks of state space are therefore N²)
        N_velocity_grid: number of division of a velocity axis between -max_speed and +max_speed
        N_theta_grid: number of division of the theta range
    """
    
    def __init__(self,N_space_grid=10,N_theta_grid=8,N_theta_dot_grid=1,N_velocity_grid=8,max_vel0=40,max_vel1=40,max_space0=500,max_space1=500,max_theta=np.pi,max_theta_dot=10):
        
        self.N_space_grid=N_space_grid
        self.N_velocity_grid=N_velocity_grid
        self.N_theta_grid=N_theta_grid
        self.N_theta_dot_grid=N_theta_dot_grid
        
        self.max_vel0=max_vel0
        self.max_vel1=max_vel1
        self.max_space0=max_space0
        self.max_space1=max_space1
        self.max_theta=max_theta
        self.max_theta_dot=max_theta_dot
        
        self.name="A Discrete Agent"
        
    
    def convert_state_to_discrete_space(self,state):
        """Convert a Parachutist state into discrete indices for our agent

        Args:
            state : Parachutist state

        Returns:
            discrete_state: List of indices of our discretized state space
        """
        pos0,pos1,theta,theta_dot,vel0,vel1=state
        return [round((pos0+self.max_space0)/(2*self.N_space_grid*self.max_space0)), # Negative and positive pos0 so 2*max_space0 chunk size
                round(pos1/(self.N_space_grid*self.max_space1)), # Only positive pos1
                round((theta+self.max_theta)/(self.N_theta_grid*2*self.max_theta)),
                round((theta_dot+self.max_theta_dot)/(self.N_theta_dot_grid*2*self.max_theta_dot)),
                round((vel0+self.max_vel0)/(self.N_velocity_grid*2*self.max_vel0)),
                round(vel1/(self.N_velocity_grid*self.max_vel1))]
        

class QLearning_Agent(DiscreteAgent):
    """Implement a QLearning Agent using a discrete representation of the state space given by the DiscreteAgent framework

    Args:
        DiscreteAgent (_type_): _description_
    """
    pass


if __name__ == "__main__":
    pygame.init()
    env = ParachutistEnv()
    # Set the wind of the environment:
    env.parachutist.wind=np.array([2.,0.])
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
