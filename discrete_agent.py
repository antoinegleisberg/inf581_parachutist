import numpy as np


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
        discrete_state= [round((pos0+self.max_space0)/(2*self.N_space_grid*self.max_space0)), # Negative and positive pos0 so 2*max_space0 chunk size
                round(pos1/(self.N_space_grid*self.max_space1)), # Only positive pos1
                round((theta+self.max_theta)/(self.N_theta_grid*2*self.max_theta)),
                round((theta_dot+self.max_theta_dot)/(self.N_theta_dot_grid*2*self.max_theta_dot)),
                round((vel0+self.max_vel0)/(self.N_velocity_grid*2*self.max_vel0)),
                round(vel1/(self.N_velocity_grid*self.max_vel1))]
        
        if (np.array(discrete_state)<0).any() or (np.array(discrete_state)>np.array([self.N_space_grid,self.N_space_grid,self.max_theta,self.max_theta_dot,self.max_vel0,self.max_vel1])).any():
            print("ERROR: discrete indices out of bound")
            print(discrete_state)
            raise Exception()

        return discrete_state
        

class QLearning_Agent(DiscreteAgent):
    """Implement a QLearning Agent using a discrete representation of the state space given by the DiscreteAgent framework

    Args:
        DiscreteAgent (_type_): _description_
    """
    pass