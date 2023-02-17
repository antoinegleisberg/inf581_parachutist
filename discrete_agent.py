import numpy as np
from agent_baseline import *
from test_agent import *
from env import *


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
    def __init__(self,epsilon=0.8,grid=[10,8,1,8],max_values=[40,40,500,500,np.pi,10]):
        self.grid=grid
        self.max_values=max_values
        DiscreteAgent.__init__(self,grid[0],grid[1],grid[2],grid[3],max_values[0],max_values[1],max_values[2],max_values[3],max_values[4],max_values[5])
        
        self.q_table=np.zeros(grid+[4])
        self.epsilon=epsilon
        
        self.q_table_history=[]
    
    
    def act(self,observation):
        discrete_state=self.convert_state_to_discrete_space(observation)
        p=np.random.rand()
        if p<self.epsilon:
            action=np.random.randint(5)
        else:
            action=np.argmax(self.q_table[discrete_state])

        return action
    
    
    def train(self,env=ParachutistEnv(pygame_used=False),alpha=0.1, alpha_factor=0.9995, gamma=0.99, num_episodes=1000,verbose=True):
        env.parachutist.verbose=False
        for episode_index in range(num_episodes):
            if verbose and episode_index % int(num_episodes/10) == 1:
                try:
                    env.reset()
                    test_agent(self,env,env.parachutist.wind)
                except:
                    print("Failed to simulate the agent")
                print("Episode "+str(episode_index)+"    nombre d'erreurs k:"+str(k))
            
            self.q_table_history.append(self.q_table.copy())

            # Update alpha
            if alpha_factor is not None:
                alpha = alpha * alpha_factor
            
            S=env.reset()
            A=self.act(S)
            final=False
            dS=self.convert_state_to_discrete_space(S)
            k=0
            
            while (not final) and k<100:
                try:
                    s2,R,final,_=env.step(A)
                    a2=self.act(s2)
                    ds2=self.convert_state_to_discrete_space(s2)
                    self.q_table[dS+[A]]+=alpha*(R+gamma*self.q_table[ds2+[a2]]-self.q_table[dS+[A]])
                    S=s2.copy()
                    dS=ds2.copy()
                    A=a2
                except:
                    # counting errors, if k>100 : stop
                    k+=1
            
                    
env=ParachutistEnv(pygame_used=True)
QL=QLearning_Agent()
QL.train(num_episodes=1000)
# env=ParachutistEnv()
# env.parachutist.verbose=False
# s=env.reset()
# a=QL.act(s)
# s2,r,_,_=env.step(a)
# print(s2,a,s,r)
test_agent(QL,env,env.parachutist.wind)