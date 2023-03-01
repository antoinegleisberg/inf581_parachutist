import numpy as np
"""
Implement a spatial representation of the wind
"""
#create a class that contains functions that return wind vectors
def constant_wind(x,y):
    """Return a constant wind vector
    dqn Episode 200 Average Reward 239.47025269231872 Best Reward 333.51636285593247 Last Reward 320.37510444026145
    """
    return [4.0, 0.0]

def opposite_wind(x,y):
    """Return a wind vector that is opposite to the parachute's velocity"""
    return [-4.0, 0.0]

def linear_wind(x,y):
    """Return a wind vector that is linear in y"""
    return [(250-y)/12.5-10, 0.0]




class Wind:
    def __init__(self, wind_function=constant_wind):
        self.wind_function = wind_function

    def get_wind(self, x, y):
        return self.wind_function(x,y)




    

