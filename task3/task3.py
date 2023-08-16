"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the AlgorithmManyArms class. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)
"""

import numpy as np
import math
# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE

class AlgorithmManyArms:
    def __init__(self, num_arms, horizon):
        self.n = num_arms
        self.horizon = horizon
        self.mu = 1-1/num_arms
        self.m = int(math.sqrt(num_arms-1)*math.log(num_arms-1))
        self.counts = 0
        self.now = np.random.randint(num_arms)
        self.state = [0 for i in range(num_arms)]
        self.bool = 0
        # Horizon is same as number of arms
        # START EDITING HERE
        # You can add any other variables you need here
        # END EDITING HERE
    
    def give_pull(self):
        if self.counts < self.m and self.bool == 0:
            return self.now
        elif self.counts < self.m and self.bool == 1:
            return np.random.randint(self.n)
        else:
            return self.mu
       
    
    def get_reward(self, arm_index, reward):
        self.counts += 1
        if self.counts < self.m:
            if int(reward) == 1:
                self.state[arm_index] += 1
                self.bool = 0
                self.now = arm_index
            else:
                self.bool = 1
        elif self.counts == self.m:
            self.mu = np.argmax(self.state)
        
              
        
            
        
        
        
        
        
        
        
