"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
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

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.counts = np.zeros(num_arms)
        self.wins = np.zeros(num_arms)
        self.metric = np.zeros(num_arms)
        self.total = 0
        
        # You can add any other variables you need here
        # START EDITING HERE
        # END EDITING HERE
    
    def give_pull(self):
        if self.total <= self.num_arms-1 and self.counts[self.total] == 0:
            return self.total
        else:
            return np.argmax(self.metric)
    
    def get_reward(self, arm_index, reward):
        self.total += 1
        self.counts[arm_index] += 1
        self.wins[arm_index] += reward
        t = self.total
        
        if t >= self.num_arms:
            for i in range(self.num_arms):
                self.metric[i] = (self.wins[i]/self.counts[i]) + math.sqrt(2*math.log(t)/self.counts[i])
                
                
def KL(p,q):
    if p == 0 :
        return 0
    elif p == 1 and q == 1:
        return 0
    elif q == 0 or p == 1 or q == 1:
        return float('inf')
    else:
        return p*math.log(p/q) + (1-p)*math.log((1-p)/(1-q))

def KLupperConfidenceBound(p, u, c, t):
    start = p
    end   = 1.0
    mid   = (start + end) / 2.0
    final = (math.log(t) + c*math.log(math.log(t))) / u
    
    while abs(start - end) > 1.0e-4:
        if KL(p, mid) > final:
            end = mid
        else:
            start = mid
        mid = (start + end) / 2.0      
    return mid   
        

class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.counts = np.zeros(num_arms)
        self.wins = np.zeros(num_arms)
        self.metric = np.zeros(num_arms)
        self.total = 0
        # You can add any other variables you need here
        # START EDITING HERE
        # END EDITING HERE
    
    def give_pull(self):
        if self.total <= self.num_arms-1:
            return self.total
        else:
            return np.argmax(self.metric)
        # START EDITING HERE
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        self.total += 1
        self.counts[arm_index] += 1
        self.wins[arm_index] += reward
        t = self.total
      
        if t >= self.num_arms:
            for i in range(self.num_arms):
                self.metric[i] = KLupperConfidenceBound((self.wins[i]/self.counts[i]), self.counts[i], 3, t)
        # START EDITING HERE
        # END EDITING HERE


class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.counts = np.zeros(num_arms)
        self.wins = np.zeros(num_arms)
        self.metric = np.zeros(num_arms)
        self.total = 0
        # You can add any other variables you need here
        # START EDITING HERE
        # END EDITING HERE
    
    def give_pull(self):
        if self.total <= self.num_arms-1 and self.counts[self.total] == 0:
            return self.total
        else:
            return np.argmax(self.metric)
    
    def get_reward(self, arm_index, reward):
        self.total += 1
        self.counts[arm_index] += 1
        self.wins[arm_index] += reward
        t = self.total
      
        if t >= self.num_arms:
            for i in range(self.num_arms):
                self.metric[i] = np.random.beta(self.wins[i] + 1, self.counts[i] - self.wins[i] + 1)
        
        
        
        
        
