import numpy as np
from numpy.random import randint, binomial, exponential

class network_rm(object):
    """
    Implements a simple dynamic matching environment,
    where arrivals and departures are generated stochastically.
    """
    def __init__(self,
                inventory,
                demand_values,
                demand_types,
                demand_arrivals,
                inter_arrival_time):
        #problem parameters
        self.demand_types = demand_types
        self.demand_values = demand_values
        self.demand_arrivals = demand_arrivals
        self.demand_num = len(demand_types)
        self.inter_arrival_time = inter_arrival_time
        self.inventory_size = len(inventory)
        #state space
        self.state = np.concatenate((inventory, np.zeros(self.inventory_size)))
        #the last item in the inventory is time remaining.

        # rewards
        self.last_reward = 0
        self.total_reward = 0
        self.prev_arrival = None
        # need to implement these
        self.observation_shape = self.state.shape
        self.terminate = False
        self.num_actions = 2

    def collect_reward(self):
        self.total_reward += self.last_reward
        return self.last_reward

    def observe(self):
        return self.state

    def transition(self):
        self.prev_arrival = randint(self.demand_num)
        self.state[self.inventory_size:] = self.demand_types[self.prev_arrival]

    def perform_action(self, accept):
        self.last_reward = 0
        if accept:
            if all(self.state[0:(self.inventory_size)] >= self.state[(self.inventory_size):]):
                self.state[0:(self.inventory_size)] -= self.state[(self.inventory_size):]
                self.last_reward = self.demand_values[self.prev_arrival]


        self.state[self.inventory_size-1]-= exponential(self.inter_arrival_time)
        if self.state[self.inventory_size-1] <= 0:
            self.terminate = True
            #need to make sure that we backprop the result of being in termination state
