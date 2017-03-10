import numpy as np
from numpy.random import rand, binomial

class simple_matching(object):
    """
    Implements a simple dynamic matching environment,
    where arrivals and departures are generated stochastically.
    """
    def __init__(self,
                types,
                weight_matrix,
                arrival_probabilities,
                departure_probabilities):
        self.types = types
        self.weights = weight_matrix
        self.arrival_prob = arrival_probabilities
        self.departure_prob = departure_probabilities
        self.state = np.zeros(len(types))
        self.observation_shape = self.state.shape
        self.last_match_reward = 0
        self.num_actions = len(types)**2
        self.total_reward = 0
        
    def collect_reward(self):
        return self.last_match_reward

    def observe(self):
        return self.state

    def perform_action(self, matching):
        # matching is given as a np array of the number of matches for each pair of types
        # the matching array should be symmetric.
        if (self.state >= np.sum(matching,1)).all():
            self.last_match_reward = np.sum(np.multiply(self.weights,matching))
            self.total_reward += self.last_match_reward
            self.state -= np.sum(matching,1)
        else:
            self.last_match_reward = 0
            #print("Inadmissible matching")
        self.arrivals()
        self.departures()

    def arrivals(self):
        self.state += ( rand(len(self.types)) <= self.arrival_prob )

    def departures(self):
        for i in range(len(self.types)):
            self.state[i] -= binomial(self.state[i], self.departure_prob[i])
