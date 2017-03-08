import numpy as np
from numpy.random import rand, randint

class dp_agent(object):
    """
    Implements a tabular Q-learning agent using the TD(0) algorithm
    """
    def __init__(self, num_observations, num_actions, discount_rate = 0.9):

        self.num_observations   = num_observations
        self.num_actions        = num_actions
        self.table              = np.random.rand(num_observations, num_actions)

        self.discount_rate      = discount_rate
        self.step_size          = 0.1
        self.epsilon            = 0.1
        self.last_action        = None
        self.last_observation   = None
        self.last_reward        = None
        self.new_observation    = None

    def action(self,observation):
        if rand() < self.epsilon:
            action = randint(self.num_actions) # exploration: returns a random action
        else:
            action = np.argmax(self.table[observation,:]) # returns the best action according to current Q table
        return action

    def store(self, prev_observation, prev_action, reward, new_observation):
        self.last_action = prev_action
        self.last_observation = prev_observation
        self.last_reward = reward
        self.new_observation = new_observation

    def training_step(self):
        if self.last_reward is not None:
            value_new_obs = np.max(self.table[self.new_observation,:])
            value_last_obs = np.max(self.table[self.last_observation,:])
            temporal_difference = self.last_reward + self.discount_rate * value_new_obs - value_last_obs
            self.table[self.last_observation, self.last_action] += self.step_size * temporal_difference
