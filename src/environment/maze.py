import numpy as np

class maze(object):
    """
    Implements a 2d maze environment represented as an array.
    The agent can move on the array and recieves rewards based on its position
    at the end of each turn.
    """
    def __init__(self, grid):
        self.board = grid
        self.position = np.zeros(2) #start at top left
        self.actions = [np.array([-1,0]), np.array([1,0]),np.array([0,-1]),np.array([0,1])]
        self.total_reward = 0
        self.observation_size = np.size(self.board)
        self.observation_shape = (1,)
        self.num_actions = len(self.actions)

    def collect_reward(self):
        reward = self.board[int(self.position[0]),int(self.position[1])]
        self.total_reward += reward
        return reward

    def observe(self):
        #print(self.position)
        return int(self.position[0] * len(self.board) + self.position[1])

    def perform_action(self, action_id):
        self.position += self.actions[action_id]
        for i in range(len(self.position)):
            self.position[i] = max(self.position[i], 0)
            self.position[i] = min(self.position[i], len(self.board)-1)
