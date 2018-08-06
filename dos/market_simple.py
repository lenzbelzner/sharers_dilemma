import numpy as np


class Market:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.slope = np.random.uniform(2, 4)
        self.slope_local_factor = 0 # np.random.uniform(.5, 1)

    def utilities(self, x):
        # return np.divide(x, np.abs(np.sum(x, axis=0))) ** self.slope
        return np.divide(x, np.abs(np.sum(x, axis=0)) ** self.slope)
