import numpy as np
from gym.envs.toy_text.discrete import DiscreteEnv

class GridWorldEnv(DiscreteEnv):

    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3

    def __init__(self, height, width):

        self.height = height
        self.width = width

        self.nA = 4                 # There are 4 actions: left, right, up, down
        self.nS = height * width    # Each cell of the grid is a different state.
        self.isd = np.ones((height, width))/self.nS  # The agent starts from a cell that is chosen uniformly at random.

        self.P = {state: dict() for state in range(self.nS)}  # The transition probability matrix
        for state in range(self.nS):                          # P[s][a] == [(probability, nextstate, reward, done), ...]
            if self.is_done(state):
                for action in range(self.nA):
                    self.P[state][action] = [(1.0, state, 0.0, True)]
            else:
                y, x = self.state_to_coords(state)

                next_state_up = self.coords_to_state((max([0, y-1]), x))
                next_state_down = self.coords_to_state((min([self.height-1, y+1]), x))
                next_state_right = self.coords_to_state((y, min([self.width-1, x+1])))
                next_state_left = self.coords_to_state((y, max([0, x-1])))

                self.P[state][self.UP] = [(1.0, next_state_up, -1.0, self.is_done(next_state_up))]
                self.P[state][self.DOWN] = [(1.0, next_state_down, -1.0, self.is_done(next_state_down))]
                self.P[state][self.RIGHT] = [(1.0, next_state_right, -1.0, self.is_done(next_state_right))]
                self.P[state][self.LEFT] = [(1.0, next_state_left, -1.0, self.is_done(next_state_left))]

        super().__init__(self.nS, self.nA, self.P, self.isd)

    def is_done(self, s):
        return s == 0 or s == self.nS-1

    def state_to_coords(self, s):
        y = s // self.width
        x = s % self.width
        return y, x

    def coords_to_state(self, coords):
        y, x = coords
        return y * self.width + x
