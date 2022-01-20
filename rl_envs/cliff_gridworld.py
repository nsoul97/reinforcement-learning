import numpy as np
from gym import Env
from gym.spaces import Discrete, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class CliffGridWorldEnv(Env):
    HEIGHT = 4
    WIDTH = 12

    START_POSITION = (3, 0)
    CLIFF_POSITIONS = [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10)]
    TARGET_POSITION = (3, 11)

    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3

    MOVE_CHARS = {
        UP: '↑',
        DOWN: '↓',
        RIGHT: '→',
        LEFT: '←'}

    metadata = {"render.modes": ['rgb_array']}

    def __init__(self):

        self.action_space = Discrete(4)  # There are 4 actions: U, D, R, L
        self.observation_space = Tuple([Discrete(self.HEIGHT), Discrete(self.WIDTH)])  # There are height x width states

        self.P = {(h, w): dict() for h in range(self.HEIGHT) for w in range(self.WIDTH)}  # The transition probabilities
        for state in self.P.keys():  # P[s][a] == [(probability, nextstate, reward, done), ...]
            if self._is_done(state):
                for action in range(self.action_space.n):
                    self.P[state][action] = [(1.0, state, 0.0, True)]
            elif self._is_cliff(state):
                for action in range(self.action_space.n):
                    self.P[state][action] = [(1.0, self.START_POSITION, -100.0, False)]
            else:
                y, x = state

                next_state_u = self._limit_position(y - 1, x)
                next_state_d = self._limit_position(y + 1, x)
                next_state_r = self._limit_position(y, x + 1)
                next_state_l = self._limit_position(y, x - 1)

                self.P[state][self.UP] = [(1.0, next_state_u, -1.0, self._is_done(next_state_u))]
                self.P[state][self.DOWN] = [(1.0, next_state_d, -1.0, self._is_done(next_state_d))]
                self.P[state][self.RIGHT] = [(1.0, next_state_r, -1.0, self._is_done(next_state_r))]
                self.P[state][self.LEFT] = [(1.0, next_state_l, -1.0, self._is_done(next_state_l))]

        self.curr_state = None
        self.fig, self.axes = None, None
        self.ep_moves = None
        self.last_action = None
        self.moves_hist = dict()

    def reset(self):
        self.ep_moves = 0
        self.curr_state = self.START_POSITION
        self.moves_hist['y'] = [self.curr_state[0] + 0.5]
        self.moves_hist['x'] = [self.curr_state[1] + 0.5]
        self.last_action = None
        return self.curr_state

    def step(self, action):
        _, self.curr_state, reward, done = self.P[self.curr_state][action][0]
        self.ep_moves += 1
        self.moves_hist['y'].append(self.curr_state[0] + 0.5)
        self.moves_hist['x'].append(self.curr_state[1] + 0.5)
        self.last_action = action
        return self.curr_state, reward, done, {}

    def render(self, mode="human"):
        assert mode in self.metadata['render.modes']

        if self.fig is None:
            self.fig, self.axes = plt.subplots(1, 1)

        self.axes.cla()
        self.axes.set_title(f'Move# = {self.ep_moves}')
        if self.last_action is not None:
            self.fig.suptitle(f'The agent moved: {self.MOVE_CHARS[self.last_action]}', fontsize=22)
        clrs = [[0.5 if self._is_done((h, w)) else 1.0 if self._is_cliff((h, w)) else 0.0
                for w in range(self.WIDTH)] for h in range(self.HEIGHT)]
        annot = [['A' if (h, w) == self.curr_state else '' for w in range(self.WIDTH)] for h in range(self.HEIGHT)]
        sns.heatmap(ax=self.axes, data=clrs, annot=annot, vmin=0.0, vmax=1.0, cmap='Greys',
                    cbar=False, linewidths=1, linecolor='black', annot_kws={'fontsize': 22}, fmt='')
        self.axes.tick_params(left=False, bottom=False)

        self.axes.plot(self.moves_hist['x'], self.moves_hist['y'], linewidth=2, c='r')
        self.fig.canvas.draw()

        if self._is_cliff(self.curr_state):
            self.moves_hist['y'] = []
            self.moves_hist['x'] = []

        width, height = self.fig.get_size_inches() * self.fig.get_dpi()
        return np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)

    def close(self):
        plt.close(self.fig)
        self.fig, self.axes = None, None

    def _is_done(self, s):
        return s == self.TARGET_POSITION

    def _is_cliff(self, s):
        return s in self.CLIFF_POSITIONS

    def _limit_position(self, y, x):
        y = max(0, min(self.HEIGHT - 1, y))
        x = max(0, min(self.WIDTH - 1, x))
        return y, x
