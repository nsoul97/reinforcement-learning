import numpy as np
from gym import Env
from gym.spaces import Discrete, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class WindyGridWorldEnv(Env):
    HEIGHT = 7
    WIDTH = 10

    START_POSITION = (3, 0)
    TARGET_POSITION = (3, 7)
    WINDS = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3
    UP_RIGHT = 4
    UP_LEFT = 5
    DOWN_RIGHT = 6
    DOWN_LEFT = 7
    NO_MOVE = 8

    MOVE_CHARS = {
        UP: '↑',
        DOWN: '↓',
        RIGHT: '→',
        LEFT: '←',
        UP_RIGHT: '↗',
        UP_LEFT: '↖',
        DOWN_RIGHT: '↘',
        DOWN_LEFT: '↙',
        NO_MOVE: '∅'
    }

    metadata = {"render.modes": ['rgb_array']}

    def __init__(self, moves):

        if moves == 'normal_moves':
            self.action_space = Discrete(4)  # There are 4 actions: U, D, R, L
        elif moves == 'king_moves':
            self.action_space = Discrete(8)  # There are 8 actions: U, D, R, L, UR, UL, DR, DL
        elif moves == 'king_extra_moves':
            self.action_space = Discrete(9)  # There are 9 actions: U, D, R, L, UR, UL, DR, DL, NO_MOVE
        else:
            assert 0
        self.observation_space = Tuple([Discrete(self.HEIGHT), Discrete(self.WIDTH)])  # There are height x width states

        self.P = {(h, w): dict() for h in range(self.HEIGHT) for w in range(self.WIDTH)}  # The transition probabilities
        for state in self.P.keys():  # P[s][a] == [(probability, nextstate, reward, done), ...]
            if self._is_done(state):
                for action in range(self.action_space.n):
                    self.P[state][action] = [(1.0, state, 0.0, True)]
            else:
                y, x = state

                next_state_u = self._limit_position(y - 1 - self.WINDS[x], x)
                next_state_d = self._limit_position(y + 1 - self.WINDS[x], x)
                next_state_r = self._limit_position(y - self.WINDS[x], x + 1)
                next_state_l = self._limit_position(y - self.WINDS[x], x - 1)

                self.P[state][self.UP] = [(1.0, next_state_u, -1.0, self._is_done(next_state_u))]
                self.P[state][self.DOWN] = [(1.0, next_state_d, -1.0, self._is_done(next_state_d))]
                self.P[state][self.RIGHT] = [(1.0, next_state_r, -1.0, self._is_done(next_state_r))]
                self.P[state][self.LEFT] = [(1.0, next_state_l, -1.0, self._is_done(next_state_l))]

                if moves != 'normal_moves':
                    next_state_ur = self._limit_position(y - 1 - self.WINDS[x], x + 1)
                    next_state_ul = self._limit_position(y - 1 - self.WINDS[x], x - 1)
                    next_state_dr = self._limit_position(y + 1 - self.WINDS[x], x + 1)
                    next_state_dl = self._limit_position(y + 1 - self.WINDS[x], x - 1)

                    self.P[state][self.UP_RIGHT] = [(1.0, next_state_ur, -1.0, self._is_done(next_state_ur))]
                    self.P[state][self.UP_LEFT] = [(1.0, next_state_ul, -1.0, self._is_done(next_state_ul))]
                    self.P[state][self.DOWN_RIGHT] = [(1.0, next_state_dr, -1.0, self._is_done(next_state_dr))]
                    self.P[state][self.DOWN_LEFT] = [(1.0, next_state_dl, -1.0, self._is_done(next_state_dl))]

                if moves == 'king_extra_moves':
                    next_state_nm = self._limit_position(y - self.WINDS[x], x)
                    self.P[state][self.NO_MOVE] = [(1.0, next_state_nm, -1.0, self._is_done(next_state_nm))]

        self.curr_state = None
        self.fig, self.axes = None, None
        self.ep_moves = None
        self.last_action = None
        self.moves_hist = dict()
        plt.ion()

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
        if self.last_action:
            self.fig.suptitle(f'The agent moved: {self.MOVE_CHARS[self.last_action]}', fontsize=22)
        clrs = [[0.5 if (h, w) == self.TARGET_POSITION else 0 for w in range(self.WIDTH)] for h in range(self.HEIGHT)]
        annot = [['A' if (h, w) == self.curr_state else '' for w in range(self.WIDTH)] for h in range(self.HEIGHT)]
        sns.heatmap(ax=self.axes, data=clrs, annot=annot, vmin=0.0, vmax=1.0, cmap='Greys',
                    cbar=False, linewidths=1, linecolor='black', annot_kws={'fontsize': 22}, fmt='',
                    xticklabels=self.WINDS, yticklabels=False)
        self.axes.tick_params(bottom=False)

        self.axes.plot(self.moves_hist['x'], self.moves_hist['y'], linewidth=2, c='r')
        self.fig.canvas.draw()
        width, height = self.fig.get_size_inches() * self.fig.get_dpi()
        return np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)

    def close(self):
        plt.close(self.fig)

    def _is_done(self, s):
        return s == self.TARGET_POSITION

    def _limit_position(self, y, x):
        y = max(0, min(self.HEIGHT - 1, y))
        x = max(0, min(self.WIDTH - 1, x))
        return y, x
