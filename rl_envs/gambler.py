from gym import Env, spaces


class GamblerEnv(Env):

    def __init__(self, ph):

        self.ph = ph

        self.nS = 101 # There are 101 states: having a total capital of 0, 1, ..., 100$
        self.nA = 100 # There are 100 actions: staking 0, 1, 2, ..., 99$.

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)

        # The transition probability matrix, P[s][a] == [(probability, nextstate, reward, done), ...]
        self.P = {state: {action: [] for action in range(self.nA)} for state in range(self.nS)}

        for state in range(1, 100):                                 # non-terminal states
            self.P[state][0].append((1.0, state, 0.0, False))       # no stake
            for action in range(1, min(state, 100-state) + 1):      # non-zero stake

                if state + action == 100:
                    self.P[state][action].append((self.ph, 100, 1.0, True))
                else:
                    self.P[state][action].append((self.ph, state + action, 0.0, False))

                if state - action == 0:
                    self.P[state][action].append((1 - self.ph, 0, 0.0, True))
                else:
                    self.P[state][action].append((1 - self.ph, state - action, 0.0, False))

        self.P[0][0].append((1.0, 0, 0.0, True))            #terminal states 0 and 100 (staking is not possible)
        self.P[100][0].append((1.0, 100, 0.0, True))