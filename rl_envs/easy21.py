from gym import Env
from gym.spaces import Discrete, Tuple
import numpy as np


class Easy21Env(Env):
    # There are 2 colors: red & black. Red and black cards are drawn with probability 1/3 and 2/3, respectively.
    RED = 0
    BLACK = 1
    COLORS = [RED, BLACK]
    COLOR_PROB = [1/3, 2/3]

    # There are 2 actions: hit & stick
    STICK = 0
    HIT = 1

    # The deck's cards range from 1 to 10 (uniformly distributed). There are no aces or face cards in the game.
    CARDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def __init__(self):
        self.action_space = Discrete(2)  # There are 2 actions: hit & stick
        self.observation_space = Tuple((Discrete(21), Discrete(10)))  # The state space is player sum x dealer 1st card.

        self.player_sum = None
        self.dealer_card_value = None
        self.info = None

    def reset(self, state=None):
        if state is None:
            player_card = self._draw_card(sample_color=False)  # The player draws a visible black card
            dealer_card = self._draw_card(sample_color=False)  # The dealer draws a visible black card

            self.info = {'player': [player_card], 'dealer': [dealer_card]}
            self.player_sum, _ = player_card
            self.dealer_card_value, _ = dealer_card
        else:
            self.player_sum = state[0] + 1
            self.dealer_card_value = state[1] + 1
            self.info = {'player': [self.player_sum], 'dealer': [self.dealer_card_value]}

        return self._get_obs()

    def step(self, action):

        if action == self.HIT:
            card = self._draw_card()
            self.info['player'].append(card)
            card_value, card_color = card

            if card_color == self.BLACK:
                if self.player_sum + card_value <= 21:  # The player did not go bust, the game continues.
                    self.player_sum += card_value
                else:  # The player went bust, the dealer wins.
                    return self._get_obs(), -1.0, True, self.info
            else:
                if self.player_sum - card_value >= 1:  # The player did not go bust, the game continues.
                    self.player_sum -= card_value
                else:  # The player went bust, the dealer wins.
                    return self._get_obs(), -1.0, True, self.info

            return self._get_obs(), 0.0, False, self.info

        else:
            dealer_sum = self.dealer_card_value
            while dealer_sum < 17:
                card = self._draw_card()
                self.info['dealer'].append(card)
                card_value, card_color = card

                if card_color == self.BLACK:
                    if dealer_sum + card_value <= 21:  # The dealer did not go bust, the game continues.
                        dealer_sum += card_value
                    else:  # The dealer went bust, the player wins.
                        return self._get_obs(), 1.0, True, self.info
                else:
                    if dealer_sum - card_value >= 1:  # The dealer did not go bust, the game continues.
                        dealer_sum -= card_value
                    else:  # The dealer went bust, the player wins.
                        return self._get_obs(), 1.0, True, self.info

            # Both sums are in between 1 and 21. The player and the dealer did not go bust.
            if dealer_sum < self.player_sum:  # The player wins, because his sum is closer to 21.
                return self._get_obs(), 1.0, True, self.info
            elif dealer_sum > self.player_sum:  # The player loses, because the dealer's sum is closer to 21.
                return self._get_obs(), -1.0, True, self.info
            else:  # This game is a draw, because the sums are equal.
                return self._get_obs(), 0.0, True, self.info

    def _draw_card(self, sample_color=True):
        color = np.random.choice(self.COLORS, p=self.COLOR_PROB) if sample_color else self.BLACK
        card = np.random.choice(self.CARDS)
        return card, color

    def _get_obs(self):
        return self.player_sum, self.dealer_card_value
