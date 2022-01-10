from gym import Env
from gym.utils import seeding


class BlackjackEnv(Env):

    STICK = 0
    HIT = 1

    CARDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

    def __init__(self):

        self.np_random = None

        self.dealer_card = None
        self.player_sum = None
        self.usable_ace = None
        self.player_tot_cards = None
        self.info = None

        self._seed()

    def reset(self):
        self.player_tot_cards = 0
        self.player_sum = 0
        self.usable_ace = False
        self.info = {'player': [], 'dealer': []}

        while self.player_sum < 12:
            c = self._draw_card()
            self.info['player'].append(c)
            self.player_tot_cards += 1
            if c == 1 and self.player_sum + 11 <= 21:
                self.player_sum += 11
                self.usable_ace = True
            else:
                self.player_sum += c

        self.dealer_card = self._draw_card()
        self.info['dealer'].append(self.dealer_card)
        return self._get_obs()

    def step(self, action):

        if action:                                                  # The player hits.
            c = self._draw_card()
            self.info['player'].append(c)
            self.player_tot_cards += 1
            if self.player_sum + c > 21 and self.usable_ace:        # The ace in the player's hand is no longer usable
                self.player_sum += c - 10
                self.usable_ace = False
            elif self.player_sum + c <= 21:                         # The player did not go bust in this turn
                self.player_sum += c
            else:                                                   # The player went bust in this turn
                return self._get_obs(), -1.0, True, self.info

            return self._get_obs(), 0.0, False, self.info

        else:                                                       # The player sticks. It's the dealer's turn now.
            if self.dealer_card == 1:
                dealer_ace = True
                dealer_sum = 11
            else:
                dealer_ace = False
                dealer_sum = self.dealer_card
            dealer_tot_cards = 1

            while dealer_sum < 17:                                  # The dealer hits until the sum of his cards is >=17
                c = self._draw_card()
                self.info['dealer'].append(c)
                dealer_tot_cards += 1
                if c == 1 and dealer_sum + 11 <= 21:                # The dealer draws an ace that can be used as an 11
                    dealer_sum += 11
                    dealer_ace = True
                elif dealer_sum + c > 21 and dealer_ace:            # The ace in the dealer's hand is no longer usable
                    dealer_sum += c - 10
                    dealer_ace = False
                else:
                    dealer_sum += c
            if dealer_sum > 21:                                     # The dealer went bust, the player won
                return self._get_obs(), 1.0, True, self.info
            else:                                                   # The dealer did not go bust.
                if self.player_sum > dealer_sum:                    # The player had a hand closer to 21 than the dealer
                    return self._get_obs(), 1.0, True, self.info
                elif self.player_sum < dealer_sum:                  # The dealer had a hand closer to 21 than the dealer
                    return self._get_obs(), -1.0, True, self.info
                else:                                           # The sum of the dealer's and the player's hand is equal
                    if self.player_sum < 21 or (self.player_tot_cards == 2 and dealer_tot_cards == 2):
                        return self._get_obs(), 0.0, True, self.info   # The sum was less than 21 or both had a natural 21
                    elif self.player_tot_cards == 2:
                        return self._get_obs(), 1.0, True, self.info   # Only the player has a natural blackjack.
                    elif dealer_tot_cards == 2:
                        return self._get_obs(), -1.0, True, self.info  # Only the dealer has a natural blackjack.
                    else:
                        return self._get_obs(), 0.0, True, self.info   # Neither the player nor the dealer had a natural 21.

    def _get_obs(self):
        return self.player_sum, self.dealer_card, self.usable_ace

    def _draw_card(self):
        c = self.np_random.randint(13)
        return self.CARDS[c]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
