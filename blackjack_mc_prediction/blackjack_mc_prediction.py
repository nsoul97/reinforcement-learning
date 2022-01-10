import sys
sys.path.insert(0, '..')

import numpy as np
import argparse
import pprint
import matplotlib.pyplot as plt
import matplotlib.animation as an
from mpl_toolkits.mplot3d import Axes3D
from rl_envs.blackjack import BlackjackEnv


def check_positive_int(value):
    """ Check if the given string value represents α positive integer.
        If so, return the integer value. Otherwise, raise an error with an informative message.

    :param value: The command line input string.
    :return: The integer the input string represents.
    """
    num = int(value)
    if num <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return num


def check_threshold_int(value, low, high):
    """ Check if the given string value represents αn integer in [low, high].
        If so, return the integer value. Otherwise, raise an error with an informative message.

    :param value: The command line input string.
    :param low: The low threshold of the interval.
    :param high: The high threshold of the interval.
    :return: The integer the input string represents.
    """
    num = int(value)
    if num < low or num > high:
        raise argparse.ArgumentTypeError("%s is an invalid int value in [%d, %d]" % (value, low, high))
    return num


def check_positive_float(value):
    """ Check if the given string value represents α positive decimal number.
        If so, return the float value. Otherwise, raise an error with an informative message.

    :param value: The command line input string.
    :return: The float the input string represents.
    """
    num = float(value)
    if num <= 0.0:
        raise argparse.ArgumentTypeError("%s is an invalid positive float value" % value)
    return num


def parse_args():
    """ Create a help menu with informative messages.
        Parse the arguments given in the command line and return the given or the default values.

    :return: The number of episodes to sample, the threshold of the player's strategy, the discount factor of the MC
             algorithm, a plot boolean.
    """
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--n_episodes", type=check_positive_int, default=100000,
                        help="The number of episodes to sample (DEFAULT=100000)")

    parser.add_argument("--threshold", type=lambda v: check_threshold_int(v, 12, 21), default=20,
                        help="The player hits if the sum is < threshold, otherwise he sticks. (DEFAULT=20)")

    parser.add_argument("--gamma", type=check_positive_float, default=1.0,
                        help="The discount factor of the Monte Carlo Prediction algorithm. (DEFAULT=1.0)")

    parser.add_argument("--plot", action='store_true',
                        help="Plot and save as blackjack_mcp_episodes_{n_episodes}.gif the value function per episode "
                             "of the Monte Carlo Prediction algorithm and as blackjack_mcp_results_{n_episodes}.jpg "
                             "the final results of the algorithm (DEFAULT=False)")

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='Show this help message and exit.')

    args = parser.parse_args()
    return args.n_episodes, args.threshold, args.gamma, args.plot


def plot_blackjack_results(n_episodes, v_history):
    """ Plot and save two figures, one with the final value function and another where the value function progress is
     animated.

    :param n_episodes: The number of episodes that were sampled for the MC Prediction algorithm
    :param v_history: A list containing the state value function for each episode of MC
    """

    STEP = 10000
    INTERVAL = 500

    fig1 = plt.figure(figsize=(12, 20))
    ax11 = fig1.add_subplot(211, projection='3d')
    ax12 = fig1.add_subplot(212, projection='3d')
    for ax in [ax11, ax12]:
        ax.set_xlabel('Dealer showing', fontsize=18)
        ax.set_ylabel('Player sum', fontsize=18)
        ax.set_xlim(1, 10)
        ax.set_ylim(12, 21)
        ax.set_zlim(-1, 1)
    ax11.set_title('Usable ace', fontsize=18)
    ax12.set_title('No usable ace', fontsize=18)
    fig1.suptitle(f"Monte Carlo Prediction\n", fontsize=22)

    def update(fig, ax1, ax2, episode):
        vf = v_history[episode]

        ax1.cla()
        ax2.cla()
        for ax in [ax1, ax2]:
            ax.set_xlabel('Dealer showing', fontsize=18)
            ax.set_ylabel('Player sum', fontsize=18)
            ax.set_xlim(1, 10)
            ax.set_ylim(12, 21)
            ax.set_zlim(-1, 1)
        ax1.set_title('Usable ace', fontsize=18)
        ax2.set_title('No usable ace', fontsize=18)
        fig.suptitle(f"Monte Carlo Prediction\nepisode# = {episode+1}", fontsize=22)

        player_hand = np.arange(12, 22)
        dealer_card = np.arange(1, 11)

        x, y = np.meshgrid(dealer_card, player_hand)
        v_usable_ace = np.array([[vf.get((i, j, True), 0) for j in dealer_card] for i in player_hand])
        v_no_usable_ace = np.array([[vf.get((i, j, False), 0) for j in dealer_card] for i in player_hand])

        ax1.plot_surface(x, y, v_usable_ace)
        ax2.plot_surface(x, y, v_no_usable_ace)

    anim = an.FuncAnimation(fig1, lambda frame: update(fig1, ax11, ax12, STEP * (frame+1) - 1), n_episodes//STEP,
                            repeat=False, interval=INTERVAL)
    anim.save(f'blackjack_mcp_episodes_{n_episodes}.gif', writer='imagemagick', fps=1000 / INTERVAL)

    fig2 = plt.figure(figsize=(12, 20))
    ax21 = fig2.add_subplot(211, projection='3d')
    ax22 = fig2.add_subplot(212, projection='3d')
    update(fig2, ax21, ax22, len(v_history)-1)
    fig2.savefig(f'blackjack_mcp_results_{n_episodes}.jpg')

    plt.show()


def monte_carlo_prediction(env, n_episodes, gamma, strategy_fn):
    """ The Monte Carlo Prediction Algorithm to evaluate the state value function for the given strategy.

    :param env: The Blackjack environment
    :param n_episodes: The number of episodes to sample for the MC algorithm
    :param gamma: The discount factor of MC
    :param strategy_fn: The strategy of the agent given an observation of the environment
    :return: A list containing the state value function for each episode of MC
    """

    V_hist = []
    mc_state = dict()
    for i in range(n_episodes):                         # Loop for the given number of episodes to sample

        done = False                                    # Sample an episode based on the given strategy
        episode_hist = []

        curr_obs = env.reset()
        while not done:
            action = strategy_fn(curr_obs)              # The action to take based on the given strategy
            next_obs, r, done, _ = env.step(action)
            episode_hist.append((curr_obs, r))
            curr_obs = next_obs

        episode_state_returns = dict()  # Calculate the return for the first visit of a state in the sampled episode
        g = 0
        for obs, r in reversed(episode_hist):
            g = r + gamma * g
            episode_state_returns[obs] = g

        for state, g in episode_state_returns.items():
            exp_g, n = mc_state.get(state, (0, 0))
            n += 1
            exp_g += 1/n * (g - exp_g)
            mc_state[state] = (exp_g, n)

        V_hist.append({state: mc_state[state][0] for state in mc_state})

    return V_hist


def strategy(obs, threshold):
    """ The strategy of the player. If his hand is less than the given threshold, he hits. Otherwise, he sticks.

    :param obs: The observation of the player from the Blackjack environment (player_hand, dealer_card, usable_ace)
    :param threshold: The threshold of the player's strategy.
    :return: The action the player selects
    """
    player_hand, _, _ = obs

    if player_hand < threshold:
        return BlackjackEnv.HIT
    else:
        return BlackjackEnv.STICK


def main():
    """
    Read the command line arguments, create a Blackjack environment and evaluate the state value function of the
    player's strategy without knowing the environment's dynamics. The dynamics are sampled using the Monte Carlo
    Prediction algorithm. Optionally, plot an animation demonstrating the progress of the MC algorithm and an
    image with the final value function.
    """
    n_episodes, threshold, gamma, plot = parse_args()
    env = BlackjackEnv()
    v_history = monte_carlo_prediction(env, n_episodes, gamma, strategy_fn=lambda obs: strategy(obs, threshold))

    final_v = v_history[-1]
    pp = pprint.PrettyPrinter(indent=2)
    print("State Value Function")
    pp.pprint(final_v)

    if plot:
        plot_blackjack_results(n_episodes, v_history)


if __name__=='__main__':
    main()
