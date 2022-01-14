import sys
sys.path.insert(0, '..')

import numpy as np
import argparse
import pprint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
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

    :return: The number of episodes to sample, the discount factor of the off-policy MC algorithm, a plot boolean.
    """
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--n_episodes", type=check_positive_int, default=5000000,
                        help="The number of episodes to sample using the behavior policy (DEFAULT=5000000)")

    parser.add_argument("--gamma", type=check_positive_float, default=1.0,
                        help="The discount factor of the Monte Carlo Prediction algorithm. (DEFAULT=1.0)")

    parser.add_argument("--plot", action='store_true',
                        help="Plot and save as blackjack_offpolicy_mcc_wis_v.jpg the state value function of the final "
                             "target (optimal) policy and as blackjack_offpolicy_mcc_wis_policy.jpg the target policy. "
                             "(DEFAULT=False)")

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='Show this help message and exit.')

    args = parser.parse_args()
    return args.n_episodes, args.gamma, args.plot


def plot_blackjack_results(n_episodes, policy, v):
    """ Plot and save two figures, one with the optimal target policy and another with its state value function.

    :param n_episodes: The number of episodes that were sampled for the Off-Policy MC Control with Weighted Importance
                       Sampling algorithm
    :param policy: The optimal target policy for the blackjack game
    :param v: The state value function of the optimal policy
    """

    # Plot the optimal policy
    fig1, axes1 = plt.subplots(2, 1, figsize=(20, 15))
    fig1.suptitle(f'Blackjack Optimal Policy\n(after {n_episodes} episodes)', fontsize=24)
    for ax in axes1:
        ax.tick_params(left=False, bottom=False)

    usable_ace_policy = [['H' if policy[(player_sum, dealer_card, True)] else 'S' for dealer_card in range(1, 11)]
                         for player_sum in range(21, 11, -1)]
    usable_ace_clr = [[policy[(player_sum, dealer_card, True)] for dealer_card in range(1, 11)]
                      for player_sum in range(21, 11, -1)]

    no_usable_ace_policy = [['H' if policy[(player_sum, dealer_card, False)] else 'S' for dealer_card in range(1, 11)]
                            for player_sum in range(21, 11, -1)]
    no_usable_ace_clr = [[policy[(player_sum, dealer_card, False)] for dealer_card in range(1, 11)]
                         for player_sum in range(21, 11, -1)]

    sns.heatmap(ax=axes1[0], data=usable_ace_clr, annot=usable_ace_policy, vmin=0.0, vmax=1.0, cmap='plasma',
                cbar=False, annot_kws={'fontsize': 22}, fmt='')

    sns.heatmap(ax=axes1[1], data=no_usable_ace_clr, annot=no_usable_ace_policy, vmin=0.0, vmax=1.0, cmap='plasma',
                cbar=False, annot_kws={'fontsize': 22}, fmt='')

    axes1[0].set_title('Usable ace', fontsize=22)
    axes1[1].set_title('No usable ace', fontsize=22)
    for ax in axes1:
        ax.set_xlabel('Dealer showing', fontsize=18)
        ax.set_ylabel('Player sum', fontsize=18)
        ax.set_xticklabels(['A', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
        ax.set_yticklabels(['21', '20', '19', '18', '17', '16', '15', '14', '13', '12'])

    # Plot the state value function of the optimal policy
    fig2 = plt.figure(figsize=(12, 15))
    axes2 = [fig2.add_subplot(211, projection='3d'), fig2.add_subplot(212, projection='3d')]
    fig2.suptitle(f'Blackjack State Value Function V*\n(after {n_episodes} episodes)', fontsize=24)

    x, y = np.meshgrid(range(1, 11), range(12, 22))
    v_usable_ace = np.array([[v[(player_sum, dealer_card, True)] for dealer_card in range(1, 11)]
                             for player_sum in range(12, 22)])
    v_no_usable_ace = np.array([[v[(player_sum, dealer_card, False)] for dealer_card in range(1, 11)]
                                for player_sum in range(12, 22)])

    axes2[0].plot_surface(x, y, v_usable_ace)
    axes2[1].plot_surface(x, y, v_no_usable_ace)

    axes2[0].set_title('Usable ace', fontsize=22)
    axes2[1].set_title('No usable ace', fontsize=22)
    for ax in axes2:
        ax.set_xlim(1, 10)
        ax.set_ylim(12, 21)
        ax.set_zlim(-1, 1)
        ax.set_xlabel('Dealer showing', fontsize=18)
        ax.set_ylabel('Player sum', fontsize=18)

    fig1.savefig('blackjack_offpolicy_mcc_wis_policy.jpg')
    fig2.savefig('blackjack_offpolicy_mcc_wis_v.jpg')
    plt.show()


def off_policy_monte_carlo_wis_control(env, n_episodes, gamma):
    """ An implementation of the Off-Policy (Every-Visit) Monte Carlo Control with Weighted Importance Sampling
        Algorithm. The target policy of the algorithm converges to the optimal policy for the given environment.

    :param env: The Blackjack environment.
    :param n_episodes: The number of episodes to sample using the behavior policy
    :param gamma: The discount factor of MC
    :return: The final optimal policy, its state value function and its state-action value function
    """

    def behavior_policy(s, e):
        epsilon = np.sqrt(33333/(e+33333))
        prob = np.full(env.action_space.n, epsilon/env.action_space.n)
        prob[np.argmax(q[s])] += 1-epsilon
        a = np.random.choice(env.action_space.n, p=prob)
        return a, prob[a]

    q = dict()
    c = dict()
    for e in range(n_episodes):  # Loop for the given number of episodes to sample

        done = False  # Sample an episode based on the given strategy
        episode_hist = []

        curr_obs = env.reset(env.observation_space.sample())
        while not done:
            if curr_obs not in q:
                q[curr_obs] = np.zeros(env.action_space.n)
                c[curr_obs] = np.zeros(env.action_space.n)

            action, prob = behavior_policy(curr_obs, e)
            next_obs, r, done, _ = env.step(action)
            episode_hist.append(((curr_obs, action), prob, r))
            curr_obs = next_obs

        g = 0                   # (state, action) pair in the episode
        w = 1
        for (s, a), prob, r in reversed(episode_hist):
            g = r + gamma * g
            c[s][a] += w
            q[s][a] += w / c[s][a] * (g - q[s][a])
            if a != np.argmax(q[s]):
                break
            w *= 1/prob

    policy = {state: np.argmax(q[state]) for state in q}
    v = {state: np.max(q[state]) for state in q}
    return policy, v, q


def main():
    """
    Read the command line arguments, create a Blackjack environment and find the optimal strategy for the player
    without knowing the environment's dynamics. The dynamics are sampled using the Off-Policy (Every-Visit) Monte Carlo
    Control with Weighted Importance Sampling algorithm. Optionally, plot two figures, one with the optimal target
    policy and another with its state value function.
    """
    n_episodes, gamma, plot = parse_args()
    env = BlackjackEnv()
    final_policy, v, q = off_policy_monte_carlo_wis_control(env, n_episodes, gamma)

    pp = pprint.PrettyPrinter(indent=2)
    print("Final Target Policy")
    pp.pprint(final_policy)

    print("\nState Value Function V*")
    pp.pprint(v)

    print("\nState-Action Value Function Q*")
    pp.pprint(q)

    if plot:
        plot_blackjack_results(n_episodes, final_policy, v)


if __name__ == '__main__':
    main()
