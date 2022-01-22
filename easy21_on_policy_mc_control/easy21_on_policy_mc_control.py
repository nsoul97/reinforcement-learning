import sys

sys.path.insert(0, '..')

import numpy as np
import argparse
import pprint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from rl_envs.easy21 import Easy21Env
from matplotlib import cm


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

    :return: The number of episodes to sample, the discount factor of the MC algorithm, the constant n0 of the epsilon
             parameter and a plot boolean.
    """
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--n_episodes", type=check_positive_int, default=5000000,
                        help="The number of episodes to sample (DEFAULT=5000000)")

    parser.add_argument("--gamma", type=check_positive_float, default=1.0,
                        help="The discount factor of the On-Policy Monte Carlo Control algorithm. (DEFAULT=1.0)")

    parser.add_argument("--n0", type=check_positive_float, default=100.0,
                        help="The constant n0 of the epsilon parameter: epsilon(t) = n0 / (n0 + n(S(t)). "
                             "(DEFAULT=100.0)")

    parser.add_argument("--plot", action='store_true',
                        help="Plot and save as easy21_on_policy_mcc_v.jpg the state value function of the optimal "
                             "policy and as easy21_on_policy_mcc_policy.jpg the optimal policy. (DEFAULT=False)")

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='Show this help message and exit.')

    args = parser.parse_args()
    return args.n_episodes, args.gamma, args.n0, args.plot


def plot_easy21_results(n_episodes, policy, v):
    """ Plot and save two figures, one with the optimal policy and another with its state value function.

    :param n_episodes: The number of episodes that were sampled for the On-Policy MC Control algorithm.
    :param policy: The optimal policy for the Easy21 game
    :param v: The state value function of the optimal policy
    """

    # Plot the optimal policy
    fig1, axes1 = plt.subplots(1, 1, figsize=(20, 15))
    fig1.suptitle(f'Easy21 Final Policy\n(after {n_episodes} episodes)', fontsize=24)
    axes1.tick_params(left=False, bottom=False)

    policy_annot = [['H' if policy[(player_sum, dealer_card)] else 'S' for dealer_card in range(1, 11)]
                    for player_sum in range(21, 0, -1)]
    clr = [[policy[(player_sum, dealer_card)] for dealer_card in range(1, 11)] for player_sum in range(21, 0, -1)]

    sns.heatmap(ax=axes1, data=clr, annot=policy_annot, vmin=0.0, vmax=1.0, cmap='plasma',
                cbar=False, annot_kws={'fontsize': 22}, fmt='')

    axes1.set_xlabel('Dealer showing', fontsize=18)
    axes1.set_ylabel('Player sum', fontsize=18)
    axes1.set_xticklabels(range(1, 11))
    axes1.set_yticklabels(range(21, 0, -1))

    # Plot the state value function of the optimal policy
    fig2 = plt.figure(figsize=(12, 15), tight_layout=True)
    axes2 = fig2.add_subplot(111, projection='3d')
    fig2.suptitle(f'Easy21 State Value Function V*\n(after {n_episodes} episodes)', fontsize=24)

    x, y = np.meshgrid(range(1, 11), range(1, 22))
    z = np.array([[v[(player_sum, dealer_card)] for dealer_card in range(1, 11)] for player_sum in range(1, 22)])

    axes2.plot_surface(x, y, z, cmap=cm.coolwarm)
    axes2.set_xticks(range(1, 11))
    axes2.set_yticks(range(1, 22))
    axes2.set_xlim(1, 10)
    axes2.set_ylim(1, 21)
    axes2.set_zlim(-1, 1)
    axes2.set_xlabel('Dealer showing', fontsize=18)
    axes2.set_ylabel('Player sum', fontsize=18)

    fig1.savefig('easy21_on_policy_mcc_policy.jpg')
    fig2.savefig('easy21_on_policy_mcc_v.jpg')
    plt.show()


def eps_greedy_policy(env, q, n, n0, curr_state):
    """ The epsilon-greedy policy of the agent.

    :param env: The Windy Gridworld environment
    :param q: The state-action value function
    :param n: The number of times Q(s,a) has been updated
    :param n0: A constant to calculate the epsilon parameter of the epsilon greedy policy
    :param curr_state: The current state of the agent
    :return: The action of the agent.
    """
    epsilon = n0 / (n0 + np.sum(n[curr_state]))
    prob = np.full(env.action_space.n, epsilon / env.action_space.n)
    max_ind = np.where(q[curr_state] == np.max(q[curr_state]))[0]
    prob[max_ind] += (1 - epsilon) / len(max_ind)
    return np.random.choice(env.action_space.n, p=prob)


def on_policy_monte_carlo_control(env, n_episodes, gamma, n0):
    """ An implementation of the (First-Visit) On-Policy Monte Carlo algorithm. The algorithm estimates the
        optimal policy for the given environment.

    :param env: The Easy21 environment.
    :param n_episodes: The number of episodes to sample
    :param gamma: The discount factor of MC
    :param n0: The constant n0 of the epsilon parameter
    :return: The final optimal policy, the state value function and the state-action value function
    """

    q = dict()
    n = dict()
    for e in range(n_episodes):  # Loop for the given number of episodes to sample

        done = False  # Sample an episode based on the given strategy
        episode_hist = []
        curr_obs = env.reset()
        while not done:
            if curr_obs not in q:
                q[curr_obs] = np.zeros(env.action_space.n)
                n[curr_obs] = np.zeros(env.action_space.n)

            action = eps_greedy_policy(env, q, n, n0, curr_obs)
            next_obs, r, done, _ = env.step(action)
            episode_hist.append(((curr_obs, action), r))
            curr_obs = next_obs

        episode_returns = dict()  # Calculate the return for the first visit of a (state, action) pair in the episode
        g = 0
        for (s, a), r in reversed(episode_hist):
            g = r + gamma * g
            episode_returns[(s, a)] = g

        for (s, a) in episode_returns:
            n[s][a] += 1
            alpha = 1.0 / n[s][a]
            q[s][a] += alpha * (episode_returns[(s, a)] - q[s][a])

    print(n)
    policy = {state: np.argmax(q[state]) for state in q}
    v = {state: np.max(q[state]) for state in q}
    return policy, v, q


def main():
    """ Read the command line arguments, create an Easy21 environment and find the optimal strategy for the player
        without knowing the environment's dynamics. The dynamics are sampled using the On-Policy Monte Carlo Control
        algorithm. Optionally, plot two figures, one with the final state value function and another one with the
        optimal policy.
    """
    n_episodes, gamma, n0, plot = parse_args()
    env = Easy21Env()
    final_policy, v, q = on_policy_monte_carlo_control(env, n_episodes, gamma, n0)

    pp = pprint.PrettyPrinter(indent=2)
    print("Final Policy")
    pp.pprint(final_policy)

    print("\nState Value Function V*")
    pp.pprint(v)

    print("\nState-Action Value Function Q*")
    pp.pprint(q)

    if plot:
        plot_easy21_results(n_episodes, final_policy, v)


if __name__ == '__main__':
    main()
