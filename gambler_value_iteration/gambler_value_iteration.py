import sys
sys.path.insert(0, '..')

import argparse
import pprint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as an
from rl_envs.gambler import GamblerEnv


def check_probability(value):
    """ Check if the given string value represents α positive decimal number between 0.0 and 1.0.
        If so, return the float value. Otherwise, raise an error with an informative message.

    :param value: The command line input string.
    :return: The float the input string represents.
    """
    num = float(value)
    if num < 0.0 or num > 1.0:
        raise argparse.ArgumentTypeError("%s is an invalid probability value" % value)
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

    :return: The heads probability of the coin, the discount factor gamma, the sensitivity epsilon and a plot boolean.
    """
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--ph", type=check_probability, default=0.4,
                        help="The discount factor of the value iteration algorithm. (DEFAULT=0.40)")

    parser.add_argument("--gamma", type=check_positive_float, default=1.0,
                        help="The discount factor of the value iteration algorithm. (DEFAULT=1.0)")

    parser.add_argument("--epsilon", type=check_positive_float, default=1e-5,
                        help="The value iteration algorithm terminates once the value \
                              function change is less than epsilon for all states. (DEFAULT=1e-5)")

    parser.add_argument("--plot", action='store_true',
                        help="Plot and save an animation of the value function for each step of the value iteration "
                             "algorithm (gambler_vi_animation_{ph}.mp4) and an image with the final value function, "
                             "all the optimal actions and a deterministic optimal policy (gambler_results_{ph}.jpg).")

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='Show this help message and exit.')

    args = parser.parse_args()
    return args.ph, args.gamma, args.epsilon, args.plot


def plot_gambler_results(env, V_history, opt_policies, dt_opt_policy):
    """ Plot and save two figures, one with the final value function and the corresponding optimal policies and another
        where the value function progress is animated for each step of the algorithm.

    :param env: The Gambler environment
    :param V_history: A history list that contains the value function for each step of the value iteration algorithm.
    :param opt_policies: A dictionary containing all the optimal actions (stakes) for all the states (capitals)
    :param dt_opt_policy: A dictionary containing one optimal action (stake) per state (capital). In this policy,
                          the gambler always bets the smallest non-zero amount.
    """

    INTERVAL = 10     # The milliseconds during which a single frame is displayed

    fig1, axes1 = plt.subplots(3, 1, figsize=(8, 12), tight_layout=True)
    fig1.suptitle(f"Gambler's Problem Results\nph = {env.ph:.2f}", fontsize=22)
    capital = list(range(1, 100))
    ve = [V_history[-1][s] for s in capital]
    dt_action = [dt_opt_policy[s] for s in capital]
    opt_policies_capital, opt_policies_actions = zip(*[(s, a) for s in capital for a in opt_policies[s]])

    axes1[0].plot(capital, ve)
    axes1[0].set_xlabel('Capital', fontsize=18)
    axes1[0].set_ylabel('Value Function', fontsize=18)
    axes1[0].set_xticks([1, 25, 50, 75, 99])
    axes1[0].set_xlim(0, 100)
    axes1[0].set_ylim(0.0, 1.0)

    axes1[1].scatter(capital, dt_action)
    axes1[1].set_xlabel('Capital', fontsize=18)
    axes1[1].set_ylabel('A Deterministic Final Policy\n(Stake)', fontsize=18)
    axes1[1].set_xticks([1, 25, 50, 75, 99])
    axes1[1].set_xlim(0, 100)
    axes1[1].set_ylim(-1, 51)

    axes1[2].scatter(opt_policies_capital, opt_policies_actions)
    axes1[2].set_xlabel('Capital', fontsize=18)
    axes1[2].set_ylabel('Optimal Policies\n(Possible Stakes)', fontsize=18)
    axes1[2].set_xticks([1, 25, 50, 75, 99])
    axes1[2].set_xlim(0, 100)
    axes1[2].set_ylim(-1, 51)

    for ax in axes1:
        ax.tick_params(labelsize=16)

    fig2, ax2 = plt.subplots(1, 1, tight_layout=True)
    fig2.suptitle(f'Value Iteration (ph = {env.ph:.2f})\nIteration: ', fontsize=22)

    def update(frame):
        fig2.suptitle(f'Value Iteration\nph = {env.ph:.2f}\nIteration: {frame}', fontsize=22)
        ve = [V_history[frame][s] for s in capital]

        ax2.cla()
        ax2.plot(capital, ve)
        ax2.set_xlabel('Capital', fontsize=18)
        ax2.set_ylabel('Value Function', fontsize=18)
        ax2.set_xticks([1, 25, 50, 75, 99])
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0.0, 1.0)

    anim = an.FuncAnimation(fig=fig2, func=update, frames=len(V_history), repeat=False, interval=INTERVAL)

    fig1.savefig(f'gambler_results_{env.ph:.2f}.jpg')
    anim.save(f'gambler_vi_animation_{env.ph:.2f}.mp4', fps=1000 / INTERVAL)
    plt.show()


def value_iteration(env, gamma, eps):
    """ The value iteration algorithm. The algorithm is used to find the optimal policy for the given environment and
        its corresponding state value function.

    :param env: The Gambler's environment
    :param gamma: The discount factor for the value iteration algorithm
    :param eps: The sensitivity factor for the termination of the value iteration algorithm
    :return: A history list that contains the state value function for each step of the value iteration and the
             corresponding optimal policies.
    """

    V = np.zeros(env.nS)
    V_history = [V]
    diff = np.inf

    while diff > eps:
        Q = np.zeros((env.nS, env.nA))
        for state in range(env.nS):
            for action in range(env.nA):
                if env.P[state][action]:
                    for probability, nextstate, reward, _ in env.P[state][action]:
                        Q[state, action] += probability * (reward + gamma * V[nextstate])
                else:
                    Q[state, action] = -np.inf

        V_upd = np.max(Q, axis=1)
        diff = np.max(np.abs(V_upd - V))
        V = V_upd
        V_history.append(V)

    opt_policies = {state: [action for action in range(env.nA) if abs(Q[state, action] - V[state]) <= eps]
                    for state in range(env.nS)}

    return V_history, opt_policies


def main():
    """
    Create a Gambler environment based on the command line arguments and find the optimal policies for this
    environment. Optionally, plot an animation demonstrating the progress of the value iteration algorithm and an
    image with the final value function, the optimal actions per state and an optimal deterministic policy.
    """
    ph, gamma, eps, plot = parse_args()
    env = GamblerEnv(ph)
    V_history, opt_policies = value_iteration(env, gamma, eps)
    dt_opt_policy = {state: opt_policies[state][1] if len(opt_policies[state]) >= 2 else opt_policies[state][0]
                     for state in opt_policies.keys()}  # The gambler always bets the smallest non-zero amount
    print(len(V_history))
    pp = pprint.PrettyPrinter(indent=2)
    print("Optimal Policies")
    pp.pprint(opt_policies)

    print("\nA Deterministic Optimal Policy")
    pp.pprint(dt_opt_policy)

    print("\nState Value Function")
    pp.pprint(V_history[-1])

    if plot:
        plot_gambler_results(env, V_history, opt_policies, dt_opt_policy)


if __name__ == '__main__':
    main()
