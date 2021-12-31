import sys
sys.path.insert(0, '..')

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as an
import seaborn as sns
from rl_envs.gridworld import GridWorldEnv


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

    :return: The height and width of the grid, the discount factor gamma, the sensitivity epsilon, a plot boolean.
    """
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--height", type=check_positive_int, default=4,
                        help="The height of the grid. (DEFAULT=4)")

    parser.add_argument("--width", type=check_positive_int, default=4,
                        help="The width of the grid. (DEFAULT=4)")

    parser.add_argument("--gamma", type=check_positive_float, default=1.0,
                        help="The discount factor of the iterative policy evaluation algorithm. (DEFAULT=1.0)")

    parser.add_argument("--epsilon", type=check_positive_float, default=1e-5,
                        help="The iterative policy evaluation algorithm terminates once the value function change is \
                        less than epsilon for all states. (DEFAULT=1e-5)")

    parser.add_argument("--plot", action='store_true',
                        help="Plot and save (as gridworld_ipe_animation.gif) the results of the iterative \
                        policy evaluation algorithm per iteration. (DEFAULT=False)")

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='Show this help message and exit.')

    args = parser.parse_args()
    return args.height, args.width, args.gamma, args.epsilon, args.plot


def policy_eval_animation(env, V_history):
    """ Create an animation plot that demonstrates how the iterative policy evaluation algorithm progresses in the given
        GridWorld environment.

    :param env: The GridWorld environment.
    :param V_history: The history of the iterative policy evaluation algorithms.
    """

    INTERVAL = 1000

    fig, ax = plt.subplots(1, 1)
    ax.tick_params(left=False, bottom=False)

    colors = np.zeros((env.height, env.width))
    for y in range(env.height):
        for x in range(env.width):
            if env.is_done(env.coords_to_state((y, x))):
                colors[y, x] = 0.5

    def update(frame):
        V = V_history[frame].reshape(env.height, env.width)
        ax.cla()
        fig.suptitle(f"Iterative Policy Evaluation\nIteration: {frame}")
        sns.heatmap(ax=ax, data=colors, annot=V, vmin=0.0, vmax=1.0, cmap='Greys', cbar=False, linewidths=1, linecolor='black',
                           annot_kws={'fontsize': 22}, fmt='.2f')

    anim = an.FuncAnimation(fig=fig, func=update, frames=len(V_history), repeat=False, interval=INTERVAL)
    anim.save('gridworld_ipe_animation.gif', writer='imagemagick', fps=1000/INTERVAL)
    plt.show()


def iterative_policy_evaluation(env, policy, gamma, eps):
    """ The iterative policy evaluation algorithm.

    :param env: The GridWorld environment
    :param policy: The policy to be evaluated
    :param gamma: The discount factor
    :param eps: The sensitivity factor for the termination of the algorithm
    :return: The list V_history containing the values of the environment's states for each step of the algorithm.
    """

    V_history = []

    diff = 100
    V = np.zeros(env.nS)
    V_history.append(V)
    while diff > eps:
        V_upd = np.zeros_like(V)
        for state in range(env.nS):
            for action in policy[state].keys():
                for (probability, nextstate, reward, _) in env.P[state][action]:
                    V_upd[state] += policy[state][action] * probability * (reward + gamma * V[nextstate])

        diff = np.max(np.abs(V_upd - V))
        V = V_upd
        V_history.append(V)

    return V_history


def main():
    """ Create a GridWorld environment based on the command line arguments and evaluate a random policy for this
        environment. Optionally, plot an animation demonstrating the progress of the iterative policy evaluation
        algorithm.
    """
    height, width, gamma, epsilon, plot = parse_args()
    env = GridWorldEnv(height, width)
    random_policy = {state: {action: 1.0/env.nA for action in range(env.nA)} for state in range(env.nS)}
    V_history = iterative_policy_evaluation(env, random_policy, gamma, epsilon)
    V = V_history[-1].reshape(height, width)
    print(V)

    if plot:
        policy_eval_animation(env, V_history)


if __name__ == '__main__':
    main()