import sys
sys.path.insert(0, '..')

import argparse
import pprint
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
                        help="The discount factor of the value iteration algorithm. (DEFAULT=1.0)")

    parser.add_argument("--epsilon", type=check_positive_float, default=1e-5,
                        help="The value iteration algorithm terminates once the value \
                              function change is less than epsilon for all states. (DEFAULT=1e-5)")

    parser.add_argument("--plot", action='store_true',
                        help="Plot and save an animation (gridworld_vi_animation.gif) of the value function for each "
                             "step of the value iteration algorithm and an image (gridworld_vi_policy.jpg) with the "
                             "optimal policy")

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='Show this help message and exit.')

    args = parser.parse_args()
    return args.height, args.width, args.gamma, args.epsilon, args.plot


def policy_to_annot(env, policy):
    """ Visualize the given deterministic policy. The actions up, down, right and left are represented as 'U', 'D', 'R'
        and 'L', respectively. No action is selected for the terminal states.

    :param env: The GridWorld environment
    :param policy: A 2-level dictionary with (state, action) as key and a probability as the value
    :return: A grid that visualizes the selected policy
    """
    annot = [[None for _ in range(env.width)] for _ in range(env.height)]
    for y in range(env.height):
        for x in range(env.width):
            if env.is_done(env.coords_to_state((y, x))):
                annot[y][x] = ''
            else:
                state = env.coords_to_state((y, x))
                action = list(policy[state].keys())[0]
                if action == GridWorldEnv.UP:
                    annot[y][x] = 'U'
                elif action == GridWorldEnv.DOWN:
                    annot[y][x] = 'D'
                elif action == GridWorldEnv.LEFT:
                    annot[y][x] = 'L'
                else:
                    annot[y][x] = 'R'
    return annot


def value_iteration_animation(env, V_history, opt_policy):
    """ Create an animation plot that demonstrates how the value iteration algorithm progresses in the given
        GridWorld environment. Two figures are used. In the first one, the value function progress is animated for each
        step of the algorithm. In the second one, the optimal policy is plotted for each non-terminal state.

    :param env: The GridWorld environment
    :param V_history: A history list that contains the value function for each step of the value iteration algorithm.
    :param opt_policy: A grid-like 2D-nested list containing the optimal action for each state.
    """

    INTERVAL = 1000

    fig1, ax1 = plt.subplots(1, 1, figsize=(20, 8))
    fig2, ax2 = plt.subplots(1, 1, figsize=(15, 8))

    ax1.tick_params(left=False, bottom=False)
    ax2.tick_params(left=False, bottom=False)

    fig1.suptitle('Value Iteration\nIteration: ')
    fig2.suptitle('Value Iteration\n Optimal Policy')

    colors = np.zeros((env.height, env.width))
    for y in range(env.height):
        for x in range(env.width):
            if env.is_done(env.coords_to_state((y, x))):
                colors[y, x] = 0.5

    sns.heatmap(ax=ax2, data=colors, annot=policy_to_annot(env, opt_policy), vmin=0.0, vmax=1.0, cmap='Greys',
                cbar=False, linewidths=1, linecolor='black', annot_kws={'fontsize': 22}, fmt='')

    sns.heatmap(ax=ax1, data=colors, annot=None, vmin=0.0, vmax=1.0, cmap='Greys', cbar=False, linewidths=1,
                    linecolor='black')

    def update(frame):
        fig1.suptitle(f'Value Iteration\nIteration: {frame}')
        V = V_history[frame].reshape(env.height, env.width)
        ax1.cla()
        sns.heatmap(ax=ax1, data=colors, annot=V, vmin=0.0, vmax=1.0, cmap='Greys', cbar=False, linewidths=1,
                    linecolor='black', annot_kws={'fontsize': 22}, fmt='.1f')

    anim = an.FuncAnimation(fig=fig1, func=update, frames=len(V_history), repeat=False, interval=INTERVAL)
    anim.save('gridworld_vi_animation.gif', writer='imagemagick', fps=1000 / INTERVAL)
    fig2.savefig('gridworld_vi_policy.jpg')
    plt.show()


def value_iteration(env, gamma, eps):
    """ The value iteration algorithm. The algorithm is used to find the optimal policy for the given environment and
        its corresponding state value function.

    :param env: The GridWorld environment
    :param gamma: The discount factor for the value iteration algorithm
    :param eps: The sensitivity factor for the termination of the value iteration algorithm
    :return: A history list that contains the state value function for each step of the value iteration algorithm and
             the optimal policy.
    """

    V = np.zeros(env.nS)
    V_history = [V]
    diff = np.inf

    while diff > eps:
        Q = np.zeros((env.nS, env.nA))
        for state in range(env.nS):
            for action in range(env.nA):
                for probability, nextstate, reward, _ in env.P[state][action]:
                    Q[state, action] += probability * (reward + gamma * V[nextstate])

        V_upd = np.max(Q, axis=1)
        diff = np.max(np.abs(V_upd - V))
        V = V_upd
        V_history.append(V)

    opt_policy = {state: {np.argmax(Q[state]).item(): 1.0} for state in range(env.nS)}

    return V_history, opt_policy


def main():
    """ Create a GridWorld environment based on the command line arguments and find the optimal policy for this
        environment. Optionally, plot an animation demonstrating the progress of the value iteration algorithm and an
        image with an optimal deterministic policy.
    """
    height, width, gamma, epsilon, plot = parse_args()
    env = GridWorldEnv(height, width)
    V_history, opt_policy = value_iteration(env, gamma, epsilon)
    V = V_history[-1].reshape((height, width))

    pp = pprint.PrettyPrinter(indent=2, width=env.width * 7, compact=True)
    print("Optimal Policy")
    pp.pprint(policy_to_annot(env, opt_policy))

    print("\nState Value Function")
    pp.pprint(V.reshape(height, width).tolist())

    if plot:
        value_iteration_animation(env, V_history, opt_policy)


if __name__ == '__main__':
    main()


