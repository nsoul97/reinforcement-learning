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
                        help="The discount factor of the iterative policy evaluation algorithm. (DEFAULT=1.0)")

    parser.add_argument("--epsilon", type=check_positive_float, default=1e-5,
                        help="The iterative policy evaluation algorithm terminates once the value \
                              function change is less than epsilon for all states. (DEFAULT=1e-5)")

    parser.add_argument("--plot", action='store_true',
                        help="Plot and save (as gridworld_pi_animation.gif) the policy that is selected greedily and \
                              its value function for every step of the policy iteration algorithm. (DEFAULT=False)")

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='Show this help message and exit.')

    args = parser.parse_args()
    return args.height, args.width, args.gamma, args.epsilon, args.plot


def policy_to_annot(env, policy):
    """ Visualize the given policy. The actions up, down, right and left are represented as 'U', 'D', 'R' and 'L',
        respectively. No action is selected for the terminal states. When a policy is not deterministic (possible only
        during the first step of policy iteration), the character '?' is used.

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
                if len(policy[state]) > 2:
                    annot[y][x] = '?'
                else:
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


def policy_iteration_animation(env, H):
    """ Create an animation plot that demonstrates how the policy iteration algorithm progresses in the given
        GridWorld environment. Two subplots are animated, one for the policy of the current step and another for its
        value function.

    :param env: The GridWorld environment
    :param H: A history list that contains a policy and its value function for each step of the policy iteration algorithm.
    """

    INTERVAL = 5000

    policy_history, V_history = zip(*H)

    fig, axes = plt.subplots(1, 2, figsize=(30, 8))
    axes[0].tick_params(left=False, bottom=False)
    axes[1].tick_params(left=False, bottom=False)
    fig.suptitle('Policy Iteration')

    colors = np.zeros((env.height, env.width))
    for y in range(env.height):
        for x in range(env.width):
            if env.is_done(env.coords_to_state((y, x))):
                colors[y, x] = 0.5

    for ax in axes:
        sns.heatmap(ax=ax, data=colors, annot=None, vmin=0.0, vmax=1.0, cmap='Greys', cbar=False, linewidths=1,
                    linecolor='black')
    axes[0].set_title(f"Policy\nIteration: ")
    axes[1].set_title(f"Value Function\nIteration: ")

    def update_value_func(step):
        V = V_history[step].reshape(env.height, env.width)
        axes[1].cla()
        axes[1].set_title(f"Value Function\nIteration: {step}")
        sns.heatmap(ax=axes[1], data=colors, annot=V, vmin=0.0, vmax=1.0, cmap='Greys', cbar=False, linewidths=1,
                    linecolor='black', annot_kws={'fontsize': 12}, fmt='.1f')

    def update_policy(step):
        policy = policy_history[step]
        axes[0].cla()
        axes[0].set_title(f"Policy\nIteration: {step}")
        sns.heatmap(ax=axes[0], data=colors, annot=policy_to_annot(env, policy), vmin=0.0, vmax=1.0, cmap='Greys',
                    cbar=False, linewidths=1, linecolor='black', annot_kws={'fontsize': 22}, fmt='')

    def update(frame):
        update_value_func(frame)
        update_policy(frame)

    anim = an.FuncAnimation(fig=fig, func=update, frames=len(H), repeat=False, interval=INTERVAL)
    anim.save('gridworld_pi_animation.gif', writer='imagemagick', fps=1000 / INTERVAL)
    plt.show()


def iterative_policy_evaluation(env, policy, gamma, eps):
    """ The iterative policy evaluation algorithm. The algorithm is used to evaluate a given policy.

    :param env: The GridWorld environment
    :param policy: The policy to be evaluated
    :param gamma: The discount factor
    :param eps: The sensitivity factor for the termination of the algorithm
    :return: The state value function V and the action value function Q for the given environment and the given policy.
    """

    diff = 100
    V = np.zeros(env.nS)

    while diff > eps:
        V_upd = np.zeros_like(V)
        for state in range(env.nS):
            for action in policy[state].keys():
                for (probability, nextstate, reward, _) in env.P[state][action]:
                    V_upd[state] += policy[state][action] * probability * (reward + gamma * V[nextstate])

        diff = np.max(np.abs(V_upd - V))
        V = V_upd

    Q = np.zeros((env.nS, env.nA))
    for state in range(env.nS):
        for action in range(env.nA):
            for (probability, nextstate, reward, _) in env.P[state][action]:
                Q[state, action] += probability * (reward + gamma * V[nextstate])

    return V, Q


def policy_iteration(env, gamma, eps):
    """ The policy iteration algorithm. The algorithm is used to find the optimal policy for the given environment.
        At each step of the algorithm a policy is first evaluated using the iterative policy evaluation algorithm and
        then it is improved in a greedy way.

    :param env: The GridWorld environment
    :param gamma: The discount factor for the iterative policy evaluation algorithm
    :param eps: The sensitivity factor for the termination of the iterative policy evaluation algorithm
    :return: A history list that contains a policy and its value function for each step of the policy iteration algorithm.
    """

    H = []

    policy = {state: {action: 1.0/env.nA for action in range(env.nA)} for state in range(env.nS)}
    V, Q = iterative_policy_evaluation(env, policy, gamma, eps)

    prev_tot_value = -np.inf
    curr_tot_value = np.sum(V)

    while curr_tot_value > prev_tot_value:
        H.append((policy, V))

        policy = {state: {np.argmax(Q[state]).item(): 1.0} for state in range(env.nS)}
        V, Q = iterative_policy_evaluation(env, policy, gamma, eps)

        prev_tot_value = curr_tot_value
        curr_tot_value = np.sum(V)

    return H


def main():
    """ Create a GridWorld environment based on the command line arguments and find the optimal policy for this
        environment. Optionally, plot an animation demonstrating the progress of the policy iteration algorithm.
    """
    height, width, gamma, epsilon, plot = parse_args()
    env = GridWorldEnv(height, width)
    H = policy_iteration(env, gamma, epsilon)
    opt_policy, V = H[-1]

    pp = pprint.PrettyPrinter(indent=2, width=env.width * 7, compact=True)
    print("Optimal Policy")
    pp.pprint(policy_to_annot(env, opt_policy))

    print("\nState Value Function")
    pp.pprint(V.reshape(height, width).tolist())

    if plot:
        policy_iteration_animation(env, H)


if __name__ == '__main__':
    main()


