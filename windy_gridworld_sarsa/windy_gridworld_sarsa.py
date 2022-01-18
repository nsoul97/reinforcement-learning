import sys
sys.path.insert(0, '..')

import numpy as np
import argparse
import pprint
import matplotlib.pyplot as plt
import matplotlib.animation as an
from rl_envs.windy_gridworld import WindyGridWorldEnv


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

    :return: The moves of the agent, the number of episodes, the discount factor gamma and a plot boolean.
    """
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--moves", type=str, choices=['normal_moves', 'king_moves', 'king_extra_moves'],
                        default='normal_moves', help="The moves of the normal agent are UP, DOWN, LEFT, RIGHT. "
                                                     "The moves of the king agent are UP-RIGHT, UP-LEFT, DOWN-RIGHT, "
                                                     "DOWN-LEFT. When the king's extra move is allowed, he can choose "
                                                     "NO MOVE.")

    parser.add_argument("--n_episodes", type=check_positive_int, default=200,
                        help="The number of episodes to sample (DEFAULT=200)")

    parser.add_argument("--gamma", type=check_positive_float, default=1.0,
                        help="The discount factor of the SARSA (On-Policy TD Control) algorithm. (DEFAULT=1.0)")

    parser.add_argument("--plot", action='store_true',
                        help="Plot and save (as windy_gridworld_sarsa_stats_{moves}.jpg) the statistics of the SARSA "
                             "algorithm in the Windy GridWorld over time and the optimal trajectory (as "
                             "windy_gridworld_sarsa_animation_{moves}.gif). (DEFAULT=False)")

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='Show this help message and exit.')

    args = parser.parse_args()
    return args.moves, args.n_episodes, args.gamma, args.plot


def plot_windy_gridworld_results(env, moves, n_episodes, final_policy, info):
    """ Plot a figure with the SARSA statistics and an animation with the optimal trajectory of the agent.

    :param env: The Windy Gridworld environment
    :param moves: The allowed moves of the agent
    :param n_episodes: The number of episodes that were sampled for the SARSA algorithm
    :param final_policy: The policy the SARSA algorithm converges to.
    :param info: The cumulative reward and the total timesteps per episode in the SARSA algorithm
    """
    INTERVAL = 1000

    fig1, axes1 = plt.subplots(3, 1, figsize=(15, 12))
    fig1.suptitle('SARSA Episode Statistics', fontsize=22)

    axes1[0].plot(range(1, n_episodes + 1), info['timesteps'])
    axes1[1].plot(range(1, n_episodes + 1), info['rewards'])
    axes1[2].plot(range(1, 1 + sum(info['timesteps'])),
                  np.concatenate([[e + 1] * t for e, t in enumerate(info['timesteps'])]))

    axes1[0].set_xlim(0, n_episodes + 1)
    axes1[1].set_xlim(0, n_episodes + 1)
    axes1[2].set_xlim(0, sum(info['timesteps']))
    axes1[2].set_ylim(0, n_episodes + 1)

    axes1[0].set_xlabel('Episode Length', fontsize=18)
    axes1[1].set_xlabel('Episode Reward', fontsize=18)
    axes1[2].set_xlabel('Episode', fontsize=18)

    axes1[0].set_ylabel('Episode', fontsize=18)
    axes1[1].set_ylabel('Episode', fontsize=18)
    axes1[2].set_ylabel('Time Steps', fontsize=18)

    fig1.savefig(f'windy_gridworld_sarsa_stats_{moves}.jpg')

    # Plot an animation with the optimal trajectory of the agent.
    fig2, axes2 = plt.subplots(1, 1)
    axes2.axis('off')

    s = env.reset()
    frames = [env.render('rgb_array')]
    done = False
    while not done:
        a = final_policy[s]
        s, _, done, _ = env.step(a)
        frames.append(env.render('rgb_array'))
        if env.ep_moves > 100:
            break
    env.close()

    anim = an.FuncAnimation(fig2, lambda f: axes2.imshow(frames[f]), frames=len(frames), interval=INTERVAL)
    anim.save(f'windy_gridworld_sarsa_animation_{moves}.gif', writer='imagemagick', fps=1000 / INTERVAL)
    plt.show(block=True)


def eps_greedy_policy(env, n, q, curr_state):
    """ The epsilon-greedy policy of the agent. The epsilon parameter is selected to satisfy the GLIE convergence
        criteria.

    :param env: The Windy Gridworld environment
    :param n: The number of times the Q function of a (state, action) has been updated
    :param q: The state-action value function
    :param curr_state: The current state of the agent
    :return: The action of the agent.
    """
    epsilon = 10 / (10 + np.sum(n[curr_state]))
    prob = np.full(env.action_space.n, epsilon / env.action_space.n)
    prob[np.argmax(q[curr_state])] += 1 - epsilon
    return np.random.choice(env.action_space.n, p=prob)


def sarsa(env, n_episodes, gamma):
    """ An implementation of the SARSA (On-Policy TD Control) algorithm. The agent follows an epsilon greedy policy that
        satisfies the GLIE criteria. The TD learning rate alpha satisfies the Robbins-Monro criteria. Given that, the
        algorithm converges to the optimal policy of the agent in the given environment.

    :param env: The Windy Gridworld environment
    :param n_episodes: The number of episodes to sample for the SARSA algorithm
    :param gamma: The discount factor of the SARSA algorithm
    :return: The optimal policy, its state value function V* and its state-action value function Q* and additional info
             regarding the cumulative reward and timesteps per episode.
    """
    n = dict()
    q = dict()
    info = {'timesteps': [], 'rewards': [], }

    for e in range(n_episodes):

        info['timesteps'].append(0)
        info['rewards'].append(0)

        curr_state = env.reset()
        if curr_state not in q:
            q[curr_state] = np.zeros(env.action_space.n)
            n[curr_state] = np.zeros(env.action_space.n)
        curr_action = eps_greedy_policy(env, n, q, curr_state)
        done = False
        while not done:

            next_state, reward, done, _ = env.step(curr_action)
            if next_state not in q:
                q[next_state] = np.zeros(env.action_space.n)
                n[next_state] = np.zeros(env.action_space.n)
            next_action = eps_greedy_policy(env, n, q, next_state)
            sarsa_target = reward + gamma * q[next_state][next_action]
            alpha = (10 / (n[curr_state][curr_action] + 10))**(2/3)
            q[curr_state][curr_action] += alpha * (sarsa_target - q[curr_state][curr_action])
            n[curr_state][curr_action] += 1

            curr_state = next_state
            curr_action = next_action

            info['timesteps'][-1] += 1
            info['rewards'][-1] += reward

    v = {state: np.max(q[state]) for state in q}
    policy = {state: np.argmax(q[state]) for state in q}

    return policy, v, q, info


def main():
    """
    Read the command line arguments, create a Windy GridWorld environment and find the optimal strategy for the player
    without knowing the environment's transition probability matrix. The dynamics of the environment are sampled using
    the SARSA algorithm. Optionally, plot one figure with the algorithm's statistics over time and one animation with
    the optimal trajectory, according to the optimal policy.
    """
    moves, n_episodes, gamma, plot = parse_args()
    env = WindyGridWorldEnv(moves)
    final_policy, v, q, info = sarsa(env, n_episodes, gamma)

    pp = pprint.PrettyPrinter(indent=2)
    print("Final Policy")
    pp.pprint(final_policy)

    print("\nState Value Function V*")
    pp.pprint(v)

    print("\nState-Action Value Function Q*")
    pp.pprint(q)

    if plot:
        plot_windy_gridworld_results(env, moves, n_episodes, final_policy, info)


if __name__ == '__main__':
    main()
