import sys
sys.path.insert(0, '..')

import numpy as np
import argparse
import pprint
import matplotlib.pyplot as plt
import matplotlib.animation as an
from rl_envs.cliff_gridworld import CliffGridWorldEnv


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

    :return: The number of episodes to be sampled, the discount factor gamma, the epsilon parameter of the
             epsilon-greedy behavior strategy and a plot boolean.
    """
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--n_episodes", type=check_positive_int, default=200,
                        help="The number of episodes to sample (DEFAULT=200)")

    parser.add_argument("--gamma", type=check_positive_float, default=1.0,
                        help="The discount factor of the Q-Learning (Off-Policy TD learning) algorithm. (DEFAULT=1.0)")

    parser.add_argument("--epsilon", type=check_positive_float, default=0.1,
                        help="The epsilon of the epsilon-greedy behaviour strategy b of the Q-Learning (Off-Policy TD "
                             "learning) algorithm. (DEFAULT=0.1)")

    parser.add_argument("--plot", action='store_true',
                        help="Plot and save (as cliff_gridworld_qlearning_stats.jpg) the statistics of the "
                             "Q-learning algorithm in the Cliff GridWorld over time, the optimal trajectory (as "
                             "cliff_gridworld_qlearning_animation.gif) of the agent following the final (deterministic)"
                             " target policy and the trajectory (as cliff_gridworld_epsilon_greedy_animation.gif) of "
                             "the agent following the final (stochastic) behavior strategy. (DEFAULT=False)")

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='Show this help message and exit.')

    args = parser.parse_args()
    return args.n_episodes, args.gamma, args.epsilon, args.plot


def plot_cliff_gridworld_results(env, n_episodes, final_policy, info, epsilon, q):
    """ Plot a figure with the Q-Learning statistics, an animation with the optimal trajectory of the agent based on the
        (deterministic) target policy and animation of the trajectory of the agent following the (stochastic) behavior
        policy.

    :param env: The Cliff Gridworld environment
    :param n_episodes: The number of episodes that were sampled for the Q-Learning algorithm
    :param final_policy: The target policy the Q-Learning algorithm converges to.
    :param info: The cumulative reward and the total timesteps per episode in the Q-Learning algorithm
    :param epsilon: The epsilon of the epsilon-greedy policy.
    :param q: The state-action value function Q*
    """
    INTERVAL = 1000

    fig1, axes1 = plt.subplots(3, 1, figsize=(30, 30), tight_layout=True)

    axes1[0].plot(range(1, n_episodes + 1), info['timesteps'])
    axes1[1].plot(range(1, n_episodes + 1), info['rewards'])
    axes1[2].plot(range(1, 1 + sum(info['timesteps'])),
                  np.concatenate([[e + 1] * t for e, t in enumerate(info['timesteps'])]))

    axes1[0].set_xlim(0, n_episodes + 1)
    axes1[1].set_xlim(0, n_episodes + 1)
    axes1[2].set_xlim(0, sum(info['timesteps']))
    axes1[2].set_ylim(0, n_episodes + 1)

    axes1[0].set_title('Episode Length over Time', fontsize=20)
    axes1[1].set_title('Episode Reward over Time', fontsize=20)
    axes1[2].set_title('Episode per time step', fontsize=20)

    axes1[0].set_xlabel('Episode', fontsize=18)
    axes1[1].set_xlabel('Episode', fontsize=18)
    axes1[2].set_xlabel('Time Steps', fontsize=18)

    axes1[0].set_ylabel('Episode Length', fontsize=18)
    axes1[1].set_ylabel('Episode Reward', fontsize=18)
    axes1[2].set_ylabel('Episode', fontsize=18)

    fig1.savefig('cliff_gridworld_qlearning_stats.jpg')

    # Plot an animation with the optimal trajectory of the agent.
    fig2, axes2 = plt.subplots(1, 1)
    axes2.axis('off')

    s = env.reset()
    target_frames = [env.render('rgb_array')]
    done = False
    while not done:
        a = final_policy[s]
        s, _, done, _ = env.step(a)
        target_frames.append(env.render('rgb_array'))
        if env.ep_moves > 100:
            break
    env.close()

    fig2.suptitle('Target Policy', fontsize=25)
    anim = an.FuncAnimation(fig2, lambda f: axes2.imshow(target_frames[f]), frames=len(target_frames), interval=INTERVAL,
                            repeat=True)
    anim.save('cliff_gridworld_qlearning_animation.gif', writer='imagemagick', fps=1000 / INTERVAL)

    fig3, axes3 = plt.subplots(1, 1)
    axes3.axis('off')

    s = env.reset()
    behavior_frames = [env.render('rgb_array')]
    done = False
    while not done:
        a = eps_greedy_policy(env, epsilon, q, s)
        s, _, done, _ = env.step(a)
        behavior_frames.append(env.render('rgb_array'))
        if env.ep_moves > 100:
            break
    env.close()

    fig3.suptitle('Behavior Policy', fontsize=25)
    anim = an.FuncAnimation(fig3, lambda f: axes3.imshow(behavior_frames[f]), frames=len(behavior_frames),
                            interval=INTERVAL, repeat=True)
    anim.save('cliff_gridworld_epsilon_greedy_animation.gif', writer='imagemagick', fps=1000 / INTERVAL)

    plt.show(block=True)


def eps_greedy_policy(env, epsilon, q, curr_state):
    """ The epsilon-greedy policy of the agent.

    :param env: The Windy Gridworld environment
    :param epsilon: The epsilon of the epsilon-greedy policy.
    :param q: The state-action value function
    :param curr_state: The current state of the agent
    :return: The action of the agent.
    """
    prob = np.full(env.action_space.n, epsilon / env.action_space.n)
    prob[np.argmax(q[curr_state])] += 1 - epsilon
    return np.random.choice(env.action_space.n, p=prob)


def qlearning(env, n_episodes, gamma, epsilon):
    """ An implementation of the Q-Learning (Off-Policy TD Control) algorithm. The agent follows an epsilon-greedy
        behavior policy, while he learns the target policy. The TD learning rate alpha satisfies the Robbins-Monro
        criteria. The target policy of the algorithm converges to the optimal policy of the agent in the given
        environment.

    :param env: The Windy Gridworld environment
    :param n_episodes: The number of episodes to sample for the Q-Learning algorithm based on the behavior policy b.
    :param gamma: The discount factor gamma of the Q-Learning algorithm.
    :param epsilon: The epsilon parameter of the epsilon greedy behavior policy b.
    :return: The optimal policy, its state value function V* and its state-action value function Q* and additional info
             regarding the cumulative reward per episode.
    """

    q = dict()
    n = dict()
    info = {'timesteps': [], 'rewards': []}

    for e in range(n_episodes):

        info['timesteps'].append(0)
        info['rewards'].append(0)

        curr_state = env.reset()
        if curr_state not in q:
            q[curr_state] = np.zeros(env.action_space.n)
            n[curr_state] = np.zeros(env.action_space.n)
        done = False

        while not done:
            action = eps_greedy_policy(env, epsilon, q, curr_state)
            next_state, reward, done, _ = env.step(action)
            if next_state not in q:
                q[next_state] = np.zeros(env.action_space.n)
                n[next_state] = np.zeros(env.action_space.n)

            alpha = (10 / (n[curr_state][action] + 10))**(2/3)
            q[curr_state][action] += alpha * (reward + gamma * np.max(q[next_state]) - q[curr_state][action])
            curr_state = next_state

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
    n_episodes, gamma, epsilon, plot = parse_args()
    env = CliffGridWorldEnv()
    final_target_policy, v, q, info = qlearning(env, n_episodes, gamma, epsilon)

    pp = pprint.PrettyPrinter(indent=2)
    print("Final Target Policy")
    pp.pprint(final_target_policy)

    print("\nState Value Function V*")
    pp.pprint(v)

    print("\nState-Action Value Function Q*")
    pp.pprint(q)

    if plot:
        plot_cliff_gridworld_results(env, n_episodes, final_target_policy, info, epsilon, q)


if __name__ == '__main__':
    main()
