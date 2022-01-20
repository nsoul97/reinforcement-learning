<h1>Q-Learning (Off-Policy TD Learning)</h1>

Q-Learning is a model-free algorithm, which does not assume any knowledge of MDP transitions or rewards. Given that
this is an off-policy algorithm, the actions <img src="https://latex.codecogs.com/svg.image?A_t" title="A_t" /> 
are sampled from the behavior policy <img src="https://latex.codecogs.com/svg.image?b" title="b" /> for each of the 
<img src="https://latex.codecogs.com/svg.image?n" title="n" /> episodes of the algorithm:

<!---
A_1, A_2, \dots, A_{T-1} \sim b
--->
<p align="center">
<img src="https://latex.codecogs.com/svg.image?A_1,&space;A_2,&space;\dots,&space;A_{T-1}&space;\sim&space;b" title="A_1, A_2, \dots, A_{T-1} \sim b" />
</p>

The behavior policy <img src="https://latex.codecogs.com/svg.image?b" title="b" /> is an
<img src="https://latex.codecogs.com/svg.image?\epsilon" title="\epsilon" />-greedy policy, where:

<!---
b(a|s) = \left \{\begin{array}{ll}
     1 - \epsilon + \frac{\epsilon}{|A|}, & a = \underset{a'}{argmax} \ Q(s,a')\\
     \frac{\epsilon}{|A|}, & otherwise\\
\end{array}
\right.
--->
<p align="center">
<img src="https://latex.codecogs.com/svg.image?b(a|s)&space;=&space;\left&space;\{\begin{array}{ll}&space;&space;&space;&space;&space;1&space;-&space;\epsilon&space;&plus;&space;\frac{\epsilon}{|A|},&space;&&space;a&space;=&space;\underset{a'}{argmax}&space;\&space;Q(s,a')\\&space;&space;&space;&space;&space;\frac{\epsilon}{|A|},&space;&&space;otherwise\\\end{array}\right." title="b(a|s) = \left \{\begin{array}{ll} 1 - \epsilon + \frac{\epsilon}{|A|}, & a = \underset{a'}{argmax} \ Q(s,a')\\ \frac{\epsilon}{|A|}, & otherwise\\\end{array}\right." />
</p>

with <img src="https://latex.codecogs.com/svg.image?\epsilon&space;=&space;0.1" title="\epsilon = 0.1" />.

The goal of the algorithm is to learn an optimal target strategy <img src="https://latex.codecogs.com/svg.image?\pi" title="\pi" />.
The strategy <img src="https://latex.codecogs.com/svg.image?\pi" title="\pi" /> is selected to be greedy with respect to
the state-action value function <img src="https://latex.codecogs.com/svg.image?Q" title="Q" />:

<!---
\pi(a|s) = \left \{\begin{array}{ll}
     1 , & a = \underset{a'}{argmax} \ Q(s,a')\\
     0, & otherwise\\
\end{array}\right.
--->
<p align="center">
<img src="https://latex.codecogs.com/svg.image?\pi(a|s)&space;=&space;\left&space;\{\begin{array}{ll}&space;&space;&space;&space;&space;1&space;,&space;&&space;a&space;=&space;\underset{a'}{argmax}&space;\&space;Q(s,a')\\&space;&space;&space;&space;&space;0,&space;&&space;otherwise\\\end{array}\right." title="\pi(a|s) = \left \{\begin{array}{ll} 1 , & a = \underset{a'}{argmax} \ Q(s,a')\\ 0, & otherwise\\\end{array}\right." />
</p>

Given a state <img src="https://latex.codecogs.com/svg.image?S_t" title="S_t" />, the action <img src="https://latex.codecogs.com/svg.image?A_t" title="A_t" />
is sampled based on the behavior policy <img src="https://latex.codecogs.com/svg.image?b" title="b" /> in order to update
the state-action value function <img src="https://latex.codecogs.com/svg.image?Q(S_t,&space;A_t)" title="Q(S_t, A_t)" />.
Following that, we observe the reward <img src="https://latex.codecogs.com/svg.image?R_{t&plus;1}" title="R_{t+1}" /> and
the next state <img src="https://latex.codecogs.com/svg.image?S_{t&plus;1}" title="S_{t+1}" /> and select the next action
<img src="https://latex.codecogs.com/svg.image?A'" title="A'" /> based on the target policy <img src="https://latex.codecogs.com/svg.image?\pi" title="\pi" />
(greedily). The state-action function <img src="https://latex.codecogs.com/svg.image?Q(S_t,&space;A_t)" title="Q(S_t, A_t)" />
is updated using bootstrapping:

<!---
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (Q(S_{t+1}, A') - Q(S_t, A_t))
--->
<p align="center">
<img src="https://latex.codecogs.com/svg.image?Q(S_t,&space;A_t)&space;\leftarrow&space;Q(S_t,&space;A_t)&space;&plus;&space;\alpha&space;(Q(S_{t&plus;1},&space;A')&space;-&space;Q(S_t,&space;A_t))" title="Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (Q(S_{t+1}, A') - Q(S_t, A_t))" />
</p>

Afterwards, the same process is repeated for the state <img src="https://latex.codecogs.com/svg.image?S_{t&plus;1}" title="S_{t+1}" />,
unless it is a terminal state.

The TD learning rate <img src="https://latex.codecogs.com/svg.image?\alpha" title="\alpha" />, satisfies the Robbins-Monro
criteria:

<!---
\begin{align*}
& \sum_{n=1}^{\infty} \alpha_n = \infty \\
& \sum_{n=1}^{\infty} \alpha_n^2 < \infty \\\
end{align} 
--->
<p align="center">
<img src="https://latex.codecogs.com/svg.image?\begin{align*}&&space;\sum_{n=1}^{\infty}&space;\alpha_n&space;=&space;\infty&space;\\&&space;\sum_{n=1}^{\infty}&space;\alpha_n^2&space;<&space;\infty&space;\\\end{align}&space;" title="\begin{align*}& \sum_{n=1}^{\infty} \alpha_n = \infty \\& \sum_{n=1}^{\infty} \alpha_n^2 < \infty \\\end{align} " />
</p>

since:

<!---
\alpha(s,a) = \sqrt[3]{\frac{10}{10 + N(s,a)}}^2
--->
<p align="center">
<img src="https://latex.codecogs.com/svg.image?\alpha(s,a)&space;=&space;\sqrt[3]{\frac{10}{10&space;&plus;&space;N(s,a)}}^2" title="\alpha(s,a) = \sqrt[3]{\frac{10}{10 + N(s,a)}}^2" />
</p>

As a result, the target policy of the Q-Learning algorithm converges to the optimal policy.

This exercise is based on:
- Example 6.6 of Sutton's book "Reinforcement Learning: An Introduction (2nd Edition)"

Cliff Gridworld is a 4x12 gridworld with:
- start state: (3, 0)
- goal state: (3, 11)
- the usual actions causing movement up, down, right, and left, which incur a reward of -1

Compared to a standard gridworld, in this environment, there is a cliff region. When the agent steps into a cell of this
region, he receives a reward of -100 and goes back to the start state instantly. The cells of the cliff region are:
(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9) and (3, 10).

```commandline
usage: cliff_gridworld_qlearning.py [--n_episodes N_EPISODES] [--gamma GAMMA] [--epsilon EPSILON] [--plot] [-h]

optional arguments:
  --n_episodes N_EPISODES
                        The number of episodes to sample (DEFAULT=200)
  --gamma GAMMA         The discount factor of the Q-Learning (Off-Policy TD learning) algorithm. (DEFAULT=1.0)
  --epsilon EPSILON     The epsilon of the epsilon-greedy behaviour strategy b of the Q-Learning (Off-Policy TD learning) algorithm. (DEFAULT=0.1)
  --plot                Plot and save (as cliff_gridworld_qlearning_stats.jpg) the statistics of the Q-learning algorithm in the Cliff GridWorld over time, the optimal trajectory (as cliff_gridworld_qlearning_animation.gif) of the
                        agent following the final (deterministic) target policy and the trajectory (as cliff_gridworld_epsilon_greedy_animation.gif) of the agent following the final (stochastic) behavior strategy. (DEFAULT=False)
  -h, --help            Show this help message and exit.
```

```commandline
python3 cliff_gridworld_qlearning.py
```

<p align="center">
<img src="cliff_gridworld_qlearning_stats.jpg"/>
<img src="cliff_gridworld_epsilon_greedy_animation.gif">
<img src="cliff_gridworld_qlearning_animation.gif">
</p>