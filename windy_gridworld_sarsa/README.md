<h1>SARSA (On-Policy TD Control)</h1>

SARSA is a model-free algorithm, which does not assume any knowledge of MDP transitions or rewards. This is an on-policy 
learning algorithm, since the agent follows a policy <img src="https://latex.codecogs.com/svg.image?\pi" title="\pi" />,
learns from it and improves it:

<!---
A_1, A_2, \dots, A_{T-1} \sim \pi
--->
<p align="center">
<img src="https://latex.codecogs.com/svg.image?A_1,&space;A_2,&space;\dots,&space;A_{T-1}&space;\sim&space;\pi" title="A_1, A_2, \dots, A_{T-1} \sim \pi" />
</p>

In order to address the exploration vs exploitation problem, the policy <img src="https://latex.codecogs.com/svg.image?\pi" title="\pi" />
is selected to be epsilon-greedy:

<!---
\pi(a|s) = \left \{\begin{array}{ll}
     1 - \epsilon(s) + \frac{\epsilon(s)}{|A|}, & a = \underset{a'}{argmax} \ Q(s,a')\\
     \frac{\epsilon(s)}{|A|}, & otherwise\\
\end{array}\right.
--->
<p align="center">
<img src="https://latex.codecogs.com/svg.image?\pi(a|s)&space;=&space;\left&space;\{\begin{array}{ll}&space;&space;&space;&space;&space;1&space;-&space;\epsilon(s)&space;&plus;&space;\frac{\epsilon(s)}{|A|},&space;&&space;a&space;=&space;\underset{a'}{argmax}&space;\&space;Q(s,a')\\&space;&space;&space;&space;&space;\frac{\epsilon(s)}{|A|},&space;&&space;otherwise\\\end{array}\right." title="\pi(a|s) = \left \{\begin{array}{ll} 1 - \epsilon(s) + \frac{\epsilon(s)}{|A|}, & a = \underset{a'}{argmax} \ Q(s,a')\\ \frac{\epsilon(s)}{|A|}, & otherwise\\\end{array}\right." />
</p>

with:

<!---
\epsilon(s) = \frac{10}{10+N(s)}
--->
<p align="center">
<img src="https://latex.codecogs.com/svg.image?\epsilon(s)&space;=&space;\frac{10}{10&plus;N(s)}" title="\epsilon(s) = \frac{10}{10+N(s)}" />
</p>

where <img src="https://latex.codecogs.com/svg.image?N(s)" title="N(s)" /> is the total number of updates of the 
state-action value function <img src="https://latex.codecogs.com/svg.image?Q(s,&space;\cdot)" title="Q(s, \cdot)" />.
The epsilon-greedy policy <img src="https://latex.codecogs.com/svg.image?\pi" title="\pi" /> satisfies the GLIE (Greedy 
in the Limit with Infinite Exploration) criteria:

<!---
\begin{align*}
& \underset{k \to \infty}{lim} \ N_k(s,a) \to \infty \\
& \underset{k \to \infty}{lim} \ \pi_k(a|s) = \textbf{1}(a = \underset{a' \in A}{argmax} \ Q_k(s,a'))
\end{align}
--->

<p align="center">
<img src="https://latex.codecogs.com/svg.image?\begin{align*}&&space;\underset{k&space;\to&space;\infty}{lim}&space;\&space;N_k(s,a)&space;\to&space;\infty&space;\\&&space;\underset{k&space;\to&space;\infty}{lim}&space;\&space;\pi_k(a|s)&space;=&space;\textbf{1}(a&space;=&space;\underset{a'&space;\in&space;A}{argmax}&space;\&space;Q_k(s,a'))\end{align}" title="\begin{align*}& \underset{k \to \infty}{lim} \ N_k(s,a) \to \infty \\& \underset{k \to \infty}{lim} \ \pi_k(a|s) = \textbf{1}(a = \underset{a' \in A}{argmax} \ Q_k(s,a'))\end{align}" />
</p>

In this implementation of the SARSA algorithm, the state-value function <img src="https://latex.codecogs.com/svg.image?Q" title="Q" />
is updated for every step of an episode. Given the state <img src="https://latex.codecogs.com/svg.image?S_t" title="S_t" />
and the action <img src="https://latex.codecogs.com/svg.image?A_t" title="A_t" />, we observe the reward <img src="https://latex.codecogs.com/svg.image?R_t" title="R_t" />
and the new state of the agent <img src="https://latex.codecogs.com/svg.image?S_{t&plus;1}" title="S_{t+1}" />. The action
<img src="https://latex.codecogs.com/svg.image?A_{t&plus;1}" title="A_{t+1}" /> is sampled from the epsilon greedy policy
<img src="https://latex.codecogs.com/svg.image?\pi" title="\pi" />. The SARSA algorithm uses bootstrapping to update <img src="https://latex.codecogs.com/svg.image?Q(S_t,&space;A_t)" title="Q(S_t, A_t)" />
and the proceeds to the state-action pair <img src="https://latex.codecogs.com/svg.image?(S_{t&plus;1},&space;A_{t&plus;1})" title="(S_{t+1}, A_{t+1})" />,
unless the new state <img src="https://latex.codecogs.com/svg.image?S_{t&plus;1}" title="S_{t+1}" /> is terminal:

<!---
\begin{align*}
&Q(S_t, A_t) \leftarrow Q(S_t, A_t) + a (R_t + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t))\\
&t \leftarrow t+1 
\end{align} 
--->
<p align="center">
<img src="https://latex.codecogs.com/svg.image?\begin{align*}&Q(S_t,&space;A_t)&space;\leftarrow&space;Q(S_t,&space;A_t)&space;&plus;&space;a&space;(R_t&space;&plus;&space;\gamma&space;Q(S_{t&plus;1},&space;A_{t&plus;1})&space;-&space;Q(S_t,&space;A_t))\\&t&space;\leftarrow&space;t&plus;1&space;\end{align}&space;" title="\begin{align*}&Q(S_t, A_t) \leftarrow Q(S_t, A_t) + a (R_t + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t))\\&t \leftarrow t+1 \end{align} " />
</p>

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

where <img src="https://latex.codecogs.com/svg.image?N(s,a)" title="N(s,a)" /> is the total number of updates of the 
state-action value function <img src="https://latex.codecogs.com/svg.image?Q(s,a)" title="Q(s,a)" />.


Since the GLIE and the Robbins-Monro conditions are satisfied, it is theoretically guaranteed that SARSA converges to 
the optimal policy.


This exercise is based on:
- Example 6.5 of Sutton's book "Reinforcement Learning: An Introduction (2nd Edition)"
- Exercise 6.9 of Sutton's book "Reinforcement Learning: An Introduction (2nd Edition)"
- The SARSA algorithm example presented in "Lecture 5: Model Free Control" of David Silver's Reinforcement Learning 
  Course.

Windy Gridworld is a 7x10 gridworld with:
- start state (3, 0)
- goal state (3, 7)

Compared to a standard gridworld, in this environment there is a crosswind running upward through the middle of the grid.
In the middle region the resultant next states are shifted upward by the "wind" whose strength varies from column to
column. The reward is -1 until goal state is reached. The wind's strength in the Windy Gridworld environment is 
0, 0, 0, 1, 1, 1, 2, 2, 1, 0 from the left to the right column of the grid.

We consider 3 cases:
- The agent' moves are the standard four: up, right, down, and left. (normal moves)
- The agent can also move diagonally: up, right, down, and left, up & right, up & left, down & right, down & left. (king's moves)
- The agent can also stand still: up, right, down, and left, up & right, up & left, down & right, down & left, no move (king's extra moves)

```commandline
usage: windy_gridworld_sarsa.py [--moves {normal_moves,king_moves,king_extra_moves}] [--n_episodes N_EPISODES] [--gamma GAMMA] [--plot] [-h]

optional arguments:
  --moves {normal_moves,king_moves,king_extra_moves}
                        The moves of the normal agent are UP, DOWN, LEFT, RIGHT. The moves of the king agent are UP-RIGHT, UP-LEFT, DOWN-RIGHT, DOWN-LEFT. When the king's extra move is allowed, he can choose NO MOVE.
  --n_episodes N_EPISODES
                        The number of episodes to sample (DEFAULT=200)
  --gamma GAMMA         The discount factor of the SARSA (On-Policy TD Control) algorithm. (DEFAULT=1.0)
  --plot                Plot and save (as windy_gridworld_sarsa_stats_{moves}.jpg) the statistics of the SARSA algorithm in the Windy GridWorld over time and the optimal trajectory (as windy_gridworld_sarsa_animation_{moves}.gif).
                        (DEFAULT=False)
  -h, --help            Show this help message and exit.
```

```commandline
python3 windy_gridworld_sarsa.py --plot --moves normal_moves
```
<p align="center">
<img src="windy_gridworld_sarsa_stats_normal_moves.jpg"/>
<img src="windy_gridworld_sarsa_animation_normal_moves.gif"/>
</p>

```commandline
python3 windy_gridworld_sarsa.py --plot --moves king_moves
```
<p align="center">
<img src="windy_gridworld_sarsa_stats_king_moves.jpg"/>
<img src="windy_gridworld_sarsa_animation_king_moves.gif"/>
</p>

```commandline
python3 windy_gridworld_sarsa.py --plot --moves king_extra_moves
```
<p align="center">
<img src="windy_gridworld_sarsa_stats_king_extra_moves.jpg"/>
<img src="windy_gridworld_sarsa_animation_king_extra_moves.gif"/>
</p>