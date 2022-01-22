<h1>On-Policy Monte Carlo Control</h1>

Monte Carlo is a model-free algorithm, which does not assume any knowledge of MDP transitions or rewards. Instead, the 
dynamics of the environment are learnt from episodes of experience.

We generate <img src="https://latex.codecogs.com/svg.image?n" title="n" /> episodes and, in each of them,  the actions
<img src="https://latex.codecogs.com/svg.image?A_t" title="A_t" /> are sampled following an 
<img src="https://latex.codecogs.com/svg.image?\epsilon" title="\epsilon" />-greedy policy <img src="https://latex.codecogs.com/svg.image?\pi" title="\pi" />:

<!---
A_1, A_2, \dots, A_{T-1} \sim \pi
--->
<p align="center">
<img src="https://latex.codecogs.com/svg.image?A_1,&space;A_2,&space;\dots,&space;A_{T-1}&space;\sim&space;\pi" title="A_1, A_2, \dots, A_{T-1} \sim \pi" />
</p>

The <img src="https://latex.codecogs.com/svg.image?\epsilon" title="\epsilon" />-greedy policy <img src="https://latex.codecogs.com/svg.image?\pi" title="\pi" />
is defined in the following way:

<!---
\pi(a|s) = \left \{\begin{array}{ll}
     1 - \epsilon + \frac{\epsilon}{|A|}, & a = \underset{a'}{argmax} \ Q(s,a')\\
     \frac{\epsilon}{|A|}, & otherwise\\
\end{array}\right.
--->
<p align="center">
<img src="https://latex.codecogs.com/svg.image?\pi(a|s)&space;=&space;\left&space;\{\begin{array}{ll}&space;&space;&space;&space;&space;1&space;-&space;\epsilon&space;&plus;&space;\frac{\epsilon}{|A|},&space;&&space;a&space;=&space;\underset{a'}{argmax}&space;\&space;Q(s,a')\\&space;&space;&space;&space;&space;\frac{\epsilon}{|A|},&space;&&space;otherwise\\\end{array}\right." title="\pi(a|s) = \left \{\begin{array}{ll} 1 - \epsilon + \frac{\epsilon}{|A|}, & a = \underset{a'}{argmax} \ Q(s,a')\\ \frac{\epsilon}{|A|}, & otherwise\\\end{array}\right." />
</p>

with:

<!---
\epsilon(s) = \frac{N_0}{N_0+N(s)}
--->
<p align="center">
<img src="https://latex.codecogs.com/svg.image?\epsilon(s)&space;=&space;\frac{N_0}{N_0&plus;N(s)}" title="\epsilon(s) = \frac{N_0}{N_0+N(s)}" />
</p>

where <img src="https://latex.codecogs.com/svg.image?N_0" title="N_0" /> is a constant (with 100 as its default value)
and <img src="https://latex.codecogs.com/svg.image?N(s)" title="N(s)" /> is the total number of updates of the 
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

This is an implementation of the first-visit Monte Carlo Control algorithm; if a state-action pair <img src="https://latex.codecogs.com/svg.image?(s,a)" title="(s,a)" />
is visited more than once, <img src="https://latex.codecogs.com/svg.image?(S_t,&space;A_t)&space;=&space;(S_{t'},&space;A_{t'})&space;=&space;(s,&space;a)" title="(S_t, A_t) = (S_{t'}, A_{t'}) = (s, a)" />
with <img src="https://latex.codecogs.com/svg.image?t&space;<&space;t'" title="t < t'" />, the discounted return is calculated
only the first time <img src="https://latex.codecogs.com/svg.image?t" title="t" />, during which the state-action pair 
<img src="https://latex.codecogs.com/svg.image?(s,a)" title="(s,a)" /> was visited. 

At the end of each episode, the state-action value function <img src="https://latex.codecogs.com/svg.image?Q" title="Q" />
(and therefore the policy <img src="https://latex.codecogs.com/svg.image?\pi" title="\pi" />) is updated for each of the
state-action pairs <img src="https://latex.codecogs.com/svg.image?(S_t,&space;A_t)" title="(S_t, A_t)" /> 
that were visited, such that:

<!---
\begin{align*}
& N(S_t, A_t) \leftarrow N(S_t, A_t) + 1\\
& Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \frac{1}{N(S_t, A_t)} [G_t - Q(S_t, A_t)]
\end{align}
--->

<p align="center">
<img src="https://latex.codecogs.com/svg.image?\begin{align*}&&space;N(S_t,&space;A_t)&space;\leftarrow&space;N(S_t,&space;A_t)&space;&plus;&space;1\\&&space;Q(S_t,&space;A_t)&space;\leftarrow&space;Q(S_t,&space;A_t)&space;&plus;&space;\frac{1}{N(S_t,&space;A_t)}&space;[G_t&space;-&space;Q(S_t,&space;A_t)]\end{align}" title="\begin{align*}& N(S_t, A_t) \leftarrow N(S_t, A_t) + 1\\& Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \frac{1}{N(S_t, A_t)} [G_t - Q(S_t, A_t)]\end{align}" />
</p>

where <img src="https://latex.codecogs.com/svg.image?N(s,a)" title="N(s,a)" /> is the total number of updates of the 
state-action value function <img src="https://latex.codecogs.com/svg.image?Q(s,a)" title="Q(s,a)" />.

Since the GLIE conditions are satisfied, it is theoretically guaranteed that On-Policy Monte Carlo Control algorithm
converges to the optimal policy after an infinite number of episodes. Nevertheless, in practice the algorithm did not 
converge to an optimal policy after 5M episodes for the given exercise.

This exercise is based on:
- Exercise 1 and 2 of the HW assignment in David Silver's Reinforcement Learning Course.

```commandline
usage: easy21_on_policy_mc_control.py [--n_episodes N_EPISODES] [--gamma GAMMA] [--n0 N0] [--plot] [-h]

optional arguments:
  --n_episodes N_EPISODES
                        The number of episodes to sample (DEFAULT=5000000)
  --gamma GAMMA         The discount factor of the On-Policy Monte Carlo Control algorithm. (DEFAULT=1.0)
  --n0 N0               The constant n0 of the epsilon parameter: epsilon(t) = n0 / (n0 + n(S(t)). (DEFAULT=100.0)
  --plot                Plot and save as easy21_on_policy_mcc_v.jpg the state value function of the optimal policy and as easy21_on_policy_mcc_policy.jpg the optimal policy. (DEFAULT=False)
  -h, --help            Show this help message and exit.
```

```commandline
python3 easy21_on_policy_mc_control.py --plot
```

<p align="center">
<img src="easy21_on_policy_mcc_v.jpg">
<img src="easy21_on_policy_mcc_policy.jpg">
</p>
