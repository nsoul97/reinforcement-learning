<h1>Monte Carlo Prediction</h1>

Monte Carlo is a model-free algorithm, which does not assume any knowledge of MDP transitions or rewards. Instead, the 
dynamics of the environment are learnt from episodes of experience.

<img src="https://latex.codecogs.com/svg.image?n" title="n" /> episodes are sampled based on the policy to be evaluated 
<img src="https://latex.codecogs.com/svg.image?\pi" title="\pi" /> :

<!---
S_1, A_1, R_2, S_2, A_2, R_3, \dots, S_{T-1}, A_{T-1}, R_{T}, S_{T} \sim \pi
--->

<p align="center">
<img src="https://latex.codecogs.com/svg.image?S_1,&space;A_1,&space;R_2,&space;S_2,&space;A_2,&space;R_3,&space;\dots,&space;S_{T-1},&space;A_{T-1},&space;R_{T},&space;S_{T}&space;\sim&space;\pi" title="S_1, A_1, R_2, S_2, A_2, R_3, \dots, S_{T-1}, A_{T-1}, R_{T}, S_{T} \sim \pi" />
</p>

For a state <img src="https://latex.codecogs.com/svg.image?S_t" title="S_t" /> of the episode, we calculate 
the discounted return <img src="https://latex.codecogs.com/svg.image?G_t" title="G_t" />:

<!---
G_t = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{T-t-1} R_{T}
--->

<p align="center">
<img src="https://latex.codecogs.com/svg.image?G_t&space;=&space;R_{t&plus;1}&space;&plus;&space;\gamma&space;R_{t&plus;2}&space;&plus;&space;\dots&space;&plus;&space;\gamma^{T-t-1}&space;R_{T}" title="G_t = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{T-t-1} R_{T}" />
</p>

This is an implementation of the first-visit Monte Carlo prediction algorithm; if a state <img src="https://latex.codecogs.com/svg.image?s" title="s" />
is visited more than once, <img src="https://latex.codecogs.com/svg.image?S_t&space;=&space;S_{t'}&space;=&space;s" title="S_t = S_{t'} = s" />
 with <img src="https://latex.codecogs.com/svg.image?t&space;<&space;t'" title="t < t'" /> the discounted return is calculated
only for the first time <img src="https://latex.codecogs.com/svg.image?t" title="t" /> state <img src="https://latex.codecogs.com/svg.image?s" title="s" /> was visited. 

Each time we visit a state for the first time in an episode:

<!---
\begin{align*}
& N(s) \leftarrow N(s) + 1 \\
& S(s) \leftarrow S(s) + G_t
\end{align}
--->

<p align="center">
<img src="https://latex.codecogs.com/svg.image?\begin{align*}&&space;N(s)&space;\leftarrow&space;N(s)&space;&plus;&space;1&space;\\&&space;S(s)&space;\leftarrow&space;S(s)&space;&plus;&space;G_t\end{align}&space;" title="\begin{align*}& N(s) \leftarrow N(s) + 1 \\& S(s) \leftarrow S(s) + G_t\end{align} " />
</p>

Therefore, by the law of big numbers, when <img src="https://latex.codecogs.com/svg.image?n&space;\to&space;\infty&space;" title="n \to \infty " />:

<!--- 
\frac{S(s)}{N(s)}\to E_{\pi}[G_t | S_t= s] = V_{\pi}(s)
--->

<p align="center">
<img src="https://latex.codecogs.com/svg.image?\frac{S(s)}{N(s)}\to&space;E_{\pi}[G_t&space;|&space;S_t=&space;s]&space;=&space;V_{\pi}(s)" title="\frac{S(s)}{N(s)}\to E_{\pi}[G_t | S_t= s] = V_{\pi}(s)" />
</p>

This is exercise is based on:
- Example 5.1 of Sutton's book "Reinforcement Learning: An Introduction (2nd Edition)"
- The Monte Carlo example presented in "Lecture 4: Model-Free Prediction" of David Silver's Reinforcement Learning Course

The object of the popular casino card game of blackjack is to obtain cards the sum of whose numerical values is as great
as possible without exceeding 21. All face cards count as 10, and an ace can count either as 1 or as 11. We consider the
version in which each player competes independently against the dealer. The game begins with two cards dealt to both
dealer and player. One of the dealer’s cards is face up and the other is face down. If the player has 21 immediately 
(an ace and a 10-card), it is called a natural. He then wins unless the dealer also has a natural, in which case the
game is a draw. If the player does not have a natural, then he can request additional cards, one by one (hits), until he
either stops (sticks) or exceeds 21 (goes bust). If he goes bust, he loses; if he sticks, then it becomes the dealer’s
turn. The dealer hits or sticks according to a fixed strategy without choice: he sticks on any sum of 17 or greater, and
hits otherwise. If the dealer goes bust, then the player wins; otherwise, the outcome—win, lose, or draw—is determined 
by whose final sum is closer to 21.

Playing blackjack is naturally formulated as an episodic finite MDP. Each game of blackjack is an episode. Rewards of
+1, 1, and 0 are given for winning, losing, and drawing, respectively. All rewards within a game are zero, and we do not
discount <img src="https://latex.codecogs.com/svg.image?(\gamma&space;=&space;1)" title="(\gamma = 1)" />; therefore 
these terminal rewards are also the returns. The player’s actions are to hit or to stick. The states depend on the
player’s cards and the dealer’s showing card. We assume that cards are dealt from an infinite deck (i.e., with 
replacement) so that there is no advantage to keeping track of the cards already dealt. If the player holds an ace that
he could count as 11 without going bust, then the ace is said to be usable. In this case it is always counted as 11 
because counting it as 1 would make the sum 11 or less, in which case there is no decision to be made because, 
obviously, the player should always hit. Thus, the player makes decisions on the basis of three variables: his current
sum (12–21), the dealer’s one showing card (ace–10), and whether or not he holds a usable ace. This makes for a total of 
200 states.

Consider the policy that sticks if the player’s sum is 20 or 21, and otherwise hits.
```commandline
usage: blackjack_mc_prediction.py [--n_episodes N_EPISODES] [--threshold THRESHOLD] [--gamma GAMMA] [--plot] [-h]

optional arguments:
  --n_episodes N_EPISODES
                        The number of episodes to sample (DEFAULT=100000)
  --threshold THRESHOLD
                        The player hits if the sum is < threshold, otherwise he sticks. (DEFAULT=20)
  --gamma GAMMA         The discount factor of the Monte Carlo Prediction algorithm. (DEFAULT=1.0)
  --plot                Plot and save as blackjack_mcp_episodes_{n_episodes}.gif the value function per episode of the Monte Carlo Prediction algorithm and as blackjack_mcp_results_{n_episodes}.jpg the final results of the algorithm
                        (DEFAULT=False)
  -h, --help            Show this help message and exit.
```

```commandline
 python3 blackjack_mc_prediction.py --n_episodes 10000 --plot
```
<p align="center">
<img src="blackjack_mcp_episodes_10000.gif">
</p>

<p align="center">
<img src="blackjack_mcp_results_10000.jpg">
</p>

```commandline
 python3 blackjack_mc_prediction.py --n_episodes 500000 --plot
```
<p align="center">
<img src="blackjack_mcp_episodes_500000.gif">
</p>

<p align="center">
<img src="blackjack_mcp_results_500000.jpg">
</p>