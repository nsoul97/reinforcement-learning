<center><h1> Iterative Policy Evaluation </h1></center>

This algorithm is used to evaluate a given policy <img src="https://latex.codecogs.com/svg.image?\pi" title="\pi" />,
based on the Bellman Expectation Equation:

<!---
v(s) = \sum_{a}\pi(a|s)(R_s^a + \gamma \sum_{s'}P_{ss'}^a \ v(s'))
-->

<center>
<img src="https://latex.codecogs.com/svg.image?v(s)&space;=&space;\sum_{a}\pi(a|s)(R_s^a&space;&plus;&space;\gamma&space;\sum_{s'}P_{ss'}^a&space;\&space;v(s'))" title="v(s) = \sum_{a}\pi(a|s)(R_s^a + \gamma \sum_{s'}P_{ss'}^a \ v(s'))" />
</center>

This implementation of the algorithm uses synchronous backups to update the value function of each state:

<!---
v_k(s) = \sum_{a}\pi(a|s)(R_s^a + \gamma \sum_{s'}P_{ss'}^a \ v_{k-1}(s'))
-->

<center>
<img src="https://latex.codecogs.com/svg.image?v_k(s)&space;=&space;\sum_{a}\pi(a|s)(R_s^a&space;&plus;&space;\gamma&space;\sum_{s'}P_{ss'}^a&space;\&space;v_{k-1}(s'))" title="v_k(s) = \sum_{a}\pi(a|s)(R_s^a + \gamma \sum_{s'}P_{ss'}^a \ v_{k-1}(s'))" />
</center>

This exercise is based on:
- Example 4.1 of Sutton's book "Reinforcement Learning: An Introduction (2nd Edition)"
- The Iterative Policy Evaluation example presented in "Lecture 3: Planning by Dynamic Programming" of David Silver's
Reinforcement Learning Course

In the GridWorld environment, the agent can move up, down, left or right from any non-terminal state. The terminal
states are the upper left and the lower right cells. The environment is deterministic, meaning that each action
deterministically causes the corresponding state transition, with the exception of actions that would take the agent off 
the grid. In this case, the state does not change. The agent follows an equiprobable random policy (all actions 
equally likely), for which the value function is evaluated using the Iterative Policy Evaluation Algorithm.

```commandline
usage: gridworld_iterative_policy_evaluation.py [--height HEIGHT] [--width WIDTH] [--gamma GAMMA] [--epsilon EPSILON] [--plot] [-h]

optional arguments:
  --height HEIGHT    The height of the grid. (DEFAULT=4)
  --width WIDTH      The width of the grid. (DEFAULT=4)
  --gamma GAMMA      The discount factor of the iterative policy evaluation algorithm. (DEFAULT=1.0)
  --epsilon EPSILON  The iterative policy evaluation algorithm terminates once the value function change is less than epsilon for all states. (DEFAULT=1e-5)
  --plot             Plot and save (as gridworld_ipe_animation.gif) the results of the iterative policy evaluation algorithm per iteration. (DEFAULT=False)
  -h, --help         Show this help message and exit.
```

The following figure is the result of the iterative policy evaluation algorithm for a grid of height H=4 and width W=4.
```commandline
python3 gridworld_iterative_policy_evaluation.py --plot
````


<center>
<img src="gridworld_ipe_animation.gif"/>
</center>