<h3>Overview [In Progress :construction: :construction:]</h3>
This repository is based on:

- [David Silver's Reinforcement Learning Course](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
- [Reinforcement Learning: An Introduction (2nd Edition)](http://incompleteideas.net/book/RLbook2018.pdf)
- The GitHub repository: https://github.com/dennybritz/reinforcement-learning

The goal of the repository is to implement popular Reinforcement Learning Algorithms, provide interesting plots and 
animations for each algorithm, including the figures that are presented in the RL bible of Sutton and Barto and in the
lectures of the RL :goat: David Silver.

This repository contains, but is not limited to, the exercises that are presented in Danny Britz' s excellent 
repository. The exercises are implemented from scratch in [Python 3](https://www.python.org/), using
[OpenAI Gym](https://gym.openai.com/) and [PyTorch](https://pytorch.org/) ML framework.

<h3>Algorithm Implementations</h3>
- <b>Planning by Dynamic Programming</b>
    - [Iterative Policy Evaluation](gridworld_iterative_policy_evaluation)
    - [Policy Iteration](gridworld_policy_iteration)
    - [Value Iteration](gridworld_value_iteration)
- <b>Model-Free Prediction</b>
    - [Monte Carlo Prediction](blackjack_mc_prediction)
- <b>Model-Free Control</b>
    - [Monte Carlo Control with Exploring Starts](blackjack_mc_control_exploring_starts)
    - [Off-Policy Monte Carlo Control with Weighted Importance Sampling](blackjack_off_policy_mc_control)