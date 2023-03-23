# Gymnasium-CartPole-QLearining
The solution to my first quiz in the "Reinforcement Learning for Mechatronics Engineers and Optimal Control" course. Is not perfect but it works. There exists many unused functions and variables atm, this is due to us being short on time and thus unable to clear everything atm.

## Usage

Just make sure that you have `numpy`, `gymnasium`, `pyglet` and `pygame` installed.

or simply use the following command:
> pip install -U numpy gymnasium pyglet pygame

## Posible modifications
You could change the `LEARNING_RATE`, `EPSILON`, `DISCOUNT`, `EPISODES`, `DISC_STEPS` to modify the performance of the model. You could also implement an `Epsilon decay` model to prioritize the greedy action in the later episodes. Feel free to open a pull request with any modifications that would allow the algorithm to run better.

### Output Figures
![Performance](https://github.com/AbduEhab/Gymnasium-CartPole-QLearining/blob/main/figures/performance.png?raw=true)
![Optimal Policy](https://github.com/AbduEhab/Gymnasium-CartPole-QLearining/blob/main/figures/optimal_policy.png?raw=true)