# Reinforcement Learning with PyTorch and Gymnasium

We implemented Deep Q Network (DQN) and Proximal Policy Optimization (PPO) algorithms from scratch using PyTorch. The agents were trained and tested in the Gymnasium environments "Lunar Landing V2" and "Cartpole".

- [Reinforcement Learning with PyTorch and Gymnasium](#reinforcement-learning-with-pytorch-and-gymnasium)
  - [Quick start](#quick-start)
  - [Repository structure](#repository-structure)
  - [Terminology](#terminology)
  - [Deep Q Network (DQN)](#deep-q-network-dqn)
    - [Q-Learning](#q-learning)
    - [Catastrophic Forgetting](#catastrophic-forgetting)
  - [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)
  - [Adversarial Imitation Learning (AIL)](#adversarial-imitation-learning-ail)
  - [Acknowledgements](#acknowledgements)

## Quick start

```sh
pip install -r requirements.txt
python3 lunar_lander_dqn.py
```

Check the common issues section if you have problems with `box2d`.

## Repository structure

The repo is structured as follows:

- `algorithms`: contains implementation for the DQN and PPO algorithms.
- `docs`: graphics used for documentation.
- `models`: saved model weights.
- `lunar_lander_random.py`: random agent in the Lunar Lander environment.
- `lunar_lander_dqn.py`: train and test agent in the Lunar Lander environment using the DQN algorithm.
- `lunar_lander_dqn_td.py`: a modified version of the DQN algorithm more resilient against catastrophic memory loss and forgetfulness.
- `cartpole_ppo.py`: train and test agent in the Cartpole environment using the PPO algorithm.

## Terminology

Common terminology used in Reinforcement Learning (and in this README) includes:

- states: represent the different situations or configurations the agent can encounter.
- actions: the set of all possible decisions or moves the agent can make.
- rewards: feedback received after taking an action, indicating the immediate benefit of that action.
- policy: a strategy used by the agent to decide which action to take given the current state.

## Deep Q Network (DQN)

A Deep Q Network (DQN) approximates the optimal action-value function (Q-function) using deep neural networks. It is designed to enable agents to learn how to behave optimally in environments with large, high-dimensional state spaces, where traditional Q-learning would be computationally infeasible.

Instead of maintaining a large Q-table, DQN uses a deep neural network to approximate the Q-function. This network takes the state as input and outputs the Q-values for all possible actions.

To stabilize training, we use a technique called Experience Replay by storing the agent's experiences (state, action, reward, next state) in a memory buffer. The DQN is trained by sampling random mini-batches from this buffer, which helps to break the correlation between consecutive experiences and reduces variance.

We can further stabilize training by using a separate target network that is a copy of the Q-network. The target network is updated less frequently and is used to compute the target Q-value in the Bellman equation. This helps in reducing oscillations and divergence during training.

During training, we use an exploration-exploitation strategy where the agent occasionally takes random actions (exploration) to discover new states and rewards, rather than always taking the action it currently believes is best (exploitation). The probability of taking a random action is controlled by the parameter ϵ, which decays over time.

### Q-Learning

Q-Learning is a value-based RL algorithm where the goal is to learn the optimal action-value function, known as the Q-function `Q(s, a)`, which estimates the expected future rewards for taking action `a` in state `s` and following the optimal policy thereafter. The Q-function is updated iteratively using the Bellman equation.

### Catastrophic Forgetting

In the Lunar Lander environment, we notice catastrophic forgetting during training around the 360k step. It's episodic return (reward) and losses can be seen in the following graphs:

![Episodic return](docs/lunar_lander_dqn/episodic_return.png)

![Loss and q values charts](docs/lunar_lander_dqn/loss.png)

We can mitigate this undesired effect by implementing a modification from recent research: a target network that our base DQN will mirror.

## Proximal Policy Optimization (PPO)

Proximal Policy Optimization (PPO) is a popular reinforcement learning algorithm that is used to train agents using policy gradient methods, which optimize a policy directly by adjusting its parameters to maximize the expected reward. Its relatively stable, efficient, and easy to implement. Some key features include:

- **clipping**: PPO introduces a clipping mechanism to prevent large, potentially destructive updates to the policy. The idea is to limit the change in the policy between updates to ensure stable learning. The objective function is clipped within a range, which helps to keep the updates conservative.
- **surrogate objective**: PPO uses a surrogate objective function that estimates the expected reward while considering the ratio of the new policy to the old policy. This objective is adjusted with the clipping, balancing exploration and exploitation without deviating too much from the current policy.

## Adversarial Imitation Learning (AIL)

Adversarial Imitation Learning (AIL) combines concepts from both imitation learning and adversarial training, inspired by Generative Adversarial Networks (GANs). It involves two competing networks: a _generator_ (the learning agent) and a _discriminator_:

- generator: attempts to produce behavior that is indistinguishable from the expert's with the goal of tricking the discriminator, who provides feedback used to train the generator.
- discriminator: tries to distinguish between the behavior generated by the agent and that of the expert.

This creates an adversarial dynamic where the agent continually improves its policy to fool the discriminator, while the discriminator gets better at identifying the differences between the agent’s behavior and the expert's.

Pros:

- No explicit reward function is needed.
- Generalization: by matching the distribution of expert behaviors rather than specific actions, AIL can generalize better to new states that were not seen in the expert demonstrations.

Cons:

- It can be quite unstable, as there are two competing objectives.
- It carries over all the downsides from GANs.

AIL is used in environments where defining the reward function is unfeasible, e.g., robotics, autonomous driving, and some game AI.

## Acknowledgements

These short projects were developed during the UEF Summer School "Artificial Intelligence (AI) for Computer Games" lectured by Prof. Ville Hautamäki and Federico Malato.

GAIL code provided by Prof. Ville Tanskanen during his guest lecture.
