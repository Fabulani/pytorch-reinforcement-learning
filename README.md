# Reinforcement Learning with PyTorch and Gymnasium

We implemented multiple Reinforcement Learning (RL) algorithms from scratch using PyTorch. The agents were trained and tested in adequate environments using the `gymnasium` library.

I used this repository to both store those projects and document my RL-learning (:eyes:). You'll find brief summaries of the different methods and algorithms, plus some scripts to train and test the agents.

Some agents and environment combinations (e.g., MARL in the foraging environment) are not expected to work. Theoretically, they can work, but it would require smarter tricks, e.g., curriculum implementation where the tasks start easy and get progressively more challenging.

## Table of contents

- [Reinforcement Learning with PyTorch and Gymnasium](#reinforcement-learning-with-pytorch-and-gymnasium)
  - [Table of contents](#table-of-contents)
  - [Quick start](#quick-start)
  - [Repository structure](#repository-structure)
  - [Terminology](#terminology)
  - [Deep Q Network (DQN)](#deep-q-network-dqn)
    - [Q-Learning](#q-learning)
    - [Catastrophic Forgetting](#catastrophic-forgetting)
  - [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)
  - [Adversarial Imitation Learning (AIL)](#adversarial-imitation-learning-ail)
  - [Multi Agent Reinforcement Learning (MARL)](#multi-agent-reinforcement-learning-marl)
    - [Central Q-Learning (CQL)](#central-q-learning-cql)
    - [Independent Q-Learning (IQL)](#independent-q-learning-iql)
  - [Common issues](#common-issues)
  - [Acknowledgements](#acknowledgements)

## Quick start

```sh
pip install -r requirements.txt
python3 lunar_lander_dqn.py
```

Check the [Common issues](#common-issues) section if you have problems with `box2d`.

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

## Multi Agent Reinforcement Learning (MARL)

Multi-Agent Reinforcement Learning (MARL) involves multiple agents interacting within a shared environment. Each agent seeks to maximize its own cumulative reward while also adapting to the actions of other agents. The key challenges in MARL include:

- coordination: agents need to learn how to work together or compete effectively;
- scalability: the complexity of the environment increases as more agents are added;
- non-stationarity: each agent's policy changes over time as other agents learn, making the environment non-stationary from each agent's perspective.

MARL methods can be categorized into different frameworks depending on whether the agents cooperate, compete, or both.

### Central Q-Learning (CQL)

Central Q-Learning (CQL) is a framework designed for MARL where a centralized entity maintains and updates a shared Q-function for all agents. This approach involves:

- centralized training: during training, a central critic (or Q-function) evaluates the actions of all agents. This central critic uses global information (e.g., the state of all agents) to compute Q-values and update policies;
- decentralized execution: during execution, each agent makes decisions based on its local observations and policy, without needing to share its actions or observations with other agents.

CQL helps to mitigate the non-stationarity problem by providing a consistent Q-value estimate during training.

### Independent Q-Learning (IQL)

In Independent Q-Learning (IQL), each agent uses its own Q-learning algorithm independently of other agents. Key aspects include:

- individual learning: each agent maintains its own Q-function and updates it based on its own experiences and observations;
- local Q-function: agents do not have access to global state information or the Q-functions of other agents. They treat other agents as part of the environment;
- challenges: the primary challenge is that the environment is non-stationary due to the concurrent learning of other agents, which can destabilize the learning process.

IQL is simpler to implement but may be less effective in highly interactive environments compared to approaches that consider the interactions between agents.

Pros:

- not too many actions;
- supports partial observability;
- trains exactly as SARL.

Cons:

- coordinating agents is very hard. Some form of communication might be necessary;
- trains `n` networks, which means it is `n` times slower than CQL.

## Common issues

1. During installation, you might get the `gymnasium[box2d]` problem, which says something like this in the error: `module not found: swig`. To fix this, follow the instructions in [this stackoverflow answer](https://stackoverflow.com/questions/76222239/pip-install-gymnasiumbox2d-not-working-on-google-colab). Basically, do the following:

   ```sh
    pip install swig
    pip install gymnasium[box2d]
    pip install -r requirements.txt
   ```

## Acknowledgements

These short projects were developed during the UEF Summer School "Artificial Intelligence (AI) for Computer Games" lectured by Prof. Ville Hautamäki and Federico Malato.

GAIL code provided by Prof. Ville Tanskanen during his guest lecture.
