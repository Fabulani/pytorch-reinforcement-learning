import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


"""
    Code loosely inspired from https://medium.com/@ym1942/proximal-policy-optimization-tutorial-f722f23beb83
    (replicates image from article tho)
"""


torch.autograd.set_detect_anomaly(True)

device = "cuda" if torch.cuda.is_available() else "cpu"

ENV_NAME = "CartPole-v1"
NUM_GAMES = 10
NUM_EPISODES = 100
GAMMA = 0.99
NUM_EPOCHS = 10
EPSILON = 0.2
LEARNING_RATE = 2.5e-4


class PPO(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n)
        )
        self.critic = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)

    def get_value(self, x):
        x = torch.tensor(x).to(device)
        return self.critic(x)

    def get_action_and_feedback(self, x, actions=None):
        x = torch.tensor(x).to(device)
        logits = self.actor(x)
        probs = torch.distributions.Categorical(logits=logits)
        if not actions:
            action = probs.sample()
        else:
            action = torch.tensor(actions)
        feedback = self.critic(x)

        return int(action) if not actions else action, probs.log_prob(action), probs.entropy(), feedback


class Rollout:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.current_index = 0

    def put(self, transition):
        self.observations.append(transition[0])
        self.actions.append(transition[1])
        self.rewards.append(transition[2])
        self.log_probs.append(transition[3])

    def reset(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.current_index = 0

    def get_transition(self):
        transition = (
            self.observations[self.current_index],
            self.actions[self.current_index],
            self.rewards[self.current_index],
            self.log_probs[self.current_index]
        )
        self.current_index += 1

        return transition

    def get(self):
        return self.observations, self.actions, self.rewards, self.log_probs

    def __len__(self):
        return len(self.actions)


def compute_clipped_loss(ratios, advantages):
    clipped_rewards = torch.clamp(ratios, min=1 - EPSILON, max=1 + EPSILON)
    L_clip = torch.mean(-torch.min(ratios * advantages, clipped_rewards * advantages))

    return L_clip


if __name__ == '__main__':
    env = gym.make(ENV_NAME, render_mode="human")
    obs, info = env.reset(seed=np.random.randint(0, 500))
    agent = PPO(env).to(device)
    rollout = Rollout()
    writer = SummaryWriter(log_dir=f"runs/ppo_cartpole_{NUM_EPISODES}")

    # Will it work without any training?
    final_mean_reward = 0.0
    for i in range(NUM_GAMES):
        obs, info = env.reset(seed=np.random.randint(0, 500))
        truncated = False
        terminated = False
        cumulative_reward = 0.0
        while not (terminated or truncated):
            action, _, _, _ = agent.get_action_and_feedback(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward
            env.render()
        print(f"Last episode reward: {cumulative_reward}")
        final_mean_reward += cumulative_reward
    print(f"Final mean reward for untrained agent: {final_mean_reward / NUM_GAMES}")
    env.close()

    # Let's train PPO
    env = gym.make(ENV_NAME)  # Re-instantiate the environment for good practice
    progress_bar = tqdm(range(NUM_EPISODES))
    obs, info = env.reset()
    terminated = False
    truncated = False
    episode_reward = 0.0
    episode_count = 0
    for ep in progress_bar:
        # Collect rollout
        while not (terminated or truncated):
            action, log_probs, _, _ = agent.get_action_and_feedback(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            transition = (obs, action, reward, log_probs)
            rollout.put(transition)
            # IMPORTANT!
            obs = next_obs
        progress_bar.set_description(f"Last episode reward: {np.sum(rollout.rewards)}")

        G = []
        T = len(rollout)
        for t in range(T):
            G_t = 0.0
            for k in range(t + 1, T):
                G_t += (GAMMA ** (k - t - 1)) * rollout.rewards[k]
            G.append(G_t)
            # Compute advantages
        G = torch.tensor(G)
        with torch.no_grad():
            A = G - agent.get_value(rollout.observations).squeeze()

        for e in range(NUM_EPOCHS):
            # Compute ratios
            _, log_probs, _, _ = agent.get_action_and_feedback(rollout.observations, actions=rollout.actions)
            r_theta = torch.exp(log_probs - torch.tensor(rollout.log_probs))
            # Compute losses
            clipped_loss = compute_clipped_loss(ratios=r_theta, advantages=A)
            # Update actor and critic
            agent.actor_optimizer.zero_grad()
            clipped_loss.backward(retain_graph=True)
            agent.actor_optimizer.step()

            # Update value net
            values = agent.get_value(torch.tensor(rollout.observations)).squeeze()
            loss = F.mse_loss(G.float(), values.float())
            agent.critic_optimizer.zero_grad()
            loss.backward()
            agent.critic_optimizer.step()

        obs, info = env.reset()
        terminated = False
        truncated = False
        rollout.reset()

    env.close()

    # Finally, we test the trained policy
    _ = input("Press a key to see the agent playing...")
    env = gym.make(ENV_NAME, render_mode="human")
    obs, info = env.reset(seed=np.random.randint(0, 500))
    final_mean_reward = 0.0
    for i in range(NUM_GAMES):
        obs, info = env.reset(seed=np.random.randint(0, 500))
        truncated = False
        terminated = False
        cumulative_reward = 0.0
        while not (terminated or truncated):
            action, _, _, _ = agent.get_action_and_feedback(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward
            env.render()
        print(f"Last episode reward: {cumulative_reward}")
        final_mean_reward += cumulative_reward
    print(f"Final mean reward for untrained agent: {final_mean_reward / NUM_GAMES}")
    env.close()
