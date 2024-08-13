import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from algorithms.dqn import QNetwork, ReplayBuffer

"""
    Based on CleanRL: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py
"""


# Parameters
ENV_NAME = "LunarLander-v2"
NUM_GAMES = 5
NUM_TIMESTEPS = 500000
START_E = 1
END_E = 0.05
BUFFER_SIZE = 10000
LEARNING_RATE = 2.5e-4
BATCH_SIZE = 128
GAMMA = 0.99


# For how long during training will we reduce E (random action chance)
duration = int(NUM_TIMESTEPS / 2)


# CUDA support
# device = "cuda" if torch.cuda.is_available() else "cpu"  # if CUDA works
device = "cpu"


def epsilon_greedy_action_selection(obs, q_net, action_space, current_eps):
    if np.random.uniform() <= current_eps:
        act = action_space.sample()
    else:
        q_values = q_net(obs)
        act = int(torch.argmax(q_values))
    return act


def linear_schedule(t: int, start_e: float = 1.0, end_e: float = 0.05, duration: int = 100000):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    # Initialize network, env, and training components
    env = gym.make(ENV_NAME, render_mode="human")
    q_net = QNetwork(env).to(device)
    replay_buffer = ReplayBuffer(buffer_limit=BUFFER_SIZE)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    mse_loss = torch.nn.MSELoss()

    # Plotting
    writer = SummaryWriter(f"runs/{ENV_NAME}_dqn_{NUM_TIMESTEPS}_timesteps")

    # Quick run to check performance without training
    final_mean_reward = 0
    for i in range(NUM_GAMES):
        obs, info = env.reset(seed=np.random.randint(0, 500))
        terminated = False
        truncated = False
        cumulative_reward = 0
        while not (terminated or truncated):
            q_values = q_net(obs)
            action = int(torch.argmax(q_values))
            obs, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward
            env.render()
        print(f"Last episode reward: {cumulative_reward:.2f}")
        final_mean_reward += cumulative_reward
    print(f"Final mean reward for untrained agent: {final_mean_reward / NUM_GAMES}")
    env.close()

    # ----- Training the agent -----
    env = gym.make(ENV_NAME)  # Good practice to re-instantiate the environment
    progress_bar = tqdm(range(NUM_TIMESTEPS))
    obs, info = env.reset()
    episode_reward = 0
    episode_count = 0
    final_mean_reward = 0

    for t in progress_bar:
        current_eps = linear_schedule(t, start_e=START_E, end_e=END_E, duration=int(NUM_TIMESTEPS / 2))
        action = epsilon_greedy_action_selection(obs, q_net, env.action_space, current_eps)
        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        last_transition = (obs, action, reward, next_obs, terminated or truncated)
        replay_buffer.put(last_transition)

        obs = next_obs  # Very important (I don't know why)

        # Training only happens if there are enough transitions stored
        if t > BATCH_SIZE:
            s_observations, s_actions, s_rewards, s_next_observations, s_dones = replay_buffer.sample(n=BATCH_SIZE)
            with torch.no_grad():
                next_q_values = q_net(s_next_observations)
            ys = (
                torch.tensor(s_rewards)
                + (GAMMA * torch.max(next_q_values, dim=1)[0]) * torch.tensor((1 - s_dones)).flatten()
            )
            ys = ys.float()

            # Get the old values
            old_q_values = q_net(s_observations).gather(1, torch.LongTensor(s_actions).unsqueeze(1)).squeeze()

            # Update weights
            loss = mse_loss(ys, old_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % 100 == 0:
                writer.add_scalar("losses/loss", loss, t)
                writer.add_scalar("losses/q_values", old_q_values.mean().item(), t)

        if terminated or truncated:
            progress_bar.set_description(f"Episode {episode_count} - Reward: {episode_reward:.2f}")
            writer.add_scalar("charts/episodic_return", episode_reward, t)
            episode_reward = 0
            episode_count += 1
            obs, info = env.reset()

        final_mean_reward += episode_reward

    final_mean_reward = final_mean_reward / NUM_TIMESTEPS
    print(f"Final mean reward during training: {final_mean_reward:.2f}")
    env.close()

    _ = input("Press ENTER to watch the agent play...")

    # ----- Test -----
    env = gym.make(ENV_NAME, render_mode="human")
    final_mean_reward = 0
    for i in range(NUM_GAMES):
        obs, info = env.reset(seed=np.random.randint(0, 500))
        truncated = False
        terminated = False
        cumulative_reward = 0
        while not (terminated or truncated):
            q_values = q_net(obs)
            action = int(torch.argmax(q_values))
            obs, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward
            env.render()
        print(f"Last episode reward: {cumulative_reward:.2f}")
        final_mean_reward += cumulative_reward
    print(f"Final mean reward for trained agent: {final_mean_reward / NUM_GAMES}")
    env.close()

    # Save weight
    torch.save(q_net.state_dict(), "models/dqn_model")  # pth, ph extensions

    # to load:
    # q_net.load_state_dict(torch.load("models/dqn_model.pth"))
