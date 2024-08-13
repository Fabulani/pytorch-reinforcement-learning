import gymnasium as gym
import numpy as np
import torch

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from algorithms.dqn import QNetwork, ReplayBuffer


"""
    CODE BASED ON CleanRL CODE: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py
    Adapted for this tutorial
"""

ENV_NAME = "LunarLander-v2"
NUM_GAMES = 10
NUM_TIMESTEPS = 500000
START_E = 1.0
END_E = 0.05
BUFFER_SIZE = 10000
LEARNING_RATE = 2.5e-4
BATCH_SIZE = 128
GAMMA = 0.99
TARGET_UPDATE_FREQUENCY = 500
TRAIN_FREQUENCY = 10
TAU = 1.0
LEARNING_STARTS = 10000

device = "cuda" if torch.cuda.is_available() else "cpu"


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


if __name__ == '__main__':
    # Initialize network, env and training components
    env = gym.make(ENV_NAME, render_mode="human")
    q_net = QNetwork(env).to(device)
    # We add a "target q network" used to stabilize training
    target_net = QNetwork(env).to(device)
    target_net.load_state_dict(q_net.state_dict())
    replay_buffer = ReplayBuffer(buffer_limit=BUFFER_SIZE)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    mse_loss = torch.nn.MSELoss()

    # Plotting
    writer = SummaryWriter(f"runs/{ENV_NAME}_dqn_td_{NUM_TIMESTEPS}_timesteps")

    # Let's see how our guy does without any training
    final_mean_reward = 0.0
    for i in range(NUM_GAMES):
        obs, info = env.reset(seed=np.random.randint(0, 500))
        truncated = False
        terminated = False
        cumulative_reward = 0.0
        while not (terminated or truncated):
            q_values = q_net(obs)
            action = int(torch.argmax(q_values))
            obs, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward
            env.render()
        print(f"Last episode reward: {cumulative_reward}")
        final_mean_reward += cumulative_reward
    print(f"Final mean reward for untrained agent: {final_mean_reward / NUM_GAMES}")
    env.close()

    # Ok, let's try to train it
    env = gym.make(ENV_NAME)   # Re-instantiate the environment for good practice
    progress_bar = tqdm(range(NUM_TIMESTEPS))
    obs, info = env.reset()
    episode_reward = 0.0
    episode_count = 0
    for t in progress_bar:
        current_eps = linear_schedule(t, start_e=START_E, end_e=END_E, duration=int(NUM_TIMESTEPS / 2))
        action = epsilon_greedy_action_selection(obs=obs, q_net=q_net, action_space=env.action_space, current_eps=current_eps)
        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        last_transition = (obs, action, reward, next_obs, terminated or truncated)
        replay_buffer.put(last_transition)

        # Train step (happens only if we have enough transitions stored!)
        if t > LEARNING_STARTS:
            if t % TRAIN_FREQUENCY == 0:
                s_observations, s_actions, s_rewards, s_next_observations, s_dones = replay_buffer.sample(n=BATCH_SIZE)
                with torch.no_grad():
                    next_q_values = target_net(s_next_observations)
                    ys = torch.tensor(s_rewards).flatten() + (GAMMA * torch.max(next_q_values, dim=1)[0]) * torch.tensor((1 - s_dones)).flatten()
                    ys = ys.float()
                # Compute old values
                old_q_values = q_net(s_observations).gather(1, torch.LongTensor(s_actions).unsqueeze(1)).squeeze()
                # Updating weights of the network
                loss = mse_loss(ys, old_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if t % 100 == 0:
                    writer.add_scalar("losses/loss", loss, t)
                    writer.add_scalar("losses/q_values", old_q_values.mean().item(), t)

            if t % TARGET_UPDATE_FREQUENCY == 0:
                # Occasionally, we perform a "soft update" of the target network
                for target_network_param, q_network_param in zip(target_net.parameters(), q_net.parameters()):
                    target_network_param.data.copy_(
                        TAU * q_network_param.data + (1.0 - TAU) * target_network_param.data
                    )

        # IMPORTANT!
        obs = next_obs

        if terminated or truncated:
            progress_bar.set_description(f"Episode {episode_count} - Reward: {episode_reward}")
            writer.add_scalar("charts/episodic_return", episode_reward, t)
            episode_reward = 0
            episode_count += 1
            obs, info = env.reset()

    env.close()

    _ = input("Press a key to see the agent playing...")
    # Time to test our trained agent!
    env = gym.make(ENV_NAME, render_mode="human")
    final_mean_reward = 0.0
    for i in range(NUM_GAMES):
        obs, info = env.reset(seed=np.random.randint(0, 500))
        truncated = False
        terminated = False
        cumulative_reward = 0.0
        while not (terminated or truncated):
            q_values = q_net(obs)
            action = int(torch.argmax(q_values))
            obs, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward
            env.render()
        print(f"Last episode reward: {cumulative_reward}")
        final_mean_reward += cumulative_reward
    print(f"Final mean reward for trained agent: {final_mean_reward / NUM_GAMES}")
    env.close()
