import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from algorithms.dqn import QNetwork, ReplayBuffer

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
MODEL = "models/lunar_lander_dqn_model_500000"


# For how long during training will we reduce E (random action chance)
duration = int(NUM_TIMESTEPS / 2)


# CUDA support
# device = "cuda" if torch.cuda.is_available() else "cpu"  # if CUDA works
device = "cpu"

if __name__ == "__main__":
    # Initialize network, env, and training components
    env = gym.make(ENV_NAME, render_mode="human")
    q_net = QNetwork(env).to(device)
    q_net.load_state_dict(torch.load(MODEL))
    replay_buffer = ReplayBuffer(buffer_limit=BUFFER_SIZE)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    mse_loss = torch.nn.MSELoss()

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
