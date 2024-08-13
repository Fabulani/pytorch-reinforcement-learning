import gymnasium as gym

NUM_GAMES = 1


if __name__ == "__main__":
    env = gym.make("LunarLander-v2", render_mode="human")
    try:
        for i in range(NUM_GAMES):
            obs, info = env.reset(seed=42)
            truncated = False
            terminated = False
            cumulative_reward = 0.0
            while not (terminated or truncated):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                cumulative_reward += reward
                env.render()
            print(f"Last episode reward: {cumulative_reward}")
    finally:
        env.close()
