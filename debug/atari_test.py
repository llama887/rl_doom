import gymnasium as gym


def atari_test():
    # Create the Atari environment
    env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")

    # Reset the environment to start
    observation, info = env.reset()

    for _ in range(1000):  # Run for 1000 timesteps
        env.render()  # Render the environment

        # Sample a random action
        action = env.action_space.sample()

        # Step through the environment with the action
        observation, reward, done, truncated, info = env.step(action)

        # If the episode is over, reset the environment
        if done or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    atari_test()
