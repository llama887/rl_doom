import time

import gymnasium as gym


def atari_test():
    env = gym.make(
        "ALE/SpaceInvaders-v5",
        render_mode="human",
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
    )
    env.metadata["render_fps"] = 6

    # Reset the environment to start
    observation, info = env.reset()

    # Set the end time to 5 seconds from now
    end_time = time.time() + 5

    while time.time() < end_time:  # Run for 5 seconds
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
