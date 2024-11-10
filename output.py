import gymnasium as gym
import numpy as np
from skimage.color import rgb2gray
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


def make_env():
    env = gym.make(
        "ALE/SpaceInvaders-v5",
        repeat_action_probability=0.0,
        full_action_space=False,
        render_mode="human",
    )
    env.metadata["render_fps"] = 30

    # Convert to grayscale and reshape
    env = gym.wrappers.TransformObservation(
        env, lambda obs: np.expand_dims(rgb2gray(obs), axis=-1)
    )
    return Monitor(env)


def watch_trained_agent():
    # Create the environment with frame stacking
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)  # Stack the last 4 grayscale frames

    # Load the trained model
    model = PPO.load("logs/best_model", env=env)

    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=10, render=True
    )
    print(f"Mean reward: {mean_reward} +/- {std_reward}")


if __name__ == "__main__":
    watch_trained_agent()
