import json

import gymnasium as gym
import numpy as np
from skimage.color import rgb2gray
from stable_baselines3 import A2C, DQN, PPO
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


def load_hyperparameters(filename="best_hyperparameters.json"):
    # Load the best hyperparameters from a JSON file
    with open(filename, "r") as f:
        params = json.load(f)
    return params


def watch_trained_agent():
    # Load the best hyperparameters
    hyperparams = load_hyperparameters()
    algorithm = hyperparams.pop("algorithm", "PPO")  # Default to PPO if not specified
    n_stack = hyperparams.pop("n_stack", 4)

    # Force n_stack to be 1 if using DQN to avoid shape mismatches
    if algorithm == "DQN":
        n_stack = 1

    # Create the environment with frame stacking
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=n_stack)

    # Load the trained model based on the specified algorithm
    if algorithm == "PPO":
        model = PPO.load("logs/best_model", env=env)
    elif algorithm == "A2C":
        model = A2C.load("logs/best_model", env=env)
    elif algorithm == "DQN":
        model = DQN.load("logs/best_model", env=env)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=10, render=True
    )
    print(f"Mean reward: {mean_reward} +/- {std_reward}")


if __name__ == "__main__":
    watch_trained_agent()
