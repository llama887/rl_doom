import json

import gymnasium as gym
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
    WarpFrame,
)
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

USE_ATARI_MAKE = True


def make_env(render_mode=None):
    env = gym.make(
        "ALE/Pong-v5",
        repeat_action_probability=0.0,
        full_action_space=False,
        render_mode=render_mode,
    )
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = EpisodicLifeEnv(env)
    return Monitor(env)


def load_hyperparameters(filename="best_hyperparameters.json"):
    # Load the best hyperparameters from a JSON file
    with open(filename, "r") as f:
        params = json.load(f)
    return params


def load_model_and_envs():
    # Load the best hyperparameters
    hyperparameters = load_hyperparameters()
    algorithm = hyperparameters.pop(
        "algorithm", "PPO"
    )  # Default to PPO if not specified
    n_stack = hyperparameters.pop("n_stack", 4)

    # Force n_stack to be 1 if using DQN to avoid shape mismatches
    if algorithm == "DQN":
        n_stack = 1

    # Choose the environment creation function
    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    # Apply VecFrameStack for frame stacking
    env = VecFrameStack(env, n_stack=n_stack)
    eval_env = VecFrameStack(eval_env, n_stack=n_stack)

    # Select the RL model based on the specified algorithm
    if algorithm == "PPO":
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            **hyperparameters,
        )
    elif algorithm == "A2C":
        model = A2C(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            **hyperparameters,
        )
    elif algorithm == "DQN":
        model = DQN(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            **hyperparameters,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return model, env, eval_env


if __name__ == "__main__":
    model, env, eval_env = load_model_and_envs()
    # Create callbacks for checkpointing and evaluation
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./logs/")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    # Train the model with callbacks
    model.learn(
        total_timesteps=10_000_000, callback=[checkpoint_callback, eval_callback]
    )

    # Save the trained model
    model.save("pong_model")

    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
