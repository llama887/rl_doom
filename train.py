import json
import gymnasium as gym
import numpy as np
from skimage.color import rgb2gray
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from gymnasium.wrappers import ResizeObservation, TransformObservation

def make_env():
    """
    Creates a Space Invaders environment with preprocessing:
    - Grayscale conversion
    - Image resizing to 84x84 pixels
    - Manual reward clipping to the range [-1, 1]
    """
    env = gym.make(
        "ALE/SpaceInvaders-v5",  # Ensure ALE prefix is correct for gymnasium Atari
        repeat_action_probability=0.0,
        full_action_space=False,
    )
    # Grayscale the observation
    env = TransformObservation(env, lambda obs: np.expand_dims(rgb2gray(obs), axis=-1))
    # Resize observation to 84x84 pixels
    env = ResizeObservation(env, shape=(84, 84))

    # Apply reward clipping manually
    class ClipRewardEnv(gym.RewardWrapper):
        def reward(self, reward):
            return np.clip(reward, -1, 1)

    env = ClipRewardEnv(env)
    return Monitor(env)

def load_hyperparameters(filename="best_hyperparameters.json"):
    with open(filename, "r") as f:
        params = json.load(f)
    return params

def train_space_invaders():
    hyperparams = load_hyperparameters()

    # Training environment
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)
    env = VecNormalize(env, norm_reward=True)

    # Evaluation environment
    eval_env = DummyVecEnv([make_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecNormalize(eval_env, norm_reward=True, training=False)

    # Initialize PPO model with TensorBoard logging
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=64,
        gamma=0.99,
        **hyperparams,
    )

    # Define checkpoint and evaluation callbacks
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./logs/")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    # Start training
    model.learn(total_timesteps=2000000, callback=[checkpoint_callback, eval_callback])
    model.save("space_invaders_model")

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

if __name__ == "__main__":
    train_space_invaders()
