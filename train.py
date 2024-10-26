import gymnasium as gym
import numpy as np
from skimage.color import rgb2gray
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


def make_env():
    env = gym.make(
        "ALE/SpaceInvaders-v5",
        repeat_action_probability=0.0,
        full_action_space=False,
    )
    env = gym.wrappers.TransformObservation(
        env, lambda obs: np.expand_dims(rgb2gray(obs), axis=-1)
    )
    return Monitor(
        env,
        directory="./logs/",
        video_callable=lambda episode_id: episode_id % 10000 == 0,
    )


def train_space_invaders():
    # Create the training environment and wrap it in VecTransposeImage
    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)

    # Create the evaluation environment and wrap it similarly
    eval_env = DummyVecEnv([make_env])
    eval_env = VecTransposeImage(eval_env)

    # Create the RL model with TensorBoard logging
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/")

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
    model.learn(total_timesteps=1000000, callback=[checkpoint_callback, eval_callback])

    # Save the trained model
    model.save("space_invaders_model")

    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")


if __name__ == "__main__":
    train_space_invaders()
