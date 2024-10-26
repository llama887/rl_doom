import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from skimage.color import rgb2gray
import numpy as np

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
        video_callable=lambda episode_id: episode_id % 10000 == 0,
    )

def watch_trained_agent():
    # Create the environment
    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)

    # Load the trained model
    model = PPO.load("space_invaders_model", env=env)

    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

if __name__ == "__main__":
    watch_trained_agent()