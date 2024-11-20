import json
import os

from gymnasium.wrappers import RecordVideo
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from train import make_env


def load_hyperparameters(filename="best_hyperparameters.json"):
    # Load the best hyperparameters from a JSON file
    with open(filename, "r") as f:
        params = json.load(f)
    return params


def evaluate_best_model(
    model_path,
    hyperparameters_file="best_hyperparameters.json",
    use_atari_make=True,
    video_folder="videos/",
    num_episodes=1,
):
    # Create the video folder if it doesn't exist
    os.makedirs(video_folder, exist_ok=True)

    # Load hyperparameters
    hyperparameters = load_hyperparameters(hyperparameters_file)
    algorithm = hyperparameters.get("algorithm", "PPO")
    n_stack = hyperparameters.get("n_stack", 4)

    # Override n_stack for DQN
    if algorithm == "DQN":
        n_stack = 1

    # Choose the environment creation function
    if use_atari_make:
        base_env = make_atari_env("PongNoFrameskip-v4", n_envs=1)
    else:
        base_env = DummyVecEnv([make_env])

    # Wrap the base environment with RecordVideo first
    video_env = RecordVideo(base_env, video_folder, episode_trigger=lambda x: True)

    # Apply VecFrameStack after video recording
    eval_env = VecFrameStack(video_env, n_stack=n_stack)

    # Load the appropriate model
    if algorithm == "PPO":
        model = PPO.load(model_path)
    elif algorithm == "A2C":
        model = A2C.load(model_path)
    elif algorithm == "DQN":
        model = DQN.load(model_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Evaluate the model and save videos
    total_rewards = []
    for episode in range(num_episodes):
        obs = eval_env.reset(options=None)
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward[0]}")

    mean_reward = sum(total_rewards) / num_episodes
    print(f"Mean reward over {num_episodes} episodes: {mean_reward}")

    # Close the environment
    eval_env.close()


if __name__ == "__main__":
    # Update the model path and hyperparameters file if needed
    best_model_path = "logs/best_model.zip"
    hyperparameters_file = "best_hyperparameters.json"

    evaluate_best_model(best_model_path, hyperparameters_file=hyperparameters_file)
