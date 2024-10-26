import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

def train_space_invaders():
    # Create the environment
    env = gym.make(
        "ALE/SpaceInvaders-v5",
        repeat_action_probability=0.0,
        full_action_space=False,
    )

    # Wrap the environment in a DummyVecEnv
    env = DummyVecEnv([lambda: env])

    # Create the RL model
    model = PPO('CnnPolicy', env, verbose=1)

    # Train the model for 100000 steps
    model.learn(total_timesteps=100000)

    # Save the trained model
    model.save("space_invaders_model")

    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

if __name__ == "__main__":
    train_space_invaders()
