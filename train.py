import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

def train_space_invaders():
    # Create the environment
    env = gym.make(
        "ALE/SpaceInvaders-v5",
        repeat_action_probability=0.0,
        full_action_space=False,
    )

    # Wrap the environment in a DummyVecEnv
    env = DummyVecEnv([lambda: env])

    # Create the RL model with TensorBoard logging
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./tensorboard_logs/")

    # Create a callback that saves the model every 10000 steps
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs/')

    # Create a callback that evaluates the model every 10000 steps
    eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=10000,
                                 deterministic=True, render=False)

    # Train the model for 100000 steps with the callbacks
    model.learn(total_timesteps=100000, callback=[checkpoint_callback, eval_callback])

    # Save the trained model
    model.save("space_invaders_model")

    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

if __name__ == "__main__":
    train_space_invaders()
