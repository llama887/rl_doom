import json

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from train import make_env


def find_valid_n_steps_and_batch_size(n_envs=1, target_rollout_size=1024):
    # Define possible values for n_steps based on factors of the target rollout size
    valid_n_steps = [i for i in range(128, 2049) if target_rollout_size % i == 0]

    # Calculate valid batch sizes that will divide the target rollout size evenly
    valid_batch_sizes = [
        i for i in [16, 32, 64, 128, 256, 512] if target_rollout_size % i == 0
    ]

    return valid_n_steps, valid_batch_sizes


def optimize_ppo(trial, n_envs=1):
    # Define a target rollout size (e.g., 1024, 2048) based on resource constraints
    target_rollout_size = 1024 * n_envs

    # Get valid n_steps and batch sizes based on the target rollout size
    valid_n_steps, valid_batch_sizes = find_valid_n_steps_and_batch_size(
        n_envs, target_rollout_size
    )

    # Sample n_steps from valid options
    n_steps = trial.suggest_categorical("n_steps", valid_n_steps)

    # Sample batch_size from valid options
    batch_size = trial.suggest_categorical("batch_size", valid_batch_sizes)

    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.1, log=True),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "gamma": trial.suggest_float("gamma", 0.9, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.95),
        "n_steps": n_steps,
        "batch_size": batch_size,
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 0.9),
    }


def objective(trial):
    # Get the hyperparameters
    params = optimize_ppo(trial)

    # Set up environments with frame stacking
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)
    eval_env = DummyVecEnv([make_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    # Initialize the model with these hyperparameters
    model = PPO("CnnPolicy", env, **params, verbose=0)

    # Train the model for a small number of timesteps for evaluation purposes
    model.learn(total_timesteps=10000)

    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5)
    return mean_reward


if __name__ == "__main__":
    # Run the hyperparameter optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # Output the best hyperparameters
    print(study.best_params)

    # Save the best parameters to a JSON file
    with open("best_hyperparameters.json", "w") as f:
        json.dump(study.best_params, f)
