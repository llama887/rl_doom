import json

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from train import make_env


def optimize_ppo(trial):
    return {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
        "ent_coef": trial.suggest_loguniform("ent_coef", 1e-8, 1e-1),
        "clip_range": trial.suggest_uniform("clip_range", 0.1, 0.4),
        "gamma": trial.suggest_uniform("gamma", 0.9, 0.999),
        "gae_lambda": trial.suggest_uniform("gae_lambda", 0.8, 0.95),
        "n_steps": trial.suggest_int("n_steps", 128, 2048, log=True),
        "max_grad_norm": trial.suggest_uniform("max_grad_norm", 0.3, 0.9),
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
