import json

import optuna
from optuna.pruners import MedianPruner
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from train import make_env


class TrialPruningCallback(BaseCallback):
    def __init__(self, trial, eval_env, n_eval_episodes=5, check_freq=5000):
        super().__init__()
        self.trial = trial
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.check_freq = check_freq
        self.best_mean_reward = -float("inf")

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            mean_reward, _ = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes
            )
            self.trial.report(mean_reward, step=self.n_calls)

            # Prune if mean reward has not improved
            if self.trial.should_prune():
                return False
            self.best_mean_reward = max(self.best_mean_reward, mean_reward)
        return True


def find_valid_n_steps(n_envs=1, target_rollout_size=1024):
    valid_n_steps = [i for i in range(128, 2049) if target_rollout_size % i == 0]
    return valid_n_steps


def optimize_ppo(trial, n_envs=1):
    # Predefined list of valid n_steps values
    target_rollout_size = 1024 * n_envs
    valid_n_steps = find_valid_n_steps(n_envs, target_rollout_size)
    n_steps = trial.suggest_categorical("n_steps", valid_n_steps)

    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-7, 1e-2, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.1, log=True),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "gamma": trial.suggest_float("gamma", 0.9, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.95),
        "n_steps": n_steps,
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 0.9),
    }


def optimize_a2c(trial, n_envs=1):
    target_rollout_size = 1024 * n_envs
    valid_n_steps = find_valid_n_steps(n_envs, target_rollout_size)
    n_steps = trial.suggest_categorical("n_steps", valid_n_steps)

    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-7, 1e-2, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.1, log=True),
        "gamma": trial.suggest_float("gamma", 0.9, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.95),
        "n_steps": n_steps,
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 0.5),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 0.9),
    }


def optimize_dqn(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-7, 1e-3, log=True),
        "buffer_size": trial.suggest_int(
            "buffer_size", 1000, 5000
        ),  # Reduced buffer size
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8]),
        "gamma": trial.suggest_float("gamma", 0.9, 0.999),
        "target_update_interval": trial.suggest_int(
            "target_update_interval", 100, 1000
        ),
        "exploration_fraction": trial.suggest_float("exploration_fraction", 0.1, 0.5),
        "exploration_final_eps": trial.suggest_float(
            "exploration_final_eps", 0.01, 0.1
        ),
        "learning_starts": trial.suggest_int("learning_starts", 1000, 10000),
    }


def objective(trial):
    algorithm = trial.suggest_categorical("algorithm", ["PPO"])

    # Define n_stack as a hyperparameter
    n_stack = trial.suggest_int("n_stack", 1, 10)

    # Force n_stack to be 1 if the algorithm is DQN to prevent shape mismatch errors
    if algorithm == "DQN":
        n_stack = 1

    # Get the hyperparameters based on the selected algorithm
    if algorithm == "PPO":
        params = optimize_ppo(trial)
    elif algorithm == "A2C":
        params = optimize_a2c(trial)
    elif algorithm == "DQN":
        params = optimize_dqn(trial)

    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=n_stack)
    eval_env = DummyVecEnv([make_env])
    eval_env = VecFrameStack(eval_env, n_stack=n_stack)

    # Initialize the model based on the selected algorithm
    if algorithm == "PPO":
        model = PPO("CnnPolicy", env, **params, verbose=0)
    elif algorithm == "A2C":
        model = A2C("CnnPolicy", env, **params, verbose=0)
    elif algorithm == "DQN":
        model = DQN("CnnPolicy", env, **params, verbose=0)

    # Set up the pruning callback
    pruning_callback = TrialPruningCallback(trial, eval_env)

    # Train the model
    model.learn(total_timesteps=1000000, callback=pruning_callback)

    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5)
    return mean_reward


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", pruner=MedianPruner())
    study.optimize(objective, n_trials=50)

    if study.best_params["algorithm"] == "DQN":
        study.best_params["n_stack"] = 1

    print(study.best_params)

    with open("best_hyperparameters.json", "w") as f:
        json.dump(study.best_params, f)
