from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from train import load_hyperparameters, make_env


def evaluate_best_model(model_path, algorithm="PPO", n_stack=4):
    # Force n_stack to be 1 if using DQN
    if algorithm == "DQN":
        n_stack = 1

    # Create the evaluation environment with human render mode
    def make_eval_env():
        return Monitor(make_env(render_mode="human"))

    eval_env = DummyVecEnv([make_eval_env])  # Wrap in a vectorized environment
    eval_env = VecFrameStack(eval_env, n_stack=n_stack)  # Apply frame stacking

    # Load the appropriate model
    if algorithm == "PPO":
        model = PPO.load(model_path)
    elif algorithm == "A2C":
        model = A2C.load(model_path)
    elif algorithm == "DQN":
        model = DQN.load(model_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Render a single evaluation episode
    obs = eval_env.reset()
    done = [False]
    while not done[0]:  # Handle batched environments
        action, _ = model.predict(obs)
        obs, rewards, done, infos = eval_env.step(action)
        eval_env.render()

    # Close the rendering
    eval_env.close()


if __name__ == "__main__":
    # Update the model path and algorithm if needed
    best_model_path = "./logs/best_model.zip"

    params = load_hyperparameters()
    algorithm = params.pop("algorithm", "PPO")
    n_stack = params.pop("n_stack", 4)

    evaluate_best_model(best_model_path, algorithm=algorithm, n_stack=n_stack)
