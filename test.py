from stable_baselines3 import PPO

from vizdoom_gym import VizDoomGymWrapper

# Load the trained model
model = PPO.load("ppo_vizdoom")

# Evaluate the model
env = VizDoomGymWrapper()

# Run for a few episodes
for episode in range(5):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Predict the action using the trained model
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    print(f"Episode {episode + 1}: Total Reward: {total_reward}")

env.close()
