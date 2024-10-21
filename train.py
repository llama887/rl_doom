from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from vizdoom_gym import VizDoomGymWrapper

# Initialize the custom VizDoom environment
env = VizDoomGymWrapper()

# Verify that the environment follows Gym API
check_env(env)

# Create the PPO model
model = PPO("CnnPolicy", env, verbose=1)

# Train the model for a specified number of time steps
model.learn(total_timesteps=10000)

# Save the model for later use
model.save("ppo_vizdoom")

# Close the environment after training
env.close()
