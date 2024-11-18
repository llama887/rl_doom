import time

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import *
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage

ENV_ID = "SpaceInvaders-v0"
NUM_ENV = 8
STEPS = 5_000


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")  # Enable render mode
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = FireResetEnv(env)
        env = WarpFrame(env)
        env = ClipRewardEnv(env)
        env = EpisodicLifeEnv(env)
        env.seed(seed + rank)
        return env

    return _init


def main():
    # Using SubprocVecEnv for better performance with parallel environments
    env = SubprocVecEnv([make_env(ENV_ID, i) for i in range(NUM_ENV)])
    env = VecTransposeImage(env)  # Enable image transposing for rendering

    model = PPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=STEPS)  # learning
    model.save(ENV_ID)  # save
    del model

    model = PPO.load(ENV_ID, env=env, verbose=1)  # load
    obs = env.reset()

    while True:  # replay
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render(mode="rgb_array")  # Use "rgb_array" for rendering
        time.sleep(1 / 60)


if __name__ == "__main__":
    main()
