import gymnasium as gym
import numpy as np
import vizdoom as vzd
from gymnasium import spaces


class VizDoomGymWrapper(gym.Env):
    def __init__(self):
        super(VizDoomGymWrapper, self).__init__()

        # Create a DoomGame instance
        self.game = vzd.DoomGame()

        # Load basic configuration file
        self.game.load_config(vzd.scenarios_path + "/basic.cfg")
        self.game.set_doom_scenario_path(vzd.scenarios_path + "/basic.wad")
        self.game.set_doom_map("map01")

        # Set the game to be played in PLAYER mode
        self.game.set_mode(vzd.Mode.PLAYER)
        self.game.init()

        # Define action and observation space
        # Assuming action space size corresponds to available buttons
        self.action_space = spaces.Discrete(self.game.get_available_buttons_size())

        # Observation space: shape (height, width, channels) from the screen buffer
        screen_shape = self.game.get_state().screen_buffer.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=screen_shape, dtype=np.uint8
        )

    def reset(self):
        """Reset the environment and return the initial observation."""
        self.game.new_episode()
        state = self.game.get_state()
        return state.screen_buffer

    def step(self, action):
        """Apply the action to the environment and return results."""
        # Convert action to the form accepted by VizDoom
        buttons_action = self.action_space.n * [0]
        buttons_action[action] = 1

        # Make action in the game
        reward = self.game.make_action(buttons_action)

        # Get the state after action
        state = self.game.get_state()

        # If episode is finished, return info
        done = self.game.is_episode_finished()

        # If done, reset screen buffer
        if not done:
            observation = state.screen_buffer
        else:
            observation = np.zeros(self.observation_space.shape)

        return observation, reward, done, {}

    def close(self):
        self.game.close()
