import vizdoom as vzd
import time

# Create a DoomGame instance
game = vzd.DoomGame()

# Load a basic configuration file (you can find more configs in the 'scenarios' folder of vizdoom)
game.load_config(vzd.scenarios_path + "/basic.cfg")

# Set the scenario (here we use the basic.wad, which comes with vizdoom)
game.set_doom_scenario_path(vzd.scenarios_path + "/basic.wad")

# Set the map to play
game.set_doom_map("map01")

# Initialize the game in headless for testing
# game.set_window_visible(False)  # Disable the window display
# game.set_mode(vzd.Mode.PLAYER)
game.init()

# Run a few episodes
episodes = 1
for i in range(episodes):
    print(f"Episode #{i + 1}")

    # Start a new episode
    game.new_episode()

    while not game.is_episode_finished():
        # Get the game state
        state = game.get_state()

        # Get the screen buffer (game frame)
        screen = state.screen_buffer

        # Perform a random action
        action = game.get_available_buttons_size() * [0]
        action[0] = 1  # Move forward

        # Make the action
        reward = game.make_action(action)

        # Print some information
        print(f"State # {state.number}")
        print(f"Action reward: {reward}")
        print("===================")

        # Sleep for a while to make it human-readable (for test purposes)
        time.sleep(0.02)

    print(f"Episode finished! Total reward: {game.get_total_reward()}")
    print("=============================")

# Close the game
game.close()
