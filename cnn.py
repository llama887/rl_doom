import json
import os.path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from torch.utils.data import DataLoader, Dataset

from train import make_env

# Check for GPU availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class RewardCNN(nn.Module):
    def __init__(self, state_shape, action_dim):
        super(RewardCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        # Calculate the flattened size after conv layers
        self.flattened_size = (
            2592  # This matches the shape printed in your debug output
        )
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(257, 1)  # Change the input size to 257 if needed

    def forward(self, state, action):
        cnn_out = self.conv1(state)
        cnn_out = F.relu(cnn_out)
        cnn_out = self.conv2(cnn_out)
        cnn_out = F.relu(cnn_out)
        cnn_out = cnn_out.view(state.size(0), -1)  # Flatten CNN output
        # print(f"Shape after flattening CNN output: {cnn_out.shape}")

        cnn_out = self.fc1(cnn_out)
        cnn_out = F.relu(cnn_out)

        # Ensure action has the correct shape (batch_size, action_dim)
        action = action.view(
            action.size(0), -1
        )  # Ensure action has shape (batch_size, 1)

        # Concatenate CNN output and action
        combined_input = torch.cat((cnn_out, action), dim=1)

        # Final fully connected layer to produce the predicted reward
        predicted_reward = self.fc2(combined_input)
        return predicted_reward


def collect_data(model, env, n_episodes=100):
    states, actions, rewards = [], [], []

    for ep in range(n_episodes):
        print(f"Starting EP: {ep}/{n_episodes}")
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, done, _ = env.step(action)
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            obs = next_obs

    return np.array(states), np.array(actions), np.array(rewards)


def Collect_Reward_Pairs(env, model, sample_steps=1000):
    # Collect data
    state_action_reward_data = []
    obs = env.reset()

    for ss in range(sample_steps):  # Number of steps to sample
        if ss % 50:
            print(f"Sampling {ss}/sample_steps")
        action, _ = model.predict(obs)  # Agent's action
        next_obs, reward, done, info = env.step(action)

        # Convert state to a PyTorch tensor and ensure it has the right shape
        state = torch.tensor(obs, dtype=torch.float32)

        # Check if the state has 5 dimensions and remove the extra one if necessary
        if len(state.shape) == 5:
            state = state.squeeze(0)  # Remove the first dimension if it is of size 1

        # Ensure the state has 4 dimensions: (batch_size, channels, height, width)
        if len(state.shape) == 4:
            state = state.permute(
                0, 3, 1, 2
            )  # Rearrange to (batch_size, channels, height, width)
        else:
            raise ValueError(
                f"Unexpected state shape: {state.shape}. Expected 4 dimensions."
            )

        # Save the state, action, and reward
        state_action_reward_data.append((state, action, reward))

        obs = next_obs
        if done:
            obs = env.reset()

    # Convert to a structured numpy array for saving
    state_action_reward_data_array = []
    for state, action, reward in state_action_reward_data:
        state_action_reward_data_array.append(
            (state.squeeze(), np.array(action), np.array(reward))
        )

    dtype = [("state", "O"), ("action", "O"), ("reward", "O")]
    structured_array = np.array(state_action_reward_data_array, dtype=dtype)

    # Save the collected data
    np.save("state_action_reward_data.npy", structured_array)

    print("Data collection complete.")
    # print("Shape of state in each tuple:")
    # print([x[0].shape for x in state_action_reward_data])

    # Check shapes of state components
    # print("Check shapes of states in the dataset:")
    # for idx, (state, action, reward) in enumerate(state_action_reward_data):
    #     print(
    #         f"Index {idx}: State shape = {state.shape}, Action = {action}, Reward = {reward}"
    #     )


class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, reward_model):
        super().__init__(env)
        self.reward_model = reward_model

    def reward(self, reward):
        state = (
            torch.tensor(self.env.state, dtype=torch.float32).unsqueeze(0).to(device)
        )
        action = (
            torch.tensor(self.env.last_action, dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )
        predicted_reward = self.reward_model(state, action).item()
        return predicted_reward


# Wrap the environment


class StateActionRewardDataset(Dataset):
    def __init__(self, states, actions, rewards):
        """
        Args:
            states (numpy.ndarray): Array of states with shape (num_samples, height, width, channels).
            actions (numpy.ndarray): Array of actions with shape (num_samples,).
            rewards (numpy.ndarray): Array of rewards with shape (num_samples,).
        """
        self.states = states
        self.actions = actions
        self.rewards = rewards

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]

        # Check the shape of state before permuting
        # print(f"Original state shape: {state.shape}")

        # Ensure state has 4 dimensions (e.g., (H, W, C)) before permuting
        if state.ndim == 4:
            state = (
                state.clone().detach().float().squeeze(0).permute(2, 0, 1)
            )  # (1, H, W, C) -> (C, H, W)
        elif state.ndim == 3:
            state = (
                state.clone().detach().float().permute(2, 0, 1)
            )  # (H, W, C) -> (C, H, W)
        else:
            raise ValueError(
                f"Unexpected state shape: {state.shape}. Expected 3 or 4 dimensions."
            )

        action = action.clone().detach().float()  # One-hot action
        reward = reward.clone().detach().float()

        return state, action, reward


class CNNModel(nn.Module):
    def __init__(self, state_shape, action_dim):
        super(CNNModel, self).__init__()
        in_channels = state_shape[
            0
        ]  # The number of channels in the input state (e.g., 6 for RGB)

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=3,
            ),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Assuming the input size is (6, 84, 84), calculate the output shape after CNN layers
        # Adjust based on your input dimensions
        self.fc_input_dim = 64 * (state_shape[1] // 8) * (state_shape[2] // 8)
        self.fc = nn.Linear(self.fc_input_dim, action_dim)

    def forward(self, x):
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.view(x.size(0), -1)  # Flatten the output
        output = self.fc(cnn_out)  # Fully connected layer for action prediction
        return output


def load_hyperparameters(filename="best_hyperparameters.json"):
    # Load the best hyperparameters from a JSON file
    with open(filename, "r") as f:
        params = json.load(f)
    return params


def calculate_accuracy(predicted, actual):
    # Convert to binary accuracy for simplicity (e.g., classify positive/negative rewards)
    predicted_classes = (predicted > 0.5).float()
    actual_classes = (actual > 0.5).float()
    return (predicted_classes == actual_classes).float().mean().item()


if __name__ == "__main__":
    # Load and preprocess data
    hyperparameters = load_hyperparameters()
    algorithm = hyperparameters.pop(
        "algorithm", "PPO"
    )  # Default to PPO if not specified
    n_stack = hyperparameters.pop("n_stack", 4)

    # Initialize environment and trained model
    # Choose the environment creation function
    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    # Apply VecFrameStack for frame stacking
    env = VecFrameStack(env, n_stack=n_stack)
    eval_env = VecFrameStack(eval_env, n_stack=n_stack)
    # Path to the saved model
    model_path = "./logs/best_model.zip"

    # Load the trained PPO model
    model = PPO.load(model_path)

    generate_npz_file = os.path.isfile("cnn_pong_data.npz")

    if not generate_npz_file:
        # Collect data from the model and environment
        states, actions, rewards = collect_data(model, env, 50)

        # Save data for supervised training in a .npz file
        np.savez("cnn_pong_data.npz", states=states, actions=actions, rewards=rewards)

        print("Data collection complete. Saved to 'cnn_pong_data.npz'.")
        # Collect_Reward_Pairs(env, model)

    data = np.load("cnn_pong_data.npz", allow_pickle=True)

    # Assuming each entry in `data` is a tuple (state, action, reward)
    # Extract states, actions, and rewards separately
    states = data["states"]
    actions = data["actions"]
    rewards = data["rewards"]

    # print("States shape:", states.shape)
    # print("Actions shape:", actions.shape)
    # print("Rewards shape:", rewards.shape)

    states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(actions, dtype=torch.float32).to(device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)

    # print("States tensor shape:", states_tensor.shape)
    # print("Actions tensor shape:", actions_tensor.shape)
    # print("Rewards tensor shape:", rewards_tensor.shape)

    # Split the dataset into training and validation sets
    train_states, val_states, train_actions, val_actions, train_rewards, val_rewards = (
        train_test_split(
            states_tensor,
            actions_tensor,
            rewards_tensor,
            test_size=0.2,
            random_state=42,
        )
    )

    # Create DataLoaders for training and validation sets
    train_dataset = StateActionRewardDataset(train_states, train_actions, train_rewards)
    val_dataset = StateActionRewardDataset(val_states, val_actions, val_rewards)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize variables for tracking losses and accuracy
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Iterate through the DataLoader
    for batch in train_loader:
        state, action, reward = batch
        print(state.shape, action.shape, reward.shape)
        break  # Print only the first batch

    # Initialize the model, loss, and optimizer
    action_dim = env.action_space.n
    state_shape = env.observation_space.shape
    print(f"action dim : {action_dim} , and state_shape : {state_shape}")

    # Initialize the model and move it to the device
    reward_model = RewardCNN(state_shape, action_dim).to(device)
    num_epochs = 10
    reward_model.train()

    # cnn_model = CNNModel(state_shape, action_dim)
    criterion = nn.MSELoss()  # Regression loss for reward
    optimizer = optim.Adam(reward_model.parameters(), lr=0.001)

    # Training loop with validation
    num_epochs = 10
    for epoch in range(num_epochs):
        # Training phase
        reward_model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        for state, action, reward in train_loader:
            state = state.to(device, dtype=torch.float32)
            action = action.to(device, dtype=torch.float32)
            reward = reward.to(device, dtype=torch.float32).view(-1, 1)

            optimizer.zero_grad()
            predicted_reward = reward_model(state, action)
            loss = criterion(predicted_reward, reward)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_accuracy += calculate_accuracy(predicted_reward, reward)

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy / len(train_loader))

        # Validation phase
        reward_model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for state, action, reward in val_loader:
                state = state.to(device, dtype=torch.float32)
                action = action.to(device, dtype=torch.float32)
                reward = reward.to(device, dtype=torch.float32).view(-1, 1)

                predicted_reward = reward_model(state, action)
                loss = criterion(predicted_reward, reward)

                val_loss += loss.item()
                val_accuracy += calculate_accuracy(predicted_reward, reward)

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy / len(val_loader))

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, "
            f"Validation Loss: {val_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, "
            f"Validation Accuracy: {val_accuracies[-1]:.4f}"
        )

    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig("loss.png")

    # Plot training and validation accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Training Accuracy")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig("accuracy.png")

    # Save the trained model
    torch.save(reward_model.state_dict(), "reward_model.pth")
    print("Training complete and model saved.")
