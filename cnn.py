import json
import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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
        self.flattened_size = 2592  # Matches the shape in your debug output
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(257, 3)  # Output 3 logits for classification

    def forward(self, state, action):
        cnn_out = F.relu(self.conv1(state))
        cnn_out = F.relu(self.conv2(cnn_out))
        cnn_out = cnn_out.view(state.size(0), -1)  # Flatten CNN output
        cnn_out = F.relu(self.fc1(cnn_out))
        action = action.view(
            action.size(0), -1
        )  # Ensure action has shape (batch_size, 1)
        combined_input = torch.cat((cnn_out, action), dim=1)
        logits = self.fc2(combined_input)  # Predict logits for three classes
        return logits


def collect_data(model, env, n_episodes=100, down_sample=False):
    states, actions, rewards = [], [], []

    for ep in tqdm(range(n_episodes)):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, done, _ = env.step(action)
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            obs = next_obs
    if down_sample:
        states, actions, rewards = downsample_rewards(states, actions, rewards)
    return np.array(states), np.array(actions), np.array(rewards)


def downsample_rewards(states, actions, rewards):
    # Ensure rewards is a NumPy array
    rewards = np.array(rewards)

    # Split data into three categories
    zero_indices = np.atleast_1d(rewards == 0).nonzero()[0]
    positive_indices = np.atleast_1d(rewards > 0).nonzero()[0]
    negative_indices = np.atleast_1d(rewards < 0).nonzero()[0]

    # Count instances
    n_positive = len(positive_indices)
    n_negative = len(negative_indices)
    n_zero = len(zero_indices)

    # Find the smallest class count excluding zeros
    min_nonzero_count = min(n_positive, n_negative)

    # Downsample zeros to match the smallest non-zero class
    zero_indices_downsampled = resample(
        zero_indices, replace=False, n_samples=min_nonzero_count, random_state=42
    )

    # Downsample the larger of positive or negative to match the smaller one
    if n_positive > n_negative:
        positive_indices_downsampled = resample(
            positive_indices,
            replace=False,
            n_samples=min_nonzero_count,
            random_state=42,
        )
        negative_indices_downsampled = negative_indices
    else:
        negative_indices_downsampled = resample(
            negative_indices,
            replace=False,
            n_samples=min_nonzero_count,
            random_state=42,
        )
        positive_indices_downsampled = positive_indices

    # Combine all selected indices
    selected_indices = np.concatenate(
        [
            zero_indices_downsampled,
            positive_indices_downsampled,
            negative_indices_downsampled,
        ]
    )

    selected_indices = np.array(selected_indices, dtype=int)

    # Shuffle the indices
    np.random.shuffle(selected_indices)

    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)

    return (
        states[selected_indices],
        actions[selected_indices],
        rewards[selected_indices],
    )


class StateActionRewardDataset(Dataset):
    def __init__(self, states, actions, rewards):
        self.states = states
        self.actions = actions
        self.labels = torch.where(
            rewards > 1e-6, 1, torch.where(rewards < -1e-6, 2, 0)
        )  # Classify rewards into 0, 1, 2

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        label = self.labels[idx]  # Get scalar value, not a tensor

        # Ensure state has shape (C, H, W)
        if state.ndim == 4:
            state = state.squeeze(0).permute(2, 0, 1)  # (1, H, W, C) -> (C, H, W)
        elif state.ndim == 3:
            state = state.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        else:
            raise ValueError(f"Unexpected state shape: {state.shape}")

        return state, action, label.item()  # Use `.item()` to ensure label is a scalar


def load_hyperparameters(filename="best_hyperparameters.json"):
    # Load the best hyperparameters from a JSON file
    with open(filename, "r") as f:
        params = json.load(f)
    return params


def calculate_accuracy(predicted, actual):
    """
    Calculate accuracy for classification tasks.

    Args:
        predicted (torch.Tensor): Logits with shape (batch_size, num_classes).
        actual (torch.Tensor): Ground truth class indices with shape (batch_size).

    Returns:
        float: Classification accuracy.
    """
    assert predicted.dim() == 2, f"Expected 2D logits, got shape {predicted.shape}"
    assert actual.dim() == 1, f"Expected 1D labels, got shape {actual.shape}"
    assert predicted.size(0) == actual.size(
        0
    ), f"Batch size mismatch: logits {predicted.size(0)}, labels {actual.size(0)}"

    predicted_classes = torch.argmax(predicted, dim=1)  # Class with highest logit
    correct = (predicted_classes == actual).float()  # Element-wise correctness
    accuracy = correct.mean().item()  # Average correctness
    return accuracy


if __name__ == "__main__":
    # Load and preprocess data
    hyperparameters = load_hyperparameters()
    algorithm = hyperparameters.pop(
        "algorithm", "PPO"
    )  # Default to PPO if not specified
    n_stack = hyperparameters.pop("n_stack", 4)

    # Initialize environment and trained model
    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    # Apply VecFrameStack for frame stacking
    env = VecFrameStack(env, n_stack=n_stack)
    eval_env = VecFrameStack(eval_env, n_stack=n_stack)

    # Path to the saved model
    model_path = "./logs/best_model.zip"
    model = PPO.load(model_path)

    generate_npz_file = os.path.isfile("cnn_pong_data.npz")

    if not generate_npz_file:
        states, actions, rewards = collect_data(model, env, 10, True)
        np.savez("cnn_pong_data.npz", states=states, actions=actions, rewards=rewards)
        print("Data collection complete. Saved to 'cnn_pong_data.npz'.")

    print("Loading data...")
    data = np.load("cnn_pong_data.npz", allow_pickle=True)
    states = data["states"]
    actions = data["actions"]
    rewards = data["rewards"]

    states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(actions, dtype=torch.float32).to(device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)

    train_states, val_states, train_actions, val_actions, train_rewards, val_rewards = (
        train_test_split(
            states_tensor,
            actions_tensor,
            rewards_tensor,
            test_size=0.2,
            random_state=42,
        )
    )

    train_dataset = StateActionRewardDataset(train_states, train_actions, train_rewards)
    val_dataset = StateActionRewardDataset(val_states, val_actions, val_rewards)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    action_dim = env.action_space.n
    state_shape = env.observation_space.shape
    print(f"Action dim: {action_dim}, State shape: {state_shape}")

    reward_model = RewardCNN(state_shape, action_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(reward_model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        reward_model.train()
        train_loss = 0.0
        train_accuracy = 0.0

        for state, action, label in train_loader:
            state = state.to(device, dtype=torch.float32)
            action = action.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.long)

            optimizer.zero_grad()
            logits = reward_model(state, action)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_accuracy += calculate_accuracy(logits, label)

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy / len(train_loader))

        reward_model.eval()
        val_loss = 0.0
        val_accuracy = 0.0

        with torch.no_grad():
            for state, action, label in val_loader:
                state = state.to(device, dtype=torch.float32)
                action = action.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.long)

                logits = reward_model(state, action)
                loss = criterion(logits, label)

                val_loss += loss.item()
                val_accuracy += calculate_accuracy(logits, label)

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy / len(val_loader))

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, "
            f"Validation Loss: {val_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, "
            f"Validation Accuracy: {val_accuracies[-1]:.4f}"
        )

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig("loss.png")

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Training Accuracy")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig("accuracy.png")

    torch.save(reward_model.state_dict(), "reward_model.pth")
    print("Training complete and model saved.")
