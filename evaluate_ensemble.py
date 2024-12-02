import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from torch.utils.data import DataLoader

from cnn import RewardCNN, StateActionRewardDataset, collect_data
from train import make_env

# Check for GPU availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_ensemble_models(model_paths, state_shape, action_dim):
    models = []
    for path in model_paths:
        model = RewardCNN(state_shape, action_dim).to(device)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        models.append(model)
    return models


def calculate_accuracy(predictions, labels):
    """
    Calculate accuracy for classification tasks based on predictions and true labels.

    Args:
        predictions (list[torch.Tensor]): List of predicted classes from the ensemble, or a single model.
        labels (list[torch.Tensor]): List of true class indices.

    Returns:
        float: Classification accuracy.
    """
    # Flatten predictions and labels into a single tensor
    if isinstance(predictions, list):
        predictions = torch.cat(predictions, dim=0)  # Combine batch-wise predictions
    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)  # Combine batch-wise labels

    assert (
        predictions.dim() == 1
    ), f"Expected 1D predictions, got shape {predictions.shape}"
    assert labels.dim() == 1, f"Expected 1D labels, got shape {labels.shape}"
    assert predictions.size(0) == labels.size(
        0
    ), f"Size mismatch: predictions {predictions.size(0)}, labels {labels.size(0)}"

    # Calculate accuracy
    correct = (predictions == labels).float()  # Element-wise correctness
    accuracy = correct.mean().item()  # Average correctness
    return accuracy


def roll_out_and_evaluate(env, model, ensemble, n_episodes=10):
    print("Rolling out trajectories...")
    states, actions, true_rewards = collect_data(model, env, n_episodes)

    print("Evaluating...")
    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
    true_rewards = torch.tensor(np.array(true_rewards), dtype=torch.float32).to(device)
    test_dataset = StateActionRewardDataset(states, actions, true_rewards)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, drop_last=True)

    # Initialize lists for true labels and predictions
    all_true_labels = []
    all_ensemble_predictions = []

    for state, action, label in test_loader:
        state = state.to(device, dtype=torch.float32)
        action = action.to(device, dtype=torch.float32)
        label = label.to(device, dtype=torch.long)

        # Store true labels for this batch
        all_true_labels.append(label)

        # Evaluate ensemble predictions for this batch
        batch_ensemble_predictions = []
        for model in ensemble:
            with torch.no_grad():
                logits = model(state, action)
                predicted_classes = torch.argmax(logits, dim=1)
                batch_ensemble_predictions.append(predicted_classes)

        # Aggregate predictions from ensemble (majority vote)
        batch_ensemble_predictions = torch.stack(batch_ensemble_predictions, dim=1)
        batch_final_predictions, _ = torch.mode(batch_ensemble_predictions, dim=1)

        # Store final predictions for this batch
        all_ensemble_predictions.append(batch_final_predictions)

    # Concatenate all batches into a single tensor
    all_true_labels = torch.cat(all_true_labels, dim=0)
    all_ensemble_predictions = torch.cat(all_ensemble_predictions, dim=0)

    # Calculate ensemble accuracy
    ensemble_accuracy = calculate_accuracy(all_ensemble_predictions, all_true_labels)

    print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")

    return (
        all_true_labels.cpu().numpy(),
        all_ensemble_predictions.cpu().numpy(),
        ensemble_accuracy,
    )


def plot_confusion_matrix(true_labels, predicted_labels, title):
    cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Zero", "Positive", "Negative"]
    )
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.savefig("figures/confusion_matrix.png")
    plt.close()


def plot_classification_report(true_labels, predicted_labels, title):
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=["Zero", "Positive", "Negative"],
        output_dict=True,
    )
    metrics = ["precision", "recall", "f1-score"]
    classes = ["Zero", "Positive", "Negative"]

    for metric in metrics:
        values = [report[c][metric] for c in classes]
        plt.bar(classes, values)
        plt.title(f"{metric.capitalize()} - {title}")
        plt.ylabel(metric.capitalize())
        plt.xlabel("Class")
        plt.ylim(0, 1)
        plt.savefig(f"figures/{metric}.png")
        plt.close()


if __name__ == "__main__":
    # Paths to saved ensemble models
    folder_path = "reward_model"
    model_paths = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, file))
    ]

    # Load and preprocess data
    with open("agent_hyperparameters.json", "r") as f:
        hyperparameters = json.load(f)

    algorithm = hyperparameters.pop(
        "algorithm", "PPO"
    )  # Default to PPO if not specified
    n_stack = hyperparameters.pop("n_stack", 4)

    # Initialize environment and trained model
    env = DummyVecEnv([make_env])

    # Apply VecFrameStack for frame stacking
    env = VecFrameStack(env, n_stack=n_stack)

    # Environment info
    action_dim = env.action_space.n
    state_shape = env.observation_space.shape
    print(f"Action dim: {action_dim}, State shape: {state_shape}")

    # Load ensemble models
    ensemble = load_ensemble_models(model_paths, state_shape, action_dim)

    model_path = "./logs/best_model.zip"
    model = PPO.load(model_path)

    # Roll out and evaluate the ensemble
    (
        true_labels,
        ensemble_predictions,
        ensemble_accuracy,
    ) = roll_out_and_evaluate(env, model, ensemble, n_episodes=10)

    # Plot confusion matrices
    print("Plotting confusion matrices...")
    plot_confusion_matrix(
        true_labels, ensemble_predictions, title="Ensemble Confusion Matrix"
    )

    # Plot classification reports for the ensemble
    print("Plotting classification report for the ensemble...")
    plot_classification_report(
        true_labels, ensemble_predictions, title="Ensemble Classification Report"
    )
