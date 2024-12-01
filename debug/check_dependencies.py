import gymnasium as gym
import sklearn
import stable_baselines3
import torch

print("Scikit learn:", sklearn.__version__)

print("Gym version:", gym.__version__)
print("Stable-Baselines3 version:", stable_baselines3.__version__)


def test_cuda():
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available!")

        # Get the number of GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")

        # Get the name of each GPU
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        # Run a simple operation on the GPU
        device = torch.device("cuda")  # Use the first GPU by default
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        y = torch.tensor([4.0, 5.0, 6.0], device=device)
        z = x + y

        print(f"Computation result on the GPU: {z}")
    else:
        print("CUDA is not available.")


if __name__ == "__main__":
    test_cuda()
