#!/bin/bash
if command -v nvidia-smi &> /dev/null; then
    GPU_FLAG="--gpus all"
else
    GPU_FLAG=""
fi
echo "GPU_FLAG set to: $GPU_FLAG"

# Function to check if SSH to GitHub works
check_ssh() {
    ssh -T git@github.com &>/dev/null
    return $?
}

# Check SSH authentication with GitHub
if check_ssh; then
    echo "SSH authentication with GitHub is already working."
else
    echo "SSH authentication failed. Attempting to authenticate with GitHub CLI."
    gh auth login
fi

# Check GitHub authentication status
gh auth status

# Set the remote URL for Git to the specified repository
git remote set-url origin https://github.com/llama887/rl_doom.git


# Run the dependency check script
python /workspaces/rl_doom/debug/check_dependencies.py
python /workspaces/rl_doom/debug/atari_test.py
