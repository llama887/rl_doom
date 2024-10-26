#!/bin/bash

# Make this script executable
chmod +x aider.sh

# Authenticate with GitHub
gh auth login

# Check GitHub authentication status
gh auth status

# Set the remote URL for Git to the specified repository
git remote set-url origin https://github.com/llama887/rl_doom.git

# Run the dependency check script
python /workspaces/rl_doom/debug/check_dependencies.py
python /workspaces/rl_doom/debug/atari_test.py
