# rl_doom
Final project for CS-GY 6923 Fall 2024. 

# Installing Dependencies
Windows users should clone the repository in WSL before following the directions below.

Dependencies are managed by the dev containers extension which can be downloaded from the [remote development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) vscode extension pack and requires [Docker Engine](https://docs.docker.com/engine/install/) to be running. After cloning the repo in vscode Ctrl+Shift+P to open and command palette and type `Reopen in Container`. Upon successfully starting the environment, it will run a short test script to verify GPU access via CUDA and that vizdoom was installed properly.

Alternatively, install dependencies directly from the requirements.txt.
```
pip install -r .devcontainer/requirements.txt
```
