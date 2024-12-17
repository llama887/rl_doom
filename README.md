# rl_pong
Final project for CS-GY 6923 Fall 2024. 

# Installing Dependencies
Windows users should clone the repository in WSL before following the directions below.

Dependencies are managed by the dev containers extension which can be downloaded from the [remote development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) vscode extension pack and requires [Docker Engine](https://docs.docker.com/engine/install/) to be running. After cloning the repo in vscode Ctrl+Shift+P to open and command palette and type `Reopen in Container`. Upon successfully starting the environment, it will run a short test script to verify GPU access via CUDA and that vizdoom was installed properly.

Alternatively, install dependencies directly from the requirements.txt.
```
pip install -r .devcontainer/requirements.txt
```
# Files


**REPORT IS IN final_writeup.pdf**


**cnn.py**: This file holds the code to train our CNN, tune its hyperparameters, and create rollouts. It expects to be passed cnn_hyperparameters.json or else it will start to generate new hyperparameters. If a cnn_pong_data.npz file is not avaliable, it will generate a new rollout using the agent weights found in logs/best_model.zip and train the CNN on the new data. Otherwise, it will train on the cnn_pong_data.npz data.

**evaluate_ensemble.py**: Used for evaluating the trained ensemble. Takes all of the model weights in the reward_model directory and treats each of them as a member of the ensemble. Will output a total accuracy and a confusion matrix.


**hyperparameter_optimization.py**: This is used for fine tuning the reinforcement learning agent hyperparameters.


**output.py**: Used to visually see the results of a trained agent given the agent weights.


**train_ensemble.sh**: Expects a flag containing the number of members expected in the ensemble and then it trains a ensemble via repeatedly deleteing the data file and running cnn.py and storing its weights and figures in seperate directories. By deleting the data file, we ensure each member of the ensemble has a unique subset of the data.


**train.py**: Trains an RL agent. Training progress can be viewed in tensorboard.
