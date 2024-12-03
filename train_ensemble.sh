#!/bin/bash

if ["$#" -ne 1]; then
    echo "Usage: $0 <number of ensembles>"
    exit 1
fi

rm -r reward_model
mkdir reward_model
rm -r figures
mkdir figures

for i in $(seq 1 $1); do
    rm cnn_pong_data.npz
    python cnn.py --hyperparams cnn_hyperparameters.json || python cnn.py --hyperparams cnn_hyperparameters.json
    mv reward_model.pth reward_model/reward_model_$i.pth
    mv loss.png figures/loss$i.png
    mv accuracy.png figures/accuracy$i.png
done

python evaluate_ensemble.py
