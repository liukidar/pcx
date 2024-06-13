#!/bin/bash

# Create a new tmux session
tmux new-session -d -s VGG5_cifar10

# Run commands within the tmux session
tmux send-keys -t VGG5_cifar10 'conda activate name_of_your_PCX_environment' C-m

tmux send-keys -t VGG5_cifar10 'python BP_CE.py VGG5_BP_CE.yaml' C-m
tmux send-keys -t VGG5_cifar10 'python BP_SE.py VGG5_BP_SE.yaml' C-m
tmux send-keys -t VGG5_cifar10 'python PC_SE.py VGG5_PCN_NN.yaml' C-m
tmux send-keys -t VGG5_cifar10 'python PC_SE.py VGG5_PCN_PN.yaml' C-m
tmux send-keys -t VGG5_cifar10 'python PC_SE.py VGG5_PCN_SE.yaml' C-m
tmux send-keys -t VGG5_cifar10 'python PC_CE.py VGG5_PCN_CE.yaml' C-m
tmux send-keys -t VGG5_cifar10 'python iPC_SE.py VGG5_iPC.yaml' C-m
