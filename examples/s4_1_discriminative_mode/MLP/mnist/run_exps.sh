#!/bin/bash

# Create a new tmux session
tmux new-session -d -s mnist

# Run commands within the tmux session
tmux send-keys -t mnist 'conda activate name_of_your_PCX_environment' C-m

tmux send-keys -t mnist 'python BP_CE.py L_BP_CE_mnist.yaml' C-m
tmux send-keys -t mnist 'python BP_CE.py L_BP_CE_mnist.yaml' C-m
tmux send-keys -t mnist 'python BP_SE.py L_BP_SE_mnist.yaml' C-m
tmux send-keys -t mnist 'python PC_SE.py L_PC_SE_mnist.yaml' C-m
tmux send-keys -t mnist 'python PC_SE.py L_PC_NN_mnist.yaml' C-m
tmux send-keys -t mnist 'python PC_SE.py L_PC_PN_mnist.yaml' C-m
tmux send-keys -t mnist 'python PC_CE.py L_PC_CE_mnist.yaml' C-m
tmux send-keys -t mnist 'python iPC_SE.py L_iPC_SE_mnist.yaml' C-m