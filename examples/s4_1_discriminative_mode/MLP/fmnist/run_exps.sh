#!/bin/bash

# Create a new tmux session
tmux new-session -d -s fmnist

# Run commands within the tmux session
tmux send-keys -t fmnist 'conda activate name_of_your_PCX_environment' C-m

tmux send-keys -t fmnist 'python BP_CE.py L_BP_CE_fmnist.yaml' C-m
tmux send-keys -t fmnist 'python BP_SE.py L_BP_SE_fmnist.yaml' C-m
tmux send-keys -t fmnist 'python PC_SE.py L_PC_SE_fmnist.yaml' C-m
tmux send-keys -t fmnist 'python PC_SE.py L_PC_NN_fmnist.yaml' C-m
tmux send-keys -t fmnist 'python PC_SE.py L_PC_PN_fmnist.yaml' C-m
tmux send-keys -t fmnist 'python PC_CE.py L_PC_CE_fmnist.yaml' C-m
tmux send-keys -t fmnist 'python iPC_SE.py L_iPC_SE_fmnist.yaml' C-m