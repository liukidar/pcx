#!/bin/bash

# Create a new tmux session
tmux new-session -d -s VGG5_tinyimagenet

# Run commands within the tmux session
tmux send-keys -t VGG5_tinyimagenet 'conda activate name_of_your_PCX_environment' C-m

tmux send-keys -t VGG5_tinyimagenet 'python BP_CE.py VGG5_BP_CE_tinyimagenet.yaml' C-m
tmux send-keys -t VGG5_tinyimagenet 'python BP_SE.py VGG5_BP_SE_tinyimagenet.yaml' C-m
tmux send-keys -t VGG5_tinyimagenet 'python PC_SE.py VGG5_PCN_NN_tinyimagenet.yaml' C-m
tmux send-keys -t VGG5_tinyimagenet 'python PC_SE.py VGG5_PCN_PN_tinyimagenet.yaml' C-m
tmux send-keys -t VGG5_tinyimagenet 'python PC_SE.py VGG5_PCN_SE_tinyimagenet.yaml' C-m
tmux send-keys -t VGG5_tinyimagenet 'python PC_CE.py VGG5_PCN_CE_tinyimagenet.yaml' C-m
tmux send-keys -t VGG5_tinyimagenet 'python iPC_SE.py VGG5_iPC_tinyimagenet.yaml' C-m
