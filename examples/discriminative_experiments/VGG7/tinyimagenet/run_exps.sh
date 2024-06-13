#!/bin/bash

# Create a new tmux session
tmux new-session -d -s VGG7_tinyimagenet

# Run commands within the tmux session
tmux send-keys -t VGG7_tinyimagenet 'conda activate name_of_your_PCX_environment' C-m

tmux send-keys -t VGG7_tinyimagenet 'python BP_CE.py VGG_BP_CE_tinyimagenet.yaml' C-m
tmux send-keys -t VGG7_tinyimagenet 'python BP_SE.py VGG_BP_SE_tinyimagenet.yaml' C-m
tmux send-keys -t VGG7_tinyimagenet 'python PC_SE.py VGG_PCN_NN_tinyimagenet.yaml' C-m
tmux send-keys -t VGG7_tinyimagenet 'python PC_SE.py VGG_PCN_PN_tinyimagenet.yaml' C-m
tmux send-keys -t VGG7_tinyimagenet 'python PC_SE.py VGG_PCN_SE_tinyimagenet.yaml' C-m
tmux send-keys -t VGG7_tinyimagenet 'python PC_CE.py VGG_PCN_CE_tinyimagenet.yaml' C-m
tmux send-keys -t VGG7_tinyimagenet 'python iPC_SE.py VGG_iPC_tinyimagenet.yaml' C-m
