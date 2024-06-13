#!/bin/bash

#!/bin/bash

# Create a new tmux session
tmux new-session -d -s VGG7_cifar100

# Run commands within the tmux session
tmux send-keys -t VGG7_cifar100 'conda activate name_of_your_PCX_environment' C-m

tmux send-keys -t VGG7_cifar100 'python BP_CE.py VGG_BP_CE_cifar100.yaml' C-m
tmux send-keys -t VGG7_cifar100 'python BP_SE.py VGG_BP_SE_cifar100.yaml' C-m
tmux send-keys -t VGG7_cifar100 'python PC_SE.py VGG_PCN_NN_cifar100.yaml' C-m
tmux send-keys -t VGG7_cifar100 'python PC_SE.py VGG_PCN_PN_cifar100.yaml' C-m
tmux send-keys -t VGG7_cifar100 'python PC_SE.py VGG_PCN_SE_cifar100.yaml' C-m
tmux send-keys -t VGG7_cifar100 'python PC_CE.py VGG_PCN_CE_cifar100.yaml' C-m
tmux send-keys -t VGG7_cifar100 'python iPC_SE.py VGG_iPC_cifar100.yaml' C-m