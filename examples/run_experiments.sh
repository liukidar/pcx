#!/bin/bash

# Define the ranges for the parameters
epochs_list=(100 200 400)
batch_sizes=(128 256 512)
learning_rates=(0.01 0.02 0.05)

# Loop over each combination of parameters
for epochs in "${epochs_list[@]}"; do
  for learning_rate in "${learning_rates[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
      # Run the Python script with the current set of parameters
      #python train_double_descent_pc.py --config ./config.json --noise_level 0.0 --batch_size $batch_size --learning_rate $learning_rate --epochs $epochs
      python train_double_descent_bp.py --config ./config.json --noise_level 0.0 --batch_size $batch_size --learning_rate $learning_rate --epochs $epochs
    done
  done
done