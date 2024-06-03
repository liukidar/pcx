#!/bin/bash

# usage:
# ./run_double_descent.sh -n 0.2 -s train_double_descent_pc.py

# Default values for parameters
#epochs_list=(500 1000)
epochs_list=(1000) # temp change
batch_sizes=(128)
learning_rates=(0.01 0.02 0.05)
noise_level=0.0
script_name="train_double_descent_bp.py"

# Parse command-line arguments
while getopts ":n:s:" opt; do
  case $opt in
    n) noise_level="$OPTARG"
    ;;
    s) script_name="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
        exit 1
    ;;
  esac
done

# Loop over each combination of parameters
for epochs in "${epochs_list[@]}"; do
  for learning_rate in "${learning_rates[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
      # Run the Python script with the current set of parameters
      python $script_name --config ./config.json --noise_level $noise_level --batch_size $batch_size --learning_rate $learning_rate --epochs $epochs
    done
  done
done
