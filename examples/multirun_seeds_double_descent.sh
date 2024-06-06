#!/bin/bash

# example usage: ./multirun_seeds_double_descent.sh -n 0.2 -s train_double_descent_pc_5runs.py

# Default values for parameters
epochs_list=(200)
batch_sizes=(256)
w_learning_rates=(0.05)
h_learning_rates=(0.01)
h_momentum_list=(0.7)
seeds=(1 2 3 4 5)

noise_level=0.0
script_name="train_double_descent_pc_5runs.py"

# Parse command-line arguments
while getopts ":n:s:" opt; do
  case $opt in
    n) noise_level="$OPTARG"
    ;;
    s) script_name="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
        echo "Usage: $0 [-n noise_level] [-s script_name]"
        exit 1
    ;;
  esac
done

# Print the usage example
if [ "$OPTIND" -eq 1 ]; then
  echo "Usage example:"
  echo "$0 -n 0.2 -s train_double_descent_pc_5runs.py"
fi

# Loop over each combination of parameters
for epochs in "${epochs_list[@]}"; do
  for batch_size in "${batch_sizes[@]}"; do
    for w_learning_rate in "${w_learning_rates[@]}"; do
      for h_learning_rate in "${h_learning_rates[@]}"; do
        for h_momentum in "${h_momentum_list[@]}"; do
          for seed in "${seeds[@]}"; do
            # Run the Python script with the current set of parameters
            python $script_name --config ./config.json --noise_level $noise_level --batch_size $batch_size --w_learning_rate $w_learning_rate --h_learning_rate $h_learning_rate --h_momentum $h_momentum --epochs $epochs --seed $seed
          done
        done
      done
    done
  done
done
