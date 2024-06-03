#!/bin/bash

# Default values for parameters
epochs_list=(500)
batch_sizes=(128)
w_learning_rates=(0.01)  
h_learning_rates=(0.01 0.03 0.05 0.1) # List of state learning rates
h_momentum_list=(0.1 0.3 0.5 0.7 0.9) # List of state optimizer momentum values

noise_level=0.0
script_name="train_double_descent_pc.py"

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
# This gets shown in the terminal if the user doesn't provide any arguments
if [ "$OPTIND" -eq 1 ]; then
  echo "Usage example:"
  echo "$0 -n 0.2 -s train_double_descent_pc.py"
fi

# Loop over each combination of parameters
for epochs in "${epochs_list[@]}"; do
  for batch_size in "${batch_sizes[@]}"; do
    for w_learning_rate in "${w_learning_rates[@]}"; do
      for h_learning_rate in "${h_learning_rates[@]}"; do
        for h_momentum in "${h_momentum_list[@]}"; do
          # Run the Python script with the current set of parameters
          python $script_name --config ./config.json --noise_level $noise_level --batch_size $batch_size --w_learning_rate $w_learning_rate --h_learning_rate $h_learning_rate --h_momentum $h_momentum --epochs $epochs
        done
      done
    done
  done
done
