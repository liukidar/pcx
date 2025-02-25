#!/bin/bash

# Base parameters
BASE_PATH="/share/amine.mcharrak/mixed_data_final"
EXP_NAME="increase_n_disc"  # or "mixed_confounding"
N_SAMPLES=2000
NUM_SEEDS=5

# Define experiments per GPU
GPU0_EXPS=(
  "10ER40_linear_mixed_seed_1"
  "15ER60_linear_mixed_seed_1"
  "10SF40_linear_mixed_seed_1"
  "15SF60_linear_mixed_seed_1"
  "10WS40_linear_mixed_seed_1"
  "15WS60_linear_mixed_seed_1"
)
GPU1_EXPS=(
  "20ER80_linear_mixed_seed_1"
)
GPU2_EXPS=(
  "20SF80_linear_mixed_seed_1"
)
GPU3_EXPS=(
  "20WS80_linear_mixed_seed_1"
)

# Function to run an experiment
run_experiment() {
  local exp_dir=$1
  local gpu=$2
  python icml_mixed_data_script.py \
    --path_name "${BASE_PATH}/${exp_dir}" \
    --n_samples ${N_SAMPLES} \
    --num_seeds ${NUM_SEEDS} \
    --gpu ${gpu} \
    --exp_name ${EXP_NAME}
}

# Run GPU 0 experiments
for exp in "${GPU0_EXPS[@]}"; do
  run_experiment "$exp" 0 &
done

# Run GPU 1 experiments
for exp in "${GPU1_EXPS[@]}"; do
  run_experiment "$exp" 1 &
done

# Run GPU 2 experiments
for exp in "${GPU2_EXPS[@]}"; do
  run_experiment "$exp" 2 &
done

# Run GPU 3 experiments
for exp in "${GPU3_EXPS[@]}"; do
  run_experiment "$exp" 3 &
done

# Wait for all background processes to finish
wait
