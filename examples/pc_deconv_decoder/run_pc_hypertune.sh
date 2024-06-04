dataset=$1
optim=$2

python -m stune --exe pc_hypertune --study pc_${dataset}_${optim}_decoder --gpus "0,1,2,3" --n_trials 16:8 --tuner ssh --config pc_${dataset}_${optim}_hypertune.yaml
