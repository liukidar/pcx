dataset=$1
optim=$2

python -m stune --exe bp_hypertune --study bp_${dataset}_${optim}_decoder --gpus "0,1,2,3" --n_trials 16:8 --tuner ssh --config bp_${dataset}_${optim}_hypertune.yaml
