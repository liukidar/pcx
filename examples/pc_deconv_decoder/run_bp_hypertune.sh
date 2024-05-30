dataset=$1

python -m stune --exe bp_hypertune --study bp_${dataset}_decoder --gpus "0,1,2,3" --n_trials 16:8 --tuner ssh --config bp_${dataset}_hypertune.yaml
