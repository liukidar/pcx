dataset=$1

python -m stune --exe pc_hypertune --study pc_${dataset}_decoder --gpus "0,1,2,3" --n_trials 16:8 --tuner ssh --config pc_${dataset}_hypertune.yaml
