study=$1

python -m stune --exe bp_hypertune --study bp_${study} --gpus "0,1,2,3" --n_trials 16:8 --tuner ssh --config bp_${study}_hypertune.yaml
