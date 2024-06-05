study=$1

python -m stune --exe pc_hypertune --study pc_${study} --gpus "0,1,2,3" --n_trials 16:8 --tuner ssh --config pc_${study}_hypertune.yaml
