study=$1

python -m stune --exe pc_hypertune --study pc_${study} --gpus "0,1,2,3,4,5,6,7" --n_trials 32:8 --tuner ssh --config pc_${study}_hypertune.yaml
