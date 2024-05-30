#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
# set max wallclock time
#SBATCH --time=47:59:00
#SBATCH --partition=medium
# set name of job
#SBATCH --job-name=iris_pcax
# qos
#SBATCH --qos=standard
#SBATCH --account=ndcn-computational-neuroscience
# set gpu 
#SBATCH --gres=gpu:4
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=gaspard.oliviers@bndu.ox.ac.uk


module load Anaconda3
source activate $DATA/envs/pcax

for IDX in `seq 28`; do
	GPU_ID=$((IDX % 4))
	echo $GPU_ID
	CUDA_VISIBLE_DEVICES=$GPU_ID wandb agent --project pcax --entity oliviers-gaspard  dgqmvr52 --count 100 &
done   

# wait for all processes to finish                        
wait 
