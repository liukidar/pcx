#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
# set max wallclock time
#SBATCH --time=11:00:00
#SBATCH --partition=short
# set name of job
#SBATCH --job-name=vae_mnist
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
source activate $DATA/mcpc

for IDX in `seq 1`; do
	GPU_ID=$((0))
	echo $GPU_ID
	CUDA_VISIBLE_DEVICES=$GPU_ID wandb agent --project vae --entity oliviers-gaspard  e12m8ery --count 500 &
done   

# wait for all processes to finish                        
wait