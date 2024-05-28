#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
# set max wallclock time
#SBATCH --time=47:59:00
#SBATCH --partition=medium
# set name of job
#SBATCH --job-name=mnist_pcax
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

wandb login 7edd27dd0e92add2b9a146291c69cb109c7e3396

for IDX in `seq 4`; do
	GPU_ID=$((IDX % 4))
	echo $GPU_ID
	CUDA_VISIBLE_DEVICES=$GPU_ID wandb agent --project pcax --entity oliviers-gaspard  pk4dzoip --count 100 &
done   

# wait for all processes to finish                        
wait