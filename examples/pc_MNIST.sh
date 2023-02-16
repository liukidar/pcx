#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=mnist-il
#SBATCH --gres=gpu:1
#SBATCH --ntasks 1 --cpus-per-task 8
#SBATCH --partition=devel
#SBATCH --output ./slurm-%j-%x.out # STDOUT

module purge
module load python/anaconda3 > /dev/null

eval "$(conda shell.bash hook)"
conda activate pcax

python pc_MNIST.py