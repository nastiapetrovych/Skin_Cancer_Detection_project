#!/bin/bash
#SBATCH -J original
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=logs/slurm-%x.%j.out
#SBATCH --exclude=falcon1,falcon2,falcon3

module load any/python/3.8.3-conda
eval "$(conda shell.bash hook)"
conda activate transformers-course  # Replace with your conda env name if different
module load broadwell/gcc/9.2.0
module load cuda/12.1.0

cd ~/Skin_Cancer_Detection_project
mkdir -p logs

python main.py