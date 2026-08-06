#!/bin/bash -l
#$ -N true_accuracies
#$ -cwd
#$ -pe sharedmem 2
#$ -l h_vmem=8G
#$ -l h_rt=00:30:00
#$ -V
#$ -o true_accuracies.out
#$ -e true_accuracies.err

module load anaconda
source activate /exports/eddie/scratch/$USER/ezkl_env

python evaluate_true_quantised_accuracies.py
