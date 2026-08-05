#!/bin/bash -l
#$ -N ezkl_deep_val
#$ -cwd
#$ -l h_rt=12:00:00
#$ -l h_vmem=256G
#$ -V

module load anaconda
source activate /exports/eddie/scratch/$USER/ezkl_env

python -u deep_validation_benchmark.py
