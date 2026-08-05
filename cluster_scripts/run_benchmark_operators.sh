#!/bin/bash -l
#$ -N ezkl_operators
#$ -cwd
#$ -l h_rt=08:00:00
#$ -l h_vmem=64G
#$ -V

module load anaconda
source activate /exports/eddie/scratch/$USER/ezkl_env

python benchmark_operators.py
