#!/bin/bash -l
#$ -N ezkl_linear_only
#$ -cwd
#$ -l h_rt=04:00:00
#$ -l h_vmem=32G
#$ -V

module load anaconda
source activate /exports/eddie/scratch/$USER/ezkl_env

# Wrap execution with /usr/bin/time -v to log peak system RAM
/usr/bin/time -v python benchmark_linear_only.py
