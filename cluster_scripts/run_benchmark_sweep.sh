#!/bin/bash -l
#$ -N ezkl_sweep
#$ -cwd
#$ -l h_rt=08:00:00
#$ -l h_vmem=64G
#$ -V

module load anaconda
source activate /exports/eddie/scratch/$USER/ezkl_env

/usr/bin/time -v python benchmark_logrows_sweep.py
