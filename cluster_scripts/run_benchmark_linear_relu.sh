#!/bin/bash -l
#$ -N ezkl_combinatorial
#$ -cwd
#$ -l h_rt=24:00:00
#$ -l h_vmem=64G
#$ -V

module load anaconda
source activate /exports/eddie/scratch/$USER/ezkl_env

# We use /usr/bin/time -v to force the OS to track the peak memory of the Rust backend
/usr/bin/time -v python benchmark_linear_relu.py
