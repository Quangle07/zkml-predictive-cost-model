#!/bin/bash -l
#$ -N ezkl_quant
#$ -cwd
#$ -l h_rt=24:00:00
#$ -l h_vmem=64G
#$ -V

module load anaconda
source activate /exports/eddie/scratch/$USER/ezkl_env

python benchmark_quantisation.py
