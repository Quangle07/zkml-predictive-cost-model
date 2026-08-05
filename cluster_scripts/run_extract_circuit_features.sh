#!/bin/bash -l
#$ -N ezkl_extract
#$ -cwd
#$ -l h_rt=02:00:00
#$ -l h_vmem=32G
#$ -V

module load anaconda
source activate /exports/eddie/scratch/$USER/ezkl_env

python extract_circuit_features.py
