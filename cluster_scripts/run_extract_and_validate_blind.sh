#!/bin/bash -l
#$ -N ezkl_val_blind
#$ -cwd
#$ -l h_rt=00:30:00
#$ -l h_vmem=16G
#$ -V

module load anaconda
source activate /exports/eddie/scratch/$USER/ezkl_env

/usr/bin/time -v python extract_and_validate_blind.py
