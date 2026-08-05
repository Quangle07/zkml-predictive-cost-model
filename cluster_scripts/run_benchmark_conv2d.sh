#!/bin/bash -l
#$ -N ezkl_conv2d
#$ -cwd
#$ -l h_rt=08:00:00
#$ -l h_vmem=32G
#$ -V

module load anaconda
source activate /exports/eddie/scratch/$USER/ezkl_env

/usr/bin/time -v python benchmark_conv2d.py
