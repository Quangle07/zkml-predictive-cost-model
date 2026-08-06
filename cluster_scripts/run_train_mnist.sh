#!/bin/bash -l
#$ -N train_mnist_models
#$ -cwd
#$ -pe sharedmem 2
#$ -l h_vmem=8G
#$ -l h_rt=00:15:00
#$ -V
#$ -o train_mnist.out
#$ -e train_mnist.err

module load anaconda
source activate /exports/eddie/scratch/$USER/ezkl_env

python train_mnist_models.py
