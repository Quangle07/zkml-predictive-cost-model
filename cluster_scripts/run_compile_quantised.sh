#$ -N compile_quantised
#$ -cwd
#$ -pe sharedmem 2
#$ -l h_vmem=16G
#$ -l h_rt=00:30:00
#$ -V
#$ -o compile_quantised.out
#$ -e compile_quantised.err

module load anaconda
source activate /exports/eddie/scratch/$USER/ezkl_env

python compile_quantised_models.py
