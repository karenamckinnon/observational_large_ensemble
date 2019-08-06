#!/bin/sh

#SBATCH --time=08:00:00
#SBATCH --output=logs/parallel.log
#SBATCH --partition=PARTITION
#SBATCH --account=ACCOUNT
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

# This example script shows how you can run the Observational Large Ensemble code using parallel
# across all the members of the CESM1-LE.
# Note that the all-caps fields need to be replaced with your credentials, etc.

export TMPDIR=TMPDIR
mkdir -p $TMPDIR

source /etc/profile.d/modules.sh
module load parallel

# Load your virtual env
source PATH_TO_VENV

srun="srun --exclusive -N1 -n1"

parallel="parallel --delay .2 -j $SLURM_NTASKS --joblog logs/runtask.log --resume"

$parallel "$srun python main.py 40 LE-{1} > logs/parallel_{1}.log" ::: {001..035} {101..105}
