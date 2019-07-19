#!/bin/sh

#SBATCH --time=08:00:00
#SBATCH --output=logs/parallel.log
#SBATCH --partition=dav
#SBATCH --account=P93300313
#SBATCH --ntasks=32

export TMPDIR=/glade/scratch/mckinnon/temp
mkdir -p $TMPDIR

source /etc/profile.d/modules.sh
module load parallel

# Load my virtual env
source /glade/work/mckinnon/venvs/venv07-01-2019/bin/activate

# Set input file
# mode_nc=HadISST.cvdp_data.1920-2017.nc
mode_nc=CESM1-CAM5-BGC-LE_#1.cvdp_data.1920-2017.nc

# the --exclusive to srun makes srun use distinct CPUs for each job step
# -N1 -n1 allocates a single core to each task
srun="srun --exclusive -N1 -n1"

# --delay .2 prevents overloading the controlling node
# -j is the number of tasks parallel runs so we set it to $SLURM_NTASKS
# --joblog makes parallel create a log of tasks that it has already run
# --resume makes parallel use the joblog to resume from where it has left off
# the combination of --joblog and --resume allow jobs to be resubmitted if
# necessary and continue from where they left off
parallel="parallel --delay .2 -j $SLURM_NTASKS --joblog logs/runtask.log --resume"

# this runs the parallel command we want
# in this case, we are running a script named runtask
# parallel uses ::: to separate options. Here {0..99} is a shell expansion
# so parallel will run the command passing the numbers 0 through 99
# via argument {1}
$parallel "$srun python create_surrogate_mode_sets.py -N 20 -S {1} -f $mode_nc > logs/parallel_{1}.log" ::: {0..49}
