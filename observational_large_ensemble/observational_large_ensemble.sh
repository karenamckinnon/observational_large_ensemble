#!/bin/bash

# TODO: command line args for ObsLE parameters

# Download data
source ./scripts/download_data.sh

# Create modes
# Note that this starts a batch job, so will not be done immediately! Comment out if already finished
sbatch ./scripts/run_modes_parallel.sbatch

# Fit regression model
# Model is fit to each month separately, so use parallel for efficiency if available

# module load parallel
# parallel -j12 'python ./scripts/fit_regression.py {1}' ::: $(seq 12)

for i in {1..12}
    do
    python ./scripts/fit_regression.py $i
done


