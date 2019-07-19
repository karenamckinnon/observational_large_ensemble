#!/bin/bash

# TODO: command line args for ObsLE parameters

# Download data
source ./scripts/download_data.sh

# Create modes
# Note that this starts a batch job, so will not be done immediately! Comment out if already finished
sbatch ./scripts/run_modes_parallel.sbatch
