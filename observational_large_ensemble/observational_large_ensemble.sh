#!/bin/bash

# TODO: command line args for ObsLE parameters

# Download data
source ./scripts/download_data.sh

# Create modes
python ./scripts/create_surrogate_mode_sets.py

# Fit regression model
# Model is fit to each month separately, so use parallel for efficiency
parallel -j12 'python ./scripts/fit_regression.py {1}' ::: $(seq 12)
