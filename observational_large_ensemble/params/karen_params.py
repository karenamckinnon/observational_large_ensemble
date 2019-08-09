"""Example parameter file for Observational Large Ensemble code."""

import numpy as np

valid_years = np.arange(1921, 2015)
cvdp_loc = '/glade/work/mckinnon/CVDP'
AMO_cutoff_freq = 1/10  # Cut off frequency for Butterworth filter of AMO (1/years)
mode_lag = 1  # number of months to lag between mode time series and climate response
workdir_base = '/glade/work/mckinnon/obsLE/parameters'
output_dir = '/glade/scratch/mckinnon/obsLE/output'
tas_dir = '/glade/work/mckinnon/BEST'
pr_dir = '/glade/work/mckinnon/GPCC'
slp_dir = '/glade/work/mckinnon/20CRv2c'