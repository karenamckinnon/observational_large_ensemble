"""Example parameter file for Observational Large Ensemble code."""

import numpy as np

version_name = 'short_pr_noF_noAMV'
valid_years = np.arange(1966, 2006)  # skip first year of CESM1-LE, and don't combine with future scenario
cvdp_loc = '/glade/work/mckinnon/CVDP'
AMO_cutoff_freq = 1/20  # Cut off frequency for Butterworth filter of AMO (1/years)
mode_lag = 0  # number of months to lag between mode time series and climate response
workdir_base = '/glade/work/mckinnon/obsLE/parameters_v-%s' % version_name
output_dir = '/glade/scratch/mckinnon/obsLE/output_v-%s' % version_name
data_dir = '/gpfs/fs1/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly'
pr_transform = 'boxcox'  # can be boxcox or log
varnames = ['pr']  # which variables to use to create the synthetic ensemble
predictors_names = ['constant', 'ENSO', 'PDO_orth']
