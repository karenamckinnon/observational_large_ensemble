"""
Create figures associated with modes.

(1) SST anomaly maps associated with modes in the observations. Maps vary monthly.

(2) Power spectra of the observed modes compared to the surrogates.
"""

from observational_large_ensemble import utils as olens_utils
import numpy as np
import xarray as xr
from glob import glob
import matplotlib.pyplot as plt

figdir = '/glade/work/mckinnon/obsLE/figs/'

# (1) SST anomaly maps associated with modes in the observations

# Load mode data
cvdp_loc = '/glade/work/mckinnon/CVDP'
modes_fname = '%s/HadISST.cvdp_data.1920-2017.nc' % cvdp_loc  # modes
df = olens_utils.create_mode_df(modes_fname)
ntime = len(df)

mode_names = 'AMO', 'ENSO', 'PDO_orth'
valid_years = np.arange(1921, 2017)
AMO_smooth_length = 21  # number of years to apply AMO smoothing. Following Simpson et al (2018) JClim

subset = np.isin(df['year'].values, valid_years)
df = df.loc[subset, :]

# Load SST data
sst_file = '/glade/work/mckinnon/HadISST/HadISST_sst.nc'
ds_sst = xr.open_dataset(sst_file)

# remove climatology
climo = ds_sst.groupby('time.month').mean('time')
ds_anoms = ds_sst.groupby('time.month') - climo

for this_mode in mode_names:

    for this_month in range(1, 13):

        savename = '%sSST_pattern_%s_month%02d.png' % (figdir, this_mode, this_month)

        this_ds = ds_anoms.sel(time=ds_anoms['time.month'] == this_month)
        this_ds = this_ds.sel(time=slice('01-%04d' % valid_years[0], '12-%04d' % valid_years[-1]))

        X = this_ds.sst

        ntime, nlat, nlon = np.shape(X)

        ice_loc = (climo.sst[climo.month == this_month, ...] < -2).squeeze()

        if this_mode != 'AMO':

            predictand = X.values
            predictors = df.loc[(df.month == this_month) & (np.isin(df.year, valid_years)), this_mode].values

            y_mat = np.matrix(predictand.reshape((int(ntime), nlat*nlon)))
            X_mat = np.matrix(predictors).T

            beta = (np.dot(np.dot((np.dot(X_mat.T, X_mat)).I, X_mat.T), y_mat))  # Max likelihood estimate

            olens_utils.plot_sst_patterns(this_ds.latitude, this_ds.longitude, beta, ice_loc, this_mode, savename)

        else:  # perform smoothing

            AMO_smoothed, valid_indices = olens_utils.smooth(df.loc[df['month'] == this_month, 'AMO'].values,
                                                             M=AMO_smooth_length)

            # set to unit standard deviation for consistency with other modes
            AMO_smoothed /= np.std(AMO_smoothed)

            X_mat_AMO = np.matrix(AMO_smoothed).T
            y_mat_AMO = np.matrix(X.values[valid_indices, ...].reshape((len(valid_indices), nlat*nlon)))
            beta = (np.dot(np.dot((np.dot(X_mat_AMO.T, X_mat_AMO)).I, X_mat_AMO.T), y_mat_AMO))

            olens_utils.plot_sst_patterns(this_ds.latitude, this_ds.longitude, beta, ice_loc, this_mode, savename)


# (2) Power spectra of the observed modes compared to the surrogates
# TODO: add red noise spectrum for context

# Reload modes for full time period
df = olens_utils.create_mode_df(modes_fname)
ntime = len(df)

# Load surrogate time series
# Produced via sbatch ./scripts/run_modes_parallel.sbatch
surr_dir = '/glade/work/mckinnon/obsLE/parameters/surrogates'
fnames = sorted(glob('%s/HadISST_surrogate_mode_time_series_*.npz' % surr_dir))
nsurr_per_file = int(fnames[0].split('_')[-2])
total_surr = len(fnames)*nsurr_per_file

AMO_surr = np.empty((ntime, nsurr_per_file, len(fnames)))
ENSO_surr = np.empty_like(AMO_surr)
PDO_orth_surr = np.empty_like(AMO_surr)

for ct, this_f in enumerate(fnames):
    fopen = np.load(this_f)
    AMO_surr[:, :, ct] = fopen['amo_surr']
    ENSO_surr[:, :, ct] = fopen['enso_surr']
    PDO_orth_surr[:, :, ct] = fopen['pdo_surr']

AMO_surr = np.reshape(AMO_surr, (ntime, total_surr))
ENSO_surr = np.reshape(ENSO_surr, (ntime, total_surr))
PDO_orth_surr = np.reshape(PDO_orth_surr, (ntime, total_surr))

mode_names = 'ENSO', 'PDO_orth', 'AMO'
for m in mode_names:
    orig_ts = df[m].values
    surr_ts = eval('%s_surr' % m)

    P, s, ci = olens_utils.pmtm(orig_ts, 1/12)

    Psurr = np.empty((len(P), total_surr))
    for ct in range(total_surr):
        Psurr[:, ct], _, _ = olens_utils.pmtm(surr_ts[:, ct], 1/12)

    P025 = np.percentile(Psurr, 2.5, axis=-1)
    P975 = np.percentile(Psurr, 97.5, axis=-1)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))

    ax.fill_between(s, P*ci[:, 0], P*ci[:, -1], color='lightgray', alpha=0.7, lw=0, label='95% confidence interval')
    ax.plot(s, P*ci[:, 0], color='k', lw=0.3)
    ax.plot(s, P*ci[:, 1], color='k', lw=0.3)
    ax.plot(s, P, color='k', label=m)

    ax.fill_between(s, P025, P975, color='lightblue', alpha=0.7, lw=0, label='Surrogate 95% range')
    ax.plot(s, np.mean(Psurr, axis=-1), color='b', label='Avg surrogate')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([1e-2, 1e-1, 1])
    ax.set_xticklabels(['100', '10', '1'])

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Period (yrs)', fontsize=20)
    plt.ylabel('Power density', fontsize=20)
    plt.legend(fontsize=16, loc='lower left')

    figname = '%s_spectra_with_surrogates.png' % m
    plt.savefig('%s%s' % (figdir, figname), dpi=200, bbox_inches='tight', orientation='landscape')
    plt.close()
