"""Create maps showing the SST anomaly patterns associated with desired modes.

Maps vary monthly.
"""

from observational_large_ensemble import utils as olens_utils
import numpy as np
import xarray as xr

figdir = '/glade/work/mckinnon/obsLE/figs/'

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
