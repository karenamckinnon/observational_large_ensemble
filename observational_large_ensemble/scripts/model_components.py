import numpy as np
import os
from observational_large_ensemble import utils as olens_utils
import xarray as xr
from glob import glob
from subprocess import check_call
import pandas as pd


def fit_linear_model(dsX, df, this_varname, AMO_smooth_length, workdir):
    """Save linear regression model parameters.

    Parameters
    ----------
    dsX : xarray.Dataset
        Dataset containing climate variable of interest
    df : pandas.dataframe
        Mode and forced time series
    this_varname : str
        Variable name for which to fit regression
    AMO_smooth_length : int
        Number of years over which to smooth AMO
    workdir : str
        Where to save output

    Returns
    -------
    Nothing. Saves regression coefficients to netcdf.
    """

    # Fit OLS model to variable X (deterministic)
    # Predictors: constant, GM-EM (forced component), ENSO, PDO, AMO
    # Since AMO is smoothed, can only use a subset of the data
    # Model fit is monthly dependent cognizant of the seasonal cycle in teleconnections

    # Smooth AMO
    AMO_smoothed, valid_indices = olens_utils.smooth(df.loc[:, 'AMO'].values,
                                                     M=AMO_smooth_length*12)

    # Reset AMO to unit standard deviation
    AMO_smoothed /= np.std(AMO_smoothed)

    df = df.loc[valid_indices, :]
    df = df.assign(AMO=AMO_smoothed)

    # Add constant
    df = df.assign(constant=np.ones(len(df)))

    # Subset data to match AMO
    da = dsX[this_varname][valid_indices, ...]
    attrs = dsX.attrs
    attrs['description'] = 'Residuals after removing constant, trend, and regression patterns from ENSO, PDO, AMO.'
    da.attrs = attrs

    predictors_names = ['constant', 'F', 'ENSO', 'PDO_orth', 'AMO']
    if (np.std(df.loc[:, 'F'].values) == 0):  # remove trend predictor, will happen for SLP
        predictors_names.remove('F')

    # Create dataset to save beta values
    ds_beta = xr.Dataset(coords={'month': np.arange(1, 13),
                                 'lat': da.lat,
                                 'lon': da.lon},
                         attrs={'description': 'Regression coefficients for %s' % this_varname})

    residual = np.empty(da.shape)
    _, nlat, nlon = np.shape(da)
    BETA = np.empty((12, nlat, nlon, len(predictors_names)))

    for month in range(1, 13):

        time_idx = da['time.month'] == month

        predictand = da.sel(time=da['time.month'] == month).values
        predictors = df.loc[df['month'] == month, predictors_names].values
        ntime, nlat, nlon = np.shape(predictand)

        y_mat = np.matrix(predictand.reshape(ntime, nlat*nlon))
        X_mat = np.matrix(predictors)

        beta = (np.dot(np.dot((np.dot(X_mat.T, X_mat)).I, X_mat.T), y_mat))  # Max likelihood estimate
        yhat = np.dot(X_mat, beta)
        residual[time_idx, ...] = np.array(y_mat - yhat).reshape((ntime, nlat, nlon))

        BETA[month-1, ...] = np.array(beta).T.reshape((nlat, nlon, len(predictors_names)))

    da_residual = da.copy(data=residual)
    da_residual.attrs
    for counter, name in enumerate(predictors_names):
        kwargs = {'beta_%s' % name: (('month', 'lat', 'lon'), BETA[..., counter])}
        ds_beta = ds_beta.assign(**kwargs)

    # Save to netcdf
    var_dir = '%s/%s' % (workdir, this_varname)
    if not os.path.isdir(var_dir):
        os.mkdir(var_dir)

    ds_beta.to_netcdf('%s/beta.nc' % var_dir)
    da_residual.to_netcdf('%s/residual.nc' % var_dir)


def get_all_surrogates(surr_dir, prefix):
    """Combine all surrogate mode time series into a single array for each.

    Parameters
    ----------
    surr_dir : str
        Directory with all surrogate files (produced via run_modes_parallel.sbatch)
    prefix : str
        Common component of all surrogate filenames

    Returns
    -------
    AMO_surr : numpy.ndarray
        Set of AMO surrogates
    ENSO_surr : numpy.ndarray
        Set of ENSO surrogates
    PDO_orth_surr : numpy.ndarray
        Set of PDO_orth surrogates, where PDO_orth is the Gram-Schmidt orthogonal version of PDO to ENSO
    mode_months : numpy.ndarray
        The month, [1, 12], for the mode time series.
    """

    # Use a single set of LE surrogates
    if 'LE-001' not in surr_dir:
        this_member = (surr_dir.split('/')[-2]).split('-')[-1]
        surr_dir = surr_dir.replace(this_member, '001')

    fnames = sorted(glob('%s/%s*.npz' % (surr_dir, prefix)))
    nsurr_per_file = int(fnames[0].split('_')[-2])
    total_surr = len(fnames)*nsurr_per_file

    fopen = np.load(fnames[0])
    mode_months = fopen['months']
    ntime = len(mode_months)

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

    return AMO_surr, ENSO_surr, PDO_orth_surr, mode_months


def combine_variability(varnames, workdir, output_dir, n_members, block_use_mo,
                        AMO_surr, ENSO_surr, PDO_orth_surr, mode_months, valid_years,
                        mode_lag, AMO_smooth_length, long_varnames, data_names):

    for var_ct, this_varname in enumerate(varnames):

        if not os.path.isdir('%s/%s' % (output_dir, this_varname)):
            cmd = 'mkdir -p %s/%s' % (output_dir, this_varname)
            check_call(cmd.split())

        # Set seed so sampling of residuals is consistent between variables
        np.random.seed(123)
        this_dir = '%s/%s' % (workdir, this_varname)
        fname_epsilon = '%s/residual.nc' % this_dir
        ds = xr.open_dataset(fname_epsilon)
        ntime, nlat, nlon = np.shape(ds[this_varname])

        fname_beta = '%s/beta.nc' % this_dir
        ds_beta = xr.open_dataset(fname_beta)

        # Keeping three year blocks together, perform block bootstrap
        nblocks = int(np.floor(ntime/block_use_mo))

        for kk in range(n_members):
            # Choose the starting points of the blocks
            # Blocks must always start with the same month, so as not to swap monthly sensitivities
            potential_starts = np.arange(ntime - nblocks + 1)[::12]
            these_starts = np.random.choice(potential_starts, nblocks, replace=True)

            # Figure out when we'll need to add additional points
            leftovers = ntime - nblocks*block_use_mo

            new_idx = np.array([np.arange(s, s + block_use_mo) for s in these_starts]).flatten().astype(int)
            if leftovers > 0:
                start_idx = np.random.choice(potential_starts, 1, replace=True)
                new_idx = np.hstack((new_idx, np.arange(start_idx, start_idx + leftovers))).astype(int)

            # Resampled residual = new climate noise
            climate_noise = ds[this_varname][new_idx, ...]
            climate_noise = climate_noise.assign_coords(time=ds.time)

            # Pull out a climate mode surrogate time series
            AMO_ts = AMO_surr[:, kk]
            ENSO_ts = ENSO_surr[:, kk]
            PDO_orth_ts = PDO_orth_surr[:, kk]

            mode_df = pd.DataFrame({'month': mode_months,
                                    'AMO': AMO_ts,
                                    'ENSO': ENSO_ts,
                                    'PDO_orth': PDO_orth_ts})

            # Use the indices for one month before the climate response
            df_shifted = olens_utils.shift_df(mode_df, mode_lag, ['month'])
            # Subselect to the correct number of years
            # Note that for the surrogate modes, the month matters, but the year is meaningless
            df_shifted = df_shifted.loc[:len(valid_years)*12, :]

            # Smooth AMO, and subselect rest to same period
            AMO_smoothed, valid_indices = olens_utils.smooth(df_shifted.loc[:, 'AMO'].values,
                                                             M=AMO_smooth_length*12)

            # Reset AMO to unit standard deviation
            AMO_smoothed /= np.std(AMO_smoothed)

            df_shifted = df_shifted.loc[valid_indices, :]
            df_shifted = df_shifted.assign(AMO=AMO_smoothed)

            # Match the mode month to the climate noise time series
            modes_idx = np.searchsorted(ds_beta.month, climate_noise['time.month'])

            # Add a constant
            df_shifted = df_shifted.assign(constant=np.ones(len(df_shifted)))

            # Ensure that the months are lined up correctly
            assert (df_shifted.month.values == climate_noise['time.month'].values).all()
            AMO = ds_beta.beta_AMO[modes_idx, ...]*df_shifted['AMO'][:, np.newaxis, np.newaxis]
            ENSO = ds_beta.beta_ENSO[modes_idx, ...]*df_shifted['ENSO'][:, np.newaxis, np.newaxis]
            PDO_orth = ds_beta.beta_PDO_orth[modes_idx, ...]*df_shifted['PDO_orth'][:, np.newaxis, np.newaxis]
            mean = ds_beta.beta_constant[modes_idx, ...]*df_shifted['constant'][:, np.newaxis, np.newaxis]

            detrended_values = climate_noise.copy(data=climate_noise.values + AMO + ENSO + PDO_orth + mean)
            description = ('Member %03d of the Observational Large Ensemble ' % (kk + 1) +
                           'for %s. ' % (long_varnames[var_ct]) +
                           'Data is from %s. The forced component must be added separately.' % data_names[var_ct])
            detrended_values.attrs['description'] = description
            filename = '%s/%s/%s_member%03d.nc' % (output_dir, this_varname, this_varname, kk + 1)
            detrended_values.to_netcdf(filename)
