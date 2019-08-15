import numpy as np
import os
from observational_large_ensemble import utils as olens_utils
import xarray as xr
from subprocess import check_call
import pandas as pd


def fit_linear_model(dsX, df, this_varname, workdir):
    """Save linear regression model parameters.

    Parameters
    ----------
    dsX : xarray.Dataset
        Dataset containing climate variable of interest
    df : pandas.dataframe
        Mode and forced time series
    this_varname : str
        Variable name for which to fit regression
    workdir : str
        Where to save output

    Returns
    -------
    Nothing. Saves regression coefficients to netcdf.
    """

    # Fit OLS model to variable X (deterministic)
    # Predictors: constant, GM-EM (forced component), ENSO, PDO, AMO
    # Model fit is monthly dependent cognizant of the seasonal cycle in teleconnections

    # Add constant
    df = df.assign(constant=np.ones(len(df)))

    da = dsX[this_varname]
    attrs = dsX.attrs
    attrs['description'] = 'Residuals after removing constant, trend, and regression patterns from ENSO, PDO, AMO.'
    da.attrs = attrs

    predictors_names = ['constant', 'F', 'ENSO', 'PDO_orth', 'AMO_lowpass']
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


def combine_variability(varnames, workdir, output_dir, n_members, block_use_mo,
                        AMO_surr, ENSO_surr, PDO_orth_surr, mode_months, valid_years,
                        mode_lag, long_varnames, data_names):

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
            potential_starts = np.arange(ntime - block_use_mo)[::12]
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
                                    'AMO_lowpass': AMO_ts,
                                    'ENSO': ENSO_ts,
                                    'PDO_orth': PDO_orth_ts})

            # Use the indices for one month before the climate response
            df_shifted = olens_utils.shift_df(mode_df, mode_lag, ['month'])
            # Ensure that the months are lined up correctly
            start_idx = np.where(df_shifted['month'] == climate_noise['time.month'].values[0])[0][0]
            df_shifted = df_shifted[start_idx:]
            # Subselect to the correct number of years
            # Note that for the surrogate modes, the month matters, but the year is meaningless
            df_shifted = df_shifted[:len(valid_years)*12]

            # Match the mode month to the climate noise time series
            modes_idx = np.searchsorted(ds_beta.month, climate_noise['time.month'])

            # Add a constant
            df_shifted = df_shifted.assign(constant=np.ones(len(df_shifted)))

            # Add the forced trend
            if this_varname != 'slp':
                forced_file = '%s/%s/%s_forced.nc' % (output_dir, this_varname, this_varname)
                daF = xr.open_dataarray(forced_file)
                # Check that the time aligns
                assert (daF['time.month'] == climate_noise['time.month']).all()
                assert (daF['time.year'] == climate_noise['time.year']).all()

            assert (df_shifted.month.values == climate_noise['time.month'].values).all()
            AMO_lowpass = (ds_beta.beta_AMO_lowpass[modes_idx, ...].values *
                           df_shifted['AMO_lowpass'][:, np.newaxis, np.newaxis])
            ENSO = (ds_beta.beta_ENSO[modes_idx, ...].values *
                    df_shifted['ENSO'][:, np.newaxis, np.newaxis])
            PDO_orth = (ds_beta.beta_PDO_orth[modes_idx, ...].values *
                        df_shifted['PDO_orth'][:, np.newaxis, np.newaxis])
            mean = (ds_beta.beta_constant[modes_idx, ...].values *
                    df_shifted['constant'][:, np.newaxis, np.newaxis])

            if this_varname != 'slp':
                data = climate_noise.values + AMO_lowpass + ENSO + PDO_orth + mean + daF.values
            else:
                data = climate_noise.values + AMO_lowpass + ENSO + PDO_orth + mean

            if this_varname == 'pr':
                data[data < 0] = 0  # precipitation can't be negative

            new_values = climate_noise.copy(data=data)
            description = ('Member %04d of the Observational Large Ensemble ' % (kk + 1) +
                           'for %s. ' % (long_varnames[var_ct]) +
                           'Data is from %s.' % data_names[var_ct])
            new_values.attrs['description'] = description
            filename = '%s/%s/%s_member%04d.nc' % (output_dir, this_varname, this_varname, kk + 1)
            new_values.to_netcdf(filename)


def create_surrogate_modes(cvdp_file, AMO_cutoff_freq, this_seed, n_ens_members):
    """Create random mode sets.

    Parameters
    ----------
    cvdp_file : str
        Full file path to CVDP netcdf containing observed or in-model modes
    AMO_cutoff_freq : float
        Cut off frequency for Butterworth filter of AMO (1/years)
    this_seed : int
        Random seed for reproducibility
    n_ens_members : int
        Number of mode sets to create

    Returns
    -------
    enso_surr : numpy.ndarray
        Array (ntime x n_ens_members) of surrogate ENSO time series
    pdo_surr : numpy.ndarray
        Array (ntime x n_ens_members) of surrogate orthogonalized PDO time series
    amo_surr : numpy.ndarray
        Array (ntime x n_ens_members) of surrogate low-passed AMO time series
    months : numpy.ndarray
        Months associated with the surrogate time series. Important when fit_seasonal=True
    """

    # Load original versions
    df = olens_utils.create_mode_df(cvdp_file, AMO_cutoff_freq)
    ntime = len(df)
    months = df['month'].values

    np.random.seed(this_seed)

    enso_surr = np.empty((ntime, n_ens_members))
    pdo_surr = np.empty_like(enso_surr)
    amo_surr = np.empty_like(pdo_surr)

    for kk in range(n_ens_members):
        # ENSO (accounting for seasonality of variance)
        tmp = olens_utils.iaaft(df['ENSO'].values, fit_seasonal=True)
        while type(tmp) == int:  # case of no convergence
            tmp = olens_utils.iaaft(df['ENSO'].values, fit_seasonal=True)
        enso_surr[:, kk] = tmp[0]

        # PDO
        tmp = olens_utils.iaaft(df['PDO_orth'].values)
        while type(tmp) == int:  # case of no convergence
            tmp = olens_utils.iaaft(df['PDO_orth'].values)
        pdo_surr[:, kk] = tmp[0]

        # AMO (create surrogates on unfiltered data)
        tmp = olens_utils.iaaft(df['AMO'].values)
        while type(tmp) == int:  # case of no convergence
            tmp = olens_utils.iaaft(df['AMO'].values)

        # Perform lowpass filter on AMO
        if AMO_cutoff_freq > 0:
            amo_lowpass = olens_utils.lowpass_butter(12, AMO_cutoff_freq, 3, tmp[0])
        else:  # no filter
            amo_lowpass = tmp[0]
        # Reset to unit sigma
        amo_lowpass /= np.std(amo_lowpass)
        amo_surr[:, kk] = amo_lowpass

    return enso_surr, pdo_surr, amo_surr, months


def save_forced_component(df, this_var, output_dir, workdir):
    """Calculate and save to netcdf the estimated forced component.

    The forced component is estimated by regressing the data onto the CESM1-LE global mean, ensemble mean
    of the same variable.

    Parameters
    ----------
    df : pd.Dataframe
        Dataframe containing time series of GM-EM forcing
    this_var : str
        Standard varname (tas, pr, slp)
    output_dir : str
        Location to save forced trend
    workdir : str
        Location of parameter files (sensitivity to forced trend at each month/gridbox)

    Returns
    -------
    Nothing. Forced component saved as netcdf.

    """

    param_file = '%s/%s/beta.nc' % (workdir, this_var)
    ds_beta = xr.open_dataset(param_file)

    # Match the month of the sensitivity parameter, beta, to the month of the forcing
    modes_idx = np.searchsorted(ds_beta.month, df['month'].values)

    # Recreate the time vector
    time = pd.date_range('%04d-%02d' % (df.year.values[0], df.month.values[0]),
                         freq='M', periods=len(df))

    # Forced component estimated as time series of CESM1-LE GM-EM times sensitivity at each
    # gridbox (estimated from OLS regression)
    F = ds_beta.beta_F[modes_idx, ...]*df['F'][:, np.newaxis, np.newaxis]
    F = F.rename({'month': 'time'})
    F = F.assign_coords(time=time)
    F = F.rename('forced_component')

    if not os.path.isdir('%s/%s' % (output_dir, this_var)):
        cmd = 'mkdir -p %s/%s' % (output_dir, this_var)
        check_call(cmd.split())
    savename = '%s/%s/%s_forced.nc' % (output_dir, this_var, this_var)
    F.attrs['description'] = ('Forced component estimated through regressing data onto '
                              'CESM1-LE global mean, ensemble mean time series.')
    F.to_netcdf(savename)

    # For ease, also save the constant term
    C = ds_beta.beta_constant
    C = C.rename('climatology')
    savename = '%s/%s/%s_climatology.nc' % (output_dir, this_var, this_var)
    C.attrs['description'] = 'Monthly mean climatology'
    C.to_netcdf(savename)
