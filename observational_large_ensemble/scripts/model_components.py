import numpy as np
import os
from observational_large_ensemble import utils as olens_utils
import xarray as xr
from subprocess import check_call
import pandas as pd


def fit_linear_model(da, df, this_varname, workdir, predictors_names):
    """Save linear regression model parameters.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray containing climate variable of interest
    df : pandas.dataframe
        Mode and forced time series
    this_varname : str
        Variable name for which to fit regression
    workdir : str
        Where to save output
    predictors_names : list
        Names of covariates to use

    Returns
    -------
    Nothing. Saves regression coefficients to netcdf.
    """

    # Fit OLS model to variable X (deterministic)
    # Predictors: constant, GM-EM (forced component), ENSO, PDO, AMO
    # Model fit is monthly dependent cognizant of the seasonal cycle in teleconnections

    # Add constant
    df = df.assign(constant=np.ones(len(df)))

    # Set all mode predictors to unit standard deviation
    df['ENSO'] /= np.std(df['ENSO'])
    df['PDO_orth'] /= np.std(df['PDO_orth'])
    df['AMO_lowpass'] /= np.std(df['AMO_lowpass'])

    attrs = da.attrs
    attrs['description'] = 'Residuals after removing constant, trend, and regression patterns from ENSO, PDO, AMO.'

    # Create dataset to save beta values
    ds_beta = xr.Dataset(coords={'month': np.arange(1, 13),
                                 'lat': da.lat,
                                 'lon': da.lon},
                         attrs={'description': 'Regression coefficients for %s' % this_varname})

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

        BETA[month-1, ...] = np.array(beta).T.reshape((nlat, nlon, len(predictors_names)))

    # project BETA onto two harmonics: annual and semi-annual
    time_vec = pd.date_range(start='1950-01-01', periods=365, freq='D')
    doy = xr.DataArray(np.arange(1, 366), coords={'time': time_vec}, dims='time')
    t_basis = (doy.groupby('time.month').mean()/365).values
    nbases = 2
    nt = len(t_basis)

    bases = np.empty((nbases, nt), dtype=complex)
    for counter in range(nbases):
        bases[counter, :] = np.exp(2*(counter + 1)*np.pi*1j*t_basis)

    coeff = 2/nt*(np.dot(bases, BETA.reshape((12, nlat*nlon*len(predictors_names)))))
    rec = np.real(np.dot(bases.T, np.conj(coeff))).reshape((12, nlat, nlon, len(predictors_names)))
    rec += np.mean(BETA, axis=0)

    # Recalculate yhat and residuals with these smoothed values
    residual = np.empty(da.shape)
    yhat = np.empty(da.shape)
    for month in range(1, 13):

        time_idx = da['time.month'] == month

        predictand = da.sel(time=da['time.month'] == month).values
        predictors = df.loc[df['month'] == month, predictors_names].values
        ntime, nlat, nlon = np.shape(predictand)

        y_mat = np.matrix(predictand.reshape(ntime, nlat*nlon))
        X_mat = np.matrix(predictors)

        beta = rec[month - 1, ...]
        beta = beta.reshape((nlat*nlon, len(predictors_names)))
        this_yhat = np.dot(X_mat, beta.T)
        residual[time_idx, ...] = np.array(y_mat - this_yhat).reshape((ntime, nlat, nlon))
        yhat[time_idx, ...] = np.array(this_yhat).reshape((ntime, nlat, nlon))

    da_residual = da.copy(data=residual)
    da_residual.attrs = attrs

    da_yhat = da.copy(data=yhat)
    da_yhat.attrs['description'] = 'Fitted values (non-residual)'
    for counter, name in enumerate(predictors_names):
        kwargs = {'beta_%s' % name: (('month', 'lat', 'lon'), rec[..., counter])}
        ds_beta = ds_beta.assign(**kwargs)

    # Save to netcdf
    var_dir = '%s/%s' % (workdir, this_varname)

    ds_beta.to_netcdf('%s/beta.nc' % var_dir)
    da_residual.to_netcdf('%s/residual.nc' % var_dir)
    da_yhat.to_netcdf('%s/yhat.nc' % var_dir)


def combine_variability(varnames, workdir, output_dir, n_members, block_use_mo,
                        AMO_surr, ENSO_surr, PDO_orth_surr, mode_months, valid_years,
                        mode_lag, long_varnames, data_names, pr_transform, predictors_names):

    for this_varname in varnames:

        if not os.path.isdir('%s/%s' % (output_dir, this_varname)):
            cmd = 'mkdir -p %s/%s' % (output_dir, this_varname)
            check_call(cmd.split())

        # Set seed so sampling of residuals is consistent between variables
        np.random.seed(123)
        this_dir = '%s/%s' % (workdir, this_varname)
        fname_epsilon = '%s/residual.nc' % this_dir
        fname_yhat = '%s/yhat.nc' % this_dir
        da = xr.open_dataarray(fname_epsilon)
        da_yhat = xr.open_dataarray(fname_yhat)
        ntime, nlat, nlon = np.shape(da)

        fname_beta = '%s/beta.nc' % this_dir
        ds_beta = xr.open_dataset(fname_beta)

        # Keeping blocks together, perform block bootstrap
        nblocks = int(np.ceil(ntime/block_use_mo))

        for kk in range(n_members):
            # Choose the starting points of the blocks
            # Blocks must always start with the same month, so as not to swap monthly sensitivities
            potential_starts = np.arange(ntime)[::12]
            these_starts = np.random.choice(potential_starts, nblocks, replace=True)

            new_idx = np.array([np.arange(s, s + block_use_mo) for s in these_starts]).flatten().astype(int)
            new_idx = new_idx % ntime  # circular bootstrap
            new_idx = new_idx[:ntime]  # since we used ceiling, need to cutoff the end

            # Resampled residual = new climate noise
            climate_noise = da[new_idx, ...]
            climate_noise = climate_noise.assign_coords(time=da.time)

            # Pull out a climate mode surrogate time series
            AMO_ts = AMO_surr[:, kk]
            ENSO_ts = ENSO_surr[:, kk]
            PDO_orth_ts = PDO_orth_surr[:, kk]

            # combine into dataframe, setting all to unity std
            mode_df = pd.DataFrame({'month': mode_months,
                                    'AMO_lowpass': AMO_ts/np.std(AMO_ts),
                                    'ENSO': ENSO_ts/np.std(ENSO_ts),
                                    'PDO_orth': PDO_orth_ts/np.std(PDO_orth_ts)})

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

            assert (df_shifted.month.values == climate_noise['time.month'].values).all()

            mean = (ds_beta.beta_constant[modes_idx, ...].values *
                    df_shifted['constant'][:, np.newaxis, np.newaxis])

            # Save one version with original F + modes, but resampled noise
            data = da_yhat.values + climate_noise.values

            # Save the mean + climate noise only for analysis
            no_modes_F = climate_noise.copy(data=data)
            if this_varname == 'pr':
                # use same lambda for all members of the LE
                this_workdir = workdir
                if 'LE-' in this_workdir:
                    tmp = this_workdir.split('/')[-1]  # will be LE-XXX
                    this_workdir = this_workdir.replace(tmp, 'LE-001')

                # model was fit on transformed precip, so translate back to original units
                no_modes_F = olens_utils.retransform(no_modes_F, pr_transform, '%s/%s' % (this_workdir, this_varname))

            description = ('Member %04d of the Observational Large Ensemble (only noise resampled) ' % (kk + 1) +
                           'for %s. ' % (long_varnames[this_varname]) +
                           'Data is from %s.' % data_names[this_varname])
            no_modes_F.attrs['description'] = description
            filename = '%s/%s/%s_member%04d_noise.nc' % (output_dir, this_varname, this_varname, kk + 1)
            no_modes_F = no_modes_F.rename(this_varname)
            no_modes_F.to_netcdf(filename)

            del no_modes_F

            # Add in the relevant components
            data = mean + climate_noise.values

            if 'F' in predictors_names:
                forced_file = '%s/%s/%s_forced.nc' % (output_dir, this_varname, this_varname)
                daF = xr.open_dataarray(forced_file)
                # Check that the time aligns
                assert (daF['time.month'] == climate_noise['time.month']).all()
                assert (daF['time.year'] == climate_noise['time.year']).all()
                data += daF.values

            if 'ENSO' in predictors_names:

                ENSO = (ds_beta.beta_ENSO[modes_idx, ...].values *
                        df_shifted['ENSO'][:, np.newaxis, np.newaxis])
                data += ENSO
            if 'PDO_orth' in predictors_names:

                PDO_orth = (ds_beta.beta_PDO_orth[modes_idx, ...].values *
                            df_shifted['PDO_orth'][:, np.newaxis, np.newaxis])
                data += PDO_orth
            if 'AMO_lowpass' in predictors_names:

                AMO_lowpass = (ds_beta.beta_AMO_lowpass[modes_idx, ...].values *
                               df_shifted['AMO_lowpass'][:, np.newaxis, np.newaxis])
                data += AMO_lowpass
            new_values = climate_noise.copy(data=data)
            nan_mask = ~np.isnan(new_values)  # NaNs get lost in transform
            if this_varname == 'pr':
                # model was fit on transformed precip, so translate back to original units
                this_workdir = workdir
                if 'LE-' in this_workdir:
                    tmp = this_workdir.split('/')[-1]  # will be LE-XXX
                    this_workdir = this_workdir.replace(tmp, 'LE-001')
                new_values = olens_utils.retransform(new_values.copy(), pr_transform,
                                                     '%s/%s' % (this_workdir, this_varname))
                new_values = new_values.where(nan_mask)
            description = ('Member %04d of the Observational Large Ensemble ' % (kk + 1) +
                           'for %s. ' % (long_varnames[this_varname]) +
                           'Data is from %s.' % data_names[this_varname])
            new_values.attrs['description'] = description
            filename = '%s/%s/%s_member%04d.nc' % (output_dir, this_varname, this_varname, kk + 1)
            new_values = new_values.rename(this_varname)
            new_values.to_netcdf(filename)


def create_surrogate_modes(cvdp_file, AMO_cutoff_freq, this_seed, n_ens_members, valid_years, workdir):
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
    valid_years : numpy.ndarray
        Array of years used to build the model
    workdir : str
        Where to save modes

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

    savename = '%s/synthetic_mode_ts.nc' % workdir
    if os.path.isfile(savename):
        ds_surr = xr.open_dataset(savename)
        enso_surr = ds_surr['ENSO_surr'].values
        pdo_surr = ds_surr['PDO_surr'].values
        amo_surr = ds_surr['AMO_surr'].values
        months = ds_surr['month'].values
    else:
        # Load original versions
        df = olens_utils.create_mode_df(cvdp_file, AMO_cutoff_freq)
        # Subset to valid years
        subset = np.isin(df['year'].values, valid_years)
        df = df.loc[subset, :]
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

            # AMO
            tmp = olens_utils.iaaft(df['AMO_lowpass'].values)
            while type(tmp) == int:  # case of no convergence
                tmp = olens_utils.iaaft(df['AMO_lowpass'].values)

            # Perform lowpass filter on AMO
            if AMO_cutoff_freq > 0:
                amo_lowpass = olens_utils.lowpass_butter(12, AMO_cutoff_freq, 3, tmp[0])
            else:  # no filter
                amo_lowpass = tmp[0]
            amo_surr[:, kk] = amo_lowpass

        if workdir is not None:
            ds_surr = xr.Dataset(data_vars={'ENSO_surr': (('month', 'member'), enso_surr),
                                            'PDO_surr': (('month', 'member'), pdo_surr),
                                            'AMO_surr': (('month', 'member'), amo_surr)},
                                 coords={'month': months, 'member': np.arange(n_ens_members)})
            ds_surr.to_netcdf(savename)

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
