import numpy as np
from datetime import datetime
import os
from netCDF4 import Dataset
import pandas as pd
from observational_large_ensemble import utils as olens_utils
import json


def fit_linear_model(varname, filename, n_ens_members, AMO_smooth_length, mode_lag, workdir_base, verbose=True):

    # Create dictionary of parameters to save in working directory
    param_dict = {'varname': varname,
                  'filename': filename,
                  'n_ens_members': n_ens_members,
                  'AMO_smooth_length': AMO_smooth_length,
                  'mode_lag': mode_lag}

    valid_years = np.arange(1921, 2017)

    # Location of CVDP output
    cvdp_loc = '/glade/work/mckinnon/CVDP'
    modes_fname = '%s/HadISST.cvdp_data.1920-2017.nc' % cvdp_loc  # modes

    # Output folder
    now = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
    workdir = '%s/%s/' % (workdir_base, now)
    if not os.path.isdir(workdir):
        os.mkdir(workdir)

    # Save parameter set to directory
    with open(workdir + 'parameter_set.json', 'w') as f:
        json.dump(param_dict, f)

    name_conversion = {'tas': 'temperature', 'pr': 'precip', 'slp': 'prmsl'}
    if AMO_smooth_length % 2 == 0:
        AMO_smooth_length += 1

    # Fit linear regression model to data
    # Put into large function later to pass dictionary of variables
    # So this applies to a single variable

    # Loop through variables
    for v, f in zip(varname, filename):
        if verbose:
            print('Beginning fit for %s' % v)
        this_varname = v
        this_filename = f

        # Get the forced component
        gm_em, gm_em_units, time, time_units = olens_utils.forced_trend(this_varname, cvdp_loc)

        # Get dataframe of modes
        df = olens_utils.create_mode_df(modes_fname)
        # Add EM, GM time series to it
        df = df.assign(F=gm_em)

        # Shift by desired lag for prediction (i.e. want Dec modes to predict Jan response)
        df1 = df.loc[:, ['year', 'month', 'season', 'F']].drop(df.head(mode_lag).index)
        df2 = df.loc[:, ['AMO', 'PDO', 'ENSO']].drop(df.tail(mode_lag).index)
        new_df = pd.concat((df1, df2), axis=1, ignore_index=True, join='inner')
        new_df.columns = ['year', 'month', 'season', 'F', 'AMO', 'PDO', 'ENSO']
        del df1, df2, df
        df = new_df
        del new_df

        # Subset to valid years
        subset = np.isin(df['year'].values, valid_years)
        df = df.loc[subset, :]

        # Load dataset being used for creating ObsLE
        ds = Dataset(this_filename, 'r')
        if this_varname not in ds.variables:
            this_varname2 = name_conversion[this_varname]
        else:
            this_varname2 = this_varname

        try:
            lat = ds['latitude'][:]
            lon = ds['longitude'][:]
        except IndexError:
            lat = ds['lat'][:]
            lon = ds['lon'][:]
        X = ds[this_varname2][:, :, :]
        X_units = ds[this_varname2].units
        X_time = ds['time'][:]
        X_time_units = ds['time'].units

        # Messy code dealing with various time units
        if X_time_units == 'year A.D.':
            X_time = X_time.compressed()  # saved as masked array, but no values are masked
            X_year = np.floor(X_time)
            subset = np.isin(X_year, valid_years)
        else:
            print('Need to code this')
            raise KeyboardInterrupt

        # Permute all data to be time, lat, lon
        lat_idx = np.where(np.isin(X.shape, len(lat)))[0][0]
        lon_idx = np.where(np.isin(X.shape, len(lon)))[0][0]
        time_idx = np.where(np.isin(X.shape, len(X_time)))[0][0]

        X = np.transpose(X, (time_idx, lat_idx, lon_idx))
        ntime, nlat, nlon = np.shape(X)

        # Subset data
        X = X[subset, :]
        X_time = X_time[subset]
        X_year = np.floor(X_time)
        X_month = (np.ceil((X_time - X_year)*12)).astype(int)

        ntime, nlat, nlon = np.shape(X)

        # Check that all dimensions look consistent
        assert len(df) == np.shape(X)[0]

        # Fit OLS model to variable X
        # Predictors: constant, GM-EM (forced component), ENSO, PDO
        # Model fit is monthly dependent cognizant of the seasonal cycle in teleconnections
        for mo in range(1, 13):  # months are 1-12
            if verbose:
                print('Month %i' % mo)
            predictand = X[X_month == mo, ...]
            predictors = df.loc[df['month'] == mo, ['F', 'ENSO', 'PDO']].values
            predictors = np.hstack((np.ones((len(predictand), 1)), predictors))
            predictors_names = 'constant', 'forcing', 'ENSO', 'PDO'

            y_mat = np.matrix(predictand.reshape((int(ntime/12), nlat*nlon)))
            X_mat = np.matrix(predictors)
            beta = (np.dot(np.dot((np.dot(X_mat.T, X_mat)).I, X_mat.T), y_mat))  # Max likelihood estimate
            yhat = np.dot(X_mat, beta)
            residual = y_mat - yhat

            # Calculate variance/covariance matrix for each gridbox for resampling
            variance_estimator = np.ma.var(np.array(residual), axis=0, ddof=2)
            denom_C = (np.dot(X_mat.T, X_mat)).I

            # Fit AMO to the residual
            # Have to treat separately because performing smoothing
            AMO_smoothed, valid_indices = olens_utils.smooth(df.loc[df['month'] == mo, 'AMO'].values,
                                                             M=AMO_smooth_length)

            # In order to allow for some covariance between AMO fit and prior regression fit,
            # will fit AMO model to various realizations of the initial fit
            np.random.seed(123)
            beta = np.array(beta)
            valid_indices = np.where(~np.isnan(beta[0, :]))[0]
            beta_AMO = np.zeros((n_ens_members, nlat*nlon))
            for kk in range(n_ens_members):
                if verbose:
                    print('Ensemble member %i' % kk)
                res_smooth = np.zeros((nlat*nlon, len(AMO_smoothed)))

                for ii in valid_indices:
                    this_beta = beta[:, ii]
                    if np.isnan(this_beta).any():
                        continue
                    # Get sample of beta terms
                    sample = np.random.multivariate_normal(this_beta, variance_estimator[ii]*denom_C)
                    yhat = np.dot(X_mat, sample).T
                    res = np.array(y_mat[:, ii] - yhat).flatten()
                    res_smooth[ii, :], _ = olens_utils.smooth(res, M=AMO_smooth_length)
                X_mat_AMO = np.matrix(AMO_smoothed).T
                y_mat_AMO = np.matrix(res_smooth).T
                beta_AMO[kk, :] = (np.dot(np.dot((np.dot(X_mat_AMO.T, X_mat_AMO)).I, X_mat_AMO.T), y_mat_AMO))

            if verbose:
                print('Beginning saves')
            # Save beta values to netcdf
            for counter, p_name in enumerate(predictors_names):
                this_beta = beta[counter, :].reshape((nlat, nlon))
                description = '%s regression model, %s term. Month %i' % (this_varname, p_name, mo)
                if p_name == 'constant':
                    units = X_units
                elif p_name == 'forcing':
                    units = 'unitless'
                else:
                    units = '%s/deg C' % X_units

                nc_varname = '%s_coeff' % p_name
                savename = '%sbeta%01d_month%02d.nc' % (workdir, counter, mo)
                olens_utils.save_2d_netcdf(lat, lon, this_beta, nc_varname, units,
                                           savename, description, overwrite=True)

            # Save residual variance
            description = 'Residual variance in %s from regression model. Month %i' % (this_varname, mo)
            units = '%s**2' % X_units
            nc_varname = 'res_variance'

            savename = '%sresidual_variance_month%02d.nc' % (workdir, mo)
            olens_utils.save_2d_netcdf(lat, lon, variance_estimator.reshape((nlat, nlon)),
                                       nc_varname, units, savename, description, overwrite=True)

            # Save AMO samples
            for kk in range(n_ens_members):
                this_beta = beta_AMO[counter, :].reshape((nlat, nlon))
                description = ('%s regression model, AMO term using %i year smoothing. Month %i' %
                               (this_varname, AMO_smooth_length, mo))
                units = '%s/deg C' % X_units

                nc_varname = 'AMO_coeff'
                savename = '%sbetaAMO_member%03d_month%02d.nc' % (workdir, kk, mo)
                olens_utils.save_2d_netcdf(lat, lon, this_beta, nc_varname, units,
                                           savename, description, overwrite=True)


if __name__ == '__main__':

    # Set of variables to analyze (user inputs)
    varname = ['tas']
    filename = ['/glade/work/mckinnon/BEST/Land_and_Ocean_LatLong1.nc']
    n_ens_members = 100
    AMO_smooth_length = 15  # number of years to apply AMO smoothing
    mode_lag = 1  # number of months to lag between mode time series and climate response
    workdir_base = '/glade/work/mckinnon/obsLE/parameters'

    fit_linear_model(varname, filename, n_ens_members, AMO_smooth_length, mode_lag, workdir_base)
