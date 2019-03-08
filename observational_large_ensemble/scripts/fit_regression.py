import numpy as np
from datetime import datetime
import os
from netCDF4 import Dataset
import pandas as pd
from observational_large_ensemble import utils as olens_utils
import json
import calendar
from cftime import utime


def fit_linear_model(X, X_units, lat, lon, X_year, X_month, df, this_varname, month, AMO_smooth_length, workdir):
    """Save linear regression model parameters.
    """

    ntime, nlat, nlon = np.shape(X)

    # Fit OLS model to variable X (deterministic)
    # Predictors: constant, GM-EM (forced component), ENSO, PDO
    # Model fit is monthly dependent cognizant of the seasonal cycle in teleconnections
    mo = int(month)

    predictand = X[X_month == mo, ...]
    predictors = df.loc[df['month'] == mo, ['F', 'ENSO', 'PDO']].values
    predictors = np.hstack((np.ones((len(predictand), 1)), predictors))
    predictors_names = 'constant', 'forcing', 'ENSO', 'PDO', 'AMO'
    y_mat = np.matrix(predictand.reshape((int(ntime/12), nlat*nlon)))
    X_mat = np.matrix(predictors)

    if (np.std(df.loc[:, 'F'].values) == 0):  # remove trend predictor
        predictors_names = predictors_names[:1] + predictors_names[2:]
        X_mat = X_mat[:, 1:]

    beta = (np.dot(np.dot((np.dot(X_mat.T, X_mat)).I, X_mat.T), y_mat))  # Max likelihood estimate
    yhat = np.dot(X_mat, beta)
    residual = y_mat - yhat
    residual = np.array(residual)

    # Fit AMO to the residual
    # Have to treat separately because performing smoothing
    AMO_smoothed, valid_indices = olens_utils.smooth(df.loc[df['month'] == mo, 'AMO'].values,
                                                     M=AMO_smooth_length)

    BETA = np.empty((len(predictors_names), nlat*nlon))
    BETA[:-1, :] = np.array(beta)

    valid_indices = np.where(~np.isnan(BETA[0, :]))[0]

    residual_smooth = np.zeros((nlat*nlon, len(AMO_smoothed)))

    for ii in valid_indices:
        residual_smooth[ii, :], _ = olens_utils.smooth(residual[:, ii],
                                                       M=AMO_smooth_length)

    X_mat_AMO = np.matrix(AMO_smoothed).T
    y_mat_AMO = np.matrix(residual_smooth).T
    BETA[-1, :] = (np.dot(np.dot((np.dot(X_mat_AMO.T, X_mat_AMO)).I, X_mat_AMO.T), y_mat_AMO))

    # Save beta values to netcdf
    var_dir = '%s%s/' % (workdir, this_varname)
    if not os.path.isdir(var_dir):
        os.mkdir(var_dir)
    for counter, p_name in enumerate(predictors_names):
        this_beta = BETA[counter, :].reshape((nlat, nlon))
        description = '%s regression model, %s term. Month %i' % (this_varname, p_name, mo)
        if p_name == 'constant':
            units = X_units
        elif p_name == 'forcing':
            units = 'unitless'
        else:
            units = '%s/deg C' % X_units

        nc_varname = '%s_coeff' % p_name
        savename = '%sbeta%01d_month%02d.nc' % (var_dir, counter, mo)
        olens_utils.save_2d_netcdf(lat, lon, this_beta, nc_varname, units,
                                   savename, description, overwrite=False)


def get_obs(this_varname, this_filename, valid_years, mode_lag, cvdp_loc):
    """Return observational data and associated time series of modes for a given variable.
    """

    # Location of CVDP output
    modes_fname = '%s/HadISST.cvdp_data.1920-2017.nc' % cvdp_loc  # modes

    # Convert non standard names to standard
    name_conversion = {'tas': 'temperature', 'pr': 'precip', 'slp': 'prmsl'}

    # Get the forced component
    # Assume that the global mean trend in sea level is zero
    if this_varname == 'slp':
        gm_em, gm_em_units, time, time_units = olens_utils.forced_trend('tas', cvdp_loc)
        gm_em *= 0
        gm_em += 1  # will replace constant
    else:
        gm_em, gm_em_units, time, time_units = olens_utils.forced_trend(this_varname, cvdp_loc)

    # If using precipitation, need number of days in month to convert units
    if this_varname == 'pr':
        gm_time = np.arange(1920 + 0.5/12, 1920 + 1/12*len(time), 1/12)
        gm_year = np.floor(gm_time)
        gm_month = np.ceil((gm_time - gm_year)*12)
        days_per_month = [calendar.monthrange(int(y), int(m))[1] for y, m in zip(gm_year, gm_month)]
        assert gm_em_units == 'mm/day'  # double check
        gm_em *= days_per_month
        gm_em_units = 'mm'

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

    # Load dataset
    ds = Dataset(this_filename, 'r')

    # Load data
    try:
        lat = ds['latitude'][:]
        lon = ds['longitude'][:]
    except IndexError:
        lat = ds['lat'][:]
        lon = ds['lon'][:]
    try:
        X = ds[this_varname][:, :, :]
        X_units = ds[this_varname].units
    except IndexError:
        alt_name = name_conversion[this_varname]
        X = ds[alt_name][:, :, :]
        X_units = ds[alt_name].units

    # Check unit consistency
    if this_varname == 'slp':
        assert X_units == 'Pa'
    if this_varname == 'pr':
        assert X_units == 'mm'

    X_time = ds['time'][:]
    X_time_units = ds['time'].units

    # Code dealing with various time units
    if X_time_units == 'year A.D.':
        X_time = X_time.compressed()  # saved as masked array, but no values are masked
        X_year = np.floor(X_time)
        X_month = (np.ceil((X_time - X_year)*12)).astype(int)

    else:
        # For more standard time formats
        cld = utime(X_time_units)
        dt = cld.num2date(X_time)
        X_year = np.array([t.year for t in dt])
        X_month = np.array([t.month for t in dt])

    # Permute all data to be time, lat, lon
    lat_idx = np.where(np.isin(X.shape, len(lat)))[0][0]
    lon_idx = np.where(np.isin(X.shape, len(lon)))[0][0]
    time_idx = np.where(np.isin(X.shape, len(X_time)))[0][0]

    X = np.transpose(X, (time_idx, lat_idx, lon_idx))
    ntime, nlat, nlon = np.shape(X)

    # Subset data
    subset = np.isin(X_year, valid_years)
    X = X[subset, :]
    X_year = X_year[subset]
    X_month = X_month[subset]

    # Also need to check if our data spans the full valid period
    subset = np.isin(df['year'].values, X_year)
    df = df.loc[subset, :]

    # Check that all dimensions look consistent
    assert len(df) == np.shape(X)[0]

    return X, X_units, lat, lon, X_year, X_month, df


def setup(varname, filename, AMO_smooth_length, mode_lag, workdir_base):

    # Create dictionary of parameters to save in working directory
    param_dict = {'varname': varname,
                  'filename': filename,
                  'AMO_smooth_length': AMO_smooth_length,
                  'mode_lag': mode_lag}

    # Output folder, named with current date
    now = datetime.strftime(datetime.now(), '%Y%m%d')
    workdir = '%s/%s/' % (workdir_base, now)
    if not os.path.isdir(workdir):
        os.mkdir(workdir)
    # Save parameter set to director
    with open(workdir + 'parameter_set.json', 'w') as f:
        json.dump(param_dict, f)

    return workdir


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('month', type=int, help='Which month to fit regression model')

    args = parser.parse_args()

    # Set of variables to analyze (user inputs)
    varname = ['tas', 'pr', 'slp']
    filename = ['/glade/work/mckinnon/BEST/Complete_TAVG_LatLong1.nc',
                '/glade/work/mckinnon/GPCC/precip.mon.total.1x1.v7.nc',
                '/glade/work/mckinnon/20CRv2c/prmsl.mon.mean.nc']
    AMO_smooth_length = 15  # number of years to apply AMO smoothing
    mode_lag = 1  # number of months to lag between mode time series and climate response
    workdir_base = '/glade/work/mckinnon/obsLE/parameters'
    valid_years = np.arange(1921, 2017)
    cvdp_loc = '/glade/work/mckinnon/CVDP'

    # Need odd-window for AMO
    if AMO_smooth_length % 2 == 0:
        AMO_smooth_length += 1

    # Save parameter files
    workdir = setup(varname, filename, AMO_smooth_length, mode_lag, workdir_base)

    # Get data and modes
    for v, f in zip(varname, filename):
        X, X_units, lat, lon, X_year, X_month, df = get_obs(v, f, valid_years, mode_lag, cvdp_loc)
        fit_linear_model(X, X_units, lat, lon, X_year, X_month, df, v, args.month, AMO_smooth_length, workdir)
