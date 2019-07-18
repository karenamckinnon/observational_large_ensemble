import numpy as np
from datetime import datetime
import os
from netCDF4 import Dataset
from observational_large_ensemble import utils as olens_utils
import json
import calendar
from cftime import utime
import pandas as pd
import xarray as xr


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
    for counter, name in enumerate(predictors_names):
        kwargs = {'beta_%s' % name: (('month', 'lat', 'lon'), BETA[..., counter])}
        ds_beta = ds_beta.assign(**kwargs)

    # Save to netcdf
    var_dir = '%s%s/' % (workdir, this_varname)
    if not os.path.isdir(var_dir):
        os.mkdir(var_dir)

    ds_beta.to_netcdf('%sbeta.nc' % var_dir)
    da_residual.to_netcdf('%sresidual.nc' % var_dir)


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

    # Shift modes in time
    df_shifted = olens_utils.shift_df(df, mode_lag, ['year', 'month', 'season', 'F'])

    # Subset to valid years
    subset = np.isin(df_shifted['year'].values, valid_years)
    df_shifted = df_shifted.loc[subset, :]

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
    subset = np.isin(df_shifted['year'].values, X_year)
    df_shifted = df_shifted.loc[subset, :]

    # Check that all dimensions look consistent
    assert len(df_shifted) == np.shape(X)[0]

    # Put into dataset
    time = pd.date_range(start='%04d-%02d' % (X_year[0], X_month[0]),
                         freq='M', periods=len(X_year))
    dsX = xr.Dataset(data_vars={this_varname: (('time', 'lat', 'lon'), X)},
                     coords={'time': time,
                             'lat': lat,
                             'lon': lon},
                     attrs={'%s units' % this_varname: X_units})

    return dsX, df_shifted, df


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

    # Set of variables to analyze (user inputs)
    varname = ['tas', 'pr', 'slp']
    filename = ['/glade/work/mckinnon/BEST/Complete_TAVG_LatLong1.nc',
                '/glade/work/mckinnon/GPCC/precip.mon.total.1x1.v2018.nc',
                '/glade/work/mckinnon/20CRv2c/prmsl.mon.mean.nc']
    AMO_smooth_length = 21  # number of years to apply AMO smoothing
    mode_lag = 1  # number of months to lag between mode time series and climate response
    workdir_base = '/glade/work/mckinnon/obsLE/parameters'
    valid_years = np.arange(1921, 2015)
    cvdp_loc = '/glade/work/mckinnon/CVDP'

    # Need odd-window for AMO
    if AMO_smooth_length % 2 == 0:
        AMO_smooth_length += 1

    # Save parameter files
    workdir = setup(varname, filename, AMO_smooth_length, mode_lag, workdir_base)

    # Get data and modes
    for v, f in zip(varname, filename):
        dsX, df_shifted, _ = get_obs(v, f, valid_years, mode_lag, cvdp_loc)
        fit_linear_model(dsX, df_shifted, v, AMO_smooth_length, workdir)
