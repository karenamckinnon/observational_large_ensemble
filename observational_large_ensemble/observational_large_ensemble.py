import numpy as np
from datetime import datetime
import os
from observational_large_ensemble import utils as olens_utils
import json
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
    var_dir = '%s%s/' % (workdir, this_varname)
    if not os.path.isdir(var_dir):
        os.mkdir(var_dir)

    ds_beta.to_netcdf('%sbeta.nc' % var_dir)
    da_residual.to_netcdf('%sresidual.nc' % var_dir)


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
        dsX, df_shifted, _ = olens_utils.get_obs(v, f, valid_years, mode_lag, cvdp_loc)
        fit_linear_model(dsX, df_shifted, v, AMO_smooth_length, workdir)
