import numpy as np
import os
from subprocess import check_call
from glob import glob
from datetime import timedelta
import xarray as xr
import calendar
# import spharm


def fit_trend(da, this_month, N):
    this_da = da.sel(time=da['time.month'] == this_month)
    X = this_da['time.year'].values
    y = this_da.values
    ntime, nlat, nlon = np.shape(y)
    y = y.reshape((ntime, nlat*nlon))

    X = X[-N:]
    y = y[-N:, ...]

    X = np.vstack((np.ones(len(X)), X))
    X_mat = np.matrix(X).T
    y_mat = np.matrix(y)
    beta = np.array(np.dot(np.dot((np.dot(X_mat.T, X_mat)).I, X_mat.T), y_mat))[-1, :]

    return beta.reshape((nlat, nlon))


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ens_name', type=str, help='Ensemble to use.')
    parser.add_argument('--month', type=int, help='Month to use.')
    parser.add_argument('--nyrs', type=int, help='Number of years to calculate trend')
    parser.add_argument('--var', type=str, help='which variable to use (tas, pr, or slp)')

    args = parser.parse_args()

    this_month = int(args.month)
    N = int(args.nyrs)
    ens_name = str(args.ens_name)
    var = str(args.var)

    # varnames = ['tas', 'pr', 'slp']
    long_varnames = ['near surface air temperature', 'precipitation', 'sea level pressure']

    name_conversion = {'tas': 'TREFHT', 'pr': 'PRECC', 'slp': 'PSL'}

    cesm_directory = '/gpfs/fs1/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly'
    obsle_directory = '/glade/scratch/mckinnon/obsLE/output'

    tmp_dir = '/glade/scratch/mckinnon/temp'
    if not os.path.isdir(tmp_dir):
        cmd = 'mkdir %s' % tmp_dir
        check_call(cmd.split())

    # Historical filenames for CESM. Will need to append part of RCP8.5 to get full period
    valid_members = np.hstack((np.arange(1, 36), np.arange(101, 106)))
    nmembers = len(valid_members)
    ensemble_names = 'Obs-LE', 'CESM1-LE'

    # get time, nlat, nlon from a random file
    dummy_file = '/glade/scratch/mckinnon/obsLE/output/LE-001/tas/tas_member001.nc'
    ds_dummy = xr.open_dataset(dummy_file)
    start_time = '%04d-%02d' % (ds_dummy['time.year'][0], ds_dummy['time.month'][0])
    end_time = '%04d-%02d' % (ds_dummy['time.year'][-1], ds_dummy['time.month'][-1])
    nlat = len(ds_dummy['lat'])
    nlon = len(ds_dummy['lon'])

    cesm_var = name_conversion[var]

    # Get data - need to regrid model to data
#    dummy_file = '/glade/work/mckinnon/obsLE/output/obs/%s/%s_member001.nc' % (var, var)
#    da = xr.open_dataarray(dummy_file)
#    obs_lat = da['lat'].values
#    obs_lon = da['lon'].values
#    nlatO = len(obs_lat)
#    nlonO = len(obs_lon)
#    spO = spharm.Spharmt(nlonO, nlatO, gridtype='regular', legfunc='computed')
#
#    dummy_file = '/glade/scratch/mckinnon/obsLE/output/LE-001/%s/%s_member001.nc' % (var, var)
#    da = xr.open_dataarray(dummy_file)
#    _, nlatM, nlonM = np.shape(da)
#    spM = spharm.Spharmt(nlonM, nlatM, gridtype='regular', legfunc='computed')
#
#    # Switch lon to 0, 360
#    # Switch lat to increasing
#    obs_lon[obs_lon < 0] += 360
#    idx_lon = np.argsort(obs_lon)
#    lon = obs_lon[idx_lon]
#
#    idx_lat = np.argsort(obs_lat)
#    lat = obs_lat[idx_lat]
#
#    nlat = len(lat)
#    nlon = len(lon)

    savedir = '/glade/work/mckinnon/obsLE/validation/%s/%s' % (ens_name, var)
    if not os.path.isdir(savedir):
        cmd = 'mkdir -p %s' % savedir
        check_call(cmd.split())

    savename = '%s/%s_%02dyrs_month%02d.npz' % (savedir, var, N, this_month)
    if ens_name == 'CESM1-LE':

        BETA = np.empty((len(valid_members), nlat, nlon))
        for counter, this_member in enumerate(valid_members):

            hist_str = '%s/%s/b.e11.B20TRC5CNBDRD.f09_g16.%03d.cam.h0.%s.??????-200512.nc' % (cesm_directory,
                                                                                              cesm_var,
                                                                                              this_member,
                                                                                              cesm_var)
            hist_file = glob(hist_str)[0]

            future_str = '%s/%s/b.e11.BRCP85C5CNBDRD.f09_g16.%03d.cam.h0.%s.200601-??????.nc' % (cesm_directory,
                                                                                                 cesm_var,
                                                                                                 this_member,
                                                                                                 cesm_var)
            future_file = glob(future_str)[0]
            ds = xr.open_mfdataset([hist_file, future_file], concat_dim='time')

            if cesm_var == 'PRECC':  # need to add in large scale precip
                hist_str2 = hist_str.replace(cesm_var, 'PRECL')
                future_str2 = future_str.replace(cesm_var, 'PRECL')
                hist_file2 = glob(hist_str2)[0]
                future_file2 = glob(future_str2)[0]
                ds2 = xr.open_mfdataset([hist_file2, future_file2], concat_dim='time')
                da = ds[cesm_var]
                da = da.copy(data=(ds[cesm_var].values + ds2['PRECL'].values))
            else:
                da = ds[cesm_var]

            da = da.assign_coords(time=ds.time-timedelta(days=1))
            da = da.sel(time=slice(start_time, end_time))
            da = da.rename(var)

            X_units = da.attrs['units']
            X = da.values
            if X_units == 'K':
                # convert to celsius
                X -= 273.15
                X_units = 'deg C'
            elif X_units == 'm/s':
                # convert to mm (total over month)
                X_year = da['time.year'].values
                X_month = da['time.month'].values
                days_per_month = [calendar.monthrange(int(y), int(m))[1] for y, m in zip(X_year, X_month)]
                seconds_per_month = 60*60*24*np.array(days_per_month)
                X *= seconds_per_month[:, np.newaxis, np.newaxis]  # m per month
                X *= 1000  # mm per month
                X_units = 'mm'

            # Regrid
#            ntime = X.shape[0]
#            X_interp = np.empty((ntime, nlatO, nlonO))
#            for tt in range(ntime):
#                X_interp[tt, ...] = spharm.regrid(spM, spO, X[tt, ...])
#
            new_da = xr.DataArray(X,
                                  dims=('time', 'lat', 'lon'),
                                  coords={'time': da.time,
                                          'lat': da.lat,
                                          'lon': da.lon})

            BETA[counter, ...] = fit_trend(new_da, this_month, N)
        var_beta = np.var(BETA, axis=0)
        np.savez(savename, var_beta=var_beta, lat=da.lat.values, lon=da.lon.values)

    elif 'LE-' in ens_name:
        # Loop through each member of the Obs-LE, calculate trends, and get sigma across them
        this_dir = '%s/%s/%s' % (obsle_directory, ens_name, var)
        files = sorted(glob('%s/%s_*.nc' % (this_dir, var)))
        nmembers = len(files)
        BETA = np.empty((nmembers, nlat, nlon))
        for ct_f, f in enumerate(files):

            da = xr.open_dataarray(f)
            X = da.values

#            # Regrid
#            ntime = X.shape[0]
#            X_interp = np.empty((ntime, nlat, nlon))
#            for tt in range(ntime):
#                X_interp[tt, ...] = spharm.regrid(spM, spO, X[tt, ...])
#
#            new_da = xr.DataArray(X_interp,
#                                  dims=('time', 'lat', 'lon'),
#                                  coords={'time': da.time,
#                                          'lat': lat,
#                                          'lon': lon})

            BETA[ct_f, ...] = fit_trend(da, this_month, N)

        var_beta = np.var(BETA, axis=0)
        np.savez(savename, var_beta=var_beta, lat=da.lat.values, lon=da.lon.values)
