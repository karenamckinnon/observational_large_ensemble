# CESM1-LE forced trend

# (1) Append datasets


import numpy as np
import os
from subprocess import check_call
from glob import glob
from datetime import timedelta
import xarray as xr
import calendar


varnames = ['tas', 'pr', 'slp']
long_varnames = ['near surface air temperature', 'precipitation', 'sea level pressure']

# CESM

name_conversion = {'tas': 'TREFHT', 'pr': 'PRECC', 'slp': 'PSL'}
cesm_names = list(name_conversion.values())

base_directory = '/gpfs/fs1/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly'

tmp_dir = '/glade/scratch/mckinnon/temp'
if not os.path.isdir(tmp_dir):
    cmd = 'mkdir %s' % tmp_dir
    check_call(cmd.split())

# Historical filenames for CESM. Will need to append part of RCP8.5 to get full period
valid_members = np.hstack((np.arange(1, 36), np.arange(101, 106)))


# Match time to member of Obs-LE
dummy_file = '/glade/work/mckinnon/obsLE/output/obs/tas/tas_member001.nc'
ds_dummy = xr.open_dataset(dummy_file)
start_time = '%04d-%02d' % (ds_dummy['time.year'][0], ds_dummy['time.month'][0])
end_time = '%04d-%02d' % (ds_dummy['time.year'][-1], ds_dummy['time.month'][-1])

for var in varnames:
    this_var = name_conversion[var]

    signal_dir = '/glade/work/mckinnon/obsLE/output/forced_component/%s' % var
    if not os.path.isdir(signal_dir):
        cmd = 'mkdir -p %s' % signal_dir
        check_call(cmd.split())

    savename = '%s_CESM1_EM.nc' % var
    da_list = []
    for this_member in valid_members:

        hist_str = '%s/%s/b.e11.B20TRC5CNBDRD.f09_g16.%03d.cam.h0.%s.??????-200512.nc' % (base_directory,
                                                                                          this_var,
                                                                                          this_member,
                                                                                          this_var)
        hist_file = glob(hist_str)[0]

        future_str = '%s/%s/b.e11.BRCP85C5CNBDRD.f09_g16.%03d.cam.h0.%s.200601-??????.nc' % (base_directory,
                                                                                             this_var,
                                                                                             this_member,
                                                                                             this_var)
        future_file = glob(future_str)[0]
        ds = xr.open_mfdataset([hist_file, future_file], concat_dim='time')

        if this_var == 'PRECC':  # need to add in large scale precip
            hist_str2 = hist_str.replace(this_var, 'PRECL')
            future_str2 = future_str.replace(this_var, 'PRECL')
            hist_file2 = glob(hist_str2)[0]
            future_file2 = glob(future_str2)[0]
            ds2 = xr.open_mfdataset([hist_file2, future_file2], concat_dim='time')
            da = ds[this_var]
            da = da.copy(data=(ds[this_var].values + ds2['PRECL'].values))

        else:
            da = ds[this_var]

        da = da.assign_coords(time=ds.time-timedelta(days=1))
        da = da.sel(time=slice(start_time, end_time))
        da_list.append(da)

    # Take ensemble mean
    full_da = xr.concat(da_list, dim='member')
    EM = full_da.mean(dim='member')

    # Remove time mean
    EM -= EM.mean(dim='time')

    # Change units if necessary
    X_units = EM.attrs['units']
    if X_units == 'K':
        # convert to celsius
        EM = EM.copy(data=EM[this_var] - 273.15)
        X_units = 'deg C'
    elif X_units == 'm/s':
        # convert to mm (total over month)
        X = EM[this_var]
        X_year = EM['time.year'].values
        X_month = EM['time.month'].values
        days_per_month = [calendar.monthrange(int(y), int(m))[1] for y, m in zip(X_year, X_month)]
        seconds_per_month = 60*60*24*np.array(days_per_month)
        X *= seconds_per_month[:, np.newaxis, np.newaxis]  # m per month
        X *= 1000  # mm per month
        EM = EM.copy(data=X)
        X_units = 'mm'
        EM.attrs['long_name'] = 'Total monthly precipitation'

    EM = EM.rename({this_var: var})
    EM.attrs['units'] = X_units

    EM.to_netcdf('%s/%s' % (signal_dir, savename))
