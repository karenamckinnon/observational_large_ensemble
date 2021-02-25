# Analysis
import xarray as xr
import numpy as np
from datetime import timedelta

# General
from glob import glob
import os

# My codes
import observational_large_ensemble.utils as olens_utils

version = 'main'

metrics = 'var_low', 'var_high', 'IQ_range', '85yr_event', '50yr_event'
seasons = 'DJF', 'JJA'
members = np.hstack((np.arange(1, 36), np.arange(101, 106)))
valid_years = np.arange(1921, 2006)

# for filter
fs = 1  # per year
L = 1/10.  # above or below decadal
order = 3

cesmdir = '/gpfs/fs1/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly'
obsledir = '/glade/scratch/mckinnon/obsLE/output_v-%s' % version

return_periods_save = np.array([85, 100, 200, 300])

# CESM-LE analysis
cesm_savename = '/glade/work/mckinnon/obsLE/proc/cesm_v-%s_5metrics.nc' % version
cesm_savename_ens_extremes = '/glade/work/mckinnon/obsLE/proc/cesm_v-%s_ens_extreme_metrics.nc' % version

if os.path.isfile(cesm_savename):
    ds_metrics = xr.open_dataset(cesm_savename)
    da_ens_extremes = xr.open_dataarray(cesm_savename_ens_extremes)
else:
    all_metrics = []
    # Also need to store data arrays to get rank across full ensemble
    all_cesm = []
    for m in members:
        print(m)

        # Get the CESM data
        files = glob('%s/PREC*/b.e11.B20TRC5CNBDRD.f09_g16.%03i.cam.h0.PREC*.??????-200512.nc' % (cesmdir, m))
        ds_cesm = xr.open_mfdataset(files, concat_dim='precip_type', combine='by_coords')
        da_cesm = ds_cesm['PRECC'] + ds_cesm['PRECL'] + ds_cesm['PRECSL'] + ds_cesm['PRECSC']
        da_cesm = da_cesm.assign_coords(time=da_cesm.time-timedelta(days=1))
        da_cesm = da_cesm.assign_coords({'lat': np.round(da_cesm.lat, 3)})

        # change to mm /day
        da_cesm *= 1000*24*60*60
        # resample for seasons
        da_cesm = da_cesm.resample(time='QS-DEC').mean()

        cesm = []
        da_metrics_seasons = []
        for this_season in seasons:

            # pull out desired season
            this_da_cesm = da_cesm.sel({'time': da_cesm['time.season'] == this_season})
            this_da_cesm['time'] = this_da_cesm['time.year']
            this_da_cesm = this_da_cesm.rename({'time': 'year'})
            if this_season == 'DJF':  # year is associated with december
                this_da_cesm = this_da_cesm.assign_coords({'year': this_da_cesm['year'].values + 1})

            # and desired year
            this_da_cesm = this_da_cesm.sel({'year': np.isin(this_da_cesm['year'], valid_years)})

            this_da_cesm = this_da_cesm.load()

            cesm.append(this_da_cesm)
            da_metrics = []
            for metric_name in metrics:
                da_metric = olens_utils.calc_variability_metrics(this_da_cesm, metric_name)
                da_metrics.append(da_metric)

            da_metrics = xr.concat(da_metrics, dim='metric_name')
            da_metrics['metric_name'] = np.array(metrics)
            da_metrics_seasons.append(da_metrics)

        da_metrics_seasons = xr.concat(da_metrics_seasons, dim='season')
        da_metrics_seasons['season'] = np.array(seasons)

        cesm = xr.concat(cesm, dim='season')
        cesm['season'] = np.array(seasons)

        all_metrics.append(da_metrics_seasons)
        all_cesm.append(cesm)

    all_metrics = xr.concat(all_metrics, dim='member')
    all_metrics['member'] = members
    all_metrics = all_metrics.to_dataset(dim='metric_name')
    all_metrics = all_metrics.transpose('lat', 'lon', 'member', 'season')

    all_metrics.to_netcdf(cesm_savename)

    # calculate longer return period events across full ensemble
    # return period for each season: across year and member
    all_cesm = xr.concat(all_cesm, dim='member')
    all_cesm['member'] = members
    all_cesm = all_cesm.transpose('lat', 'lon', 'season', 'member', 'year')
    da_size = all_cesm.shape
    vals = all_cesm.values.reshape((da_size[0], da_size[1], da_size[2], da_size[3]*da_size[4]))
    vals_sorted = np.sort(vals, axis=-1)[:, :, :, ::-1]
    ntime = da_size[3]*da_size[4]  # ensemble and time are used interchangeably
    RI = (ntime + 1)/np.arange(1, ntime + 1)

    da_ens_extremes = []
    for r in return_periods_save:
        RI_idx = np.argmin(np.abs(RI - r))
        event_magnitude = vals_sorted[:, :, :, RI_idx]
        da_tmp = all_cesm[:, :, :, 0, 0].copy(data=event_magnitude)
        da_ens_extremes.append(da_tmp)

    da_ens_extremes = xr.concat(da_ens_extremes, dim='return_period')
    da_ens_extremes['return_period'] = return_periods_save
    da_ens_extremes = da_ens_extremes.transpose('lat', 'lon', 'season', 'return_period')

    da_ens_extremes.to_netcdf(cesm_savename_ens_extremes)
