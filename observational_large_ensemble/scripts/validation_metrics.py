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

metrics = 'var_low', 'var_high', 'IQ_range', 'max_val', '85yr_event'
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
cesm_savename = '/glade/work/mckinnon/obsLE/proc/cesm_v-main_5metrics.nc'
cesm_savename_ens_extremes = '/glade/work/mckinnon/obsLE/proc/cesm_v-main_ens_extreme_metrics.nc'

if os.path.isfile(cesm_savename):
    ds_metrics = xr.open_dataset(cesm_savename)
    da_ens_extremes = xr.open_dataarray(cesm_savename_ens_extremes)
else:
    all_m1 = []
    all_m2 = []
    all_m3 = []
    all_m4 = []
    all_m5 = []

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

        if m == 1:  # create our metrics dataset
            ds_metrics = xr.Dataset(coords={'lat': da_cesm.lat, 'lon': da_cesm.lon,
                                            'member': members, 'season': np.array(seasons)})
        m1 = []
        m2 = []
        m3 = []
        m4 = []
        m5 = []

        cesm = []
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

            for metric_name in metrics:
                if metric_name == 'var_high':
                    vals = olens_utils.lowpass_butter(fs, L, order, this_da_cesm.values, axis=0, btype='high')
                    tmp = this_da_cesm.copy(data=vals)
                    da_metric = tmp.var('year')
                    m1.append(da_metric)

                elif metric_name == 'var_low':
                    vals = olens_utils.lowpass_butter(fs, L, order, this_da_cesm.values, axis=0, btype='low')
                    tmp = this_da_cesm.copy(data=vals)
                    da_metric = tmp.var('year')
                    m2.append(da_metric)

                elif metric_name == 'IQ_range':
                    da_metric = (this_da_cesm.quantile(q=0.75, dim='year') -
                                 this_da_cesm.quantile(q=0.25, dim='year'))
                    m3.append(da_metric)

                elif metric_name == 'max_val':
                    da_metric = this_da_cesm.max('year')
                    m4.append(da_metric)

                elif metric_name == '85yr_event':
                    ntime = len(this_da_cesm.year)
                    RI = (ntime + 1)/np.arange(1, ntime + 1)
                    vals_sorted = np.sort(this_da_cesm.values, axis=0)[::-1, :, :]
                    RI_idx = np.argmin(np.abs(RI - 85))
                    event_magnitude = vals_sorted[RI_idx, :, :]
                    da_metric = this_da_cesm[0, :, :].copy(data=event_magnitude)
                    m5.append(da_metric)

        m1 = xr.concat(m1, dim='season')
        m1['season'] = np.array(seasons)
        m2 = xr.concat(m2, dim='season')
        m2['season'] = np.array(seasons)
        m3 = xr.concat(m3, dim='season')
        m3['season'] = np.array(seasons)
        m4 = xr.concat(m4, dim='season')
        m4['season'] = np.array(seasons)
        m5 = xr.concat(m5, dim='season')
        m5['season'] = np.array(seasons)

        cesm = xr.concat(cesm, dim='season')
        cesm['season'] = np.array(seasons)

        all_m1.append(m1)
        all_m2.append(m2)
        all_m3.append(m3)
        all_m4.append(m4)
        all_m5.append(m5)
        all_cesm.append(cesm)

    all_m1 = xr.concat(all_m1, dim='member')
    all_m1['member'] = members
    all_m1 = all_m1.transpose('lat', 'lon', 'member', 'season')

    all_m2 = xr.concat(all_m2, dim='member')
    all_m2['member'] = members
    all_m2 = all_m2.transpose('lat', 'lon', 'member', 'season')

    all_m3 = xr.concat(all_m3, dim='member')
    all_m3['member'] = members
    all_m3 = all_m3.transpose('lat', 'lon', 'member', 'season')

    all_m4 = xr.concat(all_m4, dim='member')
    all_m4['member'] = members
    all_m4 = all_m4.transpose('lat', 'lon', 'member', 'season')

    all_m5 = xr.concat(all_m5, dim='member')
    all_m5['member'] = members
    all_m5 = all_m5.transpose('lat', 'lon', 'member', 'season')

    ds_metrics = ds_metrics.assign({'var_high': (('lat', 'lon', 'member', 'season'), all_m1),
                                    'var_low': (('lat', 'lon', 'member', 'season'), all_m2),
                                    'IQ_range': (('lat', 'lon', 'member', 'season'), all_m3),
                                    'max_val': (('lat', 'lon', 'member', 'season'), all_m4),
                                    '85yr_event': (('lat', 'lon', 'member', 'season'), all_m5)})

    ds_metrics.to_netcdf(cesm_savename)

    # calculate longer return period events across full ensemble
    # return period for each season: across year and member
    all_cesm = xr.concat(all_cesm, dim='member')
    all_cesm['member'] = members
    all_cesm = all_cesm.transpose('lat', 'lon', 'season', 'member', 'year')
    da_size = all_cesm.shape
    vals = all_cesm.values.reshape((da_size[0], da_size[1], da_size[2], da_size[3]*da_size[4]))
    vals_sorted = np.sort(vals, axis=-1)[:, :, :, ::-1]
    ntime = da_size[3]*da_size[4]
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

# Obs-LE analysis
obsle_savename = '/glade/work/mckinnon/obsLE/proc/obsle_v-%s_5metrics.nc' % version
obsle_savename_ens_extremes = '/glade/work/mckinnon/obsLE/proc/obsle_v-%s_ens_extreme_metrics.nc' % version

if os.path.isfile(obsle_savename):
    ds_metrics_obsle = xr.open_dataset(obsle_savename)
    da_ens_extremes_obsle = xr.open_dataarray(obsle_savename_ens_extremes)
else:
    all_m1 = []
    all_m2 = []
    all_m3 = []
    all_m4 = []
    all_m5 = []
    da_ens_extremes_obsle = []

    for m in members:
        print(m)

        # Get the Obs-LE data
        files = sorted(glob('%s/LE-%03i/pr/pr_member????.nc' % (obsledir, m)))
        da_obsle = xr.open_mfdataset(files, concat_dim='obsle_member', combine='nested')
        da_obsle = da_obsle['pr']
        da_obsle = da_obsle.assign_coords({'lat': np.round(da_obsle.lat, 3)})

        # resample for seasons
        da_obsle = da_obsle.resample(time='QS-DEC').mean()

        if m == 1:  # create our metrics dataset
            ds_metrics_obsle = xr.Dataset(coords={'lat': da_obsle.lat, 'lon': da_obsle.lon,
                                                  'member': members, 'season': np.array(seasons)})
        m1 = []
        m2 = []
        m3 = []
        m4 = []
        m5 = []
        this_da_ens_extremes_obsle = []
        for this_season in seasons:

            # pull out desired season
            this_da_obsle = da_obsle.sel({'time': da_obsle['time.season'] == this_season})
            # and desired year
            if this_season == 'DJF':  # year is associated with december
                this_da_obsle = this_da_obsle.sel({'time': np.isin(this_da_obsle['time.year'], valid_years - 1)})
            else:
                this_da_obsle = this_da_obsle.sel({'time': np.isin(this_da_obsle['time.year'], valid_years)})

            this_da_obsle = this_da_obsle.load()

            for metric_name in metrics:
                if metric_name == 'var_high':
                    vals = olens_utils.lowpass_butter(fs, L, order, this_da_obsle.values, axis=0, btype='high')
                    tmp = this_da_obsle.copy(data=vals)
                    da_metric = (tmp.var('time')).mean('obsle_member')
                    m1.append(da_metric)

                elif metric_name == 'var_low':
                    vals = olens_utils.lowpass_butter(fs, L, order, this_da_obsle.values, axis=0, btype='low')
                    tmp = this_da_obsle.copy(data=vals)
                    da_metric = (tmp.var('time')).mean('obsle_member')
                    m2.append(da_metric)

                elif metric_name == 'IQ_range':
                    da_metric = (this_da_obsle.quantile(q=0.75, dim='time') -
                                 this_da_obsle.quantile(q=0.25, dim='time')).mean('obsle_member')
                    m3.append(da_metric)

                elif metric_name == 'max_val':
                    da_metric = (this_da_obsle.max('time')).max('obsle_member')
                    m4.append(da_metric)

                elif metric_name == '85yr_event':  # calculate across all members of the Obs-LE
                    da_size = this_da_obsle.shape
                    # group time and obs-le member together
                    vals = this_da_obsle.values.reshape((da_size[0]*da_size[1], da_size[2], da_size[3]))
                    vals_sorted = np.sort(vals, axis=0)[::-1, :, :]
                    ntime = da_size[0]*da_size[1]
                    RI = (ntime + 1)/np.arange(1, ntime + 1)
                    RI_idx = np.argmin(np.abs(RI - 85))
                    event_magnitude = vals_sorted[RI_idx, :, :]
                    da_metric = this_da_obsle[0, 0, :, :].copy(data=event_magnitude)
                    m5.append(da_metric)

                    # also want to calculate the other return periods to compare to the CESM1-LE
                    da_RI = []
                    for r in return_periods_save:
                        RI_idx = np.argmin(np.abs(RI - r))
                        event_magnitude = vals_sorted[RI_idx, :, :]
                        da_tmp = this_da_obsle[0, 0, :, :].copy(data=event_magnitude)
                        da_RI.append(da_tmp)

                    da_RI = xr.concat(da_RI, dim='return_period')
                    da_RI['return_period'] = return_periods_save
                    this_da_ens_extremes_obsle.append(da_RI)

        m1 = xr.concat(m1, dim='season')
        m1['season'] = np.array(seasons)
        m2 = xr.concat(m2, dim='season')
        m2['season'] = np.array(seasons)
        m3 = xr.concat(m3, dim='season')
        m3['season'] = np.array(seasons)
        m4 = xr.concat(m4, dim='season')
        m4['season'] = np.array(seasons)
        m5 = xr.concat(m5, dim='season')
        m5['season'] = np.array(seasons)

        this_da_ens_extremes_obsle = xr.concat(this_da_ens_extremes_obsle, dim='season')
        this_da_ens_extremes_obsle['season'] = np.array(seasons)

        all_m1.append(m1)
        all_m2.append(m2)
        all_m3.append(m3)
        all_m4.append(m4)
        all_m5.append(m5)
        da_ens_extremes_obsle.append(this_da_ens_extremes_obsle)

    all_m1 = xr.concat(all_m1, dim='member')
    all_m1['member'] = members
    all_m1 = all_m1.transpose('lat', 'lon', 'member', 'season')

    all_m2 = xr.concat(all_m2, dim='member')
    all_m2['member'] = members
    all_m2 = all_m2.transpose('lat', 'lon', 'member', 'season')

    all_m3 = xr.concat(all_m3, dim='member')
    all_m3['member'] = members
    all_m3 = all_m3.transpose('lat', 'lon', 'member', 'season')

    all_m4 = xr.concat(all_m4, dim='member')
    all_m4['member'] = members
    all_m4 = all_m4.transpose('lat', 'lon', 'member', 'season')

    all_m5 = xr.concat(all_m5, dim='member')
    all_m5['member'] = members
    all_m5 = all_m5.transpose('lat', 'lon', 'member', 'season')

    da_ens_extremes_obsle = xr.concat(da_ens_extremes_obsle, dim='member')
    da_ens_extremes_obsle['member'] = members
    da_ens_extremes_obsle = da_ens_extremes_obsle.transpose('lat', 'lon', 'member', 'season', 'return_period')

    ds_metrics_obsle = ds_metrics_obsle.assign({'var_high': (('lat', 'lon', 'member', 'season'), all_m1),
                                                'var_low': (('lat', 'lon', 'member', 'season'), all_m2),
                                                'IQ_range': (('lat', 'lon', 'member', 'season'), all_m3),
                                                'max_val': (('lat', 'lon', 'member', 'season'), all_m4),
                                                '85yr_event': (('lat', 'lon', 'member', 'season'), all_m5)})

    ds_metrics_obsle.to_netcdf(obsle_savename)
    da_ens_extremes_obsle.to_netcdf(obsle_savename_ens_extremes)
