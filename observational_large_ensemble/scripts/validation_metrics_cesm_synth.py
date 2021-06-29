# Analysis
import xarray as xr
import numpy as np

# General
from glob import glob
import os

# My codes
import observational_large_ensemble.utils as olens_utils


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('member', type=int, help='Member number of the CESM1-synth to analyze')
    args = parser.parse_args()

    m = args.member
    version = 'noF'
    metrics = 'var_low', 'var_high', 'IQ_range', '150yr_event', '55yr_event', '33yr_event'
    seasons = 'DJF', 'JJA'
    members = np.hstack((np.arange(1, 36), np.arange(101, 106)))
    valid_years = np.arange(1921, 2006)

    # for filter
    fs = 1  # per year
    L = 1/10.  # above or below decadal
    order = 3

    # Gringorten formula for return periods
    b = 0.44

    obsledir = '/glade/scratch/mckinnon/obsLE/output_v-%s' % version
    savedir = '/glade/work/mckinnon/obsLE/proc'

    return_periods_save = np.array([33, 55, 150, 300, 500])

    # Obs-LE analysis
    obsle_savename = '%s/cesm_synth_v-%s_5metrics_member_%03i.nc' % (savedir, version, m)
    obsle_savename_ens_extremes = '%s/cesm_synth_v-%s_ens_extreme_metrics_member_%03i.nc' % (savedir, version, m)

    if os.path.isfile(obsle_savename):
        ds_metrics_obsle = xr.open_dataset(obsle_savename)
        da_ens_extremes_obsle = xr.open_dataarray(obsle_savename_ens_extremes)
    else:

        # Get the Obs-LE data
        files = sorted(glob('%s/LE-%03i/pr/pr_member????.nc' % (obsledir, m)))
        da_obsle = xr.open_mfdataset(files, concat_dim='obsle_member', combine='nested')
        da_obsle = da_obsle['pr']
        da_obsle = da_obsle.assign_coords({'lat': np.round(da_obsle.lat, 3)})

        # resample for seasons
        da_obsle = da_obsle.resample(time='QS-DEC').mean()

        da_metrics_seasons = []
        da_ens_extremes_obsle = []
        for this_season in seasons:
            print(this_season)
            # pull out desired season
            this_da_obsle = da_obsle.sel({'time': da_obsle['time.season'] == this_season})
            this_da_obsle['time'] = this_da_obsle['time.year']
            this_da_obsle = this_da_obsle.rename({'time': 'year'})
            if this_season == 'DJF':  # year is associated with december
                this_da_obsle = this_da_obsle.assign_coords({'year': this_da_obsle['year'].values + 1})

            # and desired year
            this_da_obsle = this_da_obsle.sel({'year': np.isin(this_da_obsle['year'], valid_years)})

            # transpose to have time, member, lat, lon
            this_da_obsle = this_da_obsle.transpose('year', 'obsle_member', 'lat', 'lon')

            this_da_obsle = this_da_obsle.load()
            da_metrics = []

            # Do some prep for return period calculations
            da_size = this_da_obsle.shape
            ntime = da_size[0]*da_size[1]  # combine year and ensemble to estimate extremes
            RI = (ntime + 1 - 2*b)/(np.arange(1, ntime + 1) - b)
            vals = this_da_obsle.values.reshape((da_size[0]*da_size[1], da_size[2], da_size[3]))
            vals_sorted = np.sort(vals, axis=0)[::-1, :, :]

            for metric_name in metrics:
                print(metric_name)
                if 'yr_event' in metric_name:  # want to append time and ensemble to get estimate
                    return_period = int(metric_name.split('yr')[0])
                    # group time and obs-le member together
                    RI_idx = np.argmin(np.abs(RI - return_period))
                    event_magnitude = vals_sorted[RI_idx, :, :]
                    da_metric = this_da_obsle[0, 0, :, :].copy(data=event_magnitude)
                else:
                    da_metric = olens_utils.calc_variability_metrics(this_da_obsle, metric_name)
                    da_metric = da_metric.mean('obsle_member')

                da_metrics.append(da_metric)

            da_metrics = xr.concat(da_metrics, dim='metric_name')
            da_metrics['metric_name'] = np.array(metrics)
            da_metrics_seasons.append(da_metrics)

            # also want to calculate the other return periods to compare to the CESM1-LE
            da_RI = []
            for r in return_periods_save:
                RI_idx = np.argmin(np.abs(RI - r))
                event_magnitude = vals_sorted[RI_idx, :, :]
                da_tmp = this_da_obsle[0, 0, :, :].copy(data=event_magnitude)
                da_RI.append(da_tmp)

            da_RI = xr.concat(da_RI, dim='return_period')
            da_RI['return_period'] = return_periods_save
            da_ens_extremes_obsle.append(da_RI)

        da_metrics_seasons = xr.concat(da_metrics_seasons, dim='season')
        da_metrics_seasons['season'] = np.array(seasons)

        da_metrics_seasons = da_metrics_seasons.to_dataset(dim='metric_name')
        da_metrics_seasons = da_metrics_seasons.transpose('lat', 'lon', 'season')

        da_ens_extremes_obsle = xr.concat(da_ens_extremes_obsle, dim='season')
        da_ens_extremes_obsle['season'] = np.array(seasons)

        da_metrics_seasons.to_netcdf(obsle_savename)
        da_ens_extremes_obsle.to_netcdf(obsle_savename_ens_extremes)
