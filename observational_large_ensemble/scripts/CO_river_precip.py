# Analysis
import xarray as xr
import numpy as np
from datetime import timedelta

# General
from glob import glob
import os

# Geometry
import salem  # noqa
import geojson
from shapely.geometry import shape
from shapely.affinity import translate

version = 'noF'
figdir = '/glade/work/mckinnon/obsLE/figs'
geom_dir = '/glade/u/home/mckinnon/observational_large_ensemble/observational_large_ensemble/shapefiles'
valid_years = np.arange(1921, 2006)
members = np.hstack((np.arange(1, 36), np.arange(101, 106)))
cesmdir = '/gpfs/fs1/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly'
obsledir = '/glade/scratch/mckinnon/obsLE/output_v-%s' % version

co_river_f = '%s/Upper_Colorado_River_Basin_Boundary.json' % geom_dir
with open(co_river_f) as f:
    geo = geojson.load(f)
    co_river_shapely = shape(geo[0]['geometry'])
co_river_shapely_360 = translate(co_river_shapely, xoff=360)

shape_use = co_river_shapely_360
region_name = 'CO_river'

this_season = 'DJF'
this_var = 'pr'
cesm_ts_savename = '/glade/work/mckinnon/obsLE/proc/cesm_%s_%s_precip_ts.nc' % (region_name, this_season)

if os.path.isfile(cesm_ts_savename):
    cesm_ts = xr.open_dataarray(cesm_ts_savename)
else:
    cesm_ts = []
    for m in members:
        print(m)
        files = glob('%s/PREC*/b.e11.B20TRC5CNBDRD.f09_g16.%03i.cam.h0.PREC*.??????-200512.nc' % (cesmdir, m))
        ds_cesm = xr.open_mfdataset(files, concat_dim='precip_type', combine='by_coords')
        da_cesm = ds_cesm['PRECC'] + ds_cesm['PRECL'] + ds_cesm['PRECSL'] + ds_cesm['PRECSC']
        da_cesm = da_cesm.assign_coords(time=da_cesm.time-timedelta(days=1))
        da_cesm = da_cesm.assign_coords({'lat': np.round(da_cesm.lat, 3)})

        # change to mm /day
        da_cesm *= 1000*24*60*60
        # resample for seasons
        da_cesm = da_cesm.resample(time='QS-DEC').mean()

        this_da_cesm = da_cesm.sel({'time': da_cesm['time.season'] == this_season})
        del da_cesm
        if this_season == 'DJF':  # year is associated with december
            this_da_cesm = this_da_cesm.sel({'time': np.isin(this_da_cesm['time.year'], valid_years - 1)})
        else:
            this_da_cesm = this_da_cesm.sel({'time': np.isin(this_da_cesm['time.year'], valid_years)})

        # subet to CA, and create time series
        nlon = len(this_da_cesm.lon)
        area_weights = np.cos(np.deg2rad(this_da_cesm.lat)).values
        area_weights = np.repeat(area_weights[:, np.newaxis], nlon, axis=-1)
        this_da_cesm = this_da_cesm.to_dataset(name=this_var)
        this_da_cesm = this_da_cesm.assign({'weights': (('lat', 'lon'), area_weights)})
        this_da_cesm = this_da_cesm.salem.roi(geometry=shape_use, crs='wgs84')

        ts = ((this_da_cesm[this_var]*this_da_cesm.weights).sum(['lat', 'lon']) /
              this_da_cesm.weights.sum(['lat', 'lon']))
        del this_da_cesm
        ts = ts.load()
        cesm_ts.append(ts)

    cesm_ts = xr.concat(cesm_ts, dim='member')
    cesm_ts['member'] = members

    cesm_ts.to_netcdf(cesm_ts_savename)


# And time series across Obs-LE
cesm_synth_ts_savename = '/glade/work/mckinnon/obsLE/proc/cesm-synth_v-%s_%s_%s_precip_ts.nc' % (version, region_name,
                                                                                                 this_season)

if os.path.isfile(cesm_synth_ts_savename):
    cesm_synth_ts = xr.open_dataarray(cesm_synth_ts_savename)
else:
    cesm_synth_ts = []
    for m in members:
        print(m)

        # Get the Obs-LE data
        files = sorted(glob('%s/LE-%03i/pr/pr_member????.nc' % (obsledir, m)))
        da_obsle = xr.open_mfdataset(files, concat_dim='obsle_member', combine='nested')
        da_obsle = da_obsle['pr']
        da_obsle = da_obsle.assign_coords({'lat': np.round(da_obsle.lat, 3)})

        # resample for seasons
        da_obsle = da_obsle.resample(time='QS-DEC').mean()

        # pull out desired season
        this_da_obsle = da_obsle.sel({'time': da_obsle['time.season'] == this_season})
        # and desired year
        if this_season == 'DJF':  # year is associated with december
            this_da_obsle = this_da_obsle.sel({'time': np.isin(this_da_obsle['time.year'], valid_years - 1)})
        else:
            this_da_obsle = this_da_obsle.sel({'time': np.isin(this_da_obsle['time.year'], valid_years)})

        # subet to CA, and create time series
        nlon = len(this_da_obsle.lon)
        area_weights = np.cos(np.deg2rad(this_da_obsle.lat)).values
        area_weights = np.repeat(area_weights[:, np.newaxis], nlon, axis=-1)
        this_da_obsle = this_da_obsle.to_dataset(name=this_var)
        this_da_obsle = this_da_obsle.assign({'weights': (('lat', 'lon'), area_weights)})
        this_da_obsle = this_da_obsle.salem.roi(geometry=shape_use, crs='wgs84')

        ts = ((this_da_obsle[this_var]*this_da_obsle.weights).sum(['lat', 'lon']) /
              this_da_obsle.weights.sum(['lat', 'lon']))

        del this_da_obsle
        ts = ts.load()
        cesm_synth_ts.append(ts)

    cesm_synth_ts = xr.concat(cesm_synth_ts, dim='member')
    cesm_synth_ts['member'] = members

    cesm_synth_ts.to_netcdf(cesm_synth_ts_savename)
