"""A set of common utilities for calculation of the Observational Large Ensemble.

"""
from scipy import signal
from netCDF4 import Dataset
import numpy as np
from glob import glob
import pandas as pd


def smooth(data, M):
    """Smooth the 1D time series data with a Hann window of length M.

    Parameters
    ----------
    data : numpy array
        One dimensional array to be smoothed
    M : int
        The length of the smoothing window. If odd, will be increased by 1.

    Returns
    -------
    smoothed : numpy array
        Smoothed version of data
    valid_indices : numpy array
        Indices associated with valid values. Can be used to subset the original time dimension for time series.

    """

    if M % 2 == 0:  # even window
        M += 1

    window = signal.windows.hann(M=M)
    window /= window.sum()

    smoothed = np.convolve(data, window, mode='valid')
    valid_indices = np.arange(len(data))[int(M/2):-int(M/2)]

    return smoothed, valid_indices


def forced_trend(varname, cvdp_loc):
    """Calculate the global mean, ensemble mean trend across the CESM Large Ensemble.

    This serves to provide the temporal shape for the estimated forced component.
    The methodology follows Dai et al (2015).

    This script draws upon the output from the CVDP as applied to the CESM1 LE. A tar file containing the
    CVDP output is available on cheyenne: /gpfs/fs1/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/CVDP/

    Parameters
    ----------
    varname : str
        Variable to calculate the EM, GM trend for (tas, pr, slp)
    cvdp_loc : str
        Full path and filename for where the CVDP output has been untarred.

    Returns
    -------
    gm_em : numpy array
        Time series of global mean, ensemble mean (across NCAR CESM1) of desired variable
    gm_em_units : str
        Units of the GM, EM variable
    time : numpy array
        Time associated with above time series
    time_units : str
        Units of the time variable

    """
    if varname not in list(('tas', 'pr', 'slp')):
        print('Variable %s not currently supported' % varname)
        return

    if not cvdp_loc.endswith('/'):
        cvdp_loc = cvdp_loc + '/'

    if varname == 'slp':
        # Need to explicitly calculate GM, EM
        print('Need to code this')

    else:
        # Can use CVDP output
        fnames = sorted(glob('%sCESM1-CAM5-BGC-LE_*.cvdp_data.1920-2017.nc' % cvdp_loc))

        cvdp_name = '%s_global_avg_mon' % varname

        nfiles = len(fnames)
        ds = Dataset(fnames[0], 'r')
        time = ds['time'][:]
        time_units = ds['time'].units
        gm_em_units = ds[cvdp_name].units

        n = len(time)
        glob_mean = np.empty((nfiles, n))
        for counter, file in enumerate(fnames):
            ds = Dataset(file, 'r')
            glob_mean[counter, :] = ds[cvdp_name][:]

        # Take average across ensemble members
        gm_em = np.mean(glob_mean, axis=0)

    return gm_em, gm_em_units, time, time_units


def create_mode_df(fname):
    """Return a dataframe with the mode time series (unfiltered) and preprocessed time columns.

    Parameters
    ----------
    fname : str
        Full path to file containing mode time series. Originally from CVDP.

    Returns
    -------
    df : pandas dataframe
        Dataframe with three modes and time variable.
    """

    ds = Dataset(fname, 'r')
    time = ds['time'][:]
    month = (time + 1) % 12
    month[month == 0] += 12
    month = month.compressed().astype(int)  # nothing is actually masked
    season = [int(m % 12 + 3)//3 for m in month]
    season_strs = ['DJF', 'MAM', 'JJA', 'SON']
    season_names = [season_strs[counter - 1] for counter in season]
    year = np.floor(1920 + (np.arange(1, len(month) + 1) - 0.5)/12).astype(int)

    amo_ts = ds['amo_timeseries_mon'][:]

    pdo_ts = ds['pdo_timeseries_mon'][:]

    enso_ts = ds['nino34'][:]

    df = pd.DataFrame(columns=['year', 'month', 'season', 'AMO', 'PDO', 'ENSO'])
    df = df.assign(year=year, month=month, season=season_names,
                   AMO=amo_ts, PDO=pdo_ts, ENSO=enso_ts)

    return df
