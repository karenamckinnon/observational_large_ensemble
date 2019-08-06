"""A set of common utilities for calculation of the Observational Large Ensemble.

"""
from scipy import signal
from netCDF4 import Dataset
import numpy as np
from glob import glob
import pandas as pd
import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import calendar
from datetime import timedelta


def lowpass_butter(fs, lowcut, order,  data, axis=-1):
    """Perform a lowpass butterworth filter on data using a forward and backward digital filter.

    Parameters
    ----------
    fs : float
        Sampling frequency of data (example: 12 for monthly data)
    lowcut : float
        Critical frequency for Butterworth filter. See scipy docs.
    order : int
        Order of filter. Note that filtfilt doubles the original filter order.
    data : numpy array
        1D vector or 2D array to be filtered
    axis : int
        Axis along which filtering is performed.

    Returns
    -------
    data_filtered : numpy array
        Filtered data

    """
    from scipy.signal import butter, filtfilt

    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')  # Coefficients for Butterworth filter
    filtered = filtfilt(b, a, data, axis=axis)

    return filtered


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


def create_mode_df(fname, AMO_cutoff_freq):
    """Return a dataframe with the mode time series (unfiltered) and preprocessed time columns.

    Parameters
    ----------
    fname : str
        Full path to file containing mode time series. Originally from CVDP.
    AMO_cutoff_freq : float
        Cut off frequency for Butterworth filter of AMO (1/years)

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

    # Create version of PDO that is orthogonal to ENSO using Gram-Schmidt method
    pdo_orth = pdo_ts - np.dot(pdo_ts, enso_ts)/np.dot(enso_ts, enso_ts)*enso_ts

    # Perform lowpass filter on AMO
    if AMO_cutoff_freq > 0:
        amo_lowpass = lowpass_butter(12, AMO_cutoff_freq, 3, amo_ts)
    else:  # no filter
        amo_lowpass = amo_ts

    # Set mode time series to unit sigma
    enso_ts /= np.std(enso_ts)
    pdo_orth /= np.std(pdo_orth)
    pdo_ts /= np.std(pdo_ts)
    amo_ts /= np.std(amo_ts)
    amo_lowpass /= np.std(amo_lowpass)

    df = pd.DataFrame(columns=['year', 'month', 'season', 'AMO', 'AMO_lowpass', 'PDO', 'ENSO', 'PDO_orth'])
    df = df.assign(year=year, month=month, season=season_names,
                   AMO=amo_ts, AMO_lowpass=amo_lowpass, PDO=pdo_ts, ENSO=enso_ts, PDO_orth=pdo_orth)

    return df


def pmtm(x, dt, nw=3, cl=0.95):
    """Returns Thomson’s multitaper power spectral density (PSD) estimate, pxx, of the input signal, x.

    Slightly modified from Peter Huybers's matlab code, pmtmPH.m

    Parameters
    ----------
    x : numpy array
        Time series to analyze
    dt : float
        Time step
    nw : float
        The time-halfbandwidth product
    cl : float
        Confidence interval to calculate and display

    Returns
    -------
    P : numpy array
        PSD estimate
    s : numpy array
        Associated frequencies
    ci : numpy array
        Associated confidence interval

    """
    from scipy.signal import windows

    nfft = np.shape(x)[0]

    nx = np.shape(x)[0]
    k = min(np.round(2.*nw), nx)
    k = int(max(k-1, 1))
    s = np.arange(0, 1/dt, 1/(nfft*dt))

    # Compute the discrete prolate spheroidal sequences
    [E, V] = windows.dpss(nx, nw, k, return_ratios=True)
    E = E.T

    # Compute the windowed DFTs.
    Pk = np.abs(np.fft.fft(E*x[:, np.newaxis], nfft, axis=0))**2

    if k > 1:
        sig2 = np.dot(x[np.newaxis, :], x[:, np.newaxis])[0][0]/nx
        # initial spectrum estimate
        P = ((Pk[:, 0] + Pk[:, 1])/2)[:, np.newaxis]
        Ptemp = np.zeros((nfft, 1))
        P1 = np.zeros((nfft, 1))
        tol = .0005*sig2/nfft
        a = sig2*(1-V)

        while (np.sum(np.abs(P - P1)/nfft) > tol):
            b = np.repeat(P, k, axis=-1)/(P*V[np.newaxis, :] + np.ones((nfft, 1))*a[np.newaxis, :])
            wk = (b**2) * (np.ones((nfft, 1))*V[np.newaxis, :])
            P1 = (np.sum(wk*Pk, axis=-1)/np.sum(wk, axis=-1))[:, np.newaxis]

            Ptemp = np.empty_like(P1)
            Ptemp[:] = P1
            P1 = np.empty_like(P)
            P1[:] = P
            P = np.empty_like(Ptemp)
            P[:] = Ptemp

        # Determine equivalent degrees of freedom, see Percival and Walden 1993.
        v = ((2*np.sum((b**2)*(np.ones((nfft, 1))*V[np.newaxis, :]), axis=-1)**2) /
             np.sum((b**4)*(np.ones((nfft, 1))*V[np.newaxis, :]**2), axis=-1))

    else:
        P = np.empty_like(Pk)
        P[:] = Pk
        v = 2*np.ones((nfft, 1))

    select = (np.arange(0, (nfft + 1)/2.)).astype('int')
    P = P[select].flatten()
    s = s[select].flatten()
    v = v[select].flatten()

    # Chi-squared 95% confidence interval
    # approximation from Chamber's et al 1983; see Percival and Walden p.256, 1993
    ci = np.empty((np.shape(v)[0], 2))
    ci[:, 0] = 1./(1-2/(9*v) - 1.96*np.sqrt(2/(9*v)))**3
    ci[:, 1] = 1./(1-2/(9*v) + 1.96*np.sqrt(2/(9*v)))**3

    return P, s, ci


def plot_spectra(P, s, ci, plot_ci=True, savename=None, **kwargs):
    """Make, display, and optionally save power spectrum plot.

    Parameters
    ----------
    P : numpy array
        PSD estimate
    s : numpy array
        Associated frequencies
    ci : numpy array
        Associated confidence interval
    plot_ci : bool
        Indicator of whether to plot confidence interval
    savename : str or None
        Full filepath to save figure
    **kwargs : Key word args
        Optional fig, ax if spectrum should be added to existing figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis handle
    """
    import matplotlib.pyplot as plt

    if 'fig' in kwargs:
        fig = kwargs['fig']
        ax = kwargs['ax']
    else:  # creat new figure
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    if plot_ci:
        ax.fill_between(s, P*ci[:, 0], P*ci[:, -1], color='lightgray', alpha=0.5, lw=0)
    ax.plot(s, P)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Frequency', fontsize=20)
    plt.ylabel('Power density', fontsize=20)

    if savename is not None:
        plt.savefig(savename)

    return fig, ax


def iaaft(x, fit_seasonal=False):
    """Return a surrogate time series based on IAAFT.

    Parameters
    ----------
    x : numpy array
        Original time series
    fit_seasonal : bool
        Should the monthly amplitudes be matched? Use True for ENSO.

    Returns
    -------
    x_new : numpy array
        Surrogate time series
    this_iter : int
        Number of iterations until convergence
    """

    # To account for some sampling variability, the seasonal cycle in ENSO variance is calculated
    # with resampling for each surrogate time series
    if fit_seasonal:
        nyrs = int(np.floor(len(x)/12))
        resampled_x = x[:(nyrs*12)].reshape((nyrs, 12))
        idx = np.random.choice(np.arange(nyrs), nyrs, replace=True)
        resampled_x = resampled_x[idx, :]
        seasonal_sigma = np.std(resampled_x, axis=0)

    xbar = np.mean(x)
    x -= xbar  # remove mean
    rank = np.argsort(x)
    x_sort = x[rank]

    I_k = np.abs(np.fft.fft(x))
    x_new = np.random.choice(x, len(x), replace=False)

    delta_criterion = 1
    criterion_new = 100
    max_iters = 1000
    this_iter = 0
    while delta_criterion > 1e-8:
        criterion_old = criterion_new
        # iteration 1: spectral adjustment
        x_old = x_new
        x_fourier = np.fft.fft(x_old)
        adjusted_coeff = I_k*x_fourier/np.abs(x_fourier)
        x_new = np.fft.ifft(adjusted_coeff)

        # iteration 2: amplitude adjustment
        x_old = x_new
        index = np.argsort(np.real(x_old))
        x_new[index] = x_sort
        x_new = np.real(x_new)

        # Rescale the seasonal standard deviations to match original data
        if fit_seasonal:
            this_sigma = np.array([np.std(x_new[mo::12]) for mo in range(12)])
            scaling = seasonal_sigma/this_sigma

            for mo in range(12):
                x_new[mo::12] = scaling[mo]*x_new[mo::12]

        criterion_new = 1/np.std(x)*np.sqrt(1/len(x)*np.sum((I_k - np.abs(x_fourier))**2))
        delta_criterion = np.abs(criterion_new - criterion_old)

        if this_iter > max_iters:
            return 0

        this_iter += 1

    x_new += xbar
    x_new = np.real(x_new)

    return x_new, this_iter


def save_2d_netcdf(lat, lon, vals, varname, units, savename, description, overwrite=False):
    """Save a two-dim (lat, lon) variable to netcdf.

    Parameters
    ----------
    lat : numpy array
        Latitude (degrees north)
    lon : numpy array
        Longitude (degrees east)
    vals : numpy array
        Data to be saved
    units : str
        Name of units for vals
    savename : str
        Full path to location where values should be saved
    description : str
        Short description of what is saved in the netcdf
    overwrite : boolean
        Indicator of whether or not to overwrite existing file

    Returns
    -------
    Nothing.

    """
    fileexists = os.path.isfile(savename)
    if (fileexists & overwrite) | (not fileexists):

        if fileexists:  # remove if we want to create a new version
            os.remove(savename)

        fout = Dataset(savename, 'w')

        nlat = len(lat)
        nlon = len(lon)
        fout.createDimension('lat', nlat)
        fout.createDimension('lon', nlon)

        latnc = fout.createVariable('lat', 'f8', ('lat',))
        lonnc = fout.createVariable('lon', 'f8', ('lon',))
        varnc = fout.createVariable(varname, 'f8', ('lat', 'lon'))

        fout.description = description

        latnc.units = 'degree_north'
        lonnc.units = 'degree_east'
        varnc.units = units

        latnc[:] = lat
        lonnc[:] = lon
        varnc[:, :] = vals

        fout.close()

    return


def shift_df(df, shift, shift_names):
    """Perform an offset between two sets of columns in a dataframe.

    Parameters
    ---------
    df : pandas dataframe
        The dataframe to shift
    shift : positive integer
        The number of offsets between the two column sets.
    shift_names : list
        Column names to shift forward. If you want to shift columns backwards, pass the complement to that set here.

    Returns
    -------
    df_shifted : pandas dataframe
        The shifted dataframe

    """

    other_names = [name for name in df.columns if name not in shift_names]

    df1 = df.loc[:, shift_names].drop(df.head(shift).index)
    df2 = df.loc[:, other_names].drop(df.tail(shift).index)
    df2.index += shift  # need to match index, otherwise concat will ignore offset
    new_df = pd.concat((df1, df2), axis=1, ignore_index=True, join='inner')
    new_df.columns = shift_names + other_names
    del df1, df2
    df_shifted = new_df
    del new_df

    # Reset index
    df_shifted.reset_index(inplace=True)
    df_shifted = df_shifted.drop(['index'], axis=1)

    return df_shifted


def plot_sst_patterns(lat, lon, beta, ice_loc, modename, savename=None):
    """Make global maps of SST anomalies regressed on mode time series.

    Parameters
    ----------
    lat : xarray.DataArray or numpy.ndarray
        Latitude values
    lon : xarray.DataArray or numpy.ndarray
        Longitude values
    beta : numpy.ndarray or numpy.matrix
        Regression coefficients to plot (nlat x nlon or nlat*nlon)
    ice_loc : xarray.DataArray or numpy.ndarray
        Indicator map of where ice is present (nlat x nlon)
    modename : str
        Name of SST mode
    savename : str or None
        Full path and filename if the plot should be saved

    Returns
    -------
    Nothing if savename is not None. Will show plot if savename is None.

    """

    nlat = len(lat)
    nlon = len(lon)

    beta = beta.reshape((nlat, nlon))
    beta[ice_loc] = np.nan

    # Create xarray dataset
    ds_beta = xr.Dataset(data_vars={'beta': (('latitude', 'longitude'), beta.reshape((nlat, nlon)))},
                         coords={'latitude': lat,
                                 'longitude': lon})

    fig = plt.figure(figsize=(20, 8))
    fig.tight_layout()

    if modename == 'AMO':
        proj = ccrs.PlateCarree(central_longitude=0)
    else:
        # shift longitude for plotting so there isn't a white line in the middle
        ds_beta = ds_beta.assign_coords(longitude=(ds_beta.longitude + 360) % 360)
        ds_beta = ds_beta.sortby('longitude')
        proj = ccrs.PlateCarree(central_longitude=180)

    ax = plt.subplot(111, projection=proj)
    to_plot = ds_beta['beta']

    to_plot.plot.contourf(ax=ax, transform=ccrs.PlateCarree(),
                          levels=np.arange(-0.6, 0.7, 0.1),
                          extend='both')

    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=16)
    cax.set_ylabel(r'SST anomalies ($^\circ$C)', fontsize=16)

    ax.add_feature(cfeature.LAND, color='lightgray', edgecolor='k')
    ax.add_feature(cfeature.LAKES, color='lightgray', edgecolor=None)
    ax.coastlines()
    ax.set_global()

    if savename is not None:
        plt.savefig(savename, dpi=300, orientation='landscape', bbox_inches='tight')
        plt.close()


def get_obs(this_varname, this_filename, valid_years, mode_lag, cvdp_file, AMO_cutoff_freq, name_conversion):
    """Return observational data and associated time series of modes for a given variable.

    Parameters
    ----------
    this_varname : str
        Standard variable name (tas, pr, or slp)
    this_filename : str
        Full path to data file
    valid_years : numpy.ndarray
        Set of years to pull from file
    mode_lag : int
        Number of months to lag the climate variable response from the mode time series
    cvdp_file : str
        Full path to CVDP data
    AMO_cutoff_freq : float
        Cut off frequency for Butterworth filter of AMO (1/years)
    name_conversion : dict
        Mapping from standard names to names in specific data sources

    Returns
    -------
    dsX : xarray.Dataset
        Dataset containing subset of variable of interest
    df_shifted : pandas.Dataframe
        Dataframe with time series of modes and forced component, shifted by "mode_lag" months
    df : pandas.Dataframe
        Unshifted version of df_shifted

    """

    # Get the forced component
    cvdp_loc = '/'.join(cvdp_file.split('/')[:-1])
    # Assume that the global mean trend in sea level is zero
    if this_varname == 'slp':
        gm_em, gm_em_units, time, time_units = forced_trend('tas', cvdp_loc)
        gm_em *= 0
        gm_em += 1  # will replace constant
    else:
        gm_em, gm_em_units, time, time_units = forced_trend(this_varname, cvdp_loc)

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
    df = create_mode_df(cvdp_file, AMO_cutoff_freq)

    # Add EM, GM time series to it
    df = df.assign(F=gm_em)

    # Shift modes in time
    df_shifted = shift_df(df, mode_lag, ['year', 'month', 'season', 'F'])

    # Subset to valid years
    subset = np.isin(df_shifted['year'].values, valid_years)
    df_shifted = df_shifted.loc[subset, :]

    # Reset the forced trend time series to a mean of zero
    # This allows for the forced trend to be straightforwardly added in later
    F = df_shifted['F'].values
    F -= np.mean(F)
    df_shifted = df_shifted.assign(F=F)

    # Load dataset
    if isinstance(this_filename, str):  # Observational data
        ds = xr.open_dataset(this_filename)
    else:  # CESM data
        if this_varname == 'pr':  # CESM splits up precipitation into convective and large scale
            ds = xr.open_mfdataset(this_filename)
            this_filename2 = [f.replace('PRECC', 'PRECL') for f in this_filename]
            ds2 = xr.open_mfdataset(this_filename2)
            # CESM output saved with one day delay, so need to move back
            ds2 = ds2.assign_coords(time=ds2.time-timedelta(days=1))
        else:
            ds = xr.open_mfdataset(this_filename)

        # CESM output saved with one day delay, so need to move back
        ds = ds.assign_coords(time=ds.time-timedelta(days=1))

    # Load data
    try:
        lat = ds['latitude'].values
        lon = ds['longitude'].values
    except KeyError:
        lat = ds['lat'].values
        lon = ds['lon'].values
    try:
        X = ds[this_varname]
        X_units = ds[this_varname].units
    except KeyError:
        alt_name = name_conversion[this_varname]
        X = ds[alt_name]
        X_units = ds[alt_name].units

    # Pull out values, since we'll be permuting the data / changing units, etc
    # For CESM1-LE precipitation, need to add up convective and large scale
    if name_conversion[this_varname] == 'PRECC':
        X = X.values + ds2.PRECL.values
    else:
        X = X.values

    X_time = ds['time']
    if 'units' in ds['time'].attrs:  # nonstandard, from BEST
        assert ds['time'].units == 'year A.D.'
        X_year = np.floor(X_time)
        X_month = (np.ceil((X_time - X_year)*12)).astype(int)
    else:
        X_year = ds['time.year']
        X_month = ds['time.month']

    # Change units if necessary
    if X_units == 'K':
        # convert to celsius
        X -= 273.15
        X_units = 'deg C'
    elif X_units == 'm/s':
        # convert to mm (total over month)
        days_per_month = [calendar.monthrange(int(y), int(m))[1] for y, m in zip(X_year, X_month)]
        seconds_per_month = 60*60*24*np.array(days_per_month)
        X *= seconds_per_month[:, np.newaxis, np.newaxis]  # m per month
        X *= 1000  # mm per month
        X_units = 'mm'

    # Check unit consistency
    if this_varname == 'slp':
        assert X_units == 'Pa'
    if this_varname == 'pr':
        assert X_units == 'mm'

    if 'climatology' in ds.variables:
        climo = ds['climatology'].values
        # Add climatology to X
        for counter, this_month in enumerate(X_month):
            X[counter, ...] += climo[this_month - 1, ...]

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
                     attrs={'units': X_units})

    return dsX, df_shifted, df


def choose_block(parameter_dir, varnames, percentile_threshold=97):
    """Calculate a block size for all variables, months, and locations using the Wilks (1997) JClim formula.

    Parameters
    ----------
    parameter_dir : str
        Parent directory for parameter files
    varnames : list
        List of (standard) variable names to be considered, i.e. ['tas', 'pr', 'slp']
    percentile_threshold : float
        The percentile of estimated blocks to use universally.

    Returns
    -------
    block_use : int
        Suggested block size in years
    block_use_mo : int
        Suggested block size in months

    """

    # Initialize with smallest block (in years)
    block_use = 1

    for this_varname in varnames:
        this_dir = '%s/%s' % (parameter_dir, this_varname)
        fname = '%s/residual.nc' % this_dir

        ds = xr.open_dataset(fname)
        _, nlat, nlon = np.shape(ds[this_varname])

        has_data = ~np.isnan(ds[this_varname][-1, ...].values)

        datavec = ds[this_varname].values[:, has_data]

        # We want to know the extent to which there is year-to-year memory (not seasonal)
        # Calculate block size for each month, gridbox

        ntime, nbox = np.shape(datavec)

        block_est = np.empty((12, nbox))

        def rhs(L):
            return (n - L + 1)**((2/3)*(1-n_eff/n))

        for i in range(12):
            for j in range(nbox):
                this_ts = datavec[i::12, j]
                # estimate rho
                rho = np.corrcoef(this_ts[1:], this_ts[:-1])[0, 1]
                n = len(this_ts)

                # Wilks equation is implicit, so need to solve iteratively
                n_eff = n*(1 - rho)/(1 + rho)
                # As per Wilks 1997, start with a guess of L = sqrt(n)
                L = int(np.sqrt(n))

                while L > rhs(L):
                    L -= 1

                while L < rhs(L):
                    L += 1

                if L > rhs(L):
                    L -= 1

                block_est[i, j] = L

        new_block = np.percentile(block_est.flatten(), percentile_threshold)
        if new_block > block_use:
            block_use = new_block

    print('Block size: %d years' % block_use)
    block_use_mo = block_use*12  # switch to months

    return block_use, block_use_mo
