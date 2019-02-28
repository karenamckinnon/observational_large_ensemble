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


def plot_spectra(P, s, ci, savename=None):
    """Make, display, and optionally save power spectrum plot.

    Parameters
    ----------
    P : numpy array
        PSD estimate
    s : numpy array
        Associated frequencies
    ci : numpy array
        Associated confidence interval
    savename : str
        Full filepath to save figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis handle
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

    ax.fill_between(s, P*ci[:, 0], P*ci[:, -1], color='lightgray', alpha=0.5, lw=0)
    ax.plot(s, P)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(np.min(s), np.max(s))
    ax.set_xticks([50**-1, 20**-1, 10**-1, 5**-1, 2**-1])
    ax.set_xticklabels(['50', '20', '10', '5', '2'])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Period (yrs)', fontsize=20)
    plt.ylabel('Power density', fontsize=20)

    if savename is not None:
        plt.savefig(savename)

    return fig, ax


def iaaft(x):
    """Return a surrogate time series based on IAAFT.

    Parameters
    ----------
    x : numpy array
        Original time series

    Returns
    -------
    x_new : numpy array
        Surrogate time series
    """

    xbar = np.mean(x)
    x -= xbar  # remove mean
    rank = np.argsort(x)
    x_sort = x[rank]

    I_k = np.abs(np.fft.fft(x))
    x_new = np.random.choice(x, len(x), replace=False)

    delta_criterion = 1
    criterion_new = 100
    while delta_criterion > 1e-8:
        criterion_old = criterion_new
        # iteration 1: spectral adjustment
        x_old = x_new
        x_fourier = np.fft.fft(x_old)
        adjusted_coeff = I_k*x_fourier/np.abs(x_fourier)
        x_new = np.fft.ifft(adjusted_coeff)

        # iteration 2: amplitude adjustment
        x_old = x_new
        index = np.argsort(np.real(x_new))
        x_new[index] = x_sort

        criterion_new = 1/np.std(x)*np.sqrt(1/len(x)*np.sum((I_k - np.abs(x_fourier))**2))
        delta_criterion = np.abs(criterion_new - criterion_old)

    x_new += xbar
    x_new = np.real(x_new)

    return x_new


def create_matched_surrogates_1d(x, y):
    """Create surrogate time series with enforced empirical coherence.

    In the spectral domain, the model is
    yhat = ahat*xhat + nhat

    Thus, results will differ if x and y are switched.

    The surrogate time series are produced via IAAFT.

    Parameters
    ----------
    x : numpy array
        The first (independent) time series
    y : numpy array
        The second (dependent) time series

    Returns
    -------
    new_x : numpy array
        A surrogate version of x
    new_y : numpy array
        A surrogate version of y
    """

    xhat = np.fft.fft(x)
    yhat = np.fft.fft(y)

    Phi_xx = np.dot(xhat[np.newaxis, :], np.conj(xhat[:, np.newaxis]))/len(xhat)
    Phi_xy = np.dot(yhat[np.newaxis, :], np.conj(xhat[:, np.newaxis]))/len(xhat)

    ahat = Phi_xy/Phi_xx
    nhat = yhat - ahat*xhat

    # Get new estimate of x
    x_surr = iaaft(x)
    # Fourier transform
    x_surr_hat = np.fft.fft(x_surr)
    # Get part of y that is coherent
    y_coherent_hat = ahat*x_surr_hat

    # Return to time domain
    y_surr = np.real(np.fft.ifft(y_coherent_hat).flatten())

    # Add random version of n
    n_surr = iaaft(np.fft.ifft(nhat).flatten())

    new_x = x_surr
    new_y = y_surr + n_surr

    return new_x, new_y


def save_2d_netcdf(lat, lon, vals, varname, units, savename, description):
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

    Returns
    -------
    Nothing.

    """
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
