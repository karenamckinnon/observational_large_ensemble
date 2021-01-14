# Analysis
import xarray as xr
import numpy as np
from scipy.signal import butter, sosfiltfilt
from glob import glob
from datetime import timedelta
from observational_large_ensemble import utils as olens_utils


procdir = '/glade/work/mckinnon/obsLE/proc'


def LFCA(da, N=30, L=1/10, fs=12, order=3, landmask=None, monthly=True):
    """Perform LFCA (as per Wills et al, 2018, GRL) on a dataarray.

    Parameters
    ----------
    da : xarray.DataArray
        Data to perform LFCA on (time x lat x lon)
    N : int
        Number of EOFs to retain
    L : float
        Cutoff frequency for lowpass filter (e.g. 1/10 for per decade)
    fs : float
        Sampling frequency (1/12 for monthly)
    order : int
        Order of the Butterworth filter
    landmask : xarray.DataArray or None
        If None, do not perform any masking
        If DataArray, indicates land locations
    monthly : bool
        If True, perform lowpass filtering for each month separately

    Returns
    -------
    LFPs : numpy.ndarray
        2D array of N spatial patterns (nlat*nlon x N)
    LFCs : numpy.ndarray
        2D array of N time series (ntime x N)

    """

    from eofs.xarray import Eof

    # remove empirical seasonal cycle
    da = da.groupby('time.month') - da.groupby('time.month').mean('time')

    ntime, nlat, nlon = da.shape

    if landmask is not None:

        # expand land mask to ntime
        lnd_mask = np.repeat(is_land.values[np.newaxis, :, :], ntime, axis=0)
        da = da.where(lnd_mask)

    coslat = np.cos(np.deg2rad(da['lat'].values)).clip(0., 1.)
    wgts = np.sqrt(coslat)[..., np.newaxis]
    solver = Eof(da, weights=wgts)

    eofs = solver.eofs(eofscaling=0)  # normalized st L2 norm = 1
    eigenvalues = solver.eigenvalues()

    # Low pass filter data
    if monthly:
        fs = 1

    nyq = 0.5 * fs  # Nyquist frequency
    low = L / nyq
    sos = butter(order, low, btype='low', output='sos')  # Coefficients for Butterworth filter
    if monthly:
        X_tilde = np.empty((da.shape))
        for kk in range(12):
            X_tilde[kk::12, :, :] = sosfiltfilt(sos, da.values[kk::12, :, :], padtype='even', axis=0)

    else:
        X_tilde = sosfiltfilt(sos, da.values, axis=0)

    a_k = eofs.values[:N, :, :].reshape((N, nlat*nlon))
    sigma_k = np.sqrt(eigenvalues.values[:N])

    if landmask is not None:
        lnd_mask_vec = is_land.values.flatten()
    else:
        lnd_mask_vec = np.ones((nlat*nlon,), dtype=bool)

    PC_tilde = np.empty((ntime, N))
    for kk in range(N):
        PC_tilde[:, kk] = 1/sigma_k[kk]*np.dot(X_tilde.reshape((ntime, nlat*nlon))[:, lnd_mask_vec],
                                               a_k[kk, lnd_mask_vec])

    R = np.dot(PC_tilde.T, PC_tilde)/(N - 1)
    R_eigvals, e_k = np.linalg.eig(R)  # eigenvalues already sorted

    # eigenvalues are in columns
    u_k = np.dot((a_k.T)/sigma_k, e_k)
    LFPs = np.dot(sigma_k*(a_k.T), e_k)

    # Time series:
    LFCs = np.dot(da.values.reshape((ntime, nlat*nlon))[:, lnd_mask_vec], u_k[lnd_mask_vec, :])

    return LFPs, LFCs


# Get land mask
land_dir = '/gpfs/fs1/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/lnd/proc/tseries/monthly/SOILWATER_10CM'
land_file = '%s/b.e11.B20TRC5CNBDRD.f09_g16.002.clm2.h0.SOILWATER_10CM.192001-200512.nc' % land_dir

ds_lnd = xr.open_dataset(land_file)['SOILWATER_10CM']
is_land = ~np.isnan(ds_lnd[0, ...])

nlat, nlon = is_land.shape

n_lfc_save = 5

all_LFP = []
all_LFC = []

valid_years = np.arange(1921, 2006)
nyrs = len(valid_years)
members = np.hstack((np.arange(1, 36), np.arange(101, 106)))
cesmdir = '/gpfs/fs1/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly'

for m in members:
    print(m)
    files = glob('%s/PREC*/b.e11.B20TRC5CNBDRD.f09_g16.%03i.cam.h0.PREC*.??????-200512.nc' % (cesmdir, m))
    ds_cesm = xr.open_mfdataset(files, concat_dim='precip_type', combine='by_coords')
    da_cesm = ds_cesm['PRECC'] + ds_cesm['PRECL'] + ds_cesm['PRECSL'] + ds_cesm['PRECSC']
    da_cesm = da_cesm.assign_coords(time=da_cesm.time-timedelta(days=1))
    da_cesm = da_cesm.sel({'time': np.isin(da_cesm['time.year'], valid_years)})
    da_cesm = da_cesm.assign_coords({'lat': np.round(da_cesm.lat, 3)})

    # change to mm /day
    da_cesm *= 1000*24*60*60
    # need to load to speed up compute
    da_cesm = da_cesm.load()

    # transform
    da_cesm = olens_utils.transform(da_cesm, 'boxcox', '/glade/work/mckinnon/obsLE/parameters/LE-%03i/pr' % m)

    LFP_save = np.empty((nlat*nlon, n_lfc_save, 12))
    LFC_save = np.empty((nyrs, n_lfc_save, 12))
    for month in range(1, 13):

        tmp_ds = da_cesm.sel(time=da_cesm['time.month'] == month)

        ntime, nlat, nlon = tmp_ds.shape

        LFPs, LFCs = LFCA(tmp_ds, fs=1, monthly=False, landmask=None)

        if np.mean(LFCs[-5:, 0]) < np.mean(LFCs[:5, 0]):
            multiplier = -1
        else:
            multiplier = 1

        LFP_save[:, :, month-1] = multiplier*LFPs[:, :n_lfc_save]
        LFC_save[:, :, month-1] = multiplier*LFCs[:, :n_lfc_save]

        del LFCs, LFPs

    LFP_da = xr.DataArray(LFP_save.reshape((nlat, nlon, n_lfc_save, 12)),
                          dims=['lat', 'lon', 'LFP', 'month'],
                          coords=[da_cesm.lat, da_cesm.lon, np.arange(1, 6), np.arange(1, 13)])

    LFC_da = xr.DataArray(LFC_save,
                          dims=['year', 'LFP', 'month'],
                          coords=[np.unique(da_cesm['time.year']), np.arange(1, 6), np.arange(1, 13)])

    all_LFP.append(LFP_da)
    all_LFC.append(LFC_da)

LFPs = xr.concat(all_LFP, dim='member')
LFCs = xr.concat(all_LFC, dim='member')

# save
LFPs.to_netcdf('%s/LFPs_precip_boxcox_CESM1-LE.nc' % procdir)
LFCs.to_netcdf('%s/LFCs_precip_boxcox_CESM1-LE.nc' % procdir)
