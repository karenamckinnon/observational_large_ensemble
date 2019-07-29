from observational_large_ensemble import utils as olens_utils
import numpy as np
import os
from subprocess import check_call


def create_surrogate_modes(n_ens_members, workdir_base, mode_nc, AMO_cutoff_freq=1/10, this_seed=123):
    """Create surrogate versions of ENSO, PDO, and AMO.

    Parameters
    ----------
    n_ens_members : int
        Number of surrogate time series to create
    workdir_base : str
        Working directory
    mode_nc : str
        Filename (from CVDP) with mode time series
    AMO_cutoff_freq : float
        Cut off frequency for Butterworth filter of AMO (1/years)
    this_seed : int
        Random seed for reproducibility.

    Returns
    -------
    Currently nothing. Saves surrogates to file.

    """

    cvdp_loc = '/glade/work/mckinnon/CVDP'
    modes_fname = '%s/%s' % (cvdp_loc, mode_nc)  # modes

    # Load original versions
    df = olens_utils.create_mode_df(modes_fname, AMO_cutoff_freq)
    ntime = len(df)

    # Create ENSO with seasonality
    np.random.seed(this_seed)
    enso_surr = np.empty((ntime, n_ens_members))
#    for kk in range(n_ens_members):
#        tmp = olens_utils.iaaft_seasonal(df['ENSO'].values)
#        while type(tmp) == int:  # case of no convergence
#            tmp = olens_utils.iaaft_seasonal(df['ENSO'].values)
#        enso_surr[:, kk] = tmp[0]

    # Create PDO and AMO using standard approach
    pdo_surr = np.empty((ntime, n_ens_members))
    amo_surr = np.empty_like(pdo_surr)
    for kk in range(n_ens_members):
        # ENSO (no seasonal behavior)
        tmp = olens_utils.iaaft(df['ENSO'].values)
        while type(tmp) == int:  # case of no convergence
            tmp = olens_utils.iaaft(df['ENSO'].values)
        enso_surr[:, kk] = tmp[0]

        # PDO
        tmp = olens_utils.iaaft(df['PDO_orth'].values)
        while type(tmp) == int:  # case of no convergence
            tmp = olens_utils.iaaft(df['PDO_orth'].values)
        pdo_surr[:, kk] = tmp[0]

        # AMO (create surrogates on unfiltered data)
        tmp = olens_utils.iaaft(df['AMO'].values)
        while type(tmp) == int:  # case of no convergence
            tmp = olens_utils.iaaft(df['AMO'].values)

        # Perform lowpass filter on AMO
        if AMO_cutoff_freq > 0:
            amo_lowpass = olens_utils.lowpass_butter(12, AMO_cutoff_freq, 3, tmp[0])
        else:  # no filter
            amo_lowpass = tmp[0]
        # Reset to unit sigma
        amo_lowpass /= np.std(amo_lowpass)
        amo_surr[:, kk] = amo_lowpass

    # Save
    savedir = '%s/surrogates_noENSOseasonality' % workdir_base
    if not os.path.isdir(savedir):
        cmd = 'mkdir -p %s' % savedir
        check_call(cmd.split())

    savename = '%s_surrogate_mode_time_series_%03d_%i.npz' % (mode_nc.split('.')[0], n_ens_members, this_seed)
    saveloc = '%s/%s' % (savedir, savename)

    # Note that the years don't mean anything for the surrogates, but the months do
    np.savez(saveloc,
             years=df['year'].values,
             months=df['month'].values,
             enso_surr=enso_surr,
             pdo_surr=pdo_surr,
             amo_surr=amo_surr)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--n_ens_members', type=int, help='Number of surrogate time series to create.')
    parser.add_argument('-S', '--seed', type=int, help='Random seed to use for reproducibility')
    parser.add_argument('-f', '--mode_nc', type=str, help='Filename for modes')
    parser.add_argument('-d', '--workdir', type=str, help='Working directory')
    parser.add_argument('-fs', '--cutoff_freq', type=float, help='Cutoff frequency for AMO (1/years)')
    args = parser.parse_args()

    create_surrogate_modes(args.n_ens_members, args.workdir, args.mode_nc, args.cutoff_freq, args.seed)
