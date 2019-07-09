from observational_large_ensemble import utils as olens_utils
import numpy as np
import os


def create_surrogate_modes(n_ens_members, workdir_base, mode_nc='HadISST.cvdp_data.1920-2017.nc', this_seed=123):
    """Create surrogate versions of ENSO, PDO, and AMO.

    Parameters
    ----------
    n_ens_members : int
        Number of surrogate time series to create
    workdir_base : str
        Working directory
    mode_nc : str
        Filename (from CVDP) with mode time series
    this_seed : int
        Random seed for reproducibility.

    Returns
    -------
    Currently nothing. Saves surrogates to file.

    """

    cvdp_loc = '/glade/work/mckinnon/CVDP'
    modes_fname = '%s/%s' % (cvdp_loc, mode_nc)  # modes

    # Load original versions
    df = olens_utils.create_mode_df(modes_fname)
    ntime = len(df)

    # Create ENSO with seasonality
    np.random.seed(this_seed)
    enso_surr = np.empty((ntime, n_ens_members))
    for kk in range(n_ens_members):
        tmp = olens_utils.iaaft_seasonal(df['ENSO'].values)
        while type(tmp) == int:  # case of no convergence
            tmp = olens_utils.iaaft_seasonal(df['ENSO'].values)
        enso_surr[:, kk] = tmp[0]

    # Create PDO and AMO using standard approach
    pdo_surr = np.empty((ntime, n_ens_members))
    amo_surr = np.empty_like(pdo_surr)
    for kk in range(n_ens_members):
        # PDO
        tmp = olens_utils.iaaft(df['PDO_orth'].values)
        while type(tmp) == int:  # case of no convergence
            tmp = olens_utils.iaaft(df['PDO_orth'].values)
        pdo_surr[:, kk] = tmp[0]

        # AMO
        tmp = olens_utils.iaaft(df['AMO'].values)
        while type(tmp) == int:  # case of no convergence
            tmp = olens_utils.iaaft(df['AMO'].values)
        amo_surr[:, kk] = tmp[0]

    # Save
    savedir = '%s/surrogates' % workdir_base
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
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
    parser.add_argument('n_ens_members', type=int, help='Number of surrogate time series to create.')
    parser.add_argument('seed', type=int, help='Random seed to use for reproducibility')
    args = parser.parse_args()

    workdir_base = '/glade/work/mckinnon/obsLE/parameters'

    create_surrogate_modes(args.n_ens_members, workdir_base, args.seed)
