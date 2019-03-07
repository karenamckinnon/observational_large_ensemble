from observational_large_ensemble import utils as olens_utils
import numpy as np


def create_surrogate_modes(n_ens_members, workdir_base):

    cvdp_loc = '/glade/work/mckinnon/CVDP'
    modes_fname = '%s/HadISST.cvdp_data.1920-2017.nc' % cvdp_loc  # modes

    # Load original versions
    df = olens_utils.create_mode_df(modes_fname)
    ntime = len(df)

    # Create paired versions of ENSO, PDO, and separate version of AMO
    np.random.seed(123)
    enso_surr = np.empty((ntime, n_ens_members))
    pdo_surr = np.empty_like(enso_surr)
    amo_surr = np.empty_like(enso_surr)
    for kk in range(n_ens_members):
        enso_surr[:, kk], pdo_surr[:, kk] = olens_utils.create_matched_surrogates_1d(df['ENSO'].values,
                                                                                     df['PDO'].values)
        amo_surr[:, kk] = olens_utils.iaaft(df['AMO'].values)

    # Save
    saveloc = '%s/surrogates/time_series_%03d.npz' % (workdir_base, n_ens_members)
    np.savez(saveloc,
             enso_surr=enso_surr,
             pdo_surr=pdo_surr,
             amo_surr=amo_surr)


if __name__ == '__main__':

    n_ens_members = 100
    workdir_base = '/glade/work/mckinnon/obsLE/parameters'

    create_surrogate_modes(n_ens_members, workdir_base)
