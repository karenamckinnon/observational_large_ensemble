from observational_large_ensemble import utils as olens_utils
import numpy as np
import os


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

    # The IAAFT algorithm is not effectively preserving the annual cycle of ENSO amplitudes
    # While not ideal, going to manually reweight the variances by month
    monthly_var_enso = (df.groupby('month').std()['ENSO'])**2
    for kk in range(n_ens_members):
        surr_df = pd.DataFrame()
        surr_df = surr_df.assign(month=df['month'].values, surrogate=enso_surr[:, kk])
        surr_var = (surr_df.groupby('month').std()['surrogate'])**2
        weights = np.sqrt(monthly_var_enso/surr_var)
        weights /= np.mean(weights)
        enso_surr[:, kk] *= np.tile(weights.values, int(len(df)/12))

    # Save
    savedir = '%s/surrogates/' % workdir_base
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    saveloc = '%stime_series_%03d.npz' % (savedir, n_ens_members)

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
    args = parser.parse_args()

    workdir_base = '/glade/work/mckinnon/obsLE/parameters'

    create_surrogate_modes(args.n_ens_members, workdir_base)
