import numpy as np
from datetime import datetime
import os
import utils as olens_utils
from scripts import model_components as mc
import json
from subprocess import check_call


def setup(varname, filename, AMO_smooth_length, mode_lag, workdir_base):

    # Create dictionary of parameters to save in working directory
    param_dict = {'varname': varname,
                  'filename': filename,
                  'AMO_smooth_length': AMO_smooth_length,
                  'mode_lag': mode_lag}

    # Output folder, named with current date
    now = datetime.strftime(datetime.now(), '%Y%m%d')
    workdir = '%s/%s' % (workdir_base, now)
    if not os.path.isdir(workdir):
        cmd = 'mkdir -p %s' % workdir
        check_call(cmd.split())
    # Save parameter set to director
    with open(workdir + '/parameter_set.json', 'w') as f:
        json.dump(param_dict, f)

    return workdir


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('n_members', type=int, help='Number of members of the Observational Large Ensemble to create.')
    parser.add_argument('case', type=str, help='Whether to analyze "obs" or "LE"')
    args = parser.parse_args()

    n_members = args.n_members

    # Parameters consistent across case
    AMO_smooth_length = 21  # number of years to apply AMO smoothing
    mode_lag = 1  # number of months to lag between mode time series and climate response

    # Need odd-window for AMO
    if AMO_smooth_length % 2 == 0:
        AMO_smooth_length += 1

    valid_years = np.arange(1921, 2015)
    cvdp_loc = '/glade/work/mckinnon/CVDP'

    varnames = ['tas', 'pr', 'slp']
    long_varnames = ['near surface air temperature', 'precipitation', 'sea level pressure']

    if args.case == 'obs':
        filenames = ['/glade/work/mckinnon/BEST/Complete_TAVG_LatLong1.nc',
                     '/glade/work/mckinnon/GPCC/precip.mon.total.1x1.v2018.nc',
                     '/glade/work/mckinnon/20CRv2c/prmsl.mon.mean.nc']
        data_names = [f.split('/')[-2] for f in filenames]

        workdir_base = '/glade/work/mckinnon/obsLE/parameters/%s' % args.case
        output_dir = '/glade/work/mckinnon/obsLE/output/%s' % args.case

        # Save parameter files
        workdir = setup(varnames, filenames, AMO_smooth_length, mode_lag, workdir_base)

        # Get data and modes
        for v, f in zip(varnames, filenames):
            dsX, df_shifted, _ = olens_utils.get_obs(v, f, valid_years, mode_lag, cvdp_loc)
            mc.fit_linear_model(dsX, df_shifted, v, AMO_smooth_length, workdir)

        # Calculate block size
        block_use, block_use_mo = olens_utils.choose_block(workdir, varnames)

        # Get surrogate modes
        AMO_surr, ENSO_surr, PDO_orth_surr, mode_months = mc.get_all_surrogates('%s/surrogates' % workdir_base)
        _, total_surr = np.shape(AMO_surr)

        # Can only make as many members as there are surrogate time series
        n_members = np.min((n_members, total_surr))
        print('Creating %d ensemble members' % n_members)

        # Put it all together, and save to netcdf files
        mc.combine_variability(varnames, workdir, output_dir, n_members, block_use_mo,
                               AMO_surr, ENSO_surr, PDO_orth_surr, mode_months, valid_years,
                               mode_lag, AMO_smooth_length, long_varnames, data_names)
