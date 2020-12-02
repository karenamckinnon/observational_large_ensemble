from datetime import datetime
import os
import utils as olens_utils
from scripts import model_components as mc
import json
from subprocess import check_call
from glob import glob


def setup(varname, filename, AMO_cutoff_freq, mode_lag, workdir_base):

    # Create dictionary of parameters to save in working directory
    param_dict = {'varname': varname,
                  'filename': filename,
                  'AMO_cutoff_freq': AMO_cutoff_freq,
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
    parser.add_argument('case', type=str, help='Whether to analyze "obs" or "LE-XXX" (e.g. LE-001)')
    args = parser.parse_args()

    n_members = args.n_members

    # Modify this to your own parameter file with paths
    from params import karen_params as params
    valid_years = params.valid_years
    cvdp_loc = params.cvdp_loc
    AMO_cutoff_freq = params.AMO_cutoff_freq
    mode_lag = params.mode_lag
    workdir_base = params.workdir_base
    output_dir = params.output_dir
    tas_dir = params.tas_dir
    pr_dir = params.pr_dir
    slp_dir = params.slp_dir

    varnames = ['tas', 'pr', 'slp']
    long_varnames = ['near surface air temperature', 'precipitation', 'sea level pressure']

    workdir_base = '%s/%s' % (workdir_base, args.case)
    output_dir = '%s/%s' % (output_dir, args.case)

    if args.case == 'obs':
        cvdp_file = '%s/HadISST.cvdp_data.1920-2017.nc' % cvdp_loc
        filenames = ['%s/Complete_TAVG_LatLong1.nc' % tas_dir,
                     '%s/precip.mon.total.1x1.v2018.nc' % pr_dir,
                     '%s/prmsl.mon.mean.nc' % slp_dir]
        data_names = [f.split('/')[-2] for f in filenames]
        name_conversion = {'tas': 'temperature', 'pr': 'precip', 'slp': 'prmsl'}
        surr_prefix = 'HadISST_surrogate_mode_time_series_020'

        # Save parameter files
        workdir = setup(varnames, filenames, AMO_cutoff_freq, mode_lag, workdir_base)

        # Get data and modes
        for v, f in zip(varnames, filenames):
            dsX, df_shifted, _ = olens_utils.get_obs(args.case, v, f, valid_years, mode_lag,
                                                     cvdp_file, AMO_cutoff_freq, name_conversion)
            mc.fit_linear_model(dsX, df_shifted, v, workdir)
            if v != 'slp':  # forced component for SLP assumed to be zero
                mc.save_forced_component(df_shifted, v, output_dir, workdir)

    elif 'LE' in args.case:
        name_conversion = {'tas': 'TREFHT', 'pr': 'PRECC', 'slp': 'PSL'}
        cesm_names = [name_conversion[v] for v in varnames]
        surr_prefix = 'CESM1-CAM5-BGC-LE_#1_surrogate_mode_time_series_020'
        this_member = int((args.case).split('-')[-1])
        cvdp_file = '%s/CESM1-CAM5-BGC-LE_#%i.cvdp_data.1920-2017.nc' % (cvdp_loc, this_member)

        base_directory = '/gpfs/fs1/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly'
        # Historical filenames for CESM. Will need to append part of RCP8.5 to get full period
        filenames = []
        for var in cesm_names:
            file_str = '%s/%s/b.e11.B20TRC5CNBDRD.f09_g16.%03d.cam.h0.%s.??????-200512.nc' % (base_directory, var,
                                                                                              this_member, var)
            this_file = glob(file_str)[0]
            filenames.append(this_file)

        data_names = ['CESM1-LE', 'CESM1-LE', 'CESM1-LE']

        # Save parameter files
        workdir = setup(varnames, filenames, AMO_cutoff_freq, mode_lag, workdir_base)

        # Get data and modes
        for v, f in zip(varnames, filenames):
            # To allow for the concatenation of multiple model sims, pass the filename as a list
            dsX, df_shifted, _ = olens_utils.get_obs(args.case, v, [f], valid_years, mode_lag,
                                                     cvdp_file, AMO_cutoff_freq, name_conversion)
            if v == 'pr':  # perform transform to normalize data
                dsX = olens_utils.transform(dsX, 'boxcox', workdir)
            mc.fit_linear_model(dsX, df_shifted, v, workdir)
            if v != 'slp':  # forced component for SLP assumed to be zero
                mc.save_forced_component(df_shifted, v, output_dir, workdir)

    # Calculate block size
    block_use, block_use_mo = olens_utils.choose_block(workdir, varnames)

    # Get surrogate modes
    this_seed = 456
    ENSO_surr, PDO_orth_surr, AMO_surr, mode_months = mc.create_surrogate_modes(cvdp_file, AMO_cutoff_freq,
                                                                                this_seed, n_members)

    # Put it all together, and save to netcdf files
    mc.combine_variability(varnames, workdir, output_dir, n_members, block_use_mo,
                           AMO_surr, ENSO_surr, PDO_orth_surr, mode_months, valid_years,
                           mode_lag, long_varnames, data_names)
