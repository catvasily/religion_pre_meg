"""
**Apply tSSS, vibrational artifact correction and head
motion correction.**
"""
import warnings
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.preprocessing import find_bad_channels_maxwell, \
        compute_average_dev_head_t, maxwell_filter
from mne.chpi import read_head_pos        

import setup_utils as su

"""
WARNING to ignore:
This filename (...) does not conform to MNE naming conventions. All raw
files should end with raw.fif, raw_sss.fif, raw_tsss.fif, _meg.fif,
_eeg.fif, _ieeg.fif, raw.fif.gz, raw_sss.fif.gz, raw_tsss.fif.gz, _meg.fif.gz,
_eeg.fif.gz or _ieeg.fif.gz
"""

def maxfilter(ss):
    """
    Top level function for the Maxwell filtering step. The following
    substeps are performed here. First, bad and flat channels are identified.
    Second, presense of a vibrational artifact is determined and
    corresponding out-projectors are calculated.
    Third, tSSS or eSSS is applied depending on the vibrational artifact
    presence.

    Args:
        ss(obj): reference to this app object

    Returns:
        None

    """
    warnings.filterwarnings('ignore', category=RuntimeWarning,
                            message=ss.args["file_name_warning"])

    STEP = 'maxfilter'

    # Get input and output data locations and subjects dict for this step
    in_dir, out_dir, subjects = su.get_subjects_and_folders(ss, STEP)
    ct_file = ss.data_host.get_ct_file()
    fc_file = ss.data_host.get_fc_file()

    config = ss.args[STEP]

    if ss.data_host.cluster_job:
        config['show_plots'] = False
        subjects = su.get_sdict_for_job(ss, subjects)

    for subj in subjects:
        for date in subjects[subj]:
            # in_,out_folder define locations for specific subj and specific date
            in_folder = su.get_subj_subfolder(in_dir, subj, date)
            out_folder = su.get_subj_subfolder(out_dir, subj, date)

            if not out_folder.exists():
                out_folder.mkdir(parents=True, exist_ok=True)

            fif_files = ss.args[STEP]['files']

            if fif_files is None:
                fif_files = list(in_folder.glob('*.fif'))
            else:
                fif_files = [in_folder / f for f in fif_files]

            # Process ERM recording, if any 
            # -----------------------------
            all_erms = [f for f in fif_files if su.is_erm_file(f)]
            # Drop band-passed ERMs from previous runs:
            erm_fif = [f for f in all_erms if not su.is_erm_band_file(f)]

            if erm_fif:
                erm_fif = erm_fif[0]
                do_eSSS = True

                # NOTE: found erm_bads are NOT added to erm_raw bad channels
                erm_raw, erm_bads = find_bads(fname = erm_fif, calibration=fc_file, cross_talk=ct_file, 
                    kwargs = config['remove_bad'], verbose = ss.args['verbose'])
                erm_bads += erm_raw.info['bads']
                erm_filter_bands(erm_raw, erm_fif, config, verbose = ss.args['verbose'])
            else:
                do_eSSS = False

            # Process all non-ERM .fif files
            # ------------------------------
            rmbad_args = config['remove_bad'].copy()

            # Need to change max-filter origin from 'meg' to 'head'
            # Still keeping 'head_pos' as null for now
            rmbad_args['coord_frame'] = 'head'

            for f in fif_files:
                if su.is_erm_file(f):
                    continue
                
                # Skip positioning FIFs, or other non-raw
                if mne.what(f) != 'raw':
                    continue

                # Read the subject's next .fif file
                raw = mne.io.read_raw_fif(f, allow_maxshield=False, preload=True,
                              on_split_missing='raise', verbose = ss.args['verbose'])

                # Compute average dev to head transformation, to be used
                # for the target head position
                pos_file = in_folder / su.get_head_pos_file(f)
                trans_fif = out_folder / (su.get_fif_task(f) + '_tmp-trans.fif')    # Tmp file for dev-head transform

                if pos_file.is_file():
                    pos = read_head_pos(pos_file) 
                    avg_dev_head_t = compute_average_dev_head_t(raw, pos, verbose=ss.args['verbose'])
                    # The center of the head in DEV coords is the offset vector of head->dev transformation
                    # (because it equals H2D*(0,0,0,1)^T). Therefore to find it one would write
                    # avg_head_pos_dev_coord = mne.transforms.invert_transform(avg_dev_head_t)['trans'][:3,3]
                    #
                    # However, destination parameter in the maxfilter call means position of the DEV origin
                    # in head coordinate system (counter-intuitively, see log). Then one may set it as follows:
                    # destination = avg_dev_head_t['trans'][:3,3]
                    # In this case, the maxfiltered data will have dev->head equal to:
                    #   new_dev_head:          I3 destination
                    #                           0       1
                    # IMPORTANTLY, in this case the head position WILL CHANGE compared to the original because
                    # org_dev_head^(-1)[:3,3] != new_dev_head^(-1)[:3,3]
                    #
                    # Finally, one can specify the .fif file with the transformation; then this one will become
                    # the head-motion-corrected dataset's dev_head.
                    avg_dev_head_t.save(trans_fif, overwrite = True)
                    destination = trans_fif
                else:
                    pos = None
                    destination = None

                # Find bads using maxfilter
                raw, bads = find_bads(raw = raw, calibration=fc_file, cross_talk=ct_file, 
                    kwargs = rmbad_args, verbose = ss.args['verbose'])

                if do_eSSS:
                    # This is a combined list of bads including ERM bads
                    all_bads = erm_bads + raw.info['bads'] + bads
                    all_bads = list(set(all_bads))  # remove dupes
                    save_plots = 'rest' in f.name   # Save projector plots only for resting state
                    erm_proj = construct_projectors(erm_fif, config, all_bads, save_plots,
                                                    verbose = ss.args['verbose'])
                    raw.info['bads'] = all_bads
                else:
                    # Proceed without calculating ERM projectors
                    erm_proj = []
                    raw.info['bads'] += bads

                if config['do_head_motion_correction']:
                    raw_mf = maxwell_filter(raw, calibration=fc_file, cross_talk=ct_file, head_pos=pos,
                                extended_proj=erm_proj,
                                destination=destination,
                                **config['maxf'],
                                verbose = ss.args['verbose']
                                )
                else:
                    raw_mf = maxwell_filter(raw, calibration=fc_file, cross_talk=ct_file, head_pos=pos,
                                extended_proj=erm_proj,
                                destination=None,
                                **config['maxf'],
                                verbose = ss.args['verbose']
                                )

                    if destination is not None:
                        # When avg_dev_head_t exists, set it as the filtered data transform
                        raw_mf.info['dev_head_t'] = avg_dev_head_t

                mf_fif = ss.data_host.get_step_out_file(STEP, f)
                outname = out_folder / mf_fif
                raw_mf.save(outname, **config["save"], verbose = ss.args['verbose'])

                # Clean up
                trans_fif.unlink(missing_ok=True)

                # Estimate data rank
                data_rank = compute_data_rank(raw_mf, tmin = 0, tmax = 100, n_jobs = -1, verbose = ss.args['verbose'])
                print(f'Rank of {mf_fif} is {data_rank}')
                print(f'{f.name} done\n')

    warnings.filterwarnings("default", category=RuntimeWarning)
    print(f'\n** {STEP} step completed **\n')                

def compute_data_rank(raw, tmin = 0, tmax = 100, n_jobs = -1, verbose = None):
    """
    Estimate rank of the data by computing covariance matrix of a short segment of the record.

    Args:
        raw(MNE Raw): the Raw data object
        tmin(float): beginning of time interfal in seconds
        tmax(float): end of time interval in seconds. If None or larger than the record length -
            the end of record will be used
        n_jobs(int): the number of jobs to run in parallel. If -1, it is set to the number of CPU cores. 
        verbose(str): MNE Python verbose level

    Returns:
        data_rank(int): the rank estimate

    """
    if tmax is None:
        tmax = raw.times[-1]
    else:
        if tmax > raw.times[-1]:
            tmax = raw.times[-1]

    # Estimate data rank
    cov = mne.compute_raw_covariance(raw, tmin=tmin, tmax=tmax, n_jobs=n_jobs,
                                     verbose=verbose)

    rank_dict = mne.compute_rank(cov, rank=None, scalings=None, info=raw.info, tol='auto', proj=True,
                                 tol_kind='absolute', on_rank_mismatch='ignore', verbose=verbose)

    return sum(rank_dict.values())

def construct_projectors(erm_fif, config, all_bads, save_plots = False,
                         verbose = None):
    """
    Construct additional projectors to be used for max-filtering based
    on empty room recording. The projectors deal with vibrational artifact
    (if any). If either ERM recording does not exist or vibrational artifact
    is not detected, an empty list `[]` will be returned. The detection is
    done by checking the variance captured by extracted projection vectors.
    Only projectors with variance above specified threshold are retained.

    Args:
        erm_fif(Path): pathname to the (full band) ERM recording
        config(dict): the maxfilter step config dictionary
        all_bads(list of str): a combined list of bad channels (ERM + current
            record)
        save_plots(bool): flag to save projector topomap plots
        verbose(str): MNE Python verbose level

    Returns:
        projs(list of Projection): a list of projection objects (may be empty)

    """
    projs = []
    threshold = config['vib_filter']['threshold']

    for band in config['vib_filter']['bands']:
        # Get ERM recording filtered to current band
        erm_band_fif = su.get_erm_band_file(erm_fif, band)
        erm_band_raw = mne.io.read_raw_fif(erm_band_fif, allow_maxshield=False, preload=True,
                      on_split_missing='raise', verbose = verbose)

        # Construct projectors
        erm_band_raw.info['bads'] = all_bads    # Use bad chans from both ERM and subject record 
        prj = mne.compute_proj_raw(erm_band_raw, **config['proj'], verbose = verbose)
        keep_prj = [p for p in prj if p['explained_var'] >= threshold]
        projs += keep_prj

        if keep_prj:
            if save_plots:
                fig = mne.viz.plot_projs_topomap(keep_prj, colorbar=True,
                                       info=erm_band_raw.info, size = 3, show = False)
                fig.suptitle(f'[{band[0]}, {band[1]}] Hz')

                if config['show_plots']:
                    plt.show()

                save_fname = erm_band_fif.parent / f'{band[0]}_{band[1]}_Hz_erm_proj.png' 
                fig.savefig(save_fname)

    if (verbose is None) or (verbose.upper() == 'INFO'):
        if projs:
            print('\nThe following external projectors will be applied for eSSS:')
            for p in projs:
                print(p)
            print('')
        else:
            print('\nNo projectors will be applied for eSSS\n')

    return projs 

def remove_bad(raw = None, fname = None, calibration=None, cross_talk=None, 
               kwargs = None, verbose = None):
    """
    Identify and remove flat and bad channels using MNE's
    `find_bad_channels_maxwell()` utility.

    Args:
        raw(MNE Raw): dataset Raw object 
        fname(Path): input .fif file pathname. If both `raw` and 
            `fname` are given, the `raw` argument will be used
        calibration(str | None): path to the '.dat' file with fine
            calibration coefficients
        cross_talk(str | None): path to the FIF file with cross-talk
            correction information.
        kwargs(dict): dictionary with arguments for the 
            `find_bad_channels_maxwell()` call (except `calibraion`, `cross_talk`
            and `verbose`)
        verbose(str): MNE's verbose level.

    Returns:
        raw(MNE Raw): updated raw with identified bad and flat channels
            added to raw.info['bads']

    """
    if raw is None:
        if fname is None:
            raise ValueError('Either raw or fname argument must be specified')

        raw = mne.io.read_raw_fif(fname, allow_maxshield=False, preload=True,
                      on_split_missing='raise', verbose = verbose)

    # Identify bad channels
    noisy_chs, flat_chs = find_bad_channels_maxwell(raw, calibration = calibration, 
                                cross_talk = cross_talk, **kwargs, verbose = verbose)

    raw.info['bads'] = raw.info['bads'] + noisy_chs + flat_chs
    return raw

def find_bads(raw = None, fname = None, calibration=None, cross_talk=None, 
               kwargs = None, verbose = None):
    """
    Construct a list of flat and bad channels using MNE's `find_bad_channels_maxwell()`
    utility.

    Args:
        raw(MNE Raw): dataset Raw object 
        fname(Path): input .fif file pathname. If both `raw` and 
            `fname` are given, the `raw` argument will be used
        calibration(str | None): path to the '.dat' file with fine
            calibration coefficients
        cross_talk(str | None): path to the FIF file with cross-talk
            correction information.
        kwargs(dict): dictionary with arguments for the 
            `find_bad_channels_maxwell()` call (except `calibraion`, `cross_talk`
            and `verbose`)
        verbose(str): MNE's verbose level.

    Returns:
        raw(MNE Raw): as is
        bads(list of str): a list of identified bad and flat channels


    """
    if raw is None:
        if fname is None:
            raise ValueError('Either raw or fname argument must be specified')

        raw = mne.io.read_raw_fif(fname, allow_maxshield=False, preload=True,
                      on_split_missing='raise', verbose = verbose)

    # Identify bad channels
    noisy_chs, flat_chs = find_bad_channels_maxwell(raw, calibration = calibration, 
                                cross_talk = cross_talk, **kwargs, verbose = verbose)

    return raw, noisy_chs + flat_chs

def erm_filter_bands(erm_raw, erm_fif, config, verbose = None):
    """
    If not already done, filter ERM recording to bands relevant to artifact
    rejections, and save filtered data as separate .fif files

    Args:
        erm_raw (MNE Raw): as is
        erm_fif (Path): full pathname of ERM .fif file
        config(dict): the maxfilter step configuration dictionary
        verbose(str): MNE verbose level

    Returns:
        None: filtered versions of the original ERM file are saved as .fif
            files to the same folder where `erm_fif` resides.

    """
    for band in config['vib_filter']['bands']:
        # Filter ERM recording to vib artifact band of interest
        erm_band_fif = su.get_erm_band_file(erm_fif, band)

        if erm_band_fif.is_file():
            continue    # Nothing to do

        band_raw = erm_raw.copy()
        band_raw.filter(l_freq = band[0], h_freq = band[1],
                       **config['vib_filter']['kwargs'], verbose = verbose)
        band_raw.save(erm_band_fif, **config['save'], verbose = verbose)
        del band_raw

