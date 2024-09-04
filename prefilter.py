"""
**Initial filtering and optional downsampling of the data,
including notching the powerline frequencies.**
"""
import warnings
import mne
from mne.chpi import (compute_chpi_amplitudes, compute_chpi_locs, 
                      compute_head_pos, write_head_pos)
import setup_utils as su

def prefilter(ss):
    """
    Filter raw files to the frequency band of interest,
    notch powerline frequencies; extract events and head positioning
    data; extract events data; downsample filtered records to
    the target sample rate.

    Args:
        ss(obj): reference to this app object

    Returns:
        None

    """
    warnings.filterwarnings('ignore', category=RuntimeWarning,
                            message=ss.args["file_name_warning"])

    STEP = 'prefilter'
    config = ss.args[STEP]
    files = su.files_to_process(ss, STEP)

    for in_fif, out_fif in files:
        # Skip non-raw .fif files, just in case
        if mne.what(in_fif) != 'raw':
            continue

        # Skip vibration bands ERM files produced by
        # maxfilter step, if any
        if su.is_erm_band_file(in_fif):
            continue

        raw = mne.io.read_raw_fif(in_fif, allow_maxshield=False, preload=True,
                      on_split_missing='raise', verbose = ss.args['verbose'])

        # Extract head pos info before filtering
        posfname = out_fif.parent / su.get_head_pos_file(in_fif)

        if config['recalc_head_pos'] or (not posfname.is_file()):
            pos = extract_head_pos(raw, config['chpi'], verbose = ss.args['verbose'])

            if pos is None:
                if ss.args['verbose'] == 'info':
                    print(f'\nNo head localization info found in file {in_fif.name}\n')
            else:
                write_head_pos(posfname, pos)

        # Filtering
        raw = filter_fif(raw, config, verbose = ss.args['verbose'])

        # Get events from the data prior to downsampling
        if su.is_erm_file(in_fif):
            raw = raw.resample(sfreq = config['target_sample_rate'], events=None, verbose = ss.args['verbose'])
            raw.save(out_fif, **config['save'], verbose = ss.args['verbose'])
        else:
            config['find_events']['min_duration'] = 2./raw.info['sfreq']
            all_events = mne.find_events(raw, **config['find_events'],
                                         verbose=ss.args['verbose'])

            # Downsampling
            raw, events = raw.resample(sfreq = config['target_sample_rate'], events=all_events, verbose = ss.args['verbose'])

            if events.shape[0] != all_events.shape[0]:
                print(f'WARNING: Some events were lost while downsampling {in_fif}')

            # Save downsampled data and events
            raw.save(out_fif, **config['save'], verbose = ss.args['verbose'])
            mne.write_events(su.events_fif(ss, out_fif), events, overwrite=True, verbose = ss.args['verbose'])

        print(f'{in_fif.name} done')

    warnings.filterwarnings("default", category=RuntimeWarning)
    print(f'\n** {STEP} step completed **\n')                

def filter_fif(raw, conf, verbose = None):
    """
    In place notch filter and bandpass filter given raw data.

    Args:
        raw(Raw): MNE Raw object to process
        conf(dict): dictionary containig parameters for the prefiltering step
        verbose(str): verbose level ('info', 'warn', 'error', ...)

    Returns:
        raw(Raw): MNE Python Raw object with filtered data.
    """
    # notch_filter() cannot notch several lines with IIR at the same time
    # So we need to do it one by one:
    kwargs = conf['notch'].copy()
    freqs = kwargs.pop('freqs')

    for f in freqs:
        raw.notch_filter(freqs = f, **kwargs, verbose=verbose)

    raw.filter(**conf['filter'], verbose=verbose)
    return raw

def extract_head_pos(raw, conf, verbose = None):
    """
    Extract and save head positioning information

    Args:
        raw(Raw): MNE Raw object to process
        conf(dict): dictionary with this substep related parameters
        verbose(str): verbose level ('info', 'warn', 'error', ...)

    Returns:
        pos(array | None): Head positioning data (shape (N,10)), or None if
            not found
    """
    try:
        amps = compute_chpi_amplitudes(raw, **conf['amps'], verbose = verbose)
    except ValueError:
        return None
            
    locs = compute_chpi_locs(raw.info, amps, **conf['locs'], verbose = verbose)
    pos = compute_head_pos(raw.info, locs, **conf['pos'], verbose = verbose)

    return pos

