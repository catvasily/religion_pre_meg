"""
**Generate low frequency source level ERFs for each subject**

This step expects that results of the `src_rec` run with `do_evoked = True`
are available.
"""
import numpy as np
from scipy.signal import resample, get_window, butter, sosfiltfilt
import setup_utils as su
from src_rec import read_roi_time_courses, write_roi_time_courses, get_fwd_input_fifs
from epochs import get_epochs_for_event
from plot_waveforms import adjust_signs

def src_erf(ss):
    """
    Generate source level low frequency ERFs.

    Args:
        ss(obj): reference to this app object

    """
    STEP = 'src_erf'
    config = ss.args[STEP]

    event_ids = ss.args['src_rec']['events_for_evoked']
    atlas = ss.args['src_rec']['atlas']

    if config['target_sample_rate'] is not None:
        config['original_sample_rate'] = ss.args['prefilter']['target_sample_rate']

    # Calculate index interval for sign adjustments
    t0 = ss.args['src_rec']['epochs']['t_range'][0] # epoch's time origin
    tstart = config['sign_adjust_interval'][0] - t0
    tend = config['sign_adjust_interval'][1] - t0
    config['isign_start'] = int(config['original_sample_rate']*tstart)
    config['isign_end'] = int(config['original_sample_rate']*tend) + 1

    # We pass a generator through a static list, to avoid processing new files
    # created by this program and added to the subjects's subfolders
    files = iter(list(su.files_to_process(ss, STEP)))

    current_folder = None
    all_results = {}
    erf_hdf5 = None

    # For static files list we could just check for the last list item...
    # Yet still using with_last item_flag() wrapper for generality
    for (in_file, out_file), is_last in with_last_item_flag(files):
        # Initialize current_folder for the 1st time
        if current_folder is None:
            current_folder = in_file.resolve().parent

        if in_file.resolve().parent != current_folder:
            # Subject/date changed, but this is not the last one yet
            # Complete processing for current subj/date
            process_collected_evokes(config, all_results, event_ids, erf_hdf5)

            # Initialize processing for the next subj/date
            current_folder = in_file.resolve().parent
            all_results = {}

        # From this point on, we are starting a new subj/date or
        # proceeding with the old one

        skip = False

        try:
            task_name = su.get_fif_task(in_file)
        except:
            # We get here with old ERF files who do not have
            # task in their names
            task_name = 'skip'  # Just any garbage name will do
            skip = True
        
        # Skip continuous data
        if not su.extract_epochs(task_name):
            skip = True

        # Skip evoked data
        if 'evoked' in in_file.stem:
            skip = True

        # Skip old erf data
        if '_erf' in in_file.stem:
            skip = True

        if skip:
            if is_last:
                process_collected_evokes(config, all_results, event_ids, erf_hdf5)
                continue    # continue still needed here, do not remove
            else:
                continue

        # At this point, the "in_file" is an epoched unaveraged taskN data
        # and in_file.parent == current_folder
        meg_subject = su.fif_subject(in_file)
        ss_fif = get_fwd_input_fifs(ss, out_file)[1]    # The source space fif file
        erf_hdf5 = out_file                             # Save output ERF file name

        # Start or continue processing current subj/date files
        all_results[task_name] = {}

        # Process the in_file
        # read_roi_time_courses() returns:
        # (label_tcs, label_names, vertno, rr, W, pz, events, events_id_dict)
        events = read_roi_time_courses(in_file)[-2]

        for eID in event_ids: 
            # Load averaged (evoked) data for this eID
            ltc_hdf5 = su.ltc_file_pathname(meg_subject, task_name, ss_fif,
                                atlas, out_file.parent, eID = eID)

            if not ltc_hdf5.is_file():
                continue

            evoked_tcs_tuple = read_roi_time_courses(ltc_hdf5)

            # events is a list of event arrays for each epoch
            # Get total count of epochs for the eID
            nepochs4eID = sum(1 for earray in events if (eID in earray[:,2]))

            # Note that label_tcs.shape = (2,nlabels,ntimes); 2nd epoch
            # is std
            all_results[task_name][eID] = dict(nepochs4eID = nepochs4eID,
                        label_tcs = evoked_tcs_tuple[0],
                        label_names = evoked_tcs_tuple[1],
                        vertno = evoked_tcs_tuple[2],
                        rr = evoked_tcs_tuple[3],
                        W = evoked_tcs_tuple[4],
                        pz = evoked_tcs_tuple[5],
                        earray = evoked_tcs_tuple[6])

        if is_last:
            process_collected_evokes(config, all_results, event_ids, erf_hdf5)
            continue    # not necessary here for now, but may be if more code
                        # is added later
    # --- and files_to_process loop

    print(f'\n** {STEP} step completed **\n')                

def with_last_item_flag(generator):
    """
    A wrapper for a generator which yields the next generator
    item and also a flag indicating if this is the last
    item available.

    Args:
        generator: as is

    Returns:
        tuple (item, bool): next item and a flag indicating
            if it is the last one

    """
    previous = next(generator, None)

    for current in generator:
        yield previous, False
        previous = current

    if previous is not None:
        yield previous, True

def process_collected_evokes(config, all_results, event_ids, erf_hdf5):
    """
    Combine evoked responses from all the task runs, process
    and save.

    Note that in the resulting cumulative .hdf5 file for any event ID, 
    parameters `vertno`, `rr`, `W` are taken from the 1st task_run in
    `all_results` that contains this `eID` (for a given subj/date 
    records set).

    Args:
        config(dict): this step's config dictionary
        all_results(dict): a dictionary `task_name->eID->{nepochs4eID,
                label_tcs, pz, earray}`
        event_ids(list of int): list of possible event IDs
        erf_hdf5(pathlike): output .hdf5 file template pathname. The
            basename here does not contain eID. For each eID, in the actual
            output pathname `.hdf5` should be replaced with `_<eID>.hdf5`

    """
    if not len(all_results):
        return

    # ------------------------------------------------------------
    # NOTE: we are averaging STDs and PZs, which does not make much
    # mathematical sense. Thus results for the latter are only 
    # qualitative estimates
    # ------------------------------------------------------------
    for eID in event_ids:
        # Create averaged evoked time course over all task runs
        navg = 0
        erf_tcs = None
        mean_pz = 0

        for tsk in all_results:
            task_dict = all_results[tsk]

            if not (eID in task_dict):
                continue

            nepochs = task_dict[eID]['nepochs4eID']
            navg += nepochs

            if erf_tcs is None:
                # Set the reference epoch for signs adjustment
                epoch0 = task_dict[eID]['label_tcs'][0]
                # Shape is (2,nlabels,ntime)
                erf_tcs = nepochs*task_dict[eID]['label_tcs']
                mean_pz = nepochs*task_dict[eID]['pz']
            else:
                # Adjust channels signs based on epoch0
                # 'e1' is the (nlabels x ntime) sign-adjusted epoch
                e1 = adjust_signs([epoch0, task_dict[eID]['label_tcs'][0]],
                        istart = config['isign_start'], iend = config['isign_end'])[1]

                # Update the ltc data
                task_dict[eID]['label_tcs'][0] = e1

                # Add to the sum with a weight
                erf_tcs += nepochs * task_dict[eID]['label_tcs']
                mean_pz += nepochs * task_dict[eID]['pz']
        # --- end of for tsk cycle

        # Create output .hdf5 file name for this eID
        out_file = erf_hdf5.parent / f'{erf_hdf5.stem}_{eID}{erf_hdf5.suffix}'

        if erf_tcs is None:
            print(f'No data found for event {eID} to generate {out_file.name}')
            continue

        # Normalize to get proper weighted averages
        erf_tcs /= navg
        mean_pz /= navg

        # Filter
        if config['fmax'] is not None:
            erf_tcs = filter_downsample(config, erf_tcs)

        # Find the 1st task run (if any) that has data for this eID
        for tsk in all_results:
            task_dict = all_results[tsk]

            if not eID in task_dict:
                task_dict = None
                continue
            else:
                break
        # --- end of for tsk cycle

        if task_dict:
            write_roi_time_courses(out_file, erf_tcs, task_dict[eID]['label_names'],
                        vertno = task_dict[eID]['vertno'],
                        rr = task_dict[eID]['rr'],
                        W = task_dict[eID]['W'],
                        pz = mean_pz,
                        events = task_dict[eID]['earray'],
                        events_id_dict = None)
    # --- end of for eID loop

def filter_downsample(cfg, data):
    """
    Resample data to a target sample rate.

    **IMPORTANT** - no antialiasing filtering is applied. This should
    have been done before calling this function.

    Args:
        cfg(dict): this steps configuration dictionary
        data(ndarray): shape `(nepochs, nchan, ntime)`

    Returns:
        filtered_downsampled_data(ndarray): shape `(nepochs, nchan, ntime)` 

    """
    # Low pass filtering
    # Calc relative frequencies
    nyq = 0.5 * cfg['original_sample_rate']
    fmax = cfg['fmax'] / nyq

    # Design the filter
    sos = butter(cfg['filter_order'], fmax, btype='lowpass', 
                        output = 'sos')  			

    # NOTE: Using sosfiltfilt() instead of just filtfilt() as recommended
    # in scipy's filtfilt() documentation
    out_data = sosfiltfilt(sos, data, axis = 2)

    if cfg['target_sample_rate'] is None:
        return out_data

    # Downsample
    # Find target number of samples for the record of the same duration
    n0 = out_data.shape[2] - 1      # Total number of time steps

    # Number of new time steps - not an integer
    n1 = (n0 / cfg['original_sample_rate'])*cfg['target_sample_rate']
    target_samples = int(np.rint(n1)) + 1       # Target number of sample points
    return resample(out_data, target_samples, axis = 2, window = cfg['taper_window'])

