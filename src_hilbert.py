"""
**Band-pass filter ROI level task data to canonical frequency bands and construct Hilbert envelopes**
"""
import re
import numpy as np
from scipy.signal import resample, get_window, butter, sosfiltfilt, hilbert
import setup_utils as su
from src_rec import read_roi_time_courses, write_roi_time_courses
from src_erf import with_last_item_flag
from epochs import get_epochs_for_event

def src_hilbert(ss):
    """
    Generate Hilbert envelopes for source reconstructed time courses in canonical
    frequency bands.

    Args:
        ss(obj): reference to this app object

    """
    STEP = 'src_hilbert'
    config = ss.args[STEP]

    if config['power'] == False:    # QQQQ Remove this when '_pwr' suffix is properly
        config['power'] = True      # added to the output .hdf5 
        print('WARNING: config["power"] reset to True; power envelopes will be calculated')

    if config['target_sample_rate'] is not None:
        config['original_sample_rate'] = ss.args['prefilter']['target_sample_rate']

    ptrn1 = 'task_run'
    ptrn2 = '-ltc.hdf5'
    ptrn3 = '-evoked-'

    # We pass a generator through a static list, to avoid processing new files
    # created by this program and added to the subjects's subfolders
    # iter() is needed to create a generator again, to be used in the
    # 'with_last_item_flag()' wrapper
    files = iter(list(su.files_to_process(ss, STEP)))

    current_folder = None
    all_results = {}
    event_ids = config['events_for_hilbert']
    bands = tuple(map(tuple,config['bands']))   # We need tuples, to use as dict keys

    for (in_file, out_file), is_last in with_last_item_flag(files):
        # Initialize current_folder for the 1st time
        if current_folder is None:
            current_folder = in_file.resolve().parent

        # Check if we are done with current subject/date
        if in_file.resolve().parent != current_folder:
            # Subject/date changed, but this is not the last one yet
            # Complete processing for current subj/date
            process_collected_results(config, all_results, bands, event_ids, henv_hdf5 = henv_hdf5)

            # Initialize processing for the next subj/date
            current_folder = in_file.resolve().parent
            all_results = {}
            
        # ------------------------------------------------------
        # From this point on, we are starting a new subj/date or
        # proceeding with the old one, and are processing the next
        # single task .hdf5 file
        # ------------------------------------------------------
        skip = False

        # Choose only task_runN .hdf5 files
        if not ptrn1 in str(in_file):
            skip = True

        # Choose only original source time courses
        # (skip evoked, erf and this step generated .hdf5
        if not ptrn2 in str(in_file):
            skip = True

        # Skip 'evoked' files which also have 'task_run' in them
        if ptrn3 in str(in_file):
            skip = True

        if skip:
            if is_last:
                # This was the last item, so processing will not be triggered
                # by the folder change. We need to order it explicitly:
                process_collected_results(config, all_results, bands, event_ids, henv_hdf5 = henv_hdf5)
                continue    # continue still needed here, because we are skipping
                            # the in_file
            else:
                continue

        # At this point, the "in_file" is an epoched unaveraged taskN data
        # and in_file.parent == current_folder
        meg_subject = su.fif_subject(in_file)       # Never mind 'fif' - works for .hdf5 as well
        task_name = su.get_fif_task(in_file)        # task_runN
        all_results[task_name] = {}
        henv_hdf5 = out_file

        # (label_tcs, label_names, vertno, rr, W, pz, events, events_id_dict)
        ltc_tuple = list(read_roi_time_courses(in_file))
        data = ltc_tuple[0]     # (nepochs, nchan, ntime)
        events_per_epoch = ltc_tuple[6] # list of 'nepochs' event arrays

        # lts_info contains info common for all task runs:
        # label_names, vertno, rr
        # Save it if not done yet:
        if not ('ltc_info' in all_results):
            all_results['ltc_info'] = ltc_tuple[1:4]    # label_names, vertno, rr

        # Save data common for current task irrespective to band and eID: W, pz
        all_results[task_name]['W'] = ltc_tuple[4]
        all_results[task_name]['pz'] = ltc_tuple[5]

        for band in bands:
            # Extract envelopes and downsample
            henv = filter_downsample(config, data, band)    # (nepochs, nchan, ntime)
            all_results[task_name][band] = {}

            # Now henv has envelopes for current band for all events
            # Group them by event IDs
            for eID in event_ids:
                all_results[task_name][band][eID] = {}
                idx = get_epochs_for_event(henv, events_per_epoch, eID, return_idx = True)[1]
                all_results[task_name][band][eID]['henv'] = henv[idx,:,:]
                # In the saved averaged envelopes we assign events from the 1st encountered epoch
                # for the eID to both the mean and STD epochs.
                # !!! IMPORTANT !!!
                # !!! EVENTS ARRAYS ARE NOT CORRECTED FOR NEW SAMPLE RATE !!!
                events4eID = [events_per_epoch[i] for i,b in enumerate(idx) if b]
                all_results[task_name][band][eID]['events'] = 2*[events4eID[0]]

        # At this point, the all_results dictionary structure is as follows:
        # -----------------------------------------------------------------
        # all_resultsi = {
        #   'ltc_info': (label_names, vertno, rr)
        #   'task_runN': {          # for N = 1,..3
        #       'W': W,
        #       'pz': pz, 
        #       ...
        #       <band>: {           # for <band> = [4.0,7.0],...
        #           ...
        #           <eID>: {
        #               'henv': nepocs4eID x nlabels x ntime
        #               'events': 2*[earray]        # earray = neventsPerEpoch x 3 
        #               }                           # MNE events array FOR ORIGINAL SAMPLE RATE
        #           ...
        #           }
        #       ...
        #       }
        #   }

        print(f'Completed processing {in_file}')
    # ---- end of for (in_file, out_file) loop ------------

    print(f'\n** {STEP} step completed **\n')                

def henv_out_file(ss, pathname, band):  # QQQQ
    """
    Given source time course .hdf5 file pathname, generate an output file
    for a given frequency band.

    NOTE: this function is currently not used!!! (should be fixed in the
    future versions). This is why '_pwr' suffix is not added to the .hdf5
    file name even when power envelopes are actually calculated.

    Args:
        ss (Obj): reference to this app's object
        pathname (Path | str): .hdf5 file pathname
        band(list): [fmin, fmax] - frequency band, Hz

    Returns:
        out_pathname(str): full pathname of the output file with band included

    """
    cfg = ss.args['src_hilbert']
    sfx = cfg['suffix']
    pwr_sfx = 'pwr_' if cfg['power'] else ''

    fmin, fmax = band
    return str(pathname).replace(sfx,f'{sfx}{pwr_sfx}{fmin:.0f}-{fmax:.0f}Hz')

def filter_downsample(cfg, data, band):
    """
    Band-pass filter, extract hilbert envelope and downsample it to a target sample rate.

    Args:
        cfg(dict): this steps configuration dictionary
        data(ndarray): shape `(nepochs, nchan, ntime)`
        band(list): [fmin,fmax] frequency band, Hz

    Returns:
        envelope(ndarray): shape `(nepochs, nchan, ntime)` 

    """
    # Bandpass filtering
    # Filter design
    sos = butter(cfg['filter_order'], band, btype='band',
                    fs=cfg['original_sample_rate'], output='sos')

    # NOTE: Using sosfiltfilt() instead of just filtfilt() as recommended
    # in scipy's filtfilt() documentation
    out_data = sosfiltfilt(sos, data, axis = 2)

    # Extract analytic signal and envelope
    analytic_signal = hilbert(out_data, axis = 2)
    del out_data

    if cfg['power']:
        envelope = np.real(analytic_signal*analytic_signal.conj())
    else:
        envelope = np.abs(analytic_signal)

    del analytic_signal

    if cfg['target_sample_rate'] is None:
        return envelope

    # Downsample
    # Find target number of samples for the record of the same duration
    n0 = envelope.shape[2] - 1      # Total number of time steps

    # Number of new time steps - not an integer
    n1 = (n0 / cfg['original_sample_rate'])*cfg['target_sample_rate']
    target_samples = int(np.rint(n1)) + 1       # Target number of sample points
    return resample(envelope, target_samples, axis = 2, window = cfg['taper_window'])

def process_collected_results(config, all_results, bands, event_ids, henv_hdf5):
    """
    Combine calculated envelopes from all the task runs, calculate means,
    save results to .hdf5 files.

    Call to this function is triggered either when all current subject's
    task files are processed and new subject is about to be started,
    or when we reached the end of the `files` collection.

    The `all_results` dictionary has the following structure::

     all_results = {
       'ltc_info': (label_names, vertno, rr)
       'task_runN': {          # for N = 1,..3
           'W': W,
           'pz': pz, 
           ...
           <band>: {           # for <band> = [4.0,7.0],...
               ...
               <eID>: {
                   'henv': nepocs4eID x nlabels x ntime
                   'events': 2*[earray]        # earray = neventsPerEpoch x 3 
                   }                           # MNE events array FOR ORIGINAL SAMPLE RATE
               ...
               }
           ...
           }
       }

    This function creates .hdf5 source files with 2 epochs: [mean envelope; STD] for every
    combination of band, eID which are saved in the subject's source time course folder.

    Note that in the resulting .hdf5 file for any event ID, the events
    for both mean and std epochs are taken from the 1st encountered epoch for this
    eID. Moreover, **events arrays correspond to the original sample rate** - not 
    the sample rate used for the envelopes.

    Note also that we average `W`, `pz` over task runs before saving - which might
    not carry much physical meaning (especially for `pz`).

    Args:
        config(dict): this step's config dictionary
        all_results(dict): a dictionary with all collected data, as described above
        bands (tuple of tuples): a tuple with frequency bands to process as
            `(...,(fMin,fMax),...)`
        event_ids(list of int): list of possible event IDs
        henv_hdf5(pathlike): output .hdf5 file template pathname. The
            basename here does not contain band or eID, but does contain 'task_run'.
            The latter will be removed, and band and eID will be appended as
            appropriate to the output base name before .hdf5

    Returns:
        None

    """
    if not len(all_results):
        return      # No task runs for this subject/date

    # Collect available task runs for this subject/date
    task_runs = [key for key in all_results if 'task' in key]
    sfx = config['suffix']
    henv_hdf5 = str(henv_hdf5)
    # Remove the task name from the template .hdf5
    file_task_name = su.get_fif_task(henv_hdf5)
    henv_hdf5 = henv_hdf5.replace(file_task_name + '-','')

    # Remove '-evoked-' that can happen to be the last listed out_file
    # (this happens if this is the last file for a subject therefore
    # triggering a call to this function; at the same time, corresponding
    # 'evoked' in_file will not be processed / added to all_results 
    # (which is the correct behavior).
    henv_hdf5 = henv_hdf5.replace('-evoked-','')

    for eID in event_ids:
        # Count epochs for this eID in each task run
        epochs_per_run = [len(all_results[task][bands[0]][eID]['henv']) for task in task_runs]
        wts = np.array(epochs_per_run)/np.sum(epochs_per_run)

        # Calculate weighted averages of W (nchans x nlabels), pz (float)
        mean_W = np.sum([all_results[task]['W']*wts[i] for i,task in enumerate(task_runs)], axis = 0)
        mean_pz = np.sum([all_results[task]['pz']*wts[i] for i,task in enumerate(task_runs)])

        for band in bands:
            # Stack epochs from all runs
            henv = np.vstack([all_results[task][band][eID]['henv'] for task in task_runs])  # nallepochs x nlabels x ntime

            mean_henv = np.mean(henv, axis = 0)         # Mean envelope, nlabels x ntimes
            std_henv = np.std(henv, axis = 0, ddof = 1) # STD of the mean envelope, nlabels x ntimes
            out_ltc = np.array([mean_henv, std_henv])   # Created "epoched" data

            # Save results to .hdf5
            fmin, fmax = band
            # QQQQ
            # !!! This file name does not properly add '_pwr' suffix when required
            # Should be using henv_out_file() function instead
            # TODO: fix it in future versions
            ltc_file = henv_hdf5.replace(sfx,f'{sfx}{fmin}-{fmax}Hz_{eID}')
            write_roi_time_courses(ltc_file, out_ltc, *all_results['ltc_info'],
                        W = mean_W, pz = mean_pz,
                        events = all_results[task_runs[0]][band][eID]['events'],    # events come from 1st task run
                        events_id_dict = None)

