"""
**Plot a grid of source waveforms for sets of channels (ROIs) and
multiple conditions**
"""
import warnings
import numpy as np
import setup_utils as su
from src_rec import read_roi_time_courses
from plot_array import plot_array

def plot_waveforms(ss):
    """
    Create an array (grid) plot of waveforms for a set of channels
    and different conditions.

    Args:
        ss(obj): reference to this app object

    """
    warnings.filterwarnings('ignore', category=RuntimeWarning,
                            message=ss.args["file_name_warning"])

    STEP = 'plot_waveforms'
    config = ss.args[STEP]

    if ss.data_host.cluster_job:
        config['show_plots'] = False

    # Supported plotting tasks
    all_tasks = ('compare_beams', 'evoked_std', 'compare_events')

    task = config['task']

    if task not in all_tasks:
        raise ValueError(f'Unknown task {task}')

    if config['files'] is None:
        raise ValueError('Files list for this step cannot be null')

    # Function to generate comparison file pathname
    get_file_to_compare = lambda in_file: str(in_file).replace( \
            config['in_dir'], config['compare_beams']['compare_dir'])

    files = su.files_to_process(ss, STEP)
    chlist = config['channels']
    SR = ss.args['prefilter']['target_sample_rate']
    t0 = ss.args['src_rec']['epochs']['t_range'][0]
    lst_events = ss.args['src_rec']['events_for_evoked']

    for in_file, out_file in files:
        save_file = out_file if config['save_plots'] else None
        label_tcs, label_names = read_roi_time_courses(in_file)[:2]
        # label_tcs is nepochs x nlabels x ntimes (for epoched data)
        # label_names (nlabels,) vector of ROI names

        data_arrays = []
        data_arrays.append(extract_channels(label_tcs[0], label_names, chlist))
        x_values = np.arange(label_tcs.shape[2])/SR + t0

        if task == 'compare_beams': 
            compare_file = get_file_to_compare(in_file)
            label_tcs1 = read_roi_time_courses(compare_file)[0]
            data_arrays.append(extract_channels(label_tcs1[0], label_names, chlist))

            data_arrays = adjust_signs(data_arrays) if config['adjust_signs'] else data_arrays
            plot_array(data_arrays, x_values=x_values, fname=save_file, chnames=chlist,
                       show=config['show_plots'], **config[task]['plot_args'])
        elif task == 'evoked_std':
            data_arrays.append(extract_channels(label_tcs[1], label_names, chlist))

            data_arrays = adjust_signs(data_arrays) if config['adjust_signs'] else data_arrays
            plot_array(data_arrays, x_values=x_values, fname=save_file, chnames=chlist,
                       show=config['show_plots'], **config[task]['plot_args'])
        elif task == 'compare_events':
            data_arrays = collect_conditions_evoked_data(in_file, lst_events, label_names, chlist)

            data_arrays = adjust_signs(data_arrays) if config['adjust_signs'] else data_arrays
            plot_array(data_arrays, x_values=x_values, fname=save_file, chnames=chlist,
                        cond_names = [f'{eID}' for eID in lst_events], 
                        show=config['show_plots'], **config[task]['plot_args'])

        if save_file:
            print(f'Saved plot to {out_file}')

    warnings.filterwarnings("default", category=RuntimeWarning)
    print(f'\n** {STEP} step completed **\n')                

def extract_channels(data, chnames, chlist):
    """
    Return a subset of waveforms corresponding to specified labels.

    NOTE:
        Raises exception if any channel in `chlist` does not belong to
        `chnames`

    Args:
        data(ndarray): shape `(nchan,ntimes)` all channels data
        chnames(list of str): channel names
        chlist(list of str): a list of channels to return

    Returns:
        chdata(ndarray): shape `(len(chlist),ntimes)` - the requested
            channels waveforms

    """
    lst = list(chnames)
    idx = [lst.index(ch) for ch in chlist]
    return  data[idx,:]

def collect_conditions_evoked_data(in_file, lst_events, label_names, chlist):
    """
    Construct a `data_arrays` list corresponding to all conditions
    evoked data for current subject and task.

    Args:
        in_file(Path): pathname to a single evoked data file. This file
            will be used to construct file paths for all the conditions.
        lst_events(list of int): a list of event codes for events to be compared
        label_names(list of str): channel names
        chlist(list of str): a list of channels to return

    Returns:
        data_arrays(list of ndarray): List of NumPy arrays of shape `(nchan, nx)`,
            one for each experiment.
    """
    PTRN = 'ltc-evoked-'
    fname = str(in_file)
    idx = fname.find(PTRN)

    if idx == -1:
        raise ValueError('Invalid input file; should be an evoked source time course .hdf5 file')

    data_arrays = []
    for eID in lst_events:
        ltc_file = fname[:idx] + PTRN + f'{eID}.hdf5'
        label_tcs = read_roi_time_courses(ltc_file)[0]
        data_arrays.append(extract_channels(label_tcs[0], label_names, chlist))

    return data_arrays

def adjust_signs(data_arrays):
    """
    For a list of identically shaped `nchan x ntimes` data arrays,
    possibly flip the signs of channels for the waveforms to better
    correlate with the first data array waveforms. This is done by calculating
    scalar product for two possible signs and choosing the sign with the
    largest product.

    Args:
        data_arrays(lst of ndarray): a list of `nchan x ntimes` data arrays

    Returns:
        data_arrays(lst of ndarray): the input list where in 2nd and further
            items some channels have their signs flipped

    """
    if len(data_arrays) <= 1:
        return data_arrays

    da0 = data_arrays[0]
    res = [da0]

    for da in data_arrays[1:]:
       s0 = np.einsum('ij,ij->i', da0, da) 
       s1 = np.einsum('ij,ij->i', da0, -da)
       flip = s1 > s0

       if np.any(flip):
           dnew = da.copy()
           dnew[flip,:] = -dnew[flip,:]
           res.append(dnew)
       else:
           res.append(da)

    return res

