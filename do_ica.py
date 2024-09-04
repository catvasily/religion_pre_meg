"""
**Perform ECG and EOG artifacts removal using ICA.**
"""
import warnings
import mne
import setup_utils as su

def do_ica(ss):
    """
    To do the ICA based artifact removal, the following steps are performed.
    First, EOG and ECG channels are filtered to respective (generally different)
    frequency bands. Then ICA model is fit to the data. Finally, ICs related to
    EOG or ECG channels are removed. Original (unfiltered) versions of the EOG
    and ECG channels are preserved in the output data.

    Args:
        ss(obj): reference to this app object

    Returns:
        None

    """
    warnings.filterwarnings('ignore', category=RuntimeWarning,
                            message=ss.args["file_name_warning"])

    STEP = 'ica'
    config = ss.args[STEP]
    config['init']['random_state'] = ss.args['seed']

    if ss.data_host.cluster_job:
        config['show_plots'] = False

    files = su.files_to_process(ss, STEP)
    for in_fif, out_fif in files:
        # Skip non-raw .fif files, just in case
        if mne.what(in_fif) != 'raw':
            continue

        # Read the subject's next .fif file
        raw = mne.io.read_raw_fif(in_fif, allow_maxshield=False, preload=True,
                      on_split_missing='raise', verbose = ss.args['verbose'])
        raw = ica_raw(raw, config, verbose = ss.args['verbose'])
        raw.save(out_fif, **config["save"], verbose = ss.args['verbose'])
        print(f'{in_fif.name} done')

    warnings.filterwarnings("default", category=RuntimeWarning)
    print(f'\n** {STEP} step completed **\n')                

def ica_raw(raw, config, verbose = None):
    """
    Perform ICA artifact removal for a single Raw record.

    Args:
        raw (MNE Raw): the input dataset to be processed in place
        config(dict): the `ica` step config dictionary
        verbose(str): MNE Python verbose level

    Returns:
        raw (MNE Raw): the input dataset modified in place

    """
    n_components = su.get_maxfiltered_rank(raw)

    if n_components is not None:
        kwargs = config['init'].copy()
        kwargs['n_components'] = n_components
        ica = mne.preprocessing.ICA(**kwargs) 
    else:
        ica = mne.preprocessing.ICA(**config['init']) 

    ica.fit(raw, **config['fit'], verbose = verbose)

    if config['show_plots']:
        ica.plot_sources(raw, **config['plot_sources'])

    eog_indices, eog_scores = ica.find_bads_eog(raw, **config['find_bads_eog'],
                                                verbose = verbose)

    ecg_indices, ecg_scores = ica.find_bads_ecg(raw, **config['find_bads_ecg'],
                                                verbose = verbose)

    ica.exclude.extend(eog_indices)
    ica.exclude.extend(ecg_indices)
    raw = ica.apply(raw, verbose = verbose) 

    return raw

