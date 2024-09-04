"""
**Event-related data processing routines**
"""
import numpy as np
import mne
from nearest_pos_def import nearestPD

def construct_epochs(ss, raw, all_events):
    """
    Construct epochs based on specified events subset.

    Args:
        ss(obj): reference to this app object
        raw (Raw): raw object representing continuous data, with only good
            sensor channels
        all_events(ndarray): (n_events,3) MNE events array from the original
            (full) raw object

    Returns:
        epochs(Epochs): the Epochs object
        events_per_epoch(list of ndarray): a list of events arrays for
            each epochs. Note that the event sample index is counted from
            the start of the epoch (not from the trigger)
        data_Cov(Covariance): MNE Covariance object for the active interval covariance
        noise_Cov(Covariance): MNE Covariance object for the control interval covariance
        inv_cov (ndarray): (nchan x nchan) (pseudo-) inverse of the data (active interval)
            covariance
        rank (int): rank of the data covariance
        pz (float): psedo-Z = SNR + 1 of the data; `pz=tr(data_cov)/tr(noise_cov)`

    """
    verbose = ss.args['verbose']
    conf = ss.args['src_rec']['epochs']

    # Add annotations to the raw object, because Epochs does not preserve
    # events for every epoch, but keeps annotations per epoch
    annots = mne.annotations_from_events(all_events, raw.info['sfreq'],
                    event_desc=None,
                    first_samp=raw.first_samp,
                    orig_time=None,
                    verbose=verbose)

    raw.set_annotations(annots, emit_warning=True, on_missing='raise',
                    verbose=verbose)

    button_events = mask_events(all_events, int(conf['btn_mask'], 16))
    non_button_events = mask_events(all_events, int(conf['non_btn_mask'], 16))

    epochs = mne.Epochs(raw, tmin = conf['t_range'][0], tmax = conf['t_range'][1],
                        events = non_button_events, # Remove button events to avoid a rare case
                        **conf['create_epochs'],    # when a button event 'spoils' a question event
                        verbose = verbose)
    # At this point, epoch.events is a standard events array which only contains
    # events used as epochs origins: samples are counted from the beginning (as in
    # the raw object), but all events except specified in conf['create_epochs']['event_id']
    # are dropped. The epoch.events_id attribute is a dict {..., 'id': id, ...}, again with
    # the event IDs only used as epochs origins.
    # Information regarding other events per epoch can be obtained only by calling
    # epochs.get_annotations_per_epoch()

    #epochs.plot(events = all_events, event_id = True, block = True)   # Debug

    events_per_epoch = get_events_per_epoch(epochs)

    data_Cov, noise_Cov, inv_cov, rank, pz = epochs_noise_and_inv_cov(epochs, conf,
                                                    verbose = verbose)

    return epochs, events_per_epoch, data_Cov, noise_Cov, inv_cov, rank, pz

def mask_events(all_events, mask):
    """
    Select events with non-zero IDs after mask is applied.

    Args:
        all_events(ndarray of int): (n_events,3) input events array
        mask(int): as is, unsigned integer

    Returns:
        masked_events(ndarray of int): (n_masked_events,3) output events array

    """
    masked_events = all_events.copy()
    masked_events[:,2] &= mask
    idx = masked_events[:,2] > 0
    return masked_events[idx,:]

def get_events_per_epoch(epochs):
    """
    Create a standard `events` array for each epoch, as if
    an epoch were a single record. The input `epochs` object
    should carry annotations inherited from the original continuous
    `raw` object.

    Args:
        epochs(Epochs): the MNE Python Epochs object

    Returns:
        events_per_epoch(list of ndarray): a list of events arrays for
            each epochs. Note that event sample index is counted from
            the start of the epoch (not from the trigger)

    """
    # Get a list of lists of tuples like (time, duration, strID):
    # [(0.0, 0.0, '41'), (1.056, 0.0, '131'), (2.366, 0.0, '1024')]
    # Note that sometimes time may be negative if some event lands
    # into a control interval
    annots = epochs.get_annotations_per_epoch()
    events_per_epoch = []

    for la in annots:
        # Events array for current epoch:
        events = np.zeros((len(la),3), dtype=int)

        for i,a in enumerate(la):
            # Get the sample number relative to the epoch start (not rel to trigger!)
            events[i,0] = epochs.time_as_index(a[0], use_rounding=False)
            events[i,2] = int(a[2])     # This is the event code

        events_per_epoch.append(events)

    return events_per_epoch
 
def epochs_noise_and_inv_cov(epochs, conf, verbose = None):
    """
    For epoched data, calculate: full (or data) covariance, which is the sensor covariance
    on active interval; noise covariance, which is the sensor covariance on
    control interval; (pseudo) inverse of the data covariance for beamformer
    weights calculations.

    The function also finds pseudo-Z of the data as a ratio `pz=tr(data_cov)/tr(noise_cov)`.

    Args:
        epochs(Epochs): MNE Python epochs object
        conf(dict): configuration dictionary corresponding to `'epochs'` key
            in the global JSON configuration file
        verbose(str): MNE Python verbose level

    Returns:
        data_cov(ndarray): `nchan x nchan` the data (active interval) covariance
        noise_cov(ndarray): `nchan x nchan` the noise (control interval) covariance
        inv_cov (ndarray): `nchan x nchan` (pseudo-) inverse of the data (active interval)
            covariance
        rank (int): rank of the data covariance
        pz (float): psedo-Z = SNR + 1 of the data; `pz=tr(data_cov)/tr(noise_cov)`

    """
    noise_Cov = mne.compute_covariance(epochs, tmin=conf['t_control'][0], tmax=conf['t_control'][1],
                           **conf['compute_cov'], verbose = verbose)

    data_Cov = mne.compute_covariance(epochs, tmin=conf['t_active'][0], tmax=conf['t_active'][1],
                           **conf['compute_cov'], verbose = verbose)

    # NOTE 1: 'nfree' attribute of the covariance is not its rank; it is found
    # as the number of samples used to calculate cov minus number of conditions.
    # In our case we subtract the mean, so nfree = nsamples - 1
    # NOTE 2: If rank arg for the compute_rank() call is 'info', the rank is **overestimated**
    # by 1 because it is taken from epochs.info, which does not take into account the mean subtraction.

    rank_dict = mne.compute_rank(data_Cov, rank=None, scalings=None, info=epochs.info, tol='auto', proj=True,
                                 tol_kind='absolute', on_rank_mismatch='ignore', verbose=verbose)

    rank = sum(rank_dict.values())

    inv_cov = invert_for_rank(data_Cov.data, rank)
    pz = np.trace(data_Cov.data) / np.trace(noise_Cov.data)

    # In case of degenerate covariance matrices after max-filtering, MNE sets all eigenvalues above known
    # rank to 0. Technically, such matrices are no longer positively defined. For them to pass the PD test,
    # use a closest positively defined matrices instead
    data_cov = nearestPD(data_Cov.data)     # Verified that the rank computed above remains the same
    noise_cov = nearestPD(noise_Cov.data)

    return data_cov, noise_cov, inv_cov, rank, pz

def invert_for_rank(data, rank):
    """
    Calculate pseudo-inverse of a Hermitian matrix using a precalculated
    rank value. Return a closest positive-definite matrix.

    Args:
        data(ndarray): a Herimitian matrix
        rank(int > 0): rank to be used for inverting

    Returns:
        inv_data(ndarray): positively defined pseudo-inverse of data

    """
    U, S, _ = np.linalg.svd(data, full_matrices=False, hermitian=True)

    # Note that all S[i] are positive and sorted in a decreasing order
    U = U[:, :rank]

    # (Pseudo-) inverse of the covariance
    # Make it pos def instead of fully degenerate
    return nearestPD(U @ np.diag(1./S[:rank]) @ U.T)

