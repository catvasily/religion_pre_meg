"""
**Event-related data processing routines**
"""
import numpy as np
import mne
from nearest_pos_def import nearestPD

def construct_epochs(ss, raw, all_events, skip_cov_calc = False):
    """
    Construct epochs based on specified events subset.

    Args:
        ss(obj): reference to this app object
        raw (Raw): raw object representing continuous data, with only good
            sensor channels
        all_events(ndarray): (n_events,3) MNE events array from the original
            (full) raw object
        skip_cov_calc(bool): flag to skip covariance and rank calculations;
            defautl `False`

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

    NOTE:
        When `skip_cov_calc = True`, returned variables `data_Cov, noise_Cov, inv_cov, rank, pz`
        are all set to `None`

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

    conf['create_epochs']['baseline'] = conf['t_control']
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

    if skip_cov_calc:
        data_Cov, noise_Cov, inv_cov, rank, pz = None,None,None,None,None
    else:
        rcond = ss.args['src_rec']['beam']['rcond']
        data_Cov, noise_Cov, inv_cov, rank, pz = epochs_noise_and_inv_cov(epochs, conf,
                                                    rcond = rcond, verbose = verbose)

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
        events_per_epoch(list of ndarray): a list of events arrays, one for
            each epoch. Each ndarray is `nevents_in_epoch x 3` array of ints
            as described in MNE Python docs. Note that the event sample index
            is counted from the start of the epoch (not from the trigger)

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

def get_epochs_for_event(epochs, events_per_epoch, eID):
    """
    Return a subset of epochs which contain specified event.

    Args:
        epochs(Epochs): the MNE Python Epochs object
        events_per_epoch(list of ndarray): a list of events arrays, one for
            each epoch. Each ndarray is `nevents_in_epoch x 3` array of ints
            as described in MNE Python docs. Note that the event sample index
            is counted from the start of the epoch (not from the trigger)
        eID (int): event ID (event code). Event codes are listed in the 3d
            column of the events array for the epoch: `ids = events[:,2] 

    Returns:
        epochs4id(Epochs): the MNE Python Epochs object for specified event ID

    """
    if len(epochs) != len(events_per_epoch):
        raise ValueError('Lengths of "epochs" and "events_per_epoch" must match')

    idx = [(eID in earray[:,2]) for earray in events_per_epoch]
    return epochs[idx]

def epochs_noise_and_inv_cov(epochs, conf, rcond = 1e-15, verbose = None):
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
        rcond (float > 0): singular values less or equal to max(sing val) * rcond will
            be dropped
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

    inv_cov = invert_for_rank(data_Cov.data, rank, rcond = rcond)
    pz = np.trace(data_Cov.data) / np.trace(noise_Cov.data)

    # In case of degenerate covariance matrices after max-filtering, MNE sets all eigenvalues above known
    # rank to 0. Technically, such matrices are no longer positively defined. For them to pass the PD test,
    # use a closest positively defined matrices instead
    data_cov = nearestPD(data_Cov.data)     # Verified that the rank computed above remains the same
    noise_cov = nearestPD(noise_Cov.data)

    return data_cov, noise_cov, inv_cov, rank, pz

def epochs_evoked_cov(sensor_data, smin, smax):
    """
    Calculate a matrix of 2nd moments of mean (evoked) time courses
    for specified time interval:

    `C = AVG[<e(s)><e(s)^T>;smin,smax]`

    Here `e(s)` is a column vector of sensor values for the sample number `s`,
    `<..>` denotes averaging over epochs, and AVG[x(s); smin, smax] performs
    averaging over samples interval `[smin, smax)`.

    Args:
        sensor_data(ndarray): shape `(n_epochs, n_channels, n_times)` epochs time
            courses data
        smin (int): first sample of the time-averaging interval (inclusive)
        smin (int): last sample of the time-averaging interval (exclusive)

    Returns:
        C (ndarray): shape `(n_channels, n_channels)` - the `C` matrix described
            above

    """
    nepochs,nchans,nsamples = sensor_data.shape

    invalid_s = lambda s,nsamples: True if (s<0) or s>=nsamples else False

    if invalid_s(smin,nsamples) or invalid_s(smax,nsamples):
        raise ValueError('smin, smax should belong to the interval [0,nsamples)')

    if smin == smax:
        raise ValueError('smin, smax cannot be equal to each other')

    if smin > smax:
        s = smax
        smax = smin
        smin = s

    # First, average over epochs
    data = np.mean(sensor_data[:,:,smin:smax], axis = 0)   # nchans x ntime

    # This is the 2nd moments matrix averaged over ntime samples
    return (data @ data.T) / (smax - smin)

def invert_for_rank(data, max_rank, rcond = 1e-15):
    """
    Calculate pseudo-inverse of a Hermitian matrix using a precalculated
    maximum rank value or tolerance condition (`rcond`). Return a closest
    positive-definite matrix.

    Args:
        data(ndarray): a Herimitian matrix
        max_rank(int > 0): maximum rank to be used for inverting
        rcond (float > 0): singular values less or equal to max(sing val) * rcond will
            be dropped

    Returns:
        inv_data(ndarray): positively defined pseudo-inverse of data

    """
    U, S, _ = np.linalg.svd(data, full_matrices=False, hermitian=True)

    # Drop close to 0 singular values and corresponding columns of U. Note that all S[i]
    # are positive and sorted in decreasing order
    t = rcond * S[0]
    U = U[:, S > t]
    rank = U.shape[1]	# The actual rank of the data covariance

    if rank > max_rank:
        rank = max_rank
        U = U[:, :rank]

    # (Pseudo-) inverse of the covariance
    # Make it pos def instead of fully degenerate
    return nearestPD(U @ np.diag(1./S[:rank]) @ U.T)

