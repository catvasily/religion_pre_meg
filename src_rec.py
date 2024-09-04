"""
**Perform automatic MRI/HEAD coordinate systems coregistration
and beamformer source reconstruction**

NOTE that this code uses external beamformer reconstruction routines located
in folder `beam_python`. This folder is expected to be found at the same level
of the files hierarchy as a folder with this source file.
"""
import warnings
import sys
import os.path as path
import numpy as np
import json
import h5py        # Needed to save/load files in .hdf5 format
import mne
from mne.coreg import Coregistration
from mne.io import read_info
import setup_utils as su
from bem_model import src_space_fif, bem_sol_pathname
from nearest_pos_def import nearestPD
from epochs import construct_epochs

__file__ = path.realpath(__file__)    # expand potentially relative path to a full one
sys.path.append(path.dirname(path.dirname(__file__))+ "/beam-python")

from construct_mcmv_weights import is_pos_def, construct_single_source_weights

_LEFT_HEMI_ZERO = -1_111_111
""" This value is interpreted as the vertex #0 of the left hemisphere surface
"""

def src_rec(ss):
    """
    Perform beamformer source reconstruction.

    Args:
        ss(obj): reference to this app object

    """
    warnings.filterwarnings('ignore', category=RuntimeWarning,
                            message=ss.args["file_name_warning"])

    STEP = 'src_rec'
    config = ss.args[STEP]

    if ss.data_host.cluster_job:
        config['show_plots'] = False

    files = su.files_to_process(ss, STEP)
    for in_fif, out_fif in files:
        # Skip non-raw .fif files, just in case
        if mne.what(in_fif) != 'raw':
            continue

        trans_fif = su.get_trans_file_pathname(out_fif)
        meg_subject = su.fif_subject(in_fif)

        # Ensure HEAD_>MRI transformation file is available
        if trans_fif.is_file():
            print(f'coregister_mri(): using HEAD->MRI transformation file {trans_fif}')
        else:
            # Construct HEAD -> MRI transform, if not done already 
            info = read_info(in_fif)
            coregister_mri(config['coreg'], info, meg_subject, ss.data_host.get_mri_subjects_dir(),
                           trans_fif, config['show_plots'], verbose = ss.args['verbose'])

        # Get pathnames to some files needed for forward calculations
        lst_fwd_fifs = get_fwd_input_fifs(ss, out_fif)   # fwd_fif, ss_fif, bem_sol_fif

        # Get forward solutions; calculate those if not done already
        fwd = get_forward_solutions(config, in_fif, trans_fif, *lst_fwd_fifs,
                          verbose = ss.args['verbose'])

        # Compute beamformer solutions for all sources
        # --------------------------------------------
        raw = mne.io.read_raw_fif(in_fif, allow_maxshield=False, preload=True,
                                  on_split_missing='raise', verbose=ss.args['verbose'])

        # Get events, if any
        events_file = su.events_fif(ss, out_fif)    # Path to the events file

        if events_file.is_file():
            all_events = mne.read_events(events_file,
                                include=None, exclude=None, mask=None,
                                mask_type='and',
                                return_event_id=False, verbose=ss.args['verbose'])
        else:
            print(f'WARNING: did not find events file for {in_fif}; extracting events now.')

            # Get events from the current step's data
            config['epochs']['find_events']['min_duration'] = 2./raw.info['sfreq']
            all_events = mne.find_events(raw, **config['epochs']['find_events'],
                                         verbose=ss.args['verbose'])

        # Ensure that bad channels are dropped; keep only sensor channels 
        picks = config['sensor_channels_for_src_rec']
        raw = raw.pick(picks, exclude = 'bads')

        task_name = su.get_fif_task(in_fif)

        if su.extract_epochs(task_name):
            epochs, events_per_epoch, data_cov, noise_cov, inv_cov, rank, pz = \
                construct_epochs(ss, raw, all_events)

            sensor_data, W, U = beamformer_stc_epochs(epochs, fwd, inv_cov, noise_cov,
                                            pz = pz, units=config['beam']['units'], verbose = ss.args['verbose'])
        else:
            # Get the events array with sample numbers relative to the data start,
            # not relative to the time when Electa system's hardware started 
            events_for_stc = all_events.copy()
            events_for_stc[:,0] -= raw.first_samp
            _, sensor_data, data_cov, W, U, pz = beamformer_stc_continuous(raw, fwd, **config['beam'],
                                                   verbose = ss.args['verbose'])

        # Compute ROI time courses
        # ------------------------
        labels, label_coms_vtc, label_coms_rr = get_labels(meg_subject, config, ss.data_host,
                                           fwd, verbose = ss.args['verbose'])
        label_tcs, label_wts, is_epochs = beam_extract_label_time_course(sensor_data, data_cov, labels, fwd, W,
                                    mode = config["roi_time_course_method"],
                                    verbose = ss.args['verbose'])

        ltc_hdf5 = su.ltc_file_pathname(meg_subject, task_name, lst_fwd_fifs[1], config['atlas'], out_fif.parent)
        label_names = [l.name for l in labels]

        write_roi_time_courses(ltc_hdf5, label_tcs, label_names,
            vertno = label_coms_vtc, rr = label_coms_rr, W = label_wts, pz = pz,
                               events = events_per_epoch if is_epochs else events_for_stc)

        if ss.args['verbose'].upper() == 'INFO':
            print(f'ROI time courses saved to {ltc_hdf5}')

        print(f'Processing of {in_fif} completed\n')

        """ DEBUG
        # Verify hdf5 read/write:
        label_tcs1, label_names1, vertno1, rr1, W1, pz1, events1, _ = read_roi_time_courses(ltc_hdf5)
        assert np.all(np.equal(label_tcs, label_tcs1))
        assert np.all(np.equal(label_coms_vtc, vertno1))
        assert np.all(np.equal(label_coms_rr, rr1))
        assert np.all(np.equal(label_wts, W1))
        assert np.isclose(pz,pz1)
        assert np.all(np.equal(label_names, label_names1))

        if is_epochs:
            assert all([np.equal(e,e1).all() for e,e1 in zip(events_per_epoch,events1)])
        else:
            assert np.equal(events_for_stc, events1).all()
        """

def coregister_mri(config, info, subject, subjects_dir, trans_fif, show_plots = True, verbose = None):
    """
    Perform automated HEAD - MRI coordinate systems coregistration.

    Args:
        config(dict): coregistration sub-step configuration dictionary
        info(mne.Info): the info object corresponding to the input raw .fif file
        subject(str): subject ID on MEG side
        subjects_dir(Path): path to the root of FreeSurfer subjects folder 
        trans_fif(Path): full pathname of .fif file to save the transformation
        show_plots(bool): flag to show the coregistration plot
        verbose(str): MNE Python verbose level

    Returns:
        None, but transformation .fif file is saved in this step's subject output
        folder.

    """
    mri_subj = su.meg2mri_subject(subject)
    coreg = Coregistration(info, mri_subj, subjects_dir, **config['init'])
    coreg.fit_fiducials(**config['fit_fiducials'], verbose=verbose)
    coreg.fit_icp(**config['icp1'], **config['fit_icp'], verbose=verbose)
    coreg.omit_head_shape_points(distance=config['omit_dist'])
    coreg.fit_icp(**config['icp2'], **config['fit_icp'], verbose=verbose)
    
    mne.write_trans(trans_fif, coreg.trans, overwrite=True, verbose=verbose)
    print(f'coregister_mri(): {subject} HEAD->MRI transformation saved as {trans_fif}')

    if verbose.upper() == 'INFO':
        dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
        print(
            f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm "
            f"/ {np.min(dists):.2f} mm / {np.max(dists):.2f} mm"
        )

    if show_plots:
        fig = mne.viz.plot_alignment(info = info, trans=coreg.trans, subject = mri_subj, 
                    subjects_dir = subjects_dir, **config['plot_alignment'], verbose = verbose)
        # view_kwargs = dict(azimuth=45, elevation=90, distance=0.6, focalpoint=(0.0, 0.0, 0.0))
        # mne.viz.set_3d_view(fig, **view_kwargs)
        input('Press <ENTER>>...')

def get_fwd_input_fifs(ss, out_fif):
    """
    Return full Path objects for .fif files needed for forward solutions
    calculation.

    Args:
        ss(obj): reference to this app object
        out_fif(Path): output .fif file pathname

    Returns:
        lst_fifs(list of Path): list of Path objects as `(fwd_fif, ss_fif, bem_sol_fif)`
            for forward solutions file, source space file and BEM solution file,
            respectively.

    """
    meg_subj = su.fif_subject(out_fif)
    mri_subj = su.meg2mri_subject(meg_subj)
    config = ss.args['bem_model']
    is_volume = config['vol_src_space']
    ico_src = config['ico_src']
    step = config['grid_step']

    # Source space base name; it is supposed to end in -src.fif:
    ss_name = src_space_fif(mri_subj, step, ico_src, is_volume)
    bem_dir = ss.data_host.get_subject_bem_dir(mri_subj)
    ss_pathname = bem_dir / ss_name

    # Forward solutions base name; it is supposed to end in -fwd.fif:
    fwd_name = ss_name.replace('-src.fif', '-fwd.fif')
    # Also, replace MRI subject with MEG subject
    fwd_name = fwd_name.replace(mri_subj, meg_subj)

    # BEM solution .fif
    bem_sol_fif = bem_sol_pathname(mri_subj, ss.data_host, config)
    # fwd_fif, ss_fif, bem_sol_fif:
    return out_fif.parent / fwd_name, ss_pathname, bem_sol_fif 

def get_forward_solutions(config, in_fif, trans_fif, fwd_fif, ss_fif, bem_sol_fif,
                          verbose = None):
    """
    Get forward solutions for the given MEG .fif record.
    Those are (re)calculated if not already present in the output
    folder.

    Args:
        config(dict): this step configuration dictionary
        in_fif(Path): input .fif file pathname
        trans_fif(Path): HEAD->MRI transformation .fif file pathname
        fwd_fif(Path): forward solutions .fif file pathname, to
            save to or to read from
        ss_fif(Path): source space .fif file pathname
        bem_sol_fif(Path): BEM solution .fif file pathname
        verbose(str): MNE Python verbose level

    Returns:
        fwd(Forward): an instance for MNE Forward class

    """
    if verbose is None:
        verbose = 'info'

    if config['recalc_forward'] or not fwd_fif.is_file():
        fwd = mne.make_forward_solution(
                  in_fif, 
                  trans = trans_fif,
                  src = ss_fif, 
                  bem = bem_sol_fif,
                  **config['make_fwd'],
                  verbose = verbose
                  )

        mne.write_forward_solution(fwd_fif, fwd, overwrite=True, verbose=verbose)

        if verbose.upper() == 'INFO':
            print(f'Forward solutions saved to file {fwd_fif}')
    else:
        if verbose.upper() == 'INFO':
            print(f'Using existing forward solution {fwd_fif}')

        fwd = mne.read_forward_solution(fwd_fif, verbose=verbose) 

    return fwd

def beamformer_stc_continuous(raw, fwd, *, return_stc = True, units = 'pz',
                           tol = 1e-2, rcond = 1e-10, verbose = None):
    """
    Reconstruct source time courses for continuous data using the single source
    minimum variance beamformer.

    IMPORTANT: if `picks` specifies only a subset of channels available in the forward
    solution `fwd`, **`fwd` object will be modified in place** as channels not included
    in `picks` will be dropped.

    Args:
        raw (mne.Raw): the raw data; it is expected to contain only
            those sensor channels that will be used to perform beamforming
        fwd (mne.Forward): forward solutions; note that those may include all meg channels,
            not only those specified by `picks` parameter
        return_stc (bool): flag to compute all source time courses and to return
            corresponding SourceEstimate object
        units (str): either "source" or "pz", to obtain timecourses in A*m or amplitude
            pseudo-Z, respectively. Note that in the 1st case source amplitudes will grow
            large for deep sources.
        tol (float > 0): tolerance (relative accuracy) in finding the noise_cov trace.
        rcond (float > 0): condition for determining rank of the covariance matrix:
            singular values less than `max(sing val) * rcond` will be considered zero.
        verbose (str or None): verbose mode (see MNE docs for details)

    Returns:
        stc (mne.SourceEstimate or None): source estimate (reconstructed source time courses),
            provided return_stc flag is True; None otherwise
        sensor_data (ndarray): `nchan x ntimes`; M/EEG channels time courses
        data_cov (ndarray): `nchan x nchan`, sensor covariance matrix adjusted to a nearest
            positive definite matrix
        W (ndarray): `nchan x nsrc` array of beamformer weights
        U (ndarray): `3 x nsrc` array of source orientations
        pz (float): data's pseudo-Z found as `pz = trace(R)/tr(N)`, where `N` is the noise covariance

    """
    # Compute sample covariance
    sensor_data = raw.get_data(    # sensor_data is nchannels x ntimes
        start=0,                # starting time sample number (int)
        stop=None,
        reject_by_annotation=None,
        return_times=False,
        units=None,             # return SI units
        verbose=verbose)

    nchan = sensor_data.shape[0]

    # We use nearestPD() because degenerate cov may have negative EVs
    # due to rounding errors
    data_cov = nearestPD(np.cov(sensor_data, rowvar=True, bias=False))

    fwd = fwd.pick_channels(raw.ch_names, ordered = False)

    # Data covariance is always degenerate due to avg ref, interpolated channels (for EEG),
    # max-filtering and out-projections, etc.
    max_rank = su.get_maxfiltered_rank(raw)
    noise_cov, inv_cov, rank, pz = construct_noise_and_inv_cov(fwd, data_cov, tol = tol,
                                        rcond = rcond, max_rank = max_rank)

    if verbose.upper() == 'INFO':
        print('Data covariance: nchan = {}, rank = {}'.format(nchan, rank))

    # W is nchan x nsrc, U is 3 x nsrc
    W, U = get_beam_weights(fwd['sol']['data'], inv_cov, noise_cov, units = units) 
    W, U, vertices = set_source_orientation_signs(fwd, W, U)

    # We normalize time sources on sqrt(PZ), which is equivalent to have tr(N) = tr(R).
    # This way time courses for all subjects will be scaled identically. Proof:
    #    pz = tr(R) / tr(N) by definition
    #    w = Rinv*h/sqrt(hRinv N Rinv h) = sqrt(pz) w1,
    #  where
    #    w1 = Rinv*h/sqrt[hRinv (pz*N) Rinv h]
    # Obviously tr(pz*N) = tr(R), thus w1 corresponds to noise with power equal to that of R.
    # So by normalizing w on sqrt(pz) we ensure tr(R) = tr(N) for all subjects.
    W /= np.sqrt(pz)

    # Create SourceEstimate object to return
    if return_stc:
        src_data = (W.T @ sensor_data)
        stc = mne.SourceEstimate(
            src_data,
            vertices, 
            tmin = raw.times[0],
            tstep = raw.times[1] - raw.times[0],
            subject='fsaverage',
            verbose=verbose)
    else:
        stc = None

    if verbose.upper() == 'INFO':
        print('Beamformer weights / time courses calculation completed.')

    return stc, sensor_data, data_cov, W, U, pz

def construct_noise_and_inv_cov(fwd, data_cov, *, tol = 1e-2, rcond = 1e-10, max_rank = None):
    """Based on the forward solutions, construct sensor-level noise covariance
    matrix assuming white, randomly oriented uncorrelated sources. Also calculate
    pseudo-inverse of the data cov and SNR.

    The basic expression for noise covariance is:

    `cov0 = const * SUM(i=1 to Nsrc){Hx Hx' + Hy Hy' + Hz Hz'}`

    where `Hx,y,z(i)` are forward solutions for i-th source with corresponding
    orientations, and `const` is defined so as data_cov - noise_cov is non-negative.

    For degenerate data_cov the noise_cov should also be degenerate with a 
    range subspace coinciding with the range of the data_cov. In this case the
    above expression should be replaced with

    `cov = P * cov0 * P`

    where `P = data_cov * pinv(data_cov)` is a projector on the `range(data_cov)`.

    The trace of the noise_cov is maximized while keeping the difference
    `data_cov - noise_cov` non-negative. The tol parameter defines how close to
    the upper boundary one should get.

    Args:
        fwd (Forward): mne Python forward solutions class
        data_cov (ndarray): nchan x nchan data covariance matrix
        tol (float > 0): tolerance in finding the noise_cov trace.
        rcond (float > 0): singular values less than max(sing val) * rcond will
            be dropped
        max_rank(int>0 | None): if specified, sets the upper bound for rank of
            the covariance. The upper bound may be known if data is max-filtered.
            In this case, any computed rank larger than max_rank will be incorrect.

    Returns:
        cov (ndarray): (nchan x nchan) noise cov matrix, such that the difference
            data_cov - noise_cov is non-negatively defined
        inv_cov (ndarray): (nchan x nchan) (pseudo-) inverse of the data cov
        rank (int): rank of the data covariance
        pz (float): psedo-Z = SNR + 1 of the data; `pz=tr(data_cov)/tr(noise_cov)`

    """
    # Reduce all calcs to a full-rank subspace of the data_cov
    # U, Vh = nchan x nchan, S = (nchan,). In fact in this case Vh' = U
    U, S, Vh = np.linalg.svd(data_cov, full_matrices=False, hermitian=True)

    # Drop close to 0 singular values and corresponding columns of U. Note that all S[i]
    # are positive and sorted in decreasing order
    t = rcond * S[0]
    U = U[:, S > t]
    rank = U.shape[1]	# The actual rank of the data covariance

    if (max_rank is not None):
        if rank > max_rank:
            rank = max_rank
            U = U[:, :rank]

    H = fwd['sol']['data']	# Should be nchan x (3*nsrc) matrix

    # Reduced unnormalized noise covariance
    uH = U.T @ H
    unoise_cov = uH @ uH.T	# unoise_cov = U' H H' U

    # Reduced data covariance
    udata_cov = np.diag(S[:rank])    # udata_cov = U' U S U' U = S

    # (Pseudo-) inverse of the covariance
    # Make it pos def instead fully degenerate
    inv_cov = nearestPD(U @ np.diag(1./S[:rank]) @ U.T)

    # Initially, normalize it with the trace of data_cov
    pwr = np.trace(udata_cov)
    unoise_cov = (pwr / np.trace(unoise_cov)) * unoise_cov

    upper = pwr    # Current upper value of the trace (not pos def)
    lower = 0.     # Current lower value of the trace (already pos def)
    tr = pwr       # New value of the trace
    tr0 = tr       # Old value of the trace
 
    while True:
        if is_pos_def(udata_cov - unoise_cov):
            lower = tr
            tr = lower + (upper - lower)/2
            is_pd = True
        else:
            upper = tr
            tr = upper - (upper - lower)/2
            is_pd = False

        assert upper > lower				# Never happens, but just in case
        ratio = tr/tr0
        unoise_cov = ratio * unoise_cov
        tr0 = tr

        if is_pd and (np.abs(ratio - 1) < tol):
            break

    # Project results back to the original sensor space
    noise_cov = nearestPD(U @ unoise_cov @ U.T)
    pz = pwr / tr
        
    return noise_cov, inv_cov, rank, pz

def set_source_orientation_signs(fwd, W, U):
    """
    Due to ambiguity of a sign of the scalar beamformer-reconstructed source orientation,
    orientations of adjacent vertices may be flipped randomly. This function properly
    aligns those orientations. For surface-based source spaces this is done by keeping
    source orientation less than 90 degrees with respect to the surface normals.

    Args:
        fwd (mne.Forward): forward solutions
        W (ndarray): shape `(nchan, nsrc)`, the original array of the beamformer weights
        U (ndarray): shape `(3, nsrc)`, the original array of source orientations

    Returns:
        W (ndarray): `(nchan, nsrc)`- the sign-corrected beamformer weights
        U (ndarray): `(3, nsrc)`- the sign-corrected source orientations
        vertices(lst of lst): a list of two lists, corresponding to source space
            vertex numbers for left and right hemisphere, respectively.

    """
    if fwd['src'][0]['type'] != 'surf':
        raise NotImplementedError('Volume source space are not supported yet')

    assert fwd['src'][0]['coord_frame'] == mne.io.constants.FIFF.FIFFV_COORD_HEAD

    # Flip orientations which are more than 90 degrees off the sourface orientations
    Usurf = list()
    vertices = list()

    for ihemi in (0, 1):
        src = fwd['src'][ihemi]
        vts = src['vertno']
        Usurf.append(src['nn'][vts,:])
        vertices.append(vts)

    Usurf = np.concatenate(Usurf, axis = 0)	# This is nsrc x 3 array
    assert Usurf.shape[0] == U.shape[1]

    # W is nchan x nsrc, U is 3 x nsrc
    signs = np.sign(np.einsum('ij,ji->i', Usurf, U))	# A vector of nsrc 1s, -1s
    W *= signs      # It does flip the columns of W, U due to np broadcast rules 
    U *= signs      # Verified 

    return W, U, vertices

def get_beam_weights(H, inv_cov, noise_cov, units):
    """Get beamformer weights matrix for a set of forward solutions

    For each source, calculate a scalar beamformer weight using a formula

    `w = const * R^(-1) h; h = [Hx, Hy, Hz]*u`

    where `R` is the data covariance matrix, `h` is a "scalar" lead field corresponding
    to the source orientation `u`. The normalization constant is selected depending
    on the `units` parameter setting:

    `units = "source": const = (h' R^(-1) h)^(-1)`

    `units = "pz":     const = [h' R^(-1) N R^(-1) h]^(-1/2)`

    In the first case absolute current dipole amplitudes (A*m) will be reconstructed.
    In the 2nd case source amplitudes will be normalized on projected noise, effectively
    representing source-level signal to noise ratio.
 

    Args:
        H (ndarray): nchan x (3*n_src) array of FS for a set of n_src sources 
        inv_cov(ndarray): nchan x nchan (pseudo-)inverse of sensor cov matrix
        noise_cov(ndarray): nchan x nchan noise cov matrix
        units (str): either "source" or "pz"

    Returns:
        W (ndarray): nchan x nsrc array of beamformer weights
        U (ndarray): 3 x nsrc array of source orientations

    """
    if units == "source":
        normalize = False
    elif units == "pz":
        normalize = True
    else:
        raise ValueError("The 'units' parameter value should be either 'source' or 'pz'")

    # Reshape forward solution matrix from nchan x (3*n_src) to n_src x n_chan x 3
    nchan, nsrc3 = H.shape
    nsrc = int(nsrc3 / 3)
    fs = np.reshape(H.T, (nsrc, 3, nchan))
    fs = np.transpose(fs, axes = (0, 2, 1))	# fs is nsrc x nchan x 3
    
    # Calculate beamformer weights; result corresponds to units = "source"
    # W is (nchan x nsrc), U is (3 x nsrc)
    W, U = construct_single_source_weights(fs, inv_cov, noise_cov, beam = "mpz", c_avg = None)

    if not normalize:
        return W

    # Normalize to get waveforms in pseudo-Z
    scales = np.sqrt(np.einsum('ns,nm,ms->s', W, noise_cov, W))	# A vector of nsrc values = sqrt(diag(W'N W))
    return W / scales, U

def beamformer_stc_epochs(epochs, fwd, inv_cov, noise_cov, pz = 1, units='pz', verbose = None):
    """
    Reconstruct source time courses for epoched data using the single source
    minimum variance beamformer.

    IMPORTANT: if epochs contain only a subset of channels available in the forward
    solution `fwd`, **`fwd` object will be modified in place** as channels not included
    in `epochs` will be dropped.

    Args:
        epochs(Epochs): as is - MNE Epochs object; it is expected to contain only
            those sensor channels that will be used to perform beamforming
        fwd (Forward): MNE forward solutions object; note fwd typically includes all
            meg channels in the system (including bad ones)
        inv_cov(ndarray): nchan x nchan (pseudo-)inverse of sensor cov matrix
        noise_cov(ndarray): nchan x nchan noise cov matrix
        pz (float): pseudo-Z of the epochs data, calculated as `tr(R)/tr(N)`, where
            `R`, `N` are data and noise covariance matrices.
        units (str): either "source" or "pz"
        verbose (str or None): verbose mode (see MNE docs for details)

    Returns:
        sensor_data (ndarray): `n_epochs x nchan x ntimes`; MEG channels time courses
        W (ndarray): `nchan x nsrc` array of beamformer weights. For any epoch number
            `i`, corresponding source time courses can be found as
            `src_epoch = W.T @ sensor_data[i,:,:]`
        U (ndarray): `3 x nsrc` array of source orientations

    """
    sensor_data = epochs.get_data(  # returned shape is (n_epochs, n_channels, n_times)
                        picks=None, # take all channels
                        item=None,  # take all epochs
                        units=None, # use original SI units
                        tmin=None,  # Get the whole time range
                        tmax=None,
                        copy=False, # Return a view of the data
                        verbose=verbose)

    fwd = fwd.pick_channels(epochs.ch_names, ordered = False)

    # W is nchan x nsrc, U is 3 x nsrc
    W, U = get_beam_weights(fwd['sol']['data'], inv_cov, noise_cov, units = units) 
    W, U, _ = set_source_orientation_signs(fwd, W, U)

    # By normalizing W on sqrt(pz) we scale time courses of all subjects to the same
    # pseudo-Z magnitude equivalent to pz = 1 (i.e. tr(R) = tr(N)). See comments in
    # beamformer_stc_continuous()
    W /= np.sqrt(pz)

    return sensor_data, W, U

def get_labels(meg_subject, config, data_host, fwd, verbose = None):
    """
    Read MRI atlas Regions Of Interest (ROIs), or 'labels' in MNE Python terms for the subject.

    Args:
        meg_subject(str): subject ID on MEG side
        config(dict): this step configuration dictionary
        data_host(DataHost): reference to `setup_utils.DataHost` object
        fwd(Forward): the forward solutions object
        verbose(str): MNE Python verbose level

    Returns:
        labels(list of Label): a list of ROI labels (constrained to the source space
            vertices)
        label_coms_vtc(ndarray of ints): 1D array of COM vertex numbers encoded as
            described in `write_roi_time_courses()`
        label_coms_rr(ndarray): shape (nlabels, 3) coordinates of ROIs COMs (centers of
            mass) in the coordinate system used by the forward solutions (typically,
            the head coordinate system)

    """
    mri_subj = su.meg2mri_subject(meg_subject)

    # Get "high definition" labels for the subject
    parc = config["parcellations"][config['atlas']]
    mri_labels = mne.read_labels_from_annot(mri_subj,       # FreeSurfer subject
                                        parc= parc,         # parcellation (atlas)
                                        subjects_dir=data_host.get_mri_subjects_dir(),
                                        **config['read_labels'],
                                        verbose=verbose)

    # Constrain labels to the source space used (low definition labels)
    src_space = fwd['src']
    labels = [l.restrict(src_space) for l in mri_labels]
    del mri_labels

    # Ensure that source space is dense enough, so that there are
    # no empty labels (ROIs)
    for l in labels:
        if not l.vertices.size:
            raise ValueError(f'An empty label {l.name} encountered. Please use a denser source space.')

    if fwd['src'][0]['type'] != 'surf':
        raise NotImplementedError('Volume source space are not supported yet')

    fs_dir = data_host.get_mri_subjects_dir()
    label_coms_vtc = get_label_coms(labels, mri_subj, fs_dir)       # Currently only for surface source spaces
    label_coms_rr = get_voxel_coords(src_space, label_coms_vtc)     # COMs coords are in source space coord system
    return labels, label_coms_vtc, label_coms_rr

def get_label_coms(labels, mri_subj, fs_dir):
    """ Calculate center-of-mass vertices for ROIs
    assuming that all vertices in ROI have identical weights.

    The ROIs are defined on the 'fsaverage' subject's original
    (that is - dense) cortical surface. So are the returned
    COM vertex numbers.

    Note:
        If the labels were restricted to a source space using
        a coarser surface than the original FreeSurfer surface, the
        returned COMs will also be vertices of this coarser surface.
        Still the vertex numbers themselves refer to the original
        dense surface. Importantly, "restricted" labels may sometimes
        have their `vertices` lists empty. In this case, **an exception
        will be thrown**.

    Args:
        labels (list of Label): list of MNE Label objects
        mri_subj(str): subject ID on MRI side
        fs_dir (str): pathname to the FreeSurfer subjects directory 

    Returns:
        label_coms (ndarray): 1D signed integer array for COM vertex numbers, with
            NEGATIVE numbers referring to the LEFT hemisphere, and non-negative
            (including 0) - to the right hemisphere. Negative vertex with number
            _LEFT_HEMI_ZERO is interpreted as vertex 0 of the left hemisphere.

    """
    lcoms = list()

    for l in labels:
        if not len(l.vertices):
            raise ValueError('Label {} has an empty vertices list; COM cannot be computed.'.\
                format(l.name))     # Happens for labels restricted to a coarse surface

        icom = l.center_of_mass(subject=mri_subj,
            restrict_vertices=True,    # Assign COM to one of label's vertices
            subjects_dir=fs_dir, surf='sphere')

        # Make left hemi voxels negative, replace zero with _LEFT_HEMI_ZERO
        if l.hemi == 'lh':
            icom = -icom if icom else _LEFT_HEMI_ZERO

        lcoms.append(icom)

    return np.array(lcoms)

def get_voxel_coords(src, vertices):
    """Given vertex numbers, return voxel spatial coordinates for
    a surface source space.

    NOTE: the coordinate system is that of the source space; it may be
    either MRI or head coordinate system.

    Args:
        src (mne.SourceSpaces): as is
        vertices (ndarray): 1D signed integer array for COM vertex numbers, with
            NEGATIVE numbers referring to the LEFT hemisphere, and non-negative
            (including 0) - to the right hemisphere. Negative vertex with number
            _LEFT_HEMI_ZERO is interpreted as vertex 0 of the left hemisphere.

    Returns:
        rr (ndarray): nvox x 3; coordinates of vertices. `nvox = len(vertices)`

    """
    ihemi = lambda i: 0 if i < 0 else 1                     # The hemisphere num 
    lh_vtx = lambda i: 0 if i == _LEFT_HEMI_ZERO else -i    # The actual vtx # in left hemi

    rr = list()

    for i in vertices:
        hemi = ihemi(i)
        vtx = i if hemi else lh_vtx(i)
        rr.append(src[hemi]['rr'][vtx,:])

    return np.array(rr)

def beam_extract_label_time_course(sensor_data, cov, labels, fwd, W, mode = 'pca_flip',
        verbose = None):
    """Compute spatial filter weights and time courses for ROIs (labels) using beamformer
    inverse solutions.

    Args:
        sensor_data (ndarray): `nchan x ntimes` or `nepochs x nchan x ntimes`; M/EEG channels
            time courses for non-epoched or epoched data, respectively
        cov (ndarray): nchan x nchan; the sensor time courses covariance matrix
        labels (list): a list of mne.Label objects for the ROIs 
        fwd (mne.Forward): forward solutions
        W (ndarray): nchan x nsrc; beamformer weights for the whole (global) source space
        mode (str): a method of constructing a single time course for the ROI; see description
            of `mne.extract_label_time_course()` function
        verbose (str): verbose mode

    Returns:
        label_tcs (ndarray): `nlabels x ntimes` or `nepochs x nlabels x ntimes` for non-epoched
            or epoched data, respectively; ROI time courses
        label_wts (ndarray): nchan x nlabels; spatial filter weights for each label
        is_epochs(Bool): `True` if epoched source time courses are returned

    """
    roi_modes = {'pca_flip': get_label_pca_weight}
    
    if not mode in roi_modes:
        raise ValueError('Mode {} is unknown or not supported'.format(mode))

    if sensor_data.ndim == 3:
        is_epochs = True
        nepochs, nchans, _ = sensor_data.shape
    else:
        is_epochs = False 
        nchans = sensor_data.shape[0]

    if verbose == 'INFO':
        print('Reconstructing ROI time courses using beamformer weights, mode = {}'.format(mode))

    func = roi_modes[mode]
    nlabels = len(labels)
    label_wts = np.zeros((nchans, nlabels))

    for i,label in enumerate(labels):
        label_wts[:, i] = func(cov, fwd, W, label)

    if is_epochs:
        label_tcs = np.einsum('cl,ect->elt', label_wts, sensor_data)
    else:
        label_tcs = label_wts.T @ sensor_data

    return label_tcs, label_wts, is_epochs
        
def get_label_pca_weight(R, fwd, W, label):
    """Return a spatial filter vector `w_pca` such that a single label 
    (ROI) time course can be found by the expression `w_pca'*b(t)`, where
    `b` is a vector of sensor time courses. 

    *Explanation*. Covariance matrix of all signals that belong to a label is
    `R_label = W_label'* R * W_label`, where R is the global sensor covariance and
    `W_label = nchan x n_label_src` are label weights.
    Let `U0 = n_label_src x 1` be the largest normalized eigenvector of R_label.
    Then label time course `s(t)` corresponding to 'pca_flip' mode is found as

    `s(t) = sign * scale * U0' * W_label' * b(t)`,

    where

    `scale = sqrt[(trace(R_label)/E0)/n_label_src]`,
    `sign = np.sign(U0'*flip)`

    E0 is the largest eigenvalue of R_label and `flip` is a flip-vector returned by 
    MNE `label_sign_flip()` function. This scaling assigns the RMS of the powers
    of all ROI sources to the returned single time course amplitude. Then it is
    clear that the expression for `w_pca` is:

    `w_pca = sign * scale * W_label * U0, w_pca = nchan x 1`

    Args:
        R (ndarray): nchan x nchan, the global sensor data covariance matrix
        fwd (mne.Forward): the forward solutions object for the whole source space
        W (ndarray): nchan x nsrc, weights matrix for the whole source space
        label (mne.Label): the `Label` object for the ROI 

    Returns:
        w_pca (ndarray): nchan x 1 weight vector for the ROI 

    """
    W_label = get_label_wts(fwd, W, label)
    R_label = W_label.T @ R @ W_label
    E, U = np.linalg.eigh(R_label)

    # The eigh() returns EVs in ascending order
    e0 = E[-1]
    U0 = U[:,-1]

    scale = np.sqrt(np.trace(R_label)/e0/W_label.shape[1])
    flip = mne.label_sign_flip(label, fwd['src'])
    sign = np.sign(U0.T @ flip)
    w_pca = sign * scale * (W_label @ U0)

    return w_pca

def get_label_wts(fwd, W, label):
    """Get a subset of spatial filter weights corresponding to specified Label (ROI)

    Args:
        fwd (mne.Forward): the global `Forward` object
        W (ndarray): nchan x nsrc, weights matrix for the whole source space
        label (mne.Label): the `Label` object for the ROI 

    Returns:
        W_label (ndarray): nchan x (n_label_src) array of ROI weights

    """
    assert fwd['sol']['data'].shape[0] == W.shape[0]
    assert int(fwd['sol']['data'].shape[1]/3) == W.shape[1]

    idx = get_label_src_idx(fwd, label)
    return W[:,idx]

def get_label_src_idx(fwd, label):
    """Get source indecies for a label (ROI)

    Forward solution matrices H and weight matrices W have
    columns (or triplets of columns) corresponding to sources
    in the whole (left + right hemisphere) source space. At the
    same time, the SourceSpaces object has separate source indexing
    for each hemisphere, and so does the ROI (label). This function
    returns a mapping from label vertices to columns of scalar H and
    scalar W.

    Args:
        fwd (mne.Forward): the global `Forward` object
        label (mne.Label): the `Label` object for the ROI 

    Returns:
        idx (1D array of ints): n_label_src-dimensional vector of indecies

    """
    # Generally follow the SourceEstimate._hemilabel_stc() source code
    if label.hemi == 'lh':
        ihemi = 0
    elif label.hemi == 'rh':
        ihemi = 1
    else:
        raise ValueError("Only single hemisphere labels are allowed")

    # Get all the source space vertices for a hemisphere, that is
    # a mapping src # -> dense (FreeSurfer) vortex #
    all_vertices = fwd["src"][ihemi]["vertno"]    # 1D array of integers

    # Index of label vertices into all vertices of a hemisphere
    # Equivalently, idx yields source numbers belonging to the label
    idx = np.nonzero(np.in1d(all_vertices, label.vertices))[0]

    # Source space vertex numbers corresponding to the label, that is the
    # mapping src # -> dense (FreeSurfer) vortex # for a label (ROI)
    # label_vertices = all_vertices[idx]

    # In forward solutions or weights data, the left and right hemis are concatenated, so
    # source ## for the right hemisphere should be shifted by a total number of
    # sources in the left hemisphere:
    if ihemi == 1:
        idx += len(fwd["src"][0]["vertno"])

    return idx

def write_roi_time_courses(ltc_file, label_tcs, label_names, vertno = None, rr = None, W = None,
                           pz = None, events = None, events_id_dict = None):
    """Save ROI (label) time courses and related data in .hdf5
    file.

    The output file will contain at least two datasets with names 'label_tcs' and
    'label_names'. If provided, ROI centers of mass (COMs) vertex numbers
    on the FreeSurface's `fsaverage` cortex surface, ROI COMs in MRI coordinates,
    ROI spatial filter weights and the MEG record overall pseudo-Z will also be saved.

    When supplied, 'events' array and corresponding 'event_id' dictionary will be saved 
    in the datasets named 'events', 'events_id_dict' respectively.
    Note that events_id_dict is saved as a JSON string corresponding to this dictionary
    object.

    When vertex numbers of the left and right hemispheres of surface source spaces need to
    be joined in a single `vertno` array, the following encoding scheme is used. `vertno`
    is expected to be a 1D signed integer array with NEGATIVE numbers referring to the
    LEFT hemisphere, and NON-NEGATIVE (including 0) numbers referring to the RIGHT
    hemisphere. A negative vertex with number `_LEFT_HEMI_ZERO` is interpreted as vertex
    0 of the left hemisphere.

    Args:
        ltc_file (Path | str): full pathname of the output .hdf5 file
        label_tcs (ndarray): `nlabels x ntimes` or `nepochs x nlabels x ntimes` for non-epoched
            or epoched data, respectively; ROI time courses
        label_names (list of str): names of ROIs
        vertno (ndarray or None): 1D signed integer array of vertex numbers corresponding
            to the ROI COMs. See above regarding the vertex numbers encoding rules.
        rr (ndarray or None): nlabels x 3; coordinates of ROI reference locations
            in head coordinates
        W (ndarray or None): nchans x nlabels; spatial filter weights for each ROI.
            Those can be used to reconstruct ROI time courses as `W.T @ sensor_data` 
        pz (float or None): data's pseudo-Z found as `pz = trace(R)/tr(N)`,
            where `N` is the noise covariance.
        events(ndarray | list of ndarray): `nevents x 3` or `[events1,...,eventsK,...]`;
            events array in MNE Python 'events' format for non-epoched data, or a list
            of such arrays for epoched data, respectively. Note that in the latter case
            the event sample index is counted from the start of the epoch (not from the trigger)
        events_id_dict(dict): dictionary event_descr -> event_id; see `event_id` parameter
            description of the MNE `Epochs` object constructor

    Returns:
        None

    """

    with h5py.File(ltc_file, 'w') as f:
        is_epochs = (label_tcs.ndim == 3)

        f.create_dataset('label_tcs', data=label_tcs)
        f.create_dataset('label_names', data=label_names)

        if not (vertno is None):
            f.create_dataset('vertno', data=vertno)

        if not (rr is None):
            f.create_dataset('rr', data=rr)

        if not (W is None):
            f.create_dataset('W', data=W)

        if not (pz is None):
            f.create_dataset('pz', data=pz)

        if events is not None:
            if is_epochs:
                # Make a list of numbers of events per epoch
                event_counts = [e.shape[0] for e in events]

                # Store those as a separate dataset
                f.create_dataset('event_counts', data=event_counts)

                # Merge all events into one ndarray
                events = np.concatenate(events, axis=0)

            f.create_dataset('events', data=events)

        if events_id_dict is not None:
            # Convert dict to JSON string
            sjson = json.dumps(events_id_dict)
            f.create_dataset('events_id_dict', data=sjson)

def read_roi_time_courses(ltc_file):
    """Read ROI (label) time courses and corresponding ROI names from .hdf5
    file.

    When vertex numbers of the left and right hemispheres of surface source spaces need to
    be joined in a single `vertno` array, the following encoding scheme is used. `vertno`
    is expected to be a 1D signed integer array with NEGATIVE numbers referring to the
    LEFT hemisphere, and NON-NEGATIVE (including 0) numbers referring to the RIGHT
    hemisphere. A negative vertex with number `_LEFT_HEMI_ZERO` is interpreted as vertex
    0 of the left hemisphere.

    Args:
        ltc_file (Path | str): full pathname of the output .hdf5 file

    Returns:
        label_tcs (ndarray): `nlabels x ntimes` or `nepochs x nlabels x ntimes` for non-epoched
            or epoched data, respectively; ROI time courses
        label_names (ndarray of str):  1 x nlabels vector of ROI names
        vertno (ndarray or None): 1D signed integer array of vertex numbers corresponding
            to the ROI COMs. See above regarding the vertex numbers encoding rules.
        rr (ndarray or None): nlabels x 3; coordinates of ROI reference locations
            in head coordinates
        W (ndarray or None): nchans x nlabels; spatial filter weights for each ROI.
            Those can be used to reconstruct ROI time courses as `W.T @ sensor_data` 
        pz (float or None): data's pseudo-Z found as `pz = trace(R)/tr(N)`,
            where `N` is the noise covariance
        events(ndarray | list of ndarray): `nevents x 3` or `[events1,...,eventsK,...]`;
            events array in MNE Python 'events' format for non-epoched data, or a list
            of such arrays for epoched data, respectively. Note that in the latter case
            the event sample index is counted from the start of the epoch (not from the trigger)
        events_id_dict(dict): dictionary event_descr -> event_id; see `event_id` parameter
            description of the MNE `Epochs` object constructor

    """
    with h5py.File(ltc_file, 'r') as f:
        label_tcs = f['label_tcs'][:]   # The actual dimensions of label_tcs will be restored
        label_names = f['label_names'].asstr()[:]

        if 'vertno' in f:
            vertno = f['vertno'][:]
        else:
            vertno = None

        if 'rr' in f:
            rr = f['rr'][:,:]
        else:
            rr = None

        if 'W' in f:
            W = f['W'][:,:]
        else:
            W = None

        if 'pz' in f:
            pz = f['pz'][()]
        else:
            pz = None

        if 'events' in f:
            events = f['events'][:,:]

            if label_tcs.ndim == 3:
                # Epoched data. Get the event counts
                event_counts = f['event_counts'][:]

                # Create a list of events per epoch
                splits = np.cumsum(event_counts)[:-1]
                events = np.split(events, splits)
        else:
            events = None

        if 'events_id_dict' in f:
            events_id_dict = json.loads(f['events_id_dict'][()])
        else:
            events_id_dict = None

    return (label_tcs, label_names, vertno, rr, W, pz, events, events_id_dict)  
 
