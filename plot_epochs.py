"""
**Create overview plots of tasks runs epoched data**
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
import mne
import setup_utils as su
from epochs import construct_epochs

def plot_epochs(ss):
    """
    Create plots of epoched data.

    Args:
        ss(obj): reference to this app object

    """
    warnings.filterwarnings('ignore', category=RuntimeWarning,
                            message=ss.args["file_name_warning"])

    STEP = 'plot_epochs'
    config = ss.args[STEP]
    find_event_args = ss.args['src_rec']['epochs']['find_events'].copy()

    if ss.data_host.cluster_job:
        config['show_plots'] = False

    files = su.files_to_process(ss, STEP)
    for in_fif, out_fif in files:
        # Skip non-raw .fif files, just in case
        if mne.what(in_fif) != 'raw':
            continue

        task_name = su.get_fif_task(in_fif)

        if not su.extract_epochs(task_name):
            continue

        raw = mne.io.read_raw_fif(in_fif, allow_maxshield=False, preload=True,
                                  on_split_missing='raise', verbose=ss.args['verbose'])

        # min_duration = 0 which is set in src_rec config creates an error
        # here for some reason. So:
        find_event_args['min_duration'] = find_event_args['shortest_event']/raw.info['sfreq']

        all_events = mne.find_events(raw, **find_event_args,
                                     verbose=ss.args['verbose'])

        raw = raw.pick(config['picks'], exclude = 'bads')

        epochs, events_per_epoch, data_cov, noise_cov, inv_cov, rank, pz = \
            construct_epochs(ss, raw, all_events, skip_cov_calc = True)

        w = config['time_res']
        fmin = config['fmin']
        fmax = config['fmax']
        df = config['f_step']
        SR = raw.info['sfreq']
        tmin = config['tmin']
        tmax = config['tmax']
        T = tmax - tmin
        freqs = np.arange(fmin, fmax, df)
        method = config['compute_tfr']['method'] 
        title_prefix = f'TFR ({method}): [{tmin},{tmax}] s, [{fmin},{fmax}] Hz - '

        for ch_type, title in dict(mag="magnetometers", grad="gradiometers").items():
            layout = mne.channels.find_layout(epochs.info, ch_type=ch_type)
            avgTFR = epochs.compute_tfr(freqs = freqs,
                                tmin=tmin, tmax=tmax,
                                picks=ch_type,
                                **config['compute_tfr'],
                                verbose=ss.args['verbose'],
                                n_cycles = calc_cycles_per_freq(freqs, w, SR, T))

            fig = avgTFR.plot_topo(picks=ch_type, tmin = tmin, tmax = tmax,
                             layout=layout, title=title_prefix + title, 
                             show = False,
                             verbose=ss.args['verbose'],**config['tfr_plot_topo'])
            fig.set_size_inches(*config['fig_size_inch'])
            fig.set_dpi(config['dpi'])

            if config['show_plots']:
                plt.show()

            out_png = str(out_fif).replace('topo_img.png',f'tfr_{ch_type}.png')
            fig.savefig(out_png, dpi = config['dpi'])
            print(f'Saved plot as {out_png}')

    warnings.filterwarnings("default", category=RuntimeWarning)
    print(f'\n** {STEP} step completed **\n')                

def calc_cycles_per_freq(lst_f, w, SR, T):
    """
    Calculate number of Morlet wavelet cycles for a specified frequency
    based on required wavelet length in time domain.

    Length of the wavelet in samples is for frequency `f` and sampling rate
    SR is

    `L = 5 pi^-1 n_cycles SR f^-1 - 1`

    This corresponds to a time window length of w = L/SR. Therefore (integer)
    number of cycles for a given time window w is

    `n_cycles = (pi/5)(f*w + f/SR)`

    Args:
        lst_f(ndarray): array of target frequencies, Hz
        w(float): wavelet window length in in seconds
        SR(float): sampling rate, Hz
        T(float): total signal duration, s

    Returns:
        lst_n(ndarray of int): a vector of n_cycle values for each
            frequency, but not exceeding the signal length in cycles

    """
    lst_n = np.rint(.25*np.pi*(w + 1/SR)*lst_f)
    n_max = np.floor(T*SR)
    return np.minimum(lst_n, n_max)

