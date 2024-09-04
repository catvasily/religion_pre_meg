"""
**A helper utility for viewing power spectra of the raw 
FIF files.**

Can be used standalone, or as a function call.
"""
import os
import os.path as path
import commentjson as cjson     # Reading JSON files with comments
import numpy as np
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog

import mne
from mne.io import read_raw_ctf, read_raw_fif

#-------------------------------------
CONFIG_JSON = "spect_raw_config.json"      # This script's input parameters
"""
Specifies JSON config file name with default settings.
"""
#-------------------------------------

__file__ = path.realpath(__file__)    # expand potentially relative path to a full one
_pathname = lambda fname: path.join(path.dirname(__file__), fname)  # full pathname to source file

def spect_raw_fif(*, dsname = None, raw = None, picks = None,
    fmin = None, fmax = None, n_fft = None, apply_projectors = None,
    xlim = None, ylim = None,
    b_logx = None, b_logy = None, show = None, verbose = None):
    """
    Function to plot **amplitude power spectral density** for sensor channels
    given the `Raw` data object or corresponding .fif file.

    All parameters except `raw` are supplied in the JSON file pointed to by
    the `CONFIG_JSON` definition. This file is supposed to reside in the same
    folder where this source file is. A subset of parameters can also be passed via
    function arguments; in this case values passed as arguments take precedence
    over those specified in the JSON file.

    Args:
        dsname(str): pathname of the data .fif file. If not supplied,
            corresponding `Raw` object should be specified via `raw` 
            argument.
        raw(mne.Raw): as is; if both `dsname` and `raw` are provided,
            `raw` will be used.
        picks(str | list of str | array) (see MNE Python docs for details) - specifies
            channels to work with.
        fmin(float): min frequency to pick from calculated spectrum, Hz
        fmax(float): max frequency to pick from calculated spectrum, Hz
        n_fft(int): FFT length for the Welch method
        apply_projectors(bool): flag to apply existing SSP projectors to data
        xlim([xmin, xmax]): X-axis (frequency) range for plotting; all
            available frequencies will be plotted if not specified.
        ylim([ymin, ymax]): Y-axis (spectrum) range for plotting; 
            calculated automatically if not specified,
        b_logx(bool): flag to use log scale for frequencies
        b_logy(bool): flag to use log scale for spectra 
        show(bool): flag to show the plot (blocks the execution until the
            plot is closed)
        verbose(str): verbose level - 'info', 'warnings', 'errors', etc.

    Return:
        fig(matplotlib figure object): the plot figure
    """
    # Create a dict of all passed args except 'dsname' and 'raw;
    passed_args = locals()
    passed_args.pop('raw')

    # Get all default args from JSON
    with open(_pathname(CONFIG_JSON), 'r') as fp:
        args = cjson.loads(fp.read())

    # Override values from JSON with the passed ones, if any
    for key in passed_args:
        val = passed_args[key]

        if val is not None:
            args[key] = val

    if (dsname is None):
        if args['fname'] is not None:
            dsname = args['path'] + '/' + args['fname']

    if args['fmax'] == 0:
        args['fmax'] = np.inf

	# 'raw' takes precedence over 'dsname'; one of them should be present
    if raw is None:
        if dsname is None:
            raise ValueError("Either the 'dsname' or the 'raw' argument should be supplied")
        else:
            raw = read_raw_fif(dsname, allow_maxshield=False, preload=True)

	# Compute PSD
    spect = raw.compute_psd(
            fmin=args['fmin'], fmax=args['fmax'],
            picks=args['picks'],
            proj = args['apply_projectors'],
            **args['kwargs_psd'],
            verbose=args['verbose'],
            n_fft = args['n_fft']
            )

	# Plot it 
    fig = spect.plot(
            picks=args['picks'],
            dB=False,       # Should be False; log scale is controlled by b_logy
            xscale='linear',# Should be 'linear'; log scale is controlled by b_log_x
            show=False, **args['kwargs_plot'])

    xlim = args['xlim']
    ylim = args['ylim']
    b_logx = args['b_logx']

    axes = fig.axes

    # If picks = 'data' or 'meg' - there will be separate plots
    # for magnetometers and gradiometers. If average = False, then the total number of axes in
    # the figure will be 4, otherwise 2. If average = True, then total number of axes will be
    # 2.
    if not args['kwargs_plot']['average']:
        nax = 1 if len(axes) <= 2 else 2
    else:
        nax = len(axes)

    for iax in range(nax):
        ax = axes[iax]

        if not (xlim is None):
            ax.set_xlim(xlim)
        elif b_logx:	# xlim is None, make sure that fmin is not 0
            if args['fmin'] <= 0:
                tmp = ax.get_xlim()
                ax.set_xlim([1., tmp[1]])

        if not (ylim is None):
            ax.set_ylim(ylim)

        if b_logx:
            ax.set_xscale('log')

        if args['b_logy']:
            ax.set_yscale('log')

        if not (dsname is None):
            _, title = os.path.split(dsname)
            fig.suptitle(title)

    if args['show']:
        plt.show()

    return fig

if __name__ == '__main__':
    with open(_pathname(CONFIG_JSON), 'r') as fp:
        args = cjson.loads(fp.read())

    if args['fname'] is None:
        root = tk.Tk()
        root.withdraw()
        dsname = filedialog.askopenfilename()
    else:
        dsname = args['path'] + '/' + args['fname']

    spect_raw_fif(dsname = dsname)

