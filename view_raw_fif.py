"""
**A simple viewer of the MNE Python raw data. Can be used both standalone and
as a function call.**
"""
import os
import os.path as path
import commentjson as cjson     # Reading JSON files with comments
import numpy as np

import tkinter as tk
from tkinter import filedialog

import mne
from mne.io import read_raw_edf, read_raw_fif

#-------------------------------------
CONFIG_JSON = "view_raw_config.json"      # This script's input parameters
"""
Specifies JSON config file name with default settings.
"""
#-------------------------------------

__file__ = path.realpath(__file__)    # expand potentially relative path to a full one
_pathname = lambda fname: path.join(path.dirname(__file__), fname)  # full pathname to source file

def view_raw_fif(*, dsname = None, raw = None, picks = None, highpass = None, lowpass = None):
    """
    Function to plot and browse channel data given the `Raw` data object or corresponding .fif file.

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
        highpass(float): viewing high pass filter cut off frequency, Hz.
        lowpass(float): viewing low pass filter cut off frequency, Hz.

    Note:
        If highpass > lowpass - then the bandstop filter is constructed

    Returns:
        Nothing

    """
    # Create a dict of all passed args except 'dsname' and 'raw;
    passed_args = locals()
    passed_args.pop('dsname')
    passed_args.pop('raw')

    # Get all default args from JSON
    with open(_pathname(CONFIG_JSON), 'r') as fp:
        args = cjson.loads(fp.read())

    # Override values from JSON with the passed ones, if any
    for key in passed_args:
        val = passed_args[key]

        if val is not None:
            args['kwargs'][key] = val

    if dsname is None:
	    dsname = args['path'] + '/' + args['fname']

    mne.viz.set_browser_backend(args['backend'])

    # 'raw' takes precedence over 'dsname'; one of them should be present
    if raw is None:
        ext = dsname[-3:].upper()

        if ext == 'FIF':
            raw = read_raw_fif(dsname, preload=args['preload'],
                verbose=args['verbose'])
        else:
            raise ValueError("Only FIF files are currently supported.")

    if args['kwargs']['events'] is not None:
        args['kwargs']['events'] = np.array(args['kwargs']['events'])
    else:
        args['kwargs']['events'] = mne.find_events(raw, **args['events_args'], 
                                                   verbose=args['verbose'])

    args['kwargs']['verbose'] = args['verbose']

    mne.viz.plot_raw(raw, **args['kwargs'])
    print(raw.info)

if __name__ == '__main__':
    # Get all default args from JSON
    with open(_pathname(CONFIG_JSON), 'r') as fp:
        args = cjson.loads(fp.read())

    if args['fname'] is None:
        root = tk.Tk()
        root.withdraw()
        dsname = filedialog.askopenfilename()
    else:
        dsname = args['path'] + '/' + args['fname']

    view_raw_fif(dsname = dsname)

