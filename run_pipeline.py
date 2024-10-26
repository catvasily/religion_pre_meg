"""
**A top level script for running MEG pipeline for
the religion project data analyses.**

To add a step to be run to this script:

1. Encapsulate all the necessary code in a dedicated function  

```
def my_step(ss):  
...
# NOTE: Step return value (if any) is not used
```

2. Add corresponding entry to the `cases` dictionary in
the EPILOGE section at the bottom of this file, in the form  

```
'my_step': my_step
```

Steps may reside in separate Python files. In that case
corresponding import statements should be added to this file.

The variables that are intended to be shared or passed between
different steps should be defined as attributes of the ss object,
as follows:

```
ss.common_var = common_var_value
```

All input parameters for the script are expected to reside
in a single JSON file specifed in a global constant 
`INPUT_JSON_FILE` as follows:

```
INPUT_JSON_FILE = '<my-file>.json'
```

Contents of the `INPUT_JSON_FILE` is available to each step
as a dictionary `ss.args`, where `ss` is a reference to this
application object passed to the step in question as the first argument.

The sequence of steps to be executed should be listed in `to_run`
key in the `INPUT_JSON_FILE` file, for example

```
to_run = ("init","step1","step6")
```

--------------------------------------------------

Available steps:
    `'input'`: As is - set all the input and configuration \
        parameters here. This step should always be run first.

    `'prefilter'`: filter raw files to band of interest, notch powerline.

    `'maxfilter'`: apply tSSS, vibrational artifact correction and head \
        motion correction.

    `'ica'`: Remove ECG and EOG artifacts using ICA.

    `'bem_model'`: MRI-side actions necessary to create BEM head conductor \
        model and the source space.

    `'src_rec'`: MEG source reconstruction (inverse solution).

    `'plot_epochs'`: create overview plots of epoched data.

    `'plot_waveforms'`: grid plots of channel waveforms

"""

import sys
import os
import os.path as path
import io
import glob
import socket
import matplotlib.pyplot as plt
#import seaborn as sns
import commentjson as cjson
import numpy as np
import h5py

from prefilter import prefilter
from maxfilter import maxfilter
from do_ica import do_ica
from bem_model import bem_model
from src_rec import src_rec
from plot_epochs import plot_epochs
from plot_waveforms import plot_waveforms
import setup_utils as su

__file__ = path.realpath(__file__)    # expand potentially relative path to a full one
_pathname = lambda fname: path.join(path.dirname(__file__), fname)  # full pathname to source file

# Add other source code folders relative to this one, to the python path:
# sys.path.append(path.dirname(path.dirname(__file__))+ '/misc')

#-------------------------------------
INPUT_JSON_FILE = "pipeline_setup.json"      # This script's input parameters
#-------------------------------------

def input(ss):
    """
    Common initial processing after reading all settings from the 
    input JSON file.

    Note that when top-level script has a command-line argument, 
    this argument is interpreted as the job number (0-based) 
    within the SLURM array job, with the total number of jobs running
    on the computer cluster being equal to `ss.args['N_ARRAY_JOBS']`.

    Args:
        ss(obj): reference to this app object

    Returns:
        Nothing

    """
    args = ss.args
    ss.data_host = su.DataHost(args)

    if len(sys.argv) == 1:   # No command line args
        ss.ijob = 0
    else:
        ss.ijob = int(sys.argv[1])

# --------------------------------------------------------
#                    EPILOGUE                             
# --------------------------------------------------------
class _app:
    # ------
    # Cases: add your steps here in the form "my_step":my_step,
    # ------
    cases = {
        # Steps to run go here:
        'input': input,
        'prefilter': prefilter,
        'maxfilter': maxfilter,
        'ica': do_ica,
        'bem_model': bem_model,
        'src_rec': src_rec,
        'plot_epochs': plot_epochs,
        'plot_waveforms': plot_waveforms,
    }

    def __call__(self, name, *args, **kwargs):
        not_found = True

        for f in self.cases:
            if f == name:
                self.cases[f](self, *args, **kwargs)
                not_found = False
                break

        if not_found:
            raise ValueError(f'Requested method "{name}" not found')

if __name__ == '__main__': 
    for c in _app.cases:
        setattr(_app, c, _app.cases[c])

    this_app = _app()

    with open(_pathname(INPUT_JSON_FILE), 'r') as fp:
        this_app.args = cjson.loads(fp.read())

    for name in this_app.args['to_run']:     
        this_app(name)

# -------------- end of Epilogue --------------------------


