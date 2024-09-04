# MEG processing pipeline for the "Religion" project

## Summary

This Python code implememts processing pipeline of the MEG recordings collected
for the *Religion* project. It starts with the raw data produced by Elekta
(MEGIN) scanner, and produces source reconstructed signal time courses for a
set of specific anatomical locations in the subject's brain. The code uses the
*MNE Python* MEG/EEG processing package.

Note that for coregistration of each subject's anatomy with the MEG system
and for the brain sources reconstruction subject's MRI data prepared by the
FreeSurfer package is also required.

The pipeline consists of many processing steps, each step typically using the
results of the previous one. The steps can be executed automaticaaly one after
another, or separately one by one; for all subjects at once or for a specific
subset; for all subject's records or just for a few. Every step usually requires
specification of numerous parameters.

In this project, all the settings for all steps are defined in a single JSON
configuration file named `pipeline_setup.json`. This way, all variations in the
runtime parameters or pipeline setup should not require changing the Python
source code. Importantly, the `pipeline_setup.json` follows an extended JSON
standard that allows comments (the original vanilla JSON has no provisions for
comments). On the Python side, this requires importing `commentjson` module
rather than the standard JSON processing module.

Note that **comments in the `pipeline_setup.json` are a crucial part of this
porject's documentation**.

Currently, the pipeline includes the following steps.
1. [**`'input'`**](#1-the-input-step) - readinging in all the input and
   configuration data.
2. [**`'prefilter'`**](#2-the-pre-filtering-step) - filtering the raw data
   to the frequency band of interest, notching out power line frequencies,
   extracting events, downsampling.
3. [**`'maxfilter'`**](#3-maxwell-filtering) - Maxwell filtering with
   vibrational artifact removal (tSSS/eSSS); head motion correction.
4. [**`'ica'`**](#4-ica---eye-blinks-and-cardiac-artifacts-removal) - eye blinks,
   muscle and cardiac artifacts removal using ICA.
5. [**`'bem_model'`**](#5-construction-of-the-bem-head-model-and-source-spaces) -
   constructing a BEM head conductor model and source spaces using the subject's
   MRI data.
6. [**`'src_rec'`**](#6-source-reconstruction) - MRI/HEAD coordinates coregistration,
   inverse solution and source time courses reconstruction for specified regions
   of interest (ROIs) in the brain.
 
More details about the configuration file and processing steps are given below.

## Setup 
### Python environment
This code runs under Python 3.10+, using MNE Python version 1.7+. To establish
a virtual environment, one needs to run the commands below.
```
# Steps to be done just once
# --------------------------
# On Digital Alliance cluster only - to load Python
    module load StdEnv/2023         # To get Python > 3.10
	module load python              # Loads version 3.11.5
	module load scipy-stack	        # ?? Version not known

# Do this at the location where the virtual environment will
# actually be stored. It does not need to be your code folder.
# One can always create a symbolic link pointing here to make
# activation/deactivation convenient.
python3 -m venv mne1.7.1-relmeg

# Do it to have simpler venv name:
ln -s <path-to-venv-storage-area>/mne1.7.1-relmeg mne

source mne/bin/activate
pip3 install --no-index --upgrade pip   # !!! --no-index is used only on cluster
pip3 install mne[hdf5]==1.7.1		# Install specific version 1.7.1

# QT-based stuff
pip3 install PyQt6                  # Fails on cedar
pip3 install mne-qt-browser # QT backend for the MNE visualizations
pip3 install psutil

# Test this minimal MNE Python installation:
python -c "import mne; mne.sys_info()"

# Packages that can potentially be installed:
# Scientific:
#   sklearn, numba, nibabel, nilearn, dipy, openmeeg, cupy, pandas
# Visualization:
#   pyvista, pyvistaqt, vtk, ipympl, ipywidgets, trame_client, trame_server,
#   trame_vtk, trame_vuetify

pip3 install commentjson    # To work with JSON files with comments
pip3 install joblib         # To use parallel jobs where possible

# This installs sklearn package (counter-intuitively)
python3 -m pip install scikit-learn # Needed for ICA
pip3 install nibabel

# Coregistration visualization
# On local machine use:
    pip3 install vtk
# On cedar, add a line when loading all other modules
    module load vtk

pip install pyvista     # Needed for coregistration visualization
    # or
pip install pyvistaqt

# To be able to generate/update documentation using sphinx
pip3 install -U sphinx

# For using read-the-docs theme for sphinx:
python3 -m pip install myst_parser
python3 -m pip install sphinx_rtd_theme

deactivate
```
### Data files organization
The following file structure is assumed for this project.

All data resides in a single project folder, referred to as the "project root"
further on.

All MEG data (both the original and the processed) resides in the folder defined
by the `meg` key in the [JSON configuration file](#json-configuration-file)
(currently "meg") under the project root.

All MRI data needed to run the pipeline, specifically the FreeSurfer output
data, may be located anywhere. This absolute location is referred to as `mri`
location in the [JSON configuration file](#json-configuration-file).

A host computer the code is being run on is referenced as `host` in the
configuration file and the source code. Depending on the host, the project
root folder and the `mri` location may vary. However,
file structure under these locations is expected to be fixed. Specifically, raw
input records for all participants should reside in the "raw" folder under the
`meg` folder. All MEG processing results go to the 
`<out_root>/<pipeline_version>` folder under the `meg` folder, where both the
`out_root` and the `pipeline_version` are assigned specific values in the
[JSON configuration file](#json-configuration-file) (currently - "preprocessed"
and "A1", respectively). Under the `<out_root>/<pipeline_version>` reside 
folders with each step results. There names are defined by values of `out_dir`
keys of respective steps. For example, the output folder of the `ica` step is
currently "icafiltered". Thus with the current settings the absolute path of
the `ica` step results will be:
`<project-root>/meg/preprocessed/A1/icafiltered`.

Both the input records in the "raw" folder and the step results folders contain
subfolders for each subject named after the subject IDs. Under the
subject ID folder reside subfolders for records collected on different dates
in YYMMDD format. For example, subfolder *240409* corresponds to a collection made
on April 9, 2024.

Similarly, the file structure and file names in the `mri` location are expected
to be in accordance with the standard FreeSurfer conventions. Note that **on the
MRI side the MEG subject ID is prepended with prefix "sub-"**. 
 
### JSON configuration file
**`pipeline_setup.json`**, a JSON file with comments, contains all configuration and
run-time parameters for every step of the pipeline. This file consists of a header
part, which defines common general settings, and dedicated keys for each pipeline
step. The latter comprise all parameters for corresponding step and are mostly
documented in the JSON file itself.

Most important keys in the header part are `to_run`, `subjects`, `N_ARRAY_JOBS`
and `hosts`.

The **`to_run`** key specifies a list of steps to be executed. The "input" step should
always be the first one, then one or more steps can follow. Note that 
the order of steps is important, because output of one step often serves as an
input to the next step.

The **`subjects`** key defines a set of subjects that will be processed. Unless `null`,
this should be a list of subject IDs on the MEG side. If `null`, all subjects
found in the input folder of a currently executing step will be processed.

The **`N_ARRAY_JOBS`** key defines the size of the _SLURM array job_ when running on
the Digital Alliance cluster. For example, when the sbatch file contains
directive
```
#SBATCH --array=0-99
```
`N_ARRAY_JOBS` should be set to `100`. Set it to `1` when running a simple job
or when running on a local computer.

Under the **`hosts`** key reside keys that define specific hosts. Which host the
code is running on is determined at run time by the pattern found in the host
name. For example, key `cedar` is used when running on the Cedar cluster;
key `ub2` - when running on hosts 'ub2004', 'ub2404', etc; key `other` -
when none of the other keys matches the host name. 

Meaning of keys specified for each host is documented in the JSON file itself.
Here we only mention the `cluster_job` key (`true` or `false`) which defines
whether the host belongs to the Digital Allience cluster. When `true`, all
interactive plotting functions will be automatically disabled at run time. 

Each step's key structure contains keys common for each step, and step-specific
keys. The common keys are `in_dir`, `out_dir`, `files` and `suffix` (except for
the `src_rec` step).

The **`in_dir`, `out_dir`** keys specify step input and output folder names. These
folders are expected to reside inside the `<out_root>/<pipeline_version>`
folder. 

The **`files`** key allows to control which input files should be processed.
If `null`, then the step will be executed for all dates and
all records that exist for a set of subjects defined by the `subjects` key
described above. If not `null`, then:
- only a single subject should be listed in the `subjects` key
- this subject must have records for only a single date
- the `files` should provide a list of the base names of this step input files
  that need be processed. All other input files will be skipped.

The **`suffix`** key defines a suffix that will be appended to the name part of
the input `.fif` file to construct the name of the output `.fif` file. For example, with
the current settings if the input file for the `ica` step is `XXXX.fif`, the output
file will be `XXXX_ica.fif`. The suffix is not used for the source reconstruction
step.

Most important step-specific keys are described below in corresponding sections.
Please refer to the comments inside the `pipeline_setup.json` file for further
details.

### Running the pipeline
After the virtual environment is activated, the pipeline can be run either on a
local machine or as a standard job on the Digital Alliance cluster by executing a
command
```
python3 run_pipeline.py
```
When running a **SLURM array job** on the cluster, please use the following command in the
corresponding `.sbatch` script:
```
python3 run_pipeline.py ${SLURM_ARRAY_TASK_ID}
```
In this case, all subjects to be processed will be distributed between 
parallel SLURM jobs as evenly as possible. Please make sure that the number
of jobs in the array defined in the `.sbatch` script matches the `N_ARRAY_JOBS`
parameter in the [JSON configuration file](#json-configuration-file).

### The source code documentation
A detailed auto-generated source code documentation can be found
[here](doc/_build/html/index.html). Note that **this link won't work as expected
if you are accessing this README file on the Github website** (in this case, the
documentation source HTML code is shown).

***<span style="color:blue">
To view the [documentation](doc/_build/html/index.html) in HTML format, one needs to:</span>***  
*<span style="color:blue">a) clone the Github repo to your local computer, and</span>*  
*<span style="color:blue">b) access this README.md file using a web browser that has the
["markdown file viewer extension"](https://chrome.google.com/webstore/detail/markdown-viewer/ckkdlimhmcjmikdlpkmbgfkaikojcbjk) installed; or</span>*   
*<span style="color:blue">c) open file `<path-to-local-repo>/doc/_build/html/index.html`
directly in you browser</span>*  
 
## 1. The input step
On this step, all the configuration data is read in and the `_app` object is
populated. Corresponding source code is found in file
[run_pipeline.py](doc/_build/html/code.html#module-run_pipeline).

## 2. The pre-filtering step
On this step, the data is filtered to the target frequency band specified by `l_freq`,
`h_freq` keys under the `prefilter/filter` key, and the power line frequency is
notched out (the `prefilter/notch/freqs` key). Additionally:
- the head positioning data is extracted from each record and saved in corresponding
  `.fif` file with `_pos` suffix
- events information is extracted and saved in `.fif` files with `_eve` suffix

Finally, the filtered raw files are downsampled to the target sampling frequency
specified by the `prefilter/target_sample_rate` key value.

The source code for this step is found in file
[prefilter.py](doc/_build/html/code.html#module-prefilter).

_Note_. After the `maxfilter` step (see below) is executed, this step's output folder
may also contain input records filtered to "vibration" bands, used to detect a
vibration artifact.

## 3. Maxwell filtering
On this step, the following operations are performed.
- Bad channels are identified in both empty room records and subject records using
  MNE Python's `find_bad_channels_maxwell()` function
- If not already done, the empty room files are bandpass-filtered to frequency
  bands relevant to vibrational artifacts rejection (please see the `maxfilter/vib_filter/bands`
  key), and saved to the _prefilter_ step output folder
- Out-projectors are constructed using principal components that capture more than
  `maxfilter/vib_filter/threshold` variance of the artifact-related bands signals
- if the head motion correction is requested (the `maxfilter/do_head_motion_correction`
  key), the average head position for each record is determined
- The eSSS maxwell filtering (the MNE Python `maxwell_filter()` method) is applied
  to the subject's pre-filtered records, using the projectors described above.
  If requested, the patients' head position is corrected to its average value
  for each recording

The source code for this step is found in file
[maxfilter.py](doc/_build/html/code.html#module-maxfilter).

## 4. ICA - eye blinks and cardiac artifacts removal
In current setup, a ***fastica*** independent component analysis technique (set by the key
`ica/init/method`) is applied to the sensor channels data to expand it into
statistically independent components (ICs). The number of components is set to be
equal to the rank of the input data - typically, around 70 ICs for the maxfiltered data.

The existing ECG and EOG channels are filtered to frequency bands specified by keys 
`ica/find_bads_ecg/l_freq, h_freq` and `ica/find_bads_eog/l_freq, h_freq` respectively.
Then the sensor data ICs that after filtering to the corresponding band have high enough
correlations with the ECG/EOG channels (see key `threshold` under `find_bads_ecg`,
`find_bads_eog`) are removed from the sensor data.

The resulting raw record have the sensor channels cleaned from the artifacts and the
ECG and EOG channels restored to their original bandwidth.

The source code for this step is found in file
[do_ica.py](doc/_build/html/code.html#module-do_ica).

## 5. Construction of the BEM head model and source spaces
This step is heavily based on the results of the subject's MRI data processing
produced by the *FreeSurfer* software package. Additionally, some *FreeSurfer*
utilites are being called by the MNE Python software at runtime, therefore a
working *FreeSurfer* installation is required in the system.

During the step execution, MNE Python functions `make_watershed_bem()`,
`make_scalp_surfaces()`, `make_bem_model()` and `make_bem_solution()` are called to
construct subject's electromagnetic head model. These functions are extensively
documented in MNE Python package; specific parameters used can be found under 
corresponding key for the `bem_model` step in the
[JSON configuration file](#json-configuration-file). A one layer BEM conductor model
is used as we are dealing with the MEG-only data (see the `bem_model/conductivity` key
setting).

Finally, MNE Python methods `setup_source_space()` or `setup_volume_source_space()`
are called to create a surface- or volume-based source spaces.

If either BEM model or source space is already found in the subject's folder on the
MRI side, corresponding calculations will be performed once again or skipped, depending 
on the value of the boolean flag  `bem_model/recalc_bem`.

The source code for this step is found in file
[bem_model.py](doc/_build/html/code.html#module-bem_model).

## 6. Source reconstruction
At this step, minimum variance beamformer inverse solution is calculated and source
reconstruction is performed, based on the BEM head conductor model and the source
spaces created on the previous step.

First, if existing MRI->HEAD transformation file for the subject is not found in the
output folder, automatic MRI/HEAD coregistration is performed using MNE Python
*Coregistration* class methods.

Next, if existing forward solutions are not found in the output folder, those are
calculated for corresponding source space using MNE Python `make_forward_solution()`
utility, and saved to the .fif file. Existing forward solutions will still be
recalculated if boolean key `src_rec/recalc_forward` is set to `true` in the JSON
configuration file.  

Then scalar minimum variance "SAM" beamformer source reconstruction is performed for
each source in the source space, yielding arrays of source orientation vectors and
beamformer weights vectors. Note that beamformer inverse solution uses full (i.e.
signal plus noise) and noise only covariance matrices to localize the sources and
to determine their orientations. These covariances are calculated differently
depending on the type of the record.

For the task records the epochs are first created, centered around the "question start"
trigger events. The trigger codes for epoching are specified in the
`src_rec/create_epochs/event_id` key. The epoch length, control and active 
time intervals relative to the trigger are set by
`src_rec/epochs/t_range,t_control,t_active` keys, respectively. In this case, the
noise covariance is calculated over the all control intervals, while the full
covariance - over all the active intervals.

For the resting state and naturalistic viewing records the noise covariance is 
constructed assuming that the noise is produced by Gaussian uncorrelated randomly
oriented brain sources uniformly distributed over the source space. The full covariance
is the MEG sensors data covariance matrix calculated over the whole record.

Finally, a single "combined" source time course is produced for each ROI using the PCA
approach. The set of ROIs to be used is controlled by the keys `src_rec/atlas` and
`src_rec/parcellations`. The ROI "centers of masses" and "combined" ROI beamformer
weight vectors are also calculated. 

Currently the source time courses are returned in pseudo-Z units, which is controlled
by the `src_rec/beam/units` setting in the JSON file (see `get_beam_weights()` function
documentation for more information about units). Additionally, **all time
courses are normalized on the global pseudo-Z value of the record**, which is calculated
as a ratio of traces of the full covariance and the noise covariance
matrices. This is done to avoid statistical biases in group analyses that
can occur due to differences in signal to noise ratios in subject records. 

The time courses are saved in HDF5 format files, together with: ROI names
and ROI centers information, the events found in the original record, the ROI
beamformer weights and the global pseudo-Z value for the corresponding
sensor data. Please refer to `write_roi_time_courses(), read_roi_time_courses()`
methods documentation for more details regarding the source time course HDF5
files. 

The source code for this step is found in file
[src_rec.py](doc/_build/html/code.html#module-src_rec).

