# MEG processing pipeline for the "Religion" project

## Summary

This Python code implements a processing pipeline for the MEG recordings collected
for the *Religion* project. It starts with the raw data collected by Elekta
(MEGIN) scanner, and produces source reconstructed signal time courses for a
set of specific anatomical locations in the subject's brain. Then a set of
statistical analyses of the results is performed. The code uses the
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
project's documentation**.

Currently, the pipeline includes the following steps.
1. [**`'input'`**](#1-the-input-step) - loading all the input and
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
7. [**`'src_erf'`**](#7-evoked-response-fields-erfs-calculation) - generating low frequency
   source level evoked response fields for each subject.
8. [**`'src_hilbert'`**](#8-constructing-hilbert-envelopes-for-task-runs-data) - constructing
   Hilbert envelopes for the task data runs.   
9. [**`'plot_epochs'`**](#9-plot-epoch-averaged-time-frequency-distributions-of-sensor-signals) -
   creating overview time-frequency plots of the epoched data.
10. [**`'plot_waveforms'`**](#10-create-a-grid-plot-of-waveforms-for-different-tasks-and-conditions) -
   creating a grid plot of waveforms for different tasks and conditions.
11. [**`'pls_analysis'`**](#11-pls-and-other-statistical-analyses-of-the-task-data) -
   PLS and other statistical analyses of evoked and induced responses.
12. [**`'plot_pls_inflated_brain'`**](#12-plot-pls-z-scores-on-an-inflated-brain-surface) -
   displaying PLS z-scores on an inflated brain surface.
13. [**`'plot_sm_fit_stats'`**](#13-plot-an-overview-of-statistical-model-fitting-results-for-all-bands-and-images) -
   plotting an overview of statistical model fitting results for all frequency bands and
   images/image pairs.
14. [**`'plot_sm_fit_inflated_brain'`**](#14-plot-statistical-model-fits-results-for-all-rois-on-an-inflated-brain-surface) -
   plotting statistical model fit results for all ROIs on an inflated brain surface.          

More details about the configuration file and the processing steps are given below.

## Setup 
### Python environment
This code runs under Python 3.10+, using MNE Python version 1.7+. To establish
a virtual environment, one needs to run the commands below.
```
Steps to be done just once
--------------------------
# On Cedar only - to activate Python environment
    module load StdEnv/2023
    module load python/3.11.5
    module load scipy-stack/2023b

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

# For using graphics with dataframes (Bonni's scripts)
python3 -m pip install seaborn

# Statsmodels package
python3 -m pip install --no-index statsmodels   # use no-index option only on CC cluster

deactivate
```
### Data files organization
The following file structure is assumed for this project.

All data with exception of MRI data resides in a single project folder, referred
to as the "project root" further on.

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
the input `.fif` (or other) file to construct the name of the output `.fif`,
`.hdf5`, etc. file. For example, with the current settings if the input file
for the `ica` step is `XXXX.fif`, the output file will be `XXXX_ica.fif`.
The suffix is not used for the source reconstruction step.

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
of jobs in the array determined in the `.sbatch` script properly matches the
`N_ARRAY_JOBS` parameter in the [JSON configuration file](#json-configuration-file).

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

### Additional sub-step: calculating evoked responses
When source reconstruction for task runs is completed, one can perform evoked (or phase-locked)
signal extraction as an additional step. To do this, it is necessary to re-run `src_rec` step
with `"do_evoked"` key in the JSON configuration set to `true`. As a result, for each event ID
listed under the `"events_for_evoked"` key in the configuration file, the averages and STDs of all
epochs corresponding to this ID will be calculated, and the result will be saved in .HDF5 file.
This will be repeated separately for each task of every subject to be processed. Assuming that 
there are four event IDs and three task runs per subject, twelve new .HDF5 files will be created
in the subject's beamformer output folder, with the following name template:
```
<subject-ID>-task_run<N>-ico-4-destrieux-ltc-evoked-<event-ID>.hdf5
```
### A note about beamformer types
Configuration key `"use_dual_state_beam"` is typically set to `false`, in which case a classic
scalar minimum variance beamformer reconstruction is performed. When set to `true`, a novel dual-state
beamformer is used, which may yield better results in certain situations. Please note that the type of
beamformer applied is not reflected in either the names or the contents of the output .hdf5 files. To
avoid overwriting existing results when using both beamformers in parallel, it is recommended to select
a dedicated output folder for each type by appropriately setting the value of the `"out_dir"` key.

### A note regarding calculating the rank of the data
The MEG sensor data used in the beamformer analysis is highly degenerate. It means that the sensor time
courses are linearly dependent. The loss of degrees of freedom in data is a result of the preprocessing
steps such as Maxwell filtering, ICA, and out-projections, aimed at reducing the noise and artifacts in
the data. The price paid is that the original rank of the data covariance matrix, which is around 300,
is typically reduced to a number slightly above 60.

If the rank is overestimated, the beamformer reconstruction may produce strong spurious sources. To avoid
this one needs to choose the rank wisely. This is done by plotting the spectrum of eigenvalues
(EVs) of the MEG data covariance matrix in descending order in logarithmic scale, identifying a 
sharp drop in their magnitudes and keeping only components with EVs to the left of that drop.
The ratio of the smallest kept EV to the maximum EV is called a condition number `rcond`. Thus found
`rcond` value is then set in this step's JSON configuration, under the key `"beam"`, sub-key `"rcond"`.
For the MEG data in question `rcond = 1e-4` was found safe to use for most subjects.

The source code for this step is found in file
[src_rec.py](doc/_build/html/code.html#module-src_rec).

## 7. Evoked response fields (ERFs) calculation
At this step low frequency source level ERFs are generated. For each subject the following operations
are performed:
1. Evoked responses from all task runs are averaged separately for each event ID
(image type)
2. The averaged data is low pass filtered
3. The filtered data is downsampled to a target ERF sample rate

To deal with the sign uncertainty of the beamformer reconstructed time courses, the signs of evoked
responses of all task runs for a given subject were aligned. This was achieved by calculating
a correlation of the evoked response for each run with that of the 1st run, on a time interval
specified by key `"sign_adjust_interval"` of this step configuration. If the correlation turned
out negative, the sign of the given evoked response was flipped before adding to the average.

The low pass filter frequency and the target sample rate are specified by configuration keys `"fmax"`
and `"target_sample_rate"`, respectively. 

Resulting ERF signals are stored as .HDF5 files in the subject's beamformer output folder, with the
following name template:
```
<subject-ID>-ico-4-destrieux-ltc-erf-<event-ID>.hdf5
```

The source code for this step is found in file
[src_erf.py](doc/_build/html/code.html#module-src_erf).

## 8. Constructing Hilbert envelopes for task runs data
ERF fields calculated at the previous step represent broad band phase-locked brain responses
to the stimuli. At the current step so called induced, or "time-locked" responses to the stimuli are
estimated as described below. Parameters for this step reside under key `"src_hilbert"` in the
JSON configuration file.
 
Source-reconstructed task runs for each subject are first band-pass filtered to canonical
frequency bands (key `"bands"`). The events to process are listed under the key
`"events_for_hilbert"`. Then for each frequency band the analytic signal is constructed and a
time course of either its squared amplitude (key `"power"` set to `true`) or of the 
amplitude itself is found. The time course is then downsampled to the sampling
rate specified by the key `"target_sample_rate"`, and saved to the .hdf5 file in the
subject's folder. 

NOTE: Current version of the code **forces the power (i.e. squared amplitude) envelope** irrespective
to the `"power"` key setting, and the output .hdf5 file name looks like this:
`45TDGV-ico-4-destrieux-ltc_henv_8.0-13.0Hz_141.hdf5`. This is done because additional 'pwr'
suffix was not included in the output .hdf5 name to properly reflect this key setting at the
time of mass data processing. However all changes required for adding the suffix are already
implemented and can be turned on in future runs - see "QQQ" comments in the code.

The source code for this step is found in file
[src_hilbert.py](doc/_build/html/code.html#module-src_hilbert).

## 9. Plot epoch-averaged time frequency distributions of sensor signals
An overview topographic plot of time-frequency power spectra of sensor signals for a given
task run of a given subject is created.

All input parameters for this step are specified under key `"plot_epochs"` of the JSON configuration
file. Please note that the top-level subject list (key `"subjects"`) should contain exactly one
subject (for example, `["45TDGV"]`), and `"plot_epochs"/"files"` key should list just a
single task run data file for this same subject (i.e. `["45TDGV_task_run1_raw_filt_tsss_ica.fif"]`).
The created plot is saved as .PNG image inside a folder specifed by the `"plot_epochs"/"out_dir"`
key.

The `"picks"` key specifies a set of channels to plot, for example `"meg"`, `"mag"`, `"grad"`, etc.
Other parameters are self-explanatory or documented inside the JSON configuration file.

The source code for this step is found in file
[plot_epochs.py](doc/_build/html/code.html#module-plot_epochs).

## 10. Create a grid plot of waveforms for different tasks and conditions
This steps generates several types of grid plots of the reconstructed signal time courses.
Similarly to the previous step, the top-level subject list (key `"subjects"`) should contain exactly
one subject (for example, `["L846DP"]`), and the `"files"` sub-key for this step should list
just a single evoked or ERF source data file for the same subject, for example
```
    "plot_waveforms": {
        ...
        "files": ["L846DP-ico-4-destrieux-ltc_erf_141.hdf5"],
        ...
    }
``` 

For all types of plots the `"channels"` sub-key specifies a list of the names ROIs to be shown.
The `"adjust_signs"` sub-key defines whether an attempt to align signs of the curves shown
within a single sub-plot is attempted. If requested, the sign alignment is done by calculating
signs of correlations between the time curves, the same way as described for the
[ERF step](#7-evoked-response-fields-erfs-calculation).

The type of the grid plot is determined by the value of the `"task"` sub-key, as follows.

### Tasks "evoked_std", "erf_std"
For these values of the `"task"` key plots of evoked or ERF responses from specified locations
together with their standard deviations will be generated. Additional plotting parameters (colors,
labels, titles) can be set under the sub-keys `"evoked_std"`, `"erf_std"` of the main key of
this step `"plot_waveforms"`. Mind to specify the correct file (`...evoked...hdf5` or `...erf...hdf5`)
with the `"files"` key.

### Task "compare_events"
In this case plots of several evoked responses for given locations are constructed. Corresponding
events are listed under the key `"src_rec"`, sub-key `"events_for_evoked"` of the source reconstruction
step. The `"files"` key in this case should list a *single* .hdf5 file with an evoked response corresponding
to any of the events; data for all other events will be found automatically. Additional plotting
parameters can be specified under the sub-key `"compare_events"`.

### Task "compare_beams"
For this task, the plots will display *evoked* data from two versions of a single .hdf5 file listed
under the `"files"`key. The 1st version of file will be taken from folder given by the
`"in_dir"` sub-key of this step. The 2nd version will be taken from the folder given by
`"compare_beams"/"compare_dir"` sub-key. Additional plotting parameters can be specified under
`"compare_beams"/"plot_args"` sub-key.

## 11. PLS and other statistical analyses of the task data
At this step various statistical analyses of brain response to presented image stimuli are performed.
The majority of them represent various PLS tests applied to the evoked (ERF) data. Some PLS tests
are applied to Hilbert envelopes of the induced responses in canonical frequency bands. Several
non-PLS statistical tasks can also be done in this step. In any case, a concrete task is specified
by the value of the sub-key `"task"` under the main key `"pls_analysis"` in the JSON configuration file. 

### PLS tasks
The PLS part of this step relies on matlab PLS package to do most of the work. Therefore one
needs matlab to be installed in the system for this code to run, together with the matlab PLS analysis
source package.

Typically, the PLS processing itself is performed remotely on the Digital Alliance Cedar cluster. 
The `.mat` file with results is then downloaded to the local machine, where heatmaps and .hdf5
files are constructed.

Most of the keys for this step in the JSON files are self-explanatory or described in the comments.
Only some important ones are mentioned here.

The time interval in seconds for the MEG data to be included in PLS is set by the sub-key
`"pls_interval"`.

When running PLS on MEG ERF data directly, the value of the sub-key `"erf_power"` should be set to
`false`. Alternatively, to do PLS with ERF powers one needs to set the value of this key to
`true`.

When one wants to actually run the PLS calculations (i.e. on Cedar), the value of a sub-key
`"return_precalculated_result"` should be set to `false`. When processing already available `.mat`
file with results (i.e. downloaded from Cedar to the local machine), the value of this sub-key
should be set to `true`.

The `"files"`, `"create_subjects_out_folders"` sub-keys are present for consistency and should
be set to `null` and `false`, respectively.

All information that PLS needs about the subjects, their affiliations with SCZ or ASD groups, and their
religiosity data is stored in `.csv` file. For every data host (i.e. Cedar or a local
machine) a full pathname to this file should be specified under key `"hosts"/"<host>"`, sub-key 
`"subjects_info_csv"` in the JSON configuration. The `.csv` file is expected to have a single
row header, and specific names for columns with various data. In particular, the column with
subject IDs should have a name given by the "pls_analysis"/"sID_colname" sub-key. Sub-key `"bias_colname"`
sets the name of the bias column. Negative values in this column correspond to SCZ group, which
is always designated as `group 0` in all analyses. Positive values correspond to the ASD group,
which is `group 1`. Sub-key `"bin_belief_colname"` sets the name for the binary "believer/non-believer"
score column. Sub-key `"santa_clara_colname"` specifies the name of a column with Santa-Clara
scores. 

As mentioned above, specific type of statistical analysis done is determined by the
`"task"` sub-key. The descriptions of varios PLS tasks are presented below.

* **`"erf_mc_2groups1img"`**. Mean-centered PLS comparing responses of 2 groups to a single image is run.
  Only a single contrast (latent variable) [0.75,-0.75] exists in this case. The ID of the image in
  question is set by the value of the `"event_id"` sub-key.
* **`"erf_mc_4groups1img"`**. Same as the previous task, but in this case 4 sub-groups are compared, which
  are formed by sub-dividing each main group into believers and non-believers. Thus each sub-group is
  specified by a pair (main group number, binary believer score). Mapping of those pairs 
  to sub-group numbers is given under the sub-key `"bias_belief_groups"`. For this task, three independent
  contrasts (latent variables) found by PLS will exist.
* **`"erf_mc_1group4img"`,`"erf_mc_2group4img"`,`"erf_mc_pooled4img"`**. Mean-centered PLS with
  four conditions (images). Condition codes are listed under the key `"img_events"`. For these tasks,
  the images are presented to a single group, or to both group 1 and group 2, or to a single pooled
  group of all subjects respectively. For the `"erf_mc_1group4img"` task the group in question is
  specified by the `"group_id"` key.
* **`"erf_contrast_2group2img"`,`"erf_contrast_2group4img"`**. The _contrast_ (non-rotated) PLS
  comparing responses of 2 groups to a pair of images, or to all four images, respectively. In this
  case the contrast is specified manually by listing the weights for each (group, condition) combination
  under the key `"contrasts"`. The weights for the group 0 are listed first, then those for group 1.
  The number of weights will be 4 for the first of these tasks (i.e. `"[1,-1,1,-1]"`),
  or 8 for the second one (i.e. `"[1,-3,1,1,-1,3,-1,-1]"`). In all cases the image events are
  listed by the key `"img_events"`.

The following tasks specify PLS analyses run on Hilbert envelopes calculated for standard frequency
bands, rather than on ERF time courses.

* **`henv_mc_4groups1img`**. Mean-centered PLS for 4 sub-groups and 1 image for Hilbert envelopes.
  The subgroups are: SCZ-believers, ASD-believers, SCZ-nonbelievers, ASD-nonbelievers.
* **`henv_std_mc_4groups1img`**. Mean-centered PLS for 4 sub-groups and 1 image for time courses of
  STDs of Hilbert envelopes rather than envelopes themselves. Same sub-groups as above.

For all PLS tasks the number of random permutations of subjects for P-values estimates is specified
by the `"num_perm"` sub-keys of the keys `"pls_options_mc"` or `"pls_options_contrast"`, depending
on the PLS type. Similarly, the number of bootstrap resamplings for the Z-scores
estimates is given by the `"num_boot"` sub-keys.

When ran locally, each of the above tasks produce a composite figure showing PLS results for a specified
contrast/latent variable. The variable number is zero-based and is set by the `"heatmap"/"latent_var"`
sub-key. The figure includes a bar plot for the contrast in question, a heatmap of the Z-scores and
a plot of median values of positive and negative Z-scores for each time moment. The median is calculated
over all the ROIs. Parameters for these plots for PLS tasks are given under sub-keys `"heatmap"`,
`"barplot"`, `"medians"`.

Non-PLS tasks are described in the next two sub-sections.

### Comparisons of distributions of correlations between brain responses and religiosity scores
As an additional sub-task, univariate statistical comparisons of distributions of correlation
coefficients for two groups of subjects are performed. The correlation coefficients are those
between the MEG responses at a given point in (ROI, time) space, and subjects' Santa-Clara
religiosity scores, for a given group.


* **`"compare_corrs_2groups1img"`**. In this task, correlation coefficients between MEG responses to
  a given image type occurring at a each point in (ROI, time) space, and the Santa-Clara religiosity
  scores are calculated for each subject. Then distributions of these coefficients for two groups
  (SCZ vs ASD) are compared. The main processing code is provided in the module `bootes_multivariate.py`
  by Bonnie Ng. Please see the source code for the details. The image ID presented to the two groups
  is specified by the key `"event_id"`. The SC scores for each participant are retrieved from the
  column designated by the key `"santa_clara_colname"` of the main .csv file, as explained above.
  All the inputs necessary to make the `bootes_multivariate()` call are provided under the key
  `"compare_corrs"` of this step's configuration.

After running, a heatmap plot similar to those for PLS tasks is produced. The settings for the this
plot are included under the `"compare_corrs"` key.

### Statistical models fitting
This group of tasks involves fitting Ordinary Least Squares (OLS) statistical model for each ROI.
The model describes brain responses to the presented image type as function of subjects age,
sex, SCZ vs ASD group, and religiosity.

Specifically, for a given subject let `Y` denote the STD of a Hilbert envelope of the induced response
**integrated over the time interval** given by the key `"pls_interval"`. Then the following two OLS tasks
are considered:

*Single image*:
```
Y ~ const + β(a)*age + β(s)*sex + β(g)*group + β(r)*religion + β(i)*group*religion
```

*A pair of images*
```
Y(2) - Y(1) ~ const + β(a)*age + β(s)*sex + β(g)*group + β(r)*religion + β(i)*group*religion
```

Here the symbols have the following meaning:
* β(...): a regression coefficient (“slope”, “correlation”) for corresponding independent variable
* age: as is;
* sex: as is;
* group: 0 for SCZ, 1 for ASD;
* religion: a binary (believer/non-believer) or Santa-Clara score

The values of interest are regression coefficients for the group, religion and
group*religion variables - that is β(g), β(r) and β(i). For each ROI, the effect size
for corresponding beta is calculated as beta itself divided by its standard error. The error is estimated
analytically and is returned with the fit results. Also, the fit function returns a formal p-value
of testing a null-hypothesis that beta in fact equals to 0. OLS model fitting code can be found
in the `stat_model_fits.py` source file.

Specific task names for stat model fitting analyses are as follows.

* **`"henv_std_sm_4groups1img"`**. Fitting stat model for 2 groups and 2 binary religiosity
  categories for a single image type, using STDs of Hilbert envelopes as the brain response.
* **`"henv_std_sm_4groups2img"`**. Fitting stat model for 2 groups and 2 binary religiosity
  categories for the difference in responses to a given pair of image types, using STDs of
  Hilbert envelopes as the brain response.
* **`"henv_std_sm_santa1img"`**. Fitting stat model for 2 groups and Santa-Clara scores as
  religiosity measure for a single image type, using STDs of Hilbert envelopes as the brain
  response.
* **`"henv_std_sm_santa2img"`**. Fitting stat model for 2 groups and Santa-Clara scores as
  religiosity measure for the difference in responses to a given pair of image types,
  using STDs of Hilbert envelopes as the brain response.

When ran locally (i.e. not on the cluster), bar plots of distributions of effect sizes over
the ROIs are produced for the three betas of interest - 3 sub-plots in one figure. Red bars
denote ROIs where effect size had formally reached the statistical significance level,
uncorrected for multiple comparisons.
  
The figure is saved in a .png file with name like `henv_std_sm_santa1img_14-21Hz_111.png` 
(for the 1-mage task) or `henv_std_sm_santa2img_14-21Hz_img_111_121.png` (for the 2-image task).

Note that for 1-image tasks the total number of such figures is 20 (5 bands x 4 images);
for 2-image tasks the total is 30 (5 bands x 6 image pairs). To have a broad overview of all
results at once in a single figure, one can run `"plot_sm_fit_stats"` step described later in
[section 13](#13-plot-statistical-model-fitting-results-distributions-over-rois).

## 12. Plot PLS z-scores on an inflated brain surface
In this step, the PLS z-scores for a specified latent variable taken at a given time moment are projected
onto an inflated brain surface.

As a result:
1. An interactive figure is displayed which can be used to observe the z-scores distribution 
   from different viewpoints. A single or several views can be shown simulateneously either in a single
   figure or in individual figures.
2. The figure(s) is saved as a .PNG file.

The plotting arguments are specified under the key `"plot_pls_inflated_brain"` in the JSON configuration
file. Most of those are self-explanatory or described in comments in the JSON itsef.

The `"time"` sub-key defines the time value in seconds for which the z-scores are plotted. Importantly,
**the actual PLS data displayed (i.e. task, latent variable, etc.) is determined by relevant settings
under the `"pls_analysis"` key**.

## 13. Plot an overview of statistical model fitting results for all bands and images
In this step, a single figure is produced which contains 20 or 30 subplots for the 1-image or 
2-image task, respectively. Each subplot reflects results for a specific choice of (frequency
band, image), or (frequency band, image pair) respectively.

Every subplot contains 4 triplets of bars. Each bar in a triplet shows the result for corresponding beta -
that is β(g), β(r) and β(i) for the (band, image or image pair) in question.  Different
triplets reflect different characteristics of the distribution of the betas over
all the ROIs, namely: 1st triplet - the average, 2nd triplet - the STD, 3rd triplet - the average
over STD ratio, and 4th triplet  - the total number of formally significant ROIs (if any)
for each beta.

Subplots are indexed row-wise from 0 to 19 for 1-image tasks and from 0 to 29 for 2-image tasks.
A combination of (frequency band, image/image pair) for a given subplot is specified by a
corresponding list element under the `"pls_analysis/"array_job_parms"` key of the PLS analysis step.
Namely, for the 1-image tasks this element should be a two-key dictionary like
`{"band": [8.0,13.0], "event_id": 111}`. For the 2-image tasks each element is a dictionary
like `{"band": [8.0,13.0], "img_events": [111,121]}`.

Other details for this step are specified under the main key `"plot_sm_fit_stats"` in the
JSON configuration file.          

The produced figure is saved to a .png file with name like
`henv_std_sm_santa1img_results_overview.png` or `henv_std_sm_santa2img_results_overview.png`,
and a textual summary is printed to the console.

## 14. Plot statistical model fits results for all ROIs on an inflated brain surface
This step is similar to [step 12](#12-plot-pls-z-scores-on-an-inflated-brain-surface) and
produces a 3-dimensional image of a distribution for the beta of interest effect size
over the ROIs. 

Specific parameters for the 3D plot are given under the key `"plot_sm_fit_inflated_brain"`
in the JSON configuration file. In particular, the beta to be plotted is given by the 
`"what"` subkey that can have values `"group", "rel", "group*rel"`. The displayed values
can be clipped so that only ROIs showing above/below threshold results are marked - see
comments for subkeys `"apply_thresholds","clip_interval"` in the configuration file.

The produced figure is interactive, and is also saved in a final static form to a .png
file with a name like `henv_std_sm_santa1img_14-21Hz_111_rel_6_views.png`.
