"""
**File structure dependent utilities and convenience functions.**
"""
from datetime import datetime
from pathlib import Path
import socket

def is_valid_sid(s):
    """
    Returns True if string `s` is a valid subject ID. Currently
    the valid sID should contain 6 characters which are eigher 
    digits or upper case letters.
    """
    if len(s) != 6:
        return False

    return all(c.isdigit() or c.isupper() for c in s)

def is_valid_date(date_str):
    """
    Given a 6-digit integer string, returns `True` if this is a valid
    YYMMDD date.
    """

    if len(date_str) != 6:
        return False

    try:
        # Try to parse the string as a date
        date_obj = datetime.strptime(date_str, '%y%m%d')
        return True

    except ValueError:
        # If parsing fails, it is not a valid date
        return False

def is_erm_file(fname):
    """
    Returns True if specified file is a (possibly processed
    version) of empty room recording.

    Args:
        fname(Path | str): the file (path)name

    Returns:
        result(bool): test result

    """
    return True if '_erm_' in Path(fname).name else False

def is_erm_band_file(fname):
    """
    Returns True if specified file is a (possibly processed
    version) of an empty room recording which was band-pass filtered
    to construct projectors. Such files are expected to have names
    ending like ..._35.0-53.0Hz.fif (that is, with band specified
    in Hz)

    Args:
        fname(Path | str): the file (path)name

    Returns:
        result(bool): test result

    """
    if not is_erm_file(fname):
        return False

    return 'Hz.fif' in Path(fname).name

def get_fif_task(fif):
    """
    Given the pathname of the .fif file, return its corresponding
    task name. Possible tasks are: 'rest', 'naturalistic' and 
    'task_runN' where `N` is `1`, `2`, or `3`.

    Args:
        fif(Path | str): the .fif file full or partial pathname

    Returns:
        task_name(str): corresponding task

    """
    stem = Path(fif).stem

    for ptrn in ('naturalistic', 'rest'):
        if  ptrn in stem:
            return ptrn

    ptrn = 'task_run'
    pos = stem.find(ptrn)

    if pos == -1:
        raise ValueError(f'Could not determine the task for .fif record {fif}')

    return stem[pos : (pos + len(ptrn) + 1)]

extract_epochs = lambda task_name: True if 'task_run' in task_name else False
"""
Test if the `task_name` requires extracting epochs from the raw data.
"""

def make_subject_dict(folder, slist):
    """
    Create a dictionary containing key-value pairs in the form 

    `'sid': ['YYMMDD', 'YYMMDD',...]`,

    where the list contains the date strings for the subject's recording sessions.

    The folder is expected to contain subject subfolders whose names satisfy
    specific format, checked by function `is_valid_sid()`.

    Args:
        folder (str): a full path to the folder containing each subject's data as a
            subfolder like 'sID / date1 date2 ... '
        slist (list | None): a list where each element is eighter sID or a list
            in the form `[sID, 'YYMMDD', 'YYMMDD']`, with `sID` being a subject ID
            string and `'YYMMDD'` - an integer string. When only `sID` is provided,
            all available dates will be added to corresponding `sID`. If `slist`
            is `None` - then all subjects and all dates in the `folder` will be
            collected.

    Returns:
        subject_dict(dict): the dictionary described above.

    """
    base = Path(folder)

    if slist is None:
        # Get a list of subfolders 
        slist = [f.name for f in base.iterdir() \
                    if f.is_dir() and is_valid_sid(f.name)]


    sdict = {}
    for s in slist:
        if isinstance(s, str):
            # Get all dates for the subject
            sdir = base / s
            dlist = [f.name for f in sdir.iterdir() \
                    if f.is_dir() and is_valid_date(f.name)]
            sdict[s] = dlist
        elif isinstance(s, list):
            sid = s[0]

            if not is_valid_sid(sid):
                raise ValueError(f'Invalid subject ID: {sid}')

            dlist = [d for d in s[1:] if is_valid_date(d)]
            sdict[sid] = dlist
        else:
            raise ValueError(f'Incorrect subject ID/dates specification: {s}')

    return sdict

def get_subjects_and_folders(ss, step):
    """
    Get input and output folders for a specified step, and construct the
    `subjects` dictionary.

    Args:
        ss(obj): reference to this app object
        step(str): the processing step name

    Returns:
        in_dir(Path): full pathname to this step input folder
        out_dir(Path): full pathname to this step output folder
        subjects(dict): subjects dictionary containing key-value pairs in the form 
            `'sid': ['YYMMDD', 'YYMMDD',...]`, where the list contains the date
            strings for the subject's recording sessions.

    """
    # Get input and output data locations for this step
    # (these are in fact Path objects)
    in_dir = ss.data_host.get_step_in_dir(step)
    out_dir = ss.data_host.get_step_out_dir(step)
    slist = ss.args['subjects']

    subjects = make_subject_dict(in_dir, slist = slist)
    return in_dir, out_dir, subjects

def get_subj_subfolder(folder, sid, date):
    """
    Given a `Path` object of a folder with all subjects' files
    construct a `Path` to the date-specific subject's data.

    Args:
        folder(Path): path to the subject files, for example '/.../meg/raw'
        sid(str): subject's ID
        date(str): integer string 'YYMMDD'

    Returns:
        path(Path): Path object for subject 'sid' files for specified date
    """
    return folder / sid / date

def get_head_pos_file(in_file):
    """
    Get base name for the head positions file, given the input raw
    .fif file name.

    Args:
        in_file (Path | str): pathname or basename of the input file

    Returns:
        basename(str): the base name of the output file
    """
    f = Path(in_file)

    # We skip everyting after '_raw' (if found) and append '_pos'
    pos = f.stem.find('_raw') + len('_raw')
    stem = f.stem[:pos] if pos != -1 else f.stem
    
    out_name = stem + '_pos' + f.suffix
    return out_name

def get_erm_band_file(f, band):
    """
    Get full pathname for ERM .fif file filtered to specified frequency
    band.

    Args:
        f(Path): full pathname of the empty room recording .fif file
        band([fmin, fmax]): frequency band, Hz

    Returns:
        erm_band_fif(Path): full pathname to the ERM recording filtered to
            specified band. The band will be appended as _NN.n-MM.mHz suffix.
    """
    append = f'_{band[0]:.1f}-{band[1]:.1f}Hz'
    out_name = f.stem + append + f.suffix
    return f.parent / out_name

def get_maxfiltered_rank(raw):
    """
    Assuming that the record was max-filtered, get the true rank of the data
    using maxfilter info saved in the raw object.

    Args:
        raw(MNE Raw): the data Raw object

    Returns:
        data_rank(int): either the rank after max_filter, or None if
            maxfilter info was not found

    """
    try:
        for h in raw.info['proc_history']:
            if 'max_info' in h:
                return h['max_info']['sss_info']['nfree']

        return None
    except:
        return None

def get_subjects_for_job(all_subjects, ijob, njobs):
    """
    Given a full list of subjects,the array job index and the total
    number of jobs in the array, return a subject list for this
    job to process.

    Args:
        all_subjects(list of str): full list of subject IDs for the SLURM
            array job
        ijob(int): 0-based index of the current job; 0 <= ijob < njobs
        njobs(int): the total number of jobs in the SLURM array job

    Returns:
        slist(list of str): list of subject IDs for this job to process

    """
    n = len(all_subjects)

    if n <= njobs:
        slist = [all_subjects[ijob]] if ijob < n else []
    else:
        nperjob = n // njobs if n % njobs == 0 else n // njobs + 1
        istart = ijob * nperjob

        if istart >= n:
            slist = []
        else:
            iend = istart + nperjob

            if iend > n:
                iend = n

            slist = all_subjects[istart:iend]

    print(f'Subjects to process for array job #{ijob}:')
    print(slist)
    print('\n')
    return slist

def get_sdict_for_job(ss, subjects):
    """
    Based on the full subjects dictionary, create subjects dictionary
    for a given array job index.

    Args:
        ss(obj): reference to this app object
        subjects(dict): subjects dictionary containing key-value pairs in the form 
            `'sid': ['YYMMDD', 'YYMMDD',...]`, where the list contains the date
            strings for the subject's recording sessions.

    Returns:
        sdict(dict): subjects dictionary for current job, in the above format

    """
    slist = list(subjects.keys())
    slist = get_subjects_for_job(slist, ss.ijob, ss.args['N_ARRAY_JOBS'])

    # Create the subject dictionary for the current job
    sdict = {s:subjects[s] for s in slist}
    return sdict

def files_to_process(ss, step):
    """
    A generator function yielding next input file to process, and corresponding
    output file. All folders needed to store the output file are created also,
    unless they already exist.

    **This generator applies to the MEG part of the pipeline only**, because the
    MRI part (including BEM model and source space construction) uses a different directory
    structure - see `mri_subjects_to_process()` for details.

    Args:
        ss(obj): reference to this app object
        step(str): name of the step being executed

    Returns:
        in_fif(Path): full pathname object for the input file to process
        out_fif(Path): full pathname object for the output file to process

    """
    # Get input and output data locations and subjects dict for this step
    in_dir, out_dir, subjects = get_subjects_and_folders(ss, step)

    # When specific files to process are listed for the pipeline step in
    # question, those should refer to a single subject and a single recording
    # date, to avoid ambiguity:
    if ss.args[step]['files'] is not None:
        if len(subjects) > 1:
            raise ValueError(\
                f'The "files" key is not null for step {step}; multiple subjects are not allowed')

        # If a single subject is specified, only a single date is allowed
        if len(list(subjects.values())[0]) > 1:
            raise ValueError(\
                f'The "files" key is not null for step {step}; multiple dates are not allowed')

    if ss.data_host.cluster_job:
        # Currently subjects refers to all subjects for the SLURM array job
        # Get a list of subjects for current job
        """QQQ - remove after testing
        slist = list(subjects.keys())
        slist = get_subjects_for_job(slist, ss.ijob, ss.args['N_ARRAY_JOBS'])

        # Create the subject dictionary for the current job
        sdict = {s:subjects[s] for s in slist}
        subjects = sdict
        """
        subjects = get_sdict_for_job(ss, subjects)

    config = ss.args[step]

    for subj in subjects:
        for date in subjects[subj]:
            # in_,out_folder define locations for specific subj and specific date
            in_folder = get_subj_subfolder(in_dir, subj, date)
            out_folder = get_subj_subfolder(out_dir, subj, date)

            if not out_folder.exists():
                out_folder.mkdir(parents=True, exist_ok=True)

            fif_files = config['files']

            if fif_files is None:
                fif_files = list(in_folder.glob('*.fif'))
            else:
                fif_files = [in_folder / f for f in fif_files]

            for in_fif in fif_files:
                out_basename = ss.data_host.get_step_out_file(step, in_fif)
                out_fif = out_folder / out_basename
                yield in_fif, out_fif


def events_fif(ss, fif):
    """
    Based on a raw .fif file full pathname, generate a full path to corresponding
    events file, which is supposed to reside in the 'prefilter' step output folder.
    The input raw .fif file may be produced by any step of the processing pipeline;
    only its subject, date and task information will be used to determine the
    events file path.

    Args:
        ss(obj): reference to this app object
        fif (Path | str): full pathname to a raw .fif file

    Reurns:
        events_fif(Path): full pathname of the events file

    """
    subject = fif_subject(fif)
    date = fif_date(fif)
    task = get_fif_task(fif)
    out_dir = ss.data_host.get_step_out_dir('prefilter')    # Prefilter step directory
    out_folder = get_subj_subfolder(out_dir, subject, date)    # Current subject's subfolder
    return out_folder / f'{subject}_{task}_eve.fif'

def mri_subjects_to_process(ss, step):
    """
    This function returns a list of subjects to process in MRI part of the
    pipeline. Note that when running array job on the cluster, for some of
    the job indices this list may be empty.

    Args:
        ss(obj): reference to this app object
        step(str): name of the step being executed

    Returns:
        subj_list(list of str): list of MRI subject IDs to process. Note that MRI subject
            ID differs from the same subject's MEG subject ID

    """
    if ss.args['subjects'] is None:
        mri_data_root = ss.data_host.get_mri_subjects_dir()
        # Get a list of subfolders 
        slist = [f.name for f in mri_data_root.iterdir() \
                    if f.is_dir() and \
                        is_valid_sid(mri2meg_subject(f.name))]
    else:
        # Convert MEG subject IDs to MRI subject IDs
        slist = [meg2mri_subject(s) for s in ss.args['subjects']]

    if ss.data_host.cluster_job:
        slist = get_subjects_for_job(slist, ss.ijob, ss.args['N_ARRAY_JOBS'])

    return slist

def meg2mri_subject(meg_subject):
    """
    For everybody's convenience, the same subject's ID is encoded
    differently on MEG and MRI side. This function converts MEG
    subject ID to MRI subject ID in this particular project.

    Args:
        meg_subject (str): MEG subject ID

    Returns:
        mri_subject (str): corresponding MRI subject ID

    """
    return 'sub-' + meg_subject

def mri2meg_subject(mri_subject):
    """
    For everybody's convenience, the same subject's ID is encoded
    differently on MEG and MRI side. This function converts MRI
    subject ID to MEG subject ID in this particular project.

    Args:
        mri_subject (str): MRI subject ID

    Returns:
        meg_subject (str): corresponding MEG subject ID

    """
    return mri_subject[4:]

def get_bem_data(ss, meg_subj):
    """
    Return a dictionary with paths to subject's BEM-related data
    required for forward solutions computation.

    Args:
        ss(obj): reference to this app object
        meg_subj(str): subject ID on the MEG side

    Returns:
        paths_dict(dict): a dictionary containing Path objects for
            the subject's BEM, mri->head transformation and source
            space data.
    """
    # QQQ Do we need this func at all?
    mri_subj = meg2mri_subject(meg_subj) 
    bem_dir = ss.data_host.get_subject_bem_dir(mri_subj)
    # NOT DONE YET!
    # Continue with creating paths to specific files in the bem_dir

fif_subject = lambda fif: fif.parent.parent.name
"""
Lambda-function for extracting MEG subject ID from the path like <path>/<meg-subj>/YYMMDD/file.ext
"""

fif_date = lambda fif: fif.parent.name
"""
Lambda-function for extracting session date from the path like <path>/<meg-subj>/YYMMDD/file.ext
"""

def get_trans_file_pathname(fif):
    """
    Given a full pathname of the fif file, construct
    a full pathname for the -trans.fif file residing in the same
    folder as the  `fif` file.

    Args:
        fif(Path): the full pathname of the .fif file. It is expected to
            be like <path>/<meg-subj>/YYMMDD/file.ext

    Returns:
        trans_fif(Path): full pathname of the transformation file, which is
            <path>/<meg-subj>/YYMMDD/<meg-subj>-trans.fif

    """
    return fif.parent / (fif_subject(fif) + '-' + fif_date(fif) + '-trans.fif') 

def ltc_file_pathname(meg_subj, task_name, ss_fif, atlas, out_dir):
    """Construct full pathname for HDF5 file with saved ROI time courses, based on the
    task, source space and the altas used. For example, for the task 'rest' and source space like
    'sub-45TDGV-ico-4-src.fif' the output .hdf5 file name will be
    '45TDGV-rest-ico-4-destrieux-ltc.hdf5'

    Note that the names of ROIs themselves depend on the atlas (parcellation) used. ROI names
    will be stored inside the generated HDF5 file.

    Args:
        meg_subj(str): subject ID on the MEG side
        task_name(str): one of 'rest', 'naturalistic' or 'task_runN',
            where `N` is `1`, `2`, or `3`
        ss_fif(Path): full pathname of this subject's source space
        atlas(str): atlas name for the ROI parcellation
        out_dir(Path): pathname of the output folder for the ltc file

    Returns:
        ltc_hdf5 (Path): full pathname of the output .hdf5 file with label
            time courses
    """
    ss_name = ss_fif.name
    mri_subj = meg2mri_subject(meg_subj)
    f = ss_name.replace(mri_subj, meg_subj + f'-{task_name}')
    f = f.replace('-src.fif', f'-{atlas}-ltc.hdf5')
    return out_dir / f

class DataHost:
    """
    This class contains the data host and file structure dependent methods and
    attributes for accessing all MEG records and other related parameters and
    data.

    **Attributes**

    Attributes:
        config(dict): the main app configuration dictionary passed to the class
            constructor
        host(str): the data host name. If current host is not listed in the 
            `config['hosts']`, `host = 'other'` will be set. 
        cluster_job(bool): flag that the host supports running python scripts as
            SLURM array of parallel jobs.
        root(Path): pathname to the root of the project data files
        pipeline_version:(str): subfolder under `root`, where all processing
            results for the current pipeline version are stored.
        meg(str): name of subfolder with project's MEG files

    **Methods**

    """
    def __init__(self, config):
        """
        Args:
            config(dict): a dictionary constructed from the main configuration JSON file.

        Returns:
        """
        self.config = config

        # Choose appropriate host name from those listed in the json:
        host_found = False
        host = socket.gethostname()

        for key in config['hosts']:
            if key in host:
                host = key
                host_found = True
                break

        if not host_found:
            host = 'other'

        self.host = host
        self.cluster_job = config['hosts'][host]['cluster_job']
        self.root = Path(config['hosts'][host]['root'])
        self.pipeline_version = config['pipeline_version']
        self.meg = config['hosts'][host]['meg']
        self.mri = config['hosts'][host]['mri']
        self.ct_file = config['hosts'][host]['ct_file']
        self.fc_file = config['hosts'][host]['fc_file']

    def get_step_in_dir(self, step):
        """
        Get input data folder for a specified step.

        Args:
            step(str): step name

        Returns:
            path(Path): full path to the folder
        """
        if step == 'prefilter':
            in_dir = self.root / self.meg / self.config[step]['in_dir']
        elif step in ('maxfilter', 'ica', 'src_rec'):
            in_dir = self.root / self.meg / self.config["out_root"] / \
                    self.pipeline_version / self.config[step]['in_dir']
        else:
            raise ValueError(f'Invalid step specified: {step}')

        return in_dir

    def get_step_out_dir(self, step):
        """
        Get output data folder for a specified step.

        Args:
            step(str): step name

        Returns:
            path(Path): full path to the folder
        """
        if step in ('prefilter', 'maxfilter', 'ica', 'src_rec'):
            out_dir = self.root / self.meg / self.config["out_root"] / \
                    self.pipeline_version / self.config[step]['out_dir']
        else:
            raise ValueError(f'Invalid step specified: {step}')

        return out_dir

    def get_step_out_file(self, step, in_file):
        """
        Get base name for the step output file, given the step 
        name and the input file name.

        Args:
            step(str): step name
            in_file (Path | str): pathname or basename of the input file

        Returns:
            basename(str): the base name of the output file

        """
        # TODO: this function does not really belong to the DataHost
        # object, because it does not use host-specific data
        f = Path(in_file)
        stem = f.stem
        ext = f.suffix

        if step in ('prefilter', 'maxfilter', 'ica', 'src_rec'):
            out_name = stem + self.config[step]['suffix'] + ext
        else:
            raise ValueError(f'Invalid step specified: {step}')

        return out_name

    def get_ct_file(self):
        """
        Return path to the MEG system cross talk .fif file
        """
        return self.root / self.meg / self.ct_file

    def get_fc_file(self):
        """
        Return path to the MEG system fine calibration data file
        """
        return self.root / self.meg / self.fc_file

    def get_mri_subjects_dir(self):
        """
        Return a path to the folder with all subjects MRI and
        FreeSurfer reconstructions data.
        """
        return self.root / self.mri 

    def get_subject_bem_dir(self, mri_subject):
        """
        Return a path to MEG subject's BEM data folder.

        Args:
            mri_subject(str): subject ID on MRI side

        Returns:
            bem_path(Path): Path object pointing to the subject's BEM
                folder.

        """
        return self.get_mri_subjects_dir() / mri_subject / 'bem'

