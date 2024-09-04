"""
** Perform all necessary steps to create a BEM based head conductor
model, including:

Creating BEM surfaces (requires FreeSurfer software installed), BEM model and
BEM solution construction; source space setup.**
"""

import warnings
import mne
import setup_utils as su

def bem_model(ss):
    """
    Perform all necessary steps to produce a BEM head conductor model and
    a source space.

    Args:
        ss(obj): reference to this app object

    """
    warnings.filterwarnings('ignore', category=RuntimeWarning,
                            message=ss.args["file_name_warning"])

    STEP = 'bem_model'
    config = ss.args[STEP]

    # NOTE: this call will properly choose subjects for array job
    # on the cluster
    mri_subjs = su.mri_subjects_to_process(ss, STEP)

    if ss.data_host.cluster_job:
        config['show_plots'] = False

    if not mri_subjs:
        print('No subjects to process - exiting')
        return

    for subj in mri_subjs:
        # Needs a runnable FreeSurfer:
        make_bem_surfaces(subj, ss.data_host, config, verbose = ss.args['verbose']) 
        make_scalp_surfaces(subj, ss.data_host, config, verbose = ss.args['verbose']) 
        construct_bem_solution(subj, ss.data_host, config, verbose = ss.args['verbose'])
        construct_source_space(subj, ss.data_host, config, verbose = ss.args['verbose'])

    warnings.filterwarnings("default", category=RuntimeWarning)
    print(f'\n** {STEP} step completed **\n')                

def make_bem_surfaces(mri_subj, data_host, config, verbose = None):
    """
    Run the watershed algorithm to create BEM surfaces. Needs FreeSurfer-generated
    brain and skull surfaces, and a runnable FreeSurfer instance. If the
    BEM surfaces already exist in subject's `bem` folder, they will be 
    recreated if `recalc_bem` flag is set in the pipeline setup JSON file.

    Args:
        mri_subj(str): subject ID on MRI side
        data_host(DataHost): the DataHost object instance
        config(dict): the `bem_model` step config dictionary
        verbose(str): MNE Python verbose level

    Returns:
        Nothing

    """
    bem_dir = data_host.get_subject_bem_dir(mri_subj)
    bem_names = ['brain.surf', 'outer_skull.surf', 'inner_skull.surf', 'outer_skin.surf',
                 mri_subj + '-head.fif']

    # Check if BEM surfaces already exist
    if not config['recalc_bem']:
        # Check if all is done already
        bem_paths = [bem_dir / f for f in bem_names]

        if all([f.is_file() for f in bem_paths]):
            print(f'make_bem_surfaces(): all surfaces for {mri_subj} already exist')
            return

    mne.bem.make_watershed_bem(mri_subj,subjects_dir=data_host.get_mri_subjects_dir(), 
                               **config['watershed'], verbose = verbose)
    print(f'\nmake_watershed_bem() for {mri_subj} completed.\n')

def make_scalp_surfaces(mri_subj, data_host, config, verbose = None):
    """
    Create scalp surfaces, needed for MEG-MRI coregistraion as per MNE Python
    docs. Needs FreeSurfer-generated brain and skull surfaces, and a runnable
    FreeSurfer instance. If the the target surfaces already exist in subject's
    `bem` folder, they will be recreated if `recalc_bem` flag is set in the
    pipeline setup JSON file.

    Args:
        mri_subj(str): subject ID on MRI side
        data_host(DataHost): the DataHost object instance
        config(dict): the `bem_model` step config dictionary
        verbose(str): MNE Python verbose level

    Returns:
        Nothing

    """
    bem_dir = data_host.get_subject_bem_dir(mri_subj)
    surf_names = [mri_subj + '-head.fif', mri_subj + '-head-dense.fif',
                  mri_subj + '-head-medium.fif', mri_subj + '-head-sparse.fif']

    # Check if BEM surfaces already exist
    if not config['recalc_bem']:
        # Check if all is done already
        surf_paths = [bem_dir / f for f in surf_names]

        if all([f.is_file() for f in surf_paths]):
            print(f'make_scalp_surfaces(): all surfaces for {mri_subj} already exist')
            return

    mne.bem.make_scalp_surfaces(mri_subj,subjects_dir=data_host.get_mri_subjects_dir(), 
                               **config['scalp'], verbose = verbose)
    print(f'\nmake_scalp_surfaces() for {mri_subj} completed.\n')

def bem_solution_fif(mri_subj, ico_bem, nlayers):
    """
    Return a basename for a BEM solution .fif file

    Args:
        mri_subj(str): subject ID on MRI side
        ico_bem (int): ico setting for BEM model: 3 (coarse), 4 (medium),
            or 5 (dense)
        nlayers(int): number of layers in the conductivity model

    Returns:
        fname(str): basename for the BEM solution .fif file

    """
    ico_dict = {3: 1280, 4: 5120, 5: 20484}
    ico_str = f'-{ico_dict[ico_bem]}'*nlayers + '-'
    return mri_subj + ico_str + 'bem-sol.fif'

def src_space_fif(mri_subj, step, ico_src, is_volume = True):
    """
    Return a basename for a source space .fif file

    Args:
        mri_subj(str): subject ID on MRI side
        step(float): grid step (m) for volume source spaces
        ico_src (int): ico setting for the surface source spaces: 3 (coarse), 4 (medium),
            or 5 (dense)
        is_volume(bool): True if a volume source space is meant; otherwise
            a surface source space is assumed
            
    Returns:
        fname(str): basename for the BEM solution .fif file

    """
    if is_volume:
        mm = int(step*1000)
        return mri_subj + f'-vol-{mm}mm-src.fif'

    token = 'vol' if is_volume else 'ico'
    return mri_subj + f'-ico-{ico_src}-src.fif'

def bem_sol_pathname(mri_subj, data_host, config):
    """
    Construct a full pathname to subject's BEM solution file.

    Args:
        mri_subj(str): subject ID on MRI side
        data_host(DataHost): the DataHost object instance
        config(dict): the `bem_model` step config dictionary

    Returns:
        pathname (Path): full pathname of the BEM solution file

    """
    conductivity = config['conductivity']
    sol_fif = bem_solution_fif(mri_subj, config['ico_bem'], len(conductivity))
    bem_dir = data_host.get_subject_bem_dir(mri_subj)
    return  bem_dir / sol_fif

def construct_bem_solution(mri_subj, data_host, config, verbose = None):
    """
    Make subject's BEM model and BEM solution and save those to the subject's
    BEM folder. If BEM solution already exists, it will only be recreated if
    'recalc_bem' flag is set in the pipeline setup JSON file.

    Args:
        mri_subj(str): subject ID on MRI side
        data_host(DataHost): the DataHost object instance
        config(dict): the `bem_model` step config dictionary
        verbose(str): MNE Python verbose level

    Returns:
        Nothing, but BEM solution file is created in the subject's BEM folder.

    """
    sol_pathname = bem_sol_pathname(mri_subj, data_host, config)

    if not config['recalc_bem'] and sol_pathname.is_file():
        print(f'construct_bem_solution(): BEM solution for {mri_subj} already exists')
        return

    model =  mne.make_bem_model(mri_subj, ico=config['ico_bem'],
                    conductivity=config['conductivity'],
                    subjects_dir=data_host.get_mri_subjects_dir(), verbose=verbose)

    bem_sol = mne.make_bem_solution(model, solver = 'mne', verbose = verbose)
    mne.write_bem_solution(sol_pathname, bem_sol, overwrite=True, verbose=verbose)
    print(f'\nconstruct_bem_solution() for {mri_subj} completed.\n')

def construct_source_space(mri_subj, data_host, config, verbose = None):
    """
    Given that BEM solution is available, construct surface or volume source
    space for the subject and save it in the subject's BEM folder. If source space
    already exists, it will only be recreated if 'recalc_bem' flag is set in
    the pipeline setup JSON file.

    Args:
        mri_subj(str): subject ID on MRI side
        data_host(DataHost): the DataHost object instance
        config(dict): the `bem_model` step config dictionary
        verbose(str): MNE Python verbose level

    Returns:
        Nothing, but source space .fif file is created in the subject's BEM folder.

    """
    is_volume = config['vol_src_space']
    ico_src = config['ico_src']
    step = config['grid_step']
    ss_fif = src_space_fif(mri_subj, step, ico_src, is_volume)
    bem_dir = data_host.get_subject_bem_dir(mri_subj)
    ss_pathname = bem_dir / ss_fif

    if not config['recalc_bem'] and ss_pathname.is_file():
        print(f'construct_source_space(): source space for {mri_subj} already exists')
        return

    subjects_dir=data_host.get_mri_subjects_dir()

    if is_volume:
        src = mne.setup_volume_source_space(
                mri_subj,
                pos=step * 1000,
                bem=bem_sol_pathname(mri_subj, data_host, config),
                subjects_dir=subjects_dir,
                **config['volume_ss'],
                verbose=verbose)
    else:
        src = mne.setup_source_space(
                mri_subj,
                spacing=f'ico{ico_src}',
                subjects_dir=subjects_dir,
                **config['surface_ss'],
                verbose=verbose)

    mne.write_source_spaces(ss_pathname, src, 
            overwrite=True, verbose=verbose)

    print(f'\nconstruct_source_space() for {mri_subj} completed.\n')

