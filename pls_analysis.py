"""
**Perform PLS analysis of epoched data** 

The following tasks are implemented.

'erf_mc_1group4img':
    Mean-centered PLS for a single group, 4 images

'erf_mc_2group4img':
    Mean-centered PLS for 2 groups and 4 images

'erf_mc_pooled4img':
    Mean-centered PLS for subjects from both groups pooled
    together, and 4 images

'erf_mc_2groups1img':
    Mean-centered PLS for 2 groups and 1 image

'erf_mc_4groups1img':
    Mean-centered PLS for 4 sub-groups and 1 image. The subgroups are:
    SCZ-believers, ASD-believers, SCZ-nonbelievers, ASD-nonbelievers

'henv_mc_4groups1img':
    Mean-centered PLS for 4 sub-groups and 1 imagei, using Hilbert
    envelopes rather than ERF responses. The subgroups are:
    SCZ-believers, ASD-believers, SCZ-nonbelievers, ASD-nonbelievers

'henv_std_mc_4groups1img':
    Mean-centered PLS for 4 sub-groups and 1 imagei, using STDs of
    Hilbert envelopes rather than ERF responses. The subgroups are:
    SCZ-believers, ASD-believers, SCZ-nonbelievers, ASD-nonbelievers

'henv_std_sm_4groups1img':
    Fitting stat model for 2 groups, 2 religiosity categories for 
    1 image, using STDs of Hilbert envelopes. 

'henv_std_sm_4groups2img':
    Fitting stat model for 2 groups, 2 religiosity categories for 
    a pair of images, using STDs of Hilbert envelopes. 

'henv_std_sm_santa1img':
    Fitting stat model for 2 groups and santa-clara religiosity scores for 
    1 image, using STDs of Hilbert envelopes. 

'henv_std_sm_santa2img':
    Fitting stat model for 2 groups and santa-clara religiosity scores for 
    a pair of images, using STDs of Hilbert envelopes. 

'erf_contrast_2group2img':
    Contrast (non-rotated) PLS for 2 groups and a pair
    of images

'erf_contrast_2group4img':
    Contrast (non-rotated) PLS for 2 groups and 4 images

'henv_std_contrast_4groups2img':
    Contrast (non-rotated) PLS for 4 groups and 2 images for
    STDs of Hilbert envelopes

'compare_corrs_2groups1img':
    Compare distributions of correlations between behavioral variable(s) and
    MEG features for two groups of subjects, for a given image

NOTE that this code uses matlab PLS package for most of the work.
One needs matlab to be installed in the system for this code to run,
as well as matlab PLS analysis source files.
"""
import pandas as pd
import numpy as np
import h5py        # Needed to save/load files in .hdf5 format
import pickle
import mne
from pathlib import Path
import matplotlib.pyplot as plt
import setup_utils as su
from src_rec import read_roi_time_courses
from run_pls import run_pls
from heatmap import plot_channel_heatmap
from plot_waveforms import adjust_signs
from compare_corr_distributions import compare_corr_distributions 
from stat_model_fits import stat_model_fits 

def pls_analysis(ss):
    """
    Run PLS using matlab PLS package.

    Args:
        ss(obj): reference to this app object

    """
    STEP = 'pls_analysis'
    config = ss.args[STEP]

    if ss.data_host.cluster_job:
        config['show_plots'] = False

    # For PLS, different array jobs mean different parameters while
    # using the full set (i.e. all) subjects at once for the analysis. This requires
    # N_ARRAY_JOBS to be set to 1 for all subject set selection operations,
    # otherwise the subjects will be split between the jobs. To resolve this
    # contradiction, we first load required PLS params to this step config based
    # on the job number, and then reset N_ARRAY_JOBS, ijob to 1 and 0,
    # respectively, to avoid subjects splitting between jobs.
    if ss.args['N_ARRAY_JOBS'] > 1:
        set_config_for_job(config, ss.ijob)
        pls_args_mat = f'pls_args{ss.ijob}.mat'
        ss.args['N_ARRAY_JOBS'] = 1
        ss.ijob = 0
    else:
        pls_args_mat = 'pls_args.mat'

    return_precalculated_result = config['return_precalculated_result']

    # Here 2nd arg to get_step_out_file() can be any valid path
    # because it is not used
    # This is a basename (string) of the .mat file with PLS results
    # for matlab to produce
    out_file = ss.data_host.get_step_out_file(STEP, config['in_dir'])
    res_mat = str(ss.data_host.get_step_out_dir(STEP) / out_file)

    if return_precalculated_result:
        if 'compare_corrs' not in config['task']:
            if '_sm_' not in config['task']:
                res = run_pls(None, None, None, None, None, 
                    res_mat = res_mat, return_precalculated_result = True)
                print(f'Loaded PLS analysis results from {res_mat}\n')
            else:
                sm_pkl_file = get_sm_results_file(ss)
                with open(sm_pkl_file, "rb") as f:
                    sm_fit_results = pickle.load(f)
                print(f'Loaded stat model fit results from {sm_pkl_file}\n')
        else:
            out_hdf5 = get_pls_results_hdf5_file(ss)
            res = read_corr_dist_results(out_hdf5)
            print(f'Loaded correlation distribution comparison analysis results from {out_hdf5}\n')
        # end of loading precalculated results branch
    else:
        # Run requested analyses
        # Get subjects groups info
        subjects_info_csv = ss.data_host.get_subjects_info_csv()
        in_dir = ss.data_host.get_step_in_dir(STEP)
        subjects = su.make_subject_dict(in_dir, slist = ss.args['subjects'])

        # Get subj2group(dict): mapping subject ID -> group #
        lstSID = list(subjects.keys())
        subj2group = get_subj_groups(config, subjects_info_csv, lstSID)

        path_to_matlab_pls = ss.data_host.path_to_matlab_pls
        options = config['pls_options_mc']
        args_mat = str(ss.data_host.get_step_out_dir(STEP) / pls_args_mat)

        if config['task'] in ('erf_mc_2groups1img','erf_mc_4groups1img', \
                'henv_mc_4groups1img','henv_std_mc_4groups1img'):
            # lst_dmat = [data0, data1]; dataI = nsubjPerGroup x nfeatures, for a single
            # event ID specified in config
            lst_dmat, lst_sid = collect_groups_for_event(ss, STEP, subj2group)

            if not all([d.shape[0] for d in lst_dmat]):
                raise ValueError('One of the groups is empty; please add more subjects')

            lst_nsubj = [da.shape[0] for da in lst_dmat]
            ncond = 1       # PLS for n groups, 1 condition
        
            print(f'PLS for {len(lst_dmat)} groups of {lst_nsubj} subjects, with {lst_dmat[0].shape[1]}-dimensional feature space.')

            res = run_pls(lst_dmat, lst_nsubj, ncond, options, path_to_matlab_pls, args_mat = args_mat,
                res_mat = res_mat, return_precalculated_result = return_precalculated_result)
        elif config['task'] == 'erf_mc_1group4img':
            # lst_dmat: shape `(nsubj*ncond, nfeatures)` - dmat for 1 group,
            lst_dmat, nsubj = prepare_1grpNimgs(ss, STEP, subj2group)
            lst_nsubj = [nsubj]
            ncond = len(config['img_events'])
            res = run_pls(lst_dmat, lst_nsubj, ncond, options, path_to_matlab_pls, args_mat = args_mat,
                res_mat = res_mat, return_precalculated_result = return_precalculated_result)
        elif config['task'] == 'erf_mc_2group4img' or \
                config['task'] == 'erf_mc_pooled4img':
            lst_dmat = []
            lst_nsubj = []
            for gID in (0,1):
                config['group_id'] = gID
                # dmat for gID has shape (nsubj*ncond, nfeatures); nsubj is a number
                # of subjects included (those that have all requested images)
                dmat, nsubj = prepare_1grpNimgs(ss, STEP, subj2group)
                lst_dmat.append(dmat)
                lst_nsubj.append(nsubj)

            ncond = len(config['img_events'])

            if config['task'] == 'erf_mc_pooled4img':
                lst_dmat, lst_nsubj = pool_groups(lst_dmat, lst_nsubj, ncond)

            res = run_pls(lst_dmat, lst_nsubj, ncond, options, path_to_matlab_pls, args_mat = args_mat,
                res_mat = res_mat, return_precalculated_result = return_precalculated_result)
        elif (config['task'] == 'erf_contrast_2group2img') or \
                (config['task'] == 'erf_contrast_2group4img') or \
                (config['task'] == 'henv_std_contrast_4groups2img'):
            lst_dmat = []
            lst_nsubj = []

            if '2group' in config['task']:
                ngroups = 2
            elif '4group' in config['task']:
                ngroups = 4
            # By design, we'll fail with ngroups undefined if none of the above
            # tests are true

            for gID in range(ngroups):
                config['group_id'] = gID
                # dmat: shape `(nsubj*ncond, nfeatures)` - dmat for 1 group,
                dmat, nsubj = prepare_1grpNimgs(ss, STEP, subj2group)
                lst_dmat.append(dmat)
                lst_nsubj.append(nsubj)

            ncond = len(config['img_events'])

            options = config['pls_options_contrast']
            options['stacked_designdata'] = np.array(config['contrasts'])[:,np.newaxis]

            res = run_pls(lst_dmat, lst_nsubj, ncond, options, path_to_matlab_pls, args_mat = args_mat,
                res_mat = res_mat, return_precalculated_result = return_precalculated_result)
        elif config['task'] == 'compare_corrs_2groups1img':
            # ccd_res is a named tuple with fields:'effect_sizes', 'hedges_g', 
            # 'CI_lower', 'CI_upper', 'bootstrap_means_A', 'bootstrap_means_B'

            """
            # QQQQQ---------------------------------------------
            subj2group['5TM3XZ'] = 0  # Just for testing
            # QQQQQ---------------------------------------------
            """

            ccd_res = compare_corr_distributions(ss, lstSID, subj2group)
            out_hdf5 = get_pls_results_hdf5_file(ss)
            res = write_corr_dist_results(out_hdf5, config['task'], config['event_id'], ccd_res)
        elif config['task'] in ('henv_std_sm_4groups1img','henv_std_sm_4groups2img',
                                'henv_std_sm_santa1img','henv_std_sm_santa2img'):
            sm_fit_results = stat_model_fits(ss)
        else:
            raise ValueError(f'Unrecognized PLS task {config["task"]}')

        print('All analyses completed\n')

    # -------------------------------------------------------
    # Printouts and heatmaps
    # -------------------------------------------------------
    if 'compare_corrs' not in config['task']:
        if '_sm_' not in config['task']:
            display_pls_results(ss, res)
        else:
            display_sm_fit_results(ss, sm_fit_results)
    else:
        display_ccd_results(ss, res)

def set_config_for_job(cfg, ijob):
    """
    When running an array job, set this step's configuration parameters
    in accordance with the job number. The `N_ARRAY_JOBS` in config should
    be equal to the number of elements in the 'array_job_parms' list, 
    where each element is a dictionary {key_i:value_i} where keys and
    values define this step's parameters to be set.

    This function is never called for `N_ARRAY_JOBS = 1`.

    Args:
        cfg(dict): this step configuration dictionary
        ijob(int): 0-based array job index

    Returns:
        None

    """
    parms_dict = cfg['array_job_parms'][ijob]

    for key in parms_dict:
        cfg[key] = parms_dict[key]


def display_pls_results(ss, res):        
    """
    Print out short summary of PLS results, and display heat maps on a
    local computer system.

    Args:
        ss(obj): reference to this app object
        res(dict): a dictionary generated by reading .mat file created
            by matlab PLS routine; see run_pls.py for fields description

    Returns:
        None

    """
    # Choose proper indexing: sometimes scalars are returned, not matrices
    smart_get = lambda x: x[:,0] if len(x.shape) else x

    np.set_printoptions(formatter={'float': lambda x: f'{x: .2f}'})
    print(f'Singular values: {smart_get(res["s"])}')                  # Returned is nlv x 1 matrix
    np.set_printoptions(formatter={'float': lambda x: f'{x: .2e}'})
    pvals = smart_get(res["perm_result"]["sprob"])
    print(f'P-values: {pvals}\n')    # sprob.shape = nlv x 1

    contrasts = res['v']                            # shape ngroups x ngroups or ncond x ncond (for 1 group)
                                                    # In general case the number of columns = nlv
    zscores = res['boot_result']['compare_u']       # shape nfeatures x nlv

    # Flip signs if necessary for the design salience to be positive for group or
    # condition 0. Explanation:
    # An SVD A = USV' can be also written as A = UDDSV'=(UD)S(VD)' 
    # where D is a diagonal matrix with 1s or -1s on diagonal, so that DD = I
    # i-th diagonal element of D = -1, then i-th column of matrix to the left has
    # it's sign flipped. Thus we can arbitrarily flip sign of a contrast column
    # provided that corresponding feature column is flipped also.
    for lv in range(contrasts.shape[1]):
        if contrasts[0,lv] < 0:
            contrasts[:,lv] = -contrasts[:,lv]
            zscores[:,lv] = -zscores[:,lv]

    # Space for positive numbers, 2 decimal places
    np.set_printoptions(formatter={'float': lambda x: f'{x: .2f}'})

    print(f'Contrasts (Y-matrix):\n{contrasts}')

    # Z-scores are nfeatures x nlv
    print(f'Z-scores:\nmin = {np.min(zscores, axis = 0)}\nmax = {np.max(zscores,axis = 0)}\n')

    if ss.data_host.host == 'cedar':
        return

    # Plot zcores colormap
    STEP = 'pls_analysis'
    config = ss.args[STEP]
    hm = config['heatmap']
    bp = config['barplot']
    ss.args['subjects'] = [hm['subject4labels']]
    files = su.files_to_process(ss, STEP)
    label_names = get_label_names(config, files)

    # Construct mapping from stored channels order to that
    # specified in our custom ordering file
    ordering_csv = ss.data_host.get_ordering_csv()
    ordering = construct_ordering(ordering_csv, label_names)
                                        
    t0,t1 = config['pls_interval']
    lv = hm['latent_var']
    SR = hm['SR']

    if SR is None:
        SR = ss.args['src_erf']['target_sample_rate']

    ntimes = int((t1-t0)*SR + 1)

    if hm['t_range'] is not None:
        t0 = max(t0, hm['t_range'][0])
        t1 = min(t1, hm['t_range'][1])

    nlabels = int(zscores.shape[0]/ntimes)
    zscores_lv = np.reshape(zscores[:,lv],(nlabels, ntimes))
    
    plot_path = ss.data_host.root / ss.data_host.meg / ss.data_host.config["out_root"] / \
                    ss.data_host.pipeline_version / hm['out_dir']

    title, pngname = construct_titles(ss)

    fig = plt.figure(figsize = hm['figsize'])
    #ax = fig.add_axes([0.125, 0.11, 0.9, 0.78])    # This is the original size of full HM plot

    # Plot contrasts bar plot
    ax_bp = fig.add_axes(bp['bbox'])
    bars = contrasts[:,lv]
    ax_bp.set_xlim(-1, len(bars))
    ax_bp.set_ylim(-1, 1)
    ax_bp.set_xticks(np.arange(len(bars)))
    ax_bp.set_xticklabels(get_bar_labels(config))
    ax_bp.set_title(f'Latent var: {lv}, ' + \
            f'P-value: {pvals[lv] if len(pvals.shape) else pvals :.1e}')
    ax_bp.grid(axis='y', linestyle='--', alpha=0.7, color='gray')
    positions = np.arange(len(bars))
    colors = ['red' if c > 0 else 'blue' for c in bars]
    ax_bp.vlines(positions, 0, bars, colors=colors, lw=12)
    ax_bp.axhline(0, color='black', linewidth=0.8)

    # Add heatmap medians plot
    ax_med = fig.add_axes(config['medians']['bbox'])
    ax_med = plot_zscores_medians(zscores_lv, ax_med, t0, SR)

    ax_hm = fig.add_axes(hm['bbox'])

    # Plot heatmap using reordered channels
    ax_hm =plot_channel_heatmap(zscores_lv[ordering,:], label_names[ordering],
                    ax = ax_hm, t0 = t0, SR = SR,
                    color_min=hm['color_min'], 
                    color_max=hm['color_max'], 
                    xlabel=hm['xlabel'],
                    ylabel=hm['ylabel'],
                    cbar_label=hm['cbar_label'],
                    cmap=hm['cmap'],
                    title=title,
                    save_file = pngname,
                    figsize = hm['figsize'],
                    fontsize_y = hm['fontsize_y'],
                    dpi = hm['dpi'],
                    show = config['show_plots'])

    # zscores are nfeatures x nlv = (nlabels*ntimes) x nlv
    # reshape them to nlv x nlabels x ntimes
    nlv = zscores.shape[1]
    zscores_3d = np.reshape(zscores.T,(nlv,nlabels,ntimes))

    out_hdf5 = get_pls_results_hdf5_file(ss) 
    write_pls_results(out_hdf5, config['task'], contrasts,
                smart_get(res['s']),                      # singular values
                smart_get(res['perm_result']['sprob']),   # p-values
                zscores_3d,
                label_names, SR,
                perms_boots = [res['perm_result']['num_perm'],res['boot_result']['num_boot']])

def display_ccd_results(ss, res):        
    """
    Print out short summary of CCD results, and display heat maps on a
    local computer system.

    Args:
        ss(obj): reference to this app object
        res(dict): a dictionary generated by reading .hdf5 file with CCD
                results.

    Returns:
        None

    The `res` dictionary contains the following fields:

        **task** (string): the corr dists task name

        **eID** (int): the event (image) ID

        **ccd_res** (CCD_Result): a named tuple with fields 'effect_sizes', \
            'hedges_g', 'CI_lower', 'CI_upper', 'bootstrap_means_A', 'bootstrap_means_B'. \
            Each field contains a vector with length `nfeatures = nlabels x ntimes` 

    """
    print(f'Read dictionary with CCD results for task \'{res["task"]}\', image ID = {res["eID"]}.')
    print(f'Effect size shape is {res["ccd_res"].effect_sizes.shape}')

    if ss.data_host.host == 'cedar':
        print('All done for now')
        return

    # Display heatmaps - lots of dirty copy/pastes from the PLS case

    # This is just for getting label names from one subject
    STEP = 'pls_analysis'
    config = ss.args[STEP]
    hm = config['heatmap']
    ss.args['subjects'] = [hm['subject4labels']]    # !!! Now subject list has a single subject
    files = su.files_to_process(ss, STEP)
    label_names = get_label_names(config, files)

    # Construct mapping from stored channels order to that
    # specified in our custom ordering file
    ordering_csv = ss.data_host.get_ordering_csv()
    ordering = construct_ordering(ordering_csv, label_names)
                                        
    t0,t1 = config['pls_interval']
    SR = hm['SR']

    if SR is None:
        SR = ss.args['src_erf']['target_sample_rate']

    ntimes = int((t1-t0)*SR + 1)

    if hm['t_range'] is not None:
        t0 = max(t0, hm['t_range'][0])
        t1 = min(t1, hm['t_range'][1])

    ccd_res = res['ccd_res']

    nlabels = int(len(ccd_res.effect_sizes)/ntimes)

    plot_path = ss.data_host.root / ss.data_host.meg / ss.data_host.config["out_root"] / \
                    ss.data_host.pipeline_version / hm['out_dir']

    # Prepare masks. MASKS SELECT POINTS TO BE INCLUDED
    pp_mask = np.logical_and(ccd_res.bootstrap_means_A >= 0,  ccd_res.bootstrap_means_B >= 0)
    pn_mask = np.logical_and(ccd_res.bootstrap_means_A >= 0,  ccd_res.bootstrap_means_B < 0)
    np_mask = np.logical_and(ccd_res.bootstrap_means_A < 0,  ccd_res.bootstrap_means_B >= 0)
    nn_mask = np.logical_and(ccd_res.bootstrap_means_A < 0,  ccd_res.bootstrap_means_B < 0)

    masks = pp_mask,pn_mask,np_mask,nn_mask
    mask_names = config['compare_corrs']['hm_subplots_titles']

    # Mask insignificant points, if asked
    ci_on = config['compare_corrs']['mask_insignif_CI']     # Flag to take CIs into accoutn

    if ci_on:
        # Mask for significant CIs
        ci_mask = np.logical_not((ccd_res.CI_lower <= 0) & (0 <= ccd_res.CI_upper))
        tmp = [np.logical_and(mm,ci_mask) for mm in masks]
        masks = tmp

    what = config['compare_corrs']['hm_what']

    # Set thresholds
    data = getattr(ccd_res, what)
    hm_alpha = config['compare_corrs']['hm_alpha']
    th_min, th_max = np.quantile(data, [hm_alpha, 1 - hm_alpha])
    th_mask = (data <= th_min) | (data >= th_max)
    tmp = [np.logical_and(mm,th_mask) for mm in masks]
    masks = tmp

    # Configure figure title and output png file name
    what_sfx = 'es' if what == 'effect_sizes' else 'hg'
    ci_sfx = 'on' if ci_on else 'off' 
    th_sfx = f'th_{hm_alpha:.2f}' if hm_alpha < 1 else 'th_off'
    sup_title, pngname = construct_titles(ss)
    pngname = Path(str(pngname).replace('.png',f'_{what_sfx}_CI_{ci_sfx}_{th_sfx}.png'))

    sup_title = f'{sup_title}; {what}; CI\'s {ci_sfx}; {th_sfx}'

    nplots = len(masks)
    
    fig, axes = plt.subplots(1,nplots, figsize=hm['figsize_ccd'])
    fig.suptitle(sup_title, fontsize=16, fontweight='bold')     # TODO: set in JSON

    for iplot in range(nplots):
        data = getattr(ccd_res, what).copy()
        data[np.logical_not(masks[iplot])] = 0.
        data = np.reshape(data,(nlabels, ntimes))

        if iplot == nplots - 1:
            cbar = True
            show = config['show_plots']
            save_file = pngname
        else:
            cbar = False
            show = False
            save_file = None


        # Plot heatmap using reordered channels
        plot_channel_heatmap(data[ordering,:], label_names[ordering],
                        ax = axes[iplot], t0 = t0, SR = SR,
                        color_min=hm['color_min'], 
                        color_max=hm['color_max'], 
                        xlabel=hm['xlabel'],
                        ylabel=None if iplot else hm['ylabel'],
                        cbar = cbar,
                        cbar_label=what.replace('_',' '),
                        cmap=hm['cmap'],
                        title=mask_names[iplot],
                        save_file = save_file,
                        figsize = hm['figsize'],
                        fontsize_y = hm['fontsize_y'],
                        dpi = hm['dpi'],
                        show = show)

def display_sm_fit_results(ss, res):        
    """
    Print out a summary of stat model fit results, and display plots on a
    local computer system.

    Args:
        ss(obj): reference to this app object
        res(dict): dictionary roi -> <fit results object>

    Returns:
        None

    """
    STEP = 'pls_analysis'
    cfg = ss.args[STEP]
    task = cfg['task']
    band = cfg['band']
    img = cfg['event_id']

    fig_title = f'Task: \'{task}\', {band} Hz, '

    if '1img' in task:
        fig_title += f'img = {cfg["img_types"][str(img)]}'
    elif '2img' in task:
        img_names = [cfg["img_types"][str(img)] for img in cfg['img_events']]
        fig_title += f'{img_names[1]} - {img_names[0]}'
    else:
        raise ValueError(f'Task \'{task}\' not recognized or not implemented')

    signif_rois = []

    for roi in res.keys():
        # -------------------------------------------------------------
        # Fit results object properties of interest:
        #   .params[<name>], where for categorical parameters names are:
        #       'C(gender)[T.M]', 'C(believer)[T.1]', 'C(group)[T.1]',
        #       'C(believer)[T.1]:C(group)[T.1]' - returns fitted value
        #       of the parameter
        #   .bse[<name>] - standard error of the parameter
        #   .f_pvalue - P-value for the F statistics
        #   .pvalues[<name>] - 2-tailed P-values for the t-statistics for
        #       fitted parms
        # -------------------------------------------------------------
        # print(res.summary())
        if res[roi].f_pvalue <=0.05:
            signif_rois.append(roi)

    if signif_rois:
        print(f'Stat-significant ROIs for {band} Hz band, image {img}:')

        for roi in signif_rois:
            print(f'ROI: {roi}')
            print(res[roi].summary())
            print('')
    else:
        print('No stat-significant ROIs found')

    # Prepare effect size data
    params_to_plot, eff_sizes, pvalues = get_eff_sizes_and_pvals(task, res)
    print_param_stats(params_to_plot, eff_sizes, pvalues)

    # Everything following is only used in plotting on local machine. Exit here if
    # running on the cluster
    if ss.data_host.cluster_job:
        return

    # Reorder results in accordance with specified ROI list
    ordering_csv = ss.data_host.get_ordering_csv()
    # ordering is a ndarray label indices yielding mapping current idx -> new idx
    ordering = construct_ordering(ordering_csv, res.keys())

    # Construct PNG file pathname
    pkl_file = ss.data_host.get_step_out_file(STEP, cfg['in_dir'])  # This is .pkl file name we are working with

    # Save plot where the heatmaps are saved
    plot_path = ss.data_host.root / ss.data_host.meg / ss.data_host.config["out_root"] / \
                    ss.data_host.pipeline_version / cfg['heatmap']['out_dir']

    pngname = plot_path / (Path(pkl_file).stem + '.png')

    # Convert eff_sizes, pvalues to arrays with shape (nparams,nlabels), reorder
    # and plot
    roi_array = np.array(list(res.keys()))
    plot_eff_sizes(roi_array[ordering], eff_sizes[:,ordering],
            pvalues[:,ordering], params_to_plot, fig_title,
            show = cfg['show_plots'], pngname = pngname)

def get_eff_sizes_and_pvals(task, res):
    """
    Return arrays of effect sizes and ROI p-values for
    parameters corresponding to group, religiosity and their
    interaction.

    Args:
        task(str): one of SM-fitting tasks
        res(dict): fitting results in the form {roi: <OLS-results object>}

    Returns:
        pnames(tuple of str): OLS model parameter names corresponding to group,
            religiosity and interaction
        eff_sizes(ndarray): shape(nparms, nrois) effect sizes for each ROI
        pvalues(ndarray): shape(nparms, nrois) t-test p-values for each ROI

    """
    if task in ('henv_std_sm_4groups1img','henv_std_sm_4groups2img'):
        pnames = 'C(group)[T.1]','C(believer)[T.1]','C(believer)[T.1]:C(group)[T.1]'
    elif task in ('henv_std_sm_santa1img','henv_std_sm_santa2img'):
        pnames = 'C(group)[T.1]', 'sc', 'sc:C(group)[T.1]'
    else:
        raise ValueError(f'Unrecognized task: {task})')

    eff_sizes = []
    pvalues = []
        
    for param in pnames:
        es = []
        pv = []

        # Collect effect sizes for param over ROIs
        for roi in res.keys():
            beta = res[roi].params[param]
            dev = res[roi].bse[param]
            es.append(beta/dev)
            pv.append(res[roi].pvalues[param])

        eff_sizes.append(es)
        pvalues.append(pv)

    eff_sizes = np.array(eff_sizes)
    pvalues = np.array(pvalues)

    return pnames, eff_sizes, pvalues

def print_param_stats(params_to_plot, eff_sizes, pvalues, alpha = 0.05):
    """
    Print out general stats for fitted parameters values distribution
    over ROIs.

    Args:
        params_to_plot(list of str): names of fitted parameters, len = nparm
        eff_sizes(ndarray): shape (nparm, nrois) - effect sizes for each fitted
            parameter
        pvalues(ndarray): shape (nparm, nrois) - t-test 2-sided p-values for each
            parameter per ROI
        alpha(float): significance threshold

    Returns:
        None

    """
    means = np.mean(eff_sizes, axis = 1)
    stds = np.std(eff_sizes, axis = 1)
    ratios = means / stds
    nsig = np.sum(pvalues < alpha, axis = 1)
    df = pd.DataFrame({'Beta':params_to_plot, 'Mean':means,'STD':stds,
                       'Mean/STD':ratios, 'n_signif':nsig})
    print(df.to_string(index=False, float_format="{:.2f}".format))
    print('')
    return
    
def plot_eff_sizes(labels,data,pvalues,subplot_titles, super_title = None,
            show = True, pngname = None, ylabel = 'eff_size', dpi = 300, alpha = 0.05):
    """
    Create bar plots of effect sizes for labels (ROIs); mark results with
    significant p-values with red color.

    Args:
        labels(list of str): ROI (label) names
        data(ndarray): shape(nplot_params,nlabels) effect sizes distributions
            over ROIs for each OLS model parameter
        pvalues(ndarray): shape (nlabels,) p-values returned by OLS fit
        subplot_titles(list of str): titles for each bar plot
        super_title(str or None): figure title
        show(bool): flag to show interactive plot
        pngname(pathlike): if not None, full path name of PNG file to save the plot
        ylabel(str): Y axis label for subplots
        dpi(int): PNG plot resolution
        alphs(float): significance level (p-value should be less than alpha to be
            significant)

    Returns:
        None

    """
    ncurves = data.shape[0]
    nlabels = len(labels)

    # Create the figure and subplots
    fig, axes = plt.subplots(ncurves, 1, figsize=(16, 9), sharex=True)

    for i, ax in enumerate(axes):
        colors = ['red' if pvalues[i, j] < alpha else 'blue' for j in range(nlabels)]
        ax.bar(range(nlabels), data[i],color=colors)  # Bar plot for each row of the array
        ax.set_title(subplot_titles[i])
        ax.set_ylabel(ylabel) 

        if i < ncurves - 1:
            ax.set_xticklabels([])  # Remove tick labels but keep ticks
        else:
            ax.set_xticks(range(nlabels))
            ax.set_xticklabels(labels, rotation=90, fontsize=8)  # Set labels for the lowest plot

    fig.suptitle(super_title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    if pngname is not None:
        plt.savefig(pngname, dpi=dpi)

    if show:
        plt.show()

def get_pls_results_hdf5_file(ss):
    """
    Return full pathname of the .hdf5 file to save PLS results

    Args:
        ss(object): a reference to this app object

    Returns:
        out_hdf5(Path): full pathname of the output .hdf5 file

    """
    STEP = 'pls_analysis'
    config = ss.args[STEP]
    out_file = ss.data_host.get_step_out_file(STEP, config['in_dir'])

    if 'compare_corrs' not in config['task']:
        out_hdf5 = out_file.replace('.mat','.hdf5') 
    else:
        out_hdf5 = out_file

    return ss.data_host.get_step_out_dir(STEP) / out_hdf5

def get_sm_results_file(ss):
    """
    Return full pathname of the .pkl file for saving/loading
    stat model fit results.

    Args:
        ss(object): a reference to this app object

    Returns:
        out_file(Path): full pathname of the output file

    """
    STEP = 'pls_analysis'
    config = ss.args[STEP]
    out_file = ss.data_host.get_step_out_file(STEP, config['in_dir'])

    return ss.data_host.get_step_out_dir(STEP) / out_file

def write_pls_results(out_hdf5, task, contrasts, singular_values, p_values, z_scores,
                label_names, SR, perms_boots):
    """
    Save PLS analysis results to .hdf5 file.

    Args:
        out_hdf5 (Path | string): full pathname to the output .hdf5 file
        task( string): PLS task name
        contrasts (ndarray): `shape(nlv,nlv)` predefined or calculated contrasts, as columns
            of the contrasts matrix
        singular_values (ndarray): `shape(nlv,)` a list of PLS task singular values (1 per
            a latent variable)
        p_values (ndarray): `shape(nlv,)` a list of p-values (1 per a latent variable)
        z_scores (ndarray): `shape(nlv,nlabels,ntimes)` Z-scores for each (lv,ROI,time_point)
        label_names (list of str): ROI names
        SR (float): sampling rate along time axis, Hz
        perms_boots (list of int): `[nperms, nboots]` numbers of permutations and boot resamples used

    Returns:
        None

    """
    with h5py.File(out_hdf5, 'w') as f:
        f.create_dataset('task', data=task.encode('utf-8')) # read it as f['task'][()].decode('utf-8')
        f.create_dataset('contrasts', data=contrasts)       # read it as f['contrasts'][:]
        f.create_dataset('singular_values', data=singular_values)
        f.create_dataset('p_values', data=p_values)
        f.create_dataset('z_scores', data=z_scores)
        f.create_dataset('label_names', data=label_names)
        f.create_dataset('SR', data=SR)                     # retrieve it later with f['SR'][()]
        f.create_dataset('perms_boots', data=perms_boots)

def read_pls_results(results_hdf5):
    """
    Read PLS results from .hdf5 file created with `write_pls_results()`.

    Args:
        results_hdf5(Path | str): full pathname of the .hdf5 file

    Returns:
        res(dict): a dictionary with the PLS results data

    The `res` dictionary contains the following fields:
        **task** (string): PLS task name

        **contrasts** (ndarray): `shape(nlv,nlv)` predefined or calculated contrasts, as columns \
            of the contrasts matrix

        **singular_values** (ndarray): `shape(nlv,)` a list of PLS task singular values (1 per \
            a latent variable)

        **p_values** (ndarray): `shape(nlv,)` a list of p-values (1 per a latent variable)

        **z_scores** (ndarray): `shape(nlv,nlabels,ntimes)` Z-scores for each (lv,ROI,time_point)

        **label_names** (list of str): ROI names

        **SR** (float): sampling rate along time axis, Hz

        **perms_boots** (list of int): `[nperms, nboots]` numbers of permutations and boot resamples used

    """
    res = {}

    with h5py.File(results_hdf5, 'r') as f:
        res['task'] = f['task'][()].decode('utf-8') # read it as f['task'][()].decode('utf-8')
        res['contrasts'] = f['contrasts'][:]        # read it as f['contrasts'][:]
        res['singular_values'] = f['singular_values'][:]
        res['p_values'] = f['p_values'][:]
        res['z_scores'] = f['z_scores'][:]          # still using [:] irrespective to the number of dimensions
        res['label_names'] = f['label_names'][:].astype(str)    # because names were stored as byte strings
        res['SR'] = f['SR'][()]                     # retrieve it later with f['SR'][()]
        res['perms_boots'] = f['perms_boots'][:]

    return res

def prepare_1grpNimgs(ss, step, subj2group):
    """
    Prepare `lst_dmat` for mean-centered 1 group, `ncond` conditions (images) PLS.
    In this case, `lst_dmat` is not a list, but just a single array of stacked
    dmats with the same subjects, for each condition (image).
    Conditions are listed in the `'img_events'` key in the JSON file.

    Args:
        ss (object): reference to this app object
        step (str): this step's name
        subj2group(dict): mapping subject ID -> group # 

    Returns:
        lst_dmat(ndarray): shape `(nsubj*ncond, nfeatures)` - dmat for 1 group,
            `ncond` conditions PLS
        nsubj(int): number of subjects in a group specified by `config['group_id']`

    """
    lst = []        # A list of dmat for each eID to stack
    eid2sid = []    # A list of lists of subject IDs for each event ID
    config = ss.args[step]
    img_events = config['img_events']
    eIDorg = config['event_id']     # Save the original value, as we'll change it
    gid = config['group_id']

    for eID in img_events:
        config['event_id'] = eID
        lst_dmat, lst_sid = collect_groups_for_event(ss, step, subj2group)
        lst.append(lst_dmat[gid])
        eid2sid.append(lst_sid[gid])

    config['event_id'] = eIDorg

    # Now lst is a list of dmats for requested group - one dmat per eID
    # eid2sid is a list of lists of subject IDs for each image ID in img_events    
    # Ensure that we have the same subjects set for each condition
    bRaiseError = False
    for i,eID in enumerate(img_events):
        for i1, e1 in enumerate(img_events[i+1:]):
            missing = find_missing_elements(eid2sid[i], eid2sid[i1])

            if missing['missing_in_list1']:
                bRaiseError = True
                print(f'Subjects {missing["missing_in_list1"]} are included for condition {e1} but missing for condition {eID}')

            if missing['missing_in_list2']:
                bRaiseError = True
                print(f'Subjects {missing["missing_in_list2"]} are included for condition {eID} but missing for condition {e1}')

    if bRaiseError:
        raise ValueError('Lists of subjects for each condition differ. Exiting...}')

    # Stack the dmats and return
    nsubj = lst[0].shape[0]
    lst_dmat = np.vstack(lst)
    return lst_dmat, nsubj
            
def include_in_pls(cfg, in_file):
    """
    Check if an input file should be used for PLS analysis. Note that 
    event ID should be set properly in the `cfg` dictionary for the
    correct file to be selected.

    Args:
        cfg(dict): this step configuration dictionary
        in_file(Path): input file pathname

    Returns:
        include(bool): the include flag

    """
    include = False
    tasks = ('erf_mc_2groups1img','erf_mc_4groups1img', 'erf_mc_1group4img',
             'erf_mc_2group4img','erf_mc_pooled4img','erf_contrast_2group2img',
             'erf_contrast_2group4img','compare_corrs_2groups1img',
             'henv_mc_4groups1img','henv_std_mc_4groups1img','henv_std_contrast_4groups2img',
             'henv_std_sm_4groups1img','henv_std_sm_4groups2img','henv_std_sm_santa1img',
             'henv_std_sm_santa2img')

    if cfg['task'] in tasks:
        if not ('henv' in cfg['task']):
            # one of ERF tasks
            if f'erf_{cfg["event_id"]}' in str(in_file):
                include = True
        else:
            # one of HENV tasks
            fmin, fmax = cfg['band']
            str_band = f'henv_{fmin:.1f}-{fmax:.1f}Hz'
            eID = cfg['event_id']

            if all([ptrn in str(in_file) for ptrn in (str_band,f'Hz_{eID}')]):
                include = True
    else:
        raise ValueError(f'Unrecognized PLS task. Valid tasks are: {tasks}')

    return include

def get_subj_groups(cfg, subjects_info_csv, lstSID):
    """
    For a list of subject IDs, construct a dictionary
    subjID -> group #.

    Args:
        cfg(dict): this step configuration dictionary
        subjects_info_csv (Pathlike): pathname of .csv file with subjects
            behavioral test results
        lstSID(list of str): list of subject IDs

    Returns:
        subj2group(dict): mapping subject ID -> group #

    """
    id_col = cfg['sID_colname']
    bias_col = cfg['bias_colname']
    believer_col = cfg['bin_belief_colname']

    df = pd.read_csv(subjects_info_csv, usecols=[id_col, bias_col, believer_col],
                dtype = {id_col:'string',bias_col:'float64',believer_col:'int64'})
    df = df[df[id_col].isin(lstSID)]

    if len(df) != len(lstSID):
        print(f'len(df)={len(df)}, len(lstSID)={len(lstSID)}')
        raise ValueError(f'lstSID contains subject IDs not listed in {subjects_info_csv}')

    # Replace bias column values with group #
    df[bias_col] = df[bias_col].apply(lambda x: 0 if x < 0 else 1)

    # Create a dictionary sID -> group #
    if '4group' not in cfg['task']:
        # Create 2 groups: SCZ (group 0) and ASD (group 1)
        subj2group = dict(zip(df[id_col], df[bias_col]))
    else:
        # Create 4 subgroups: SCZ (group 0) and ASD (group 1)
        # SCZ-believers, ASD-believers, SCZ-nonbelievers, ASD-nonbelievers
        str2tuple = lambda s: tuple(map(int, s.strip("()").split(",")))
        groups_def = cfg['bias_belief_groups']
        # groups_mapping is in the form {(m,n): <subgroup #>}
        group_mapping = {str2tuple(key):groups_def[key] for key in groups_def}
        subj2group = dict(zip(df[id_col], df[[bias_col, believer_col]].apply(tuple, axis=1).map(group_mapping)))

    return subj2group

def get_label_names(config, files):
    """
    Get a list of the channel (label) names 

    Args:
        config(dict): this step configuration dictionary
        files(generator): a generator returning file pathnames
            for subjects source (evoked, ERF) time courses.

    Returns:
        label_names(list of str): as is

    """
    # Grab the first stc file of those processed, get labels
    # from there and return immediately
    for in_file, out_file in files:
        # Choose only files names containing _erf_<eID>
        if not include_in_pls(config, in_file):
            continue

        # label_tcs is nepochs x nlabels x ntimes (for epoched data)
        # label_names (nlabels,) vector of ROI names
        include_labels = config['include_labels']
        label_names = read_selected_roi_time_courses(in_file,
                        include_labels = include_labels)[1]
        return label_names

def collect_groups_for_event(ss, step, subj2group):
    """
    Prepare time courses data for each group for event ID specified in step config.
    If sign adjustement is requested, waveform signs are aligned with the evoked time
    course of the 1st subject processed (irrespective to their group).

    Args:
        ss (object): reference to this app object
        step (str): this step's name
        subj2group(dict): mapping subject ID -> group # 

    Returns:
        lst_dmat(list of ndarray): a list of arrays of shape `(nsubj, nfeatures)`
            - one per group
        lst_sid(list of list): a list of 2 lists - sIDs of subjects included for
            each group

    """
    config = ss.args[step]
    task = config['task']

    t0 = ss.args['src_rec']['epochs']['t_range'][0] # epoch's time origin

    if ('erf' in task) or task == ('compare_corrs_2groups1img'):
        SR = ss.args['src_erf']['target_sample_rate']
    elif 'henv' in task:
        SR = ss.args['src_hilbert']['target_sample_rate']

    # NOTE: We'll fail here with SR not defined for tasks not mentioned above,
    # which is the intention

    # Calculate index interval for PLS
    tstart, tend = config['pls_interval']
    tstart -= t0
    tend -= t0
    istart = int(SR*tstart)
    iend = int(SR*tend) + 1

    # Calculate index interval for sign adjustments
    # This only will be used for erf tasks
    tstart = ss.args['src_erf']['sign_adjust_interval'][0] - t0
    tend = ss.args['src_erf']['sign_adjust_interval'][1] - t0
    isign_start = int(SR*tstart)
    isign_end = int(SR*tend) + 1

    # If erf_power is True - use square of time courses
    erf_power = False if 'henv' in task else config['erf_power']

    if erf_power or ('henv' in task):
        config['adjust_signs'] = False

    ngroups = 4 if '4group' in config['task'] else 2

    lst_data = [[] for i in range(ngroups)]
    lst_sid = [[] for i in range(ngroups)]

    files = su.files_to_process(ss, step)
    epoch0 = None       # Reference epoch for sign adjustment
    include_labels = config['include_labels']

    for in_file, out_file in files:
        # Choose only files names containing _erf_<eID>
        if not include_in_pls(config, in_file):
            continue

        # label_tcs is nepochs x nlabels x ntimes (for epoched data)
        # label_names (nlabels,) vector of ROI names
        # For ERF files nepochs = 2: 1st epoch is the evoked for condition,
        # 2nd epoch is STD
        label_tcs, label_names = read_selected_roi_time_courses(in_file,
                        include_labels = include_labels)[:2]

        if erf_power:
            # NOTE: Only square the 1st epoch (the mean). The 2nd epoch
            # (STDs) will still be the STDs of the original tcs
            label_tcs[0] = label_tcs[0] * label_tcs[0]

        if config['adjust_signs']:
            # Adjust signs of label time courses to those of the 1st subject
            # added
            if epoch0 is None:
                epoch0 = label_tcs[0]
            else:
                # Adjust channels signs based on epoch0
                # 'e1' is the (nlabels x ntime) sign-adjusted epoch
                e1 = adjust_signs([epoch0, label_tcs[0]],
                        istart = isign_start, iend = isign_end)[1]
                # Update the ltc data
                label_tcs[0] = e1

        sid = su.fif_subject(in_file)
        group = subj2group[sid]

        # Use the 1st of two epochs for PLS unless we do PLS on STDs
        epoch_idx = 0 if '_std_' not in task else 1

        lst_data[group].append(label_tcs[epoch_idx][:,istart:iend].flatten(order = 'C'))
        lst_sid[group].append(sid)

    lst_dmat = [np.array(lst) for lst in lst_data]
    del lst_data    # Likely del will have no effect here but still

    return lst_dmat, lst_sid

def find_missing_elements(list1, list2):
    """
    Compares two lists and identifies elements missing in either list.

    Args:
        list1 (list): The first list.
        list2 (list): The second list.

    Returns:
        dict: A dictionary with keys 'missing_in_list1' and 'missing_in_list2'
              containing the missing elements from each list.
    """
    missing_in_list1 = [item for item in list2 if item not in list1]
    missing_in_list2 = [item for item in list1 if item not in list2]
    
    return {
        "missing_in_list1": missing_in_list1,
        "missing_in_list2": missing_in_list2,
    }

def pool_groups(lst_dmat, lst_nsubj, ncond):
    """
    Given dmats for each group with the same set of conditions, create
    a dmat for a single pooled group for these conditions.

    Args:
        lst_dmat(list of ndarray): a list of dmats with shape
            `(nsubj*ncond, nfeatures)` - one for each group
        lst_nsubj(list of int): a list of group sizes
        ncond(int): number of conditions

    Returns:
        dmat(ndarray): shape `(nallsubj*ncond, nfeatures)`
        lst_nsubj(list of int): a list with a single element `[nallsubj]`

    """
    conds_data = []     # List of pooled results for each condition

    for icond in range(ncond):
        lst = []    # A list of groups data for current condition

        for igroup, a in enumerate(lst_dmat):
            istart = icond*lst_nsubj[igroup]
            iend = istart + lst_nsubj[igroup]
            lst.append(a[istart:iend,:])

        conds_data.append(np.vstack(lst))   # Append pooled data for condition

    dmat = np.vstack(conds_data)            # Final dmat
    return dmat, [np.sum(lst_nsubj)]

def construct_titles(ss):
    """
    Construct title for a heatmap plot.

    Args:
        ss(obj): reference to this app object

    Returns:
        title(str): heatmap plot title
        pngname(Path): full pathname for the output .PNG file

    """
    STEP = 'pls_analysis'
    config = ss.args[STEP]
    hm = config['heatmap']
    out_file = ss.data_host.get_step_out_file(STEP, config['in_dir'])
    plot_path = ss.data_host.root / ss.data_host.meg / ss.data_host.config["out_root"] / \
                    ss.data_host.pipeline_version / hm['out_dir']
    lv = hm['latent_var']

    if (config['task'] == 'erf_mc_2groups1img') or \
            (config['task'] == 'erf_mc_4groups1img') or \
            (config['task'] == 'henv_mc_4groups1img') or \
            (config['task'] == 'henv_std_mc_4groups1img') or \
            (config['task'] == 'henv_std_contrast_4groups2img') or \
            (config['task'] == 'compare_corrs_2groups1img'):
        title = f'File: {out_file}'
        pngname = plot_path / (Path(out_file).stem + '.png')
    elif (config['task'] == 'erf_mc_1group4img') or \
            (config['task'] == 'erf_mc_2group4img') or \
            (config['task'] == 'erf_mc_pooled4img'):
        title = f'File: {out_file}; latent var: {lv}'
        pngname = plot_path / (Path(out_file).stem + f'_lv{lv}' + '.png')
    elif (config['task'] == 'erf_contrast_2group2img') or \
                (config['task'] == 'erf_contrast_2group4img'):
        title = f'File: {out_file}; latent var: {lv}'
        pngname = plot_path / (Path(out_file).stem + f'_lv{lv}' + '.png')

    """
    if config['erf_power']:
        title = title + ', power'
    """

    return title, pngname

def get_bar_labels(cfg):
    """
    Return a set of names (labels) for elements of a contrast or
    design variable depending on PLS task.
    """
    four_img_tasks = ('erf_mc_1group4img','erf_mc_pooled4img')
    four_groups_tasks = ('erf_mc_4groups1img','henv_mc_4groups1img','henv_std_mc_4groups1img',
                         'henv_std_sm_4groups1img','henv_std_sm_4groups2img')

    if cfg['task'] in four_img_tasks:
        return list(cfg['img_types'].values())

    if cfg['task'] in four_groups_tasks:
        groups_dict = cfg['bias_belief_groups'] # {"(n,m)":igroup}
        l_dict = {
            '(0, 1)': 'SCZ-B',
            '(1, 1)': 'ASD-B',
            '(0, 0)': 'SCZ-N',
            '(1, 0)': 'ASD-N'
            }

        # Replace group # in groups_dict with names
        gnames_dict = {k:l_dict[k] for k in groups_dict}

        return list(gnames_dict.values())

    # TODO: other tasks
    return []

def calc_pos_median(a):
    """
    Given a 2D numpy array, calculate medians along axis 0 of only
    positive values of this array.

    Args:
        a (ndarray): `shape (m,n)`

    Returns:
        med (ndarray): `shape (n,)` the medians
    """
    poz = a.copy()
    poz[a<0] = 0
    med = []

    for col in poz.T:
        c = col[col > 0]    # Drop all non-pozitive values
        med.append(np.median(c))

    return np.array(med)
        
def plot_zscores_medians(zscores, ax, t0, SR):
    """
    Generate a plot of median values of positive and negative
    zscores as a function of time.

    Args:
        zscores(ndarray): shape (nlabels, ntimes) the zscores for a selected
            latent variable
        ax(Axes): axes object to plot to
        t0(float): time start value, s
        SR(float): sampling rate, Hz

    Returns:
        ax(Axes): axes with the plot

    """
    mpoz = calc_pos_median(zscores)     # Shape: (ntimes,)
    mneg = -calc_pos_median(-zscores)
    t = t0 + np.arange(len(mpoz)) / SR
    ax.plot(t, mpoz, color='red', linewidth=2)
    ax.plot(t, mneg, color='blue', linewidth=2)
    # ax.set_xlabel('t,s')
    # ax.set_ylabel('med(t)')
    ax.minorticks_on()
    #ax.grid(True, which='major', color='lightblue', linestyle='-', linewidth=1)  # Major grid
    ax.grid(True, which='major', color='black', linestyle='--', linewidth=0.5)  # Major grid
    ax.grid(True, which='minor', color='gray', linestyle='--', linewidth=0.5)  # Minor grid
    ax.set_title('zscores medians')

    return ax

def construct_ordering(csv, label_names):
    """
    Read ordering .csv file corresponding to selected altas, and create
    mapping from original order used by `label_names` to the order
    in .csv. 

    The .csv file should have a header (1 row), and the ordered label names
    in its 1st column.

    Args:
        csv(str | Path): full pathname of the atlas ordering .csv
        label_names(list of str): a list of labels to reorder

    Returns:
        ordering(ndarray): 1D array of label indecies

    """
    # Create mapping label name -> label index
    dd = {name:i for i,name in enumerate(label_names)}

    # Read target ordering
    df = pd.read_csv(csv, header = 0, usecols=[0])
    lst_ordered = df.iloc[:, 0].tolist()

    ordering = []
    for name in lst_ordered:
        if name in dd:
            ordering.append(dd[name])

    if len(ordering) < len(label_names):
        unused = [l for l in label_names if l not in lst_ordered]
        print(f'The following channels are not listed in atlas ordering and are dropped: {unused}')

    return np.array(ordering)

def write_corr_dist_results(out_hdf5, task, eID, ccd_res):
    """
    Save correlation distributions analyses results to .hdf5 file.

    Args:
        out_hdf5 (Path | string): full pathname to the output .hdf5 file
        task(string): the task name
        eID (int): the event (image) ID
        ccd_res (CCD_Result): a named tuple with fields: 'effect_sizes',
            'hedges_g', 'CI_lower', 'CI_upper', 'bootstrap_means_A', 'bootstrap_means_B'.
            Each field contains a vector with length `nfeatures = nlabels x ntimes` 

    Returns:
        res(dict): dictionary with fields 'task', 'eID', 'ccd_res'

    """
    with h5py.File(out_hdf5, 'w') as f:
        f.create_dataset('task', data=task.encode('utf-8')) # read it as f['task'][()].decode('utf-8')
        f.create_dataset('eID', data=eID)                   # retrieve it later with f['eID'][()]

        # Iterate over the fields of the named tuple
        for field in ccd_res._fields:
            # one can read it back as f[field][:]
            f.create_dataset(field, data=getattr(ccd_res, field))

    return {'task': task, 'eID': eID, 'ccd_res': ccd_res}

def read_corr_dist_results(results_hdf5):
    """
    Read correlation distributions analyses data saved with `write_corr_dist_results()`.

    Args:
        results_hdf5(Path | str): full pathname of the .hdf5 file

    Returns:
        res(dict): a dictionary with the results data

    The `res` dictionary contains the following fields:

        **task** ( string): the corr dists task name

        **eID** (int): the event (image) ID

        **ccd_res** (CCD_Result): a named tuple with fields 'effect_sizes', \
            'hedges_g', 'CI_lower', 'CI_upper', 'bootstrap_means_A', 'bootstrap_means_B'. \
            Each field contains a vector with length `nfeatures = nlabels x ntimes` 

    """
    from compare_corr_distributions import CCD_Result
    res = {}

    with h5py.File(results_hdf5, 'r') as f:
        res['task'] = f['task'][()].decode('utf-8') # read it as f['task'][()].decode('utf-8')
        res['eID'] = f['eID'][()]                   # retrieve it later with f['SR'][()]

        # Iterate over the fields of the named tuple
        lst = []
        for field in CCD_Result._fields:
            lst.append(f[field][:])

        res['ccd_res'] = CCD_Result(*lst)

    return res

def read_selected_roi_time_courses(ltc_file, include_labels = None):
    """
    Read selected ROI (label) time courses from .hdf5 file created using
    `write_roi_time_courses()` function from `src_rec.py`. This is a simple
    wrapper over the `read_roi_time_courses(ltc_file)` function.

    Args:
        ltc_file (Path | str): full pathname of the output .hdf5 file
        include_labels (list of str | None): if supplied, should be a list case-insensitieve
            strings representing label names. Only time courses from labels that belong
            to this list will be returned.

    Returns:
        label_tcs (ndarray): `nlabels x ntimes` or `nepochs x nlabels x ntimes` for non-epoched
            or epoched data, respectively; ROI time courses
        label_names (ndarray of str):  1 x nlabels vector of ROI names corresponding to 
            the returned time courses
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
    res = read_roi_time_courses(ltc_file)

    if include_labels is None:
        return res

    ref_lst_lower = {x.lower() for x in include_labels}     # This is a set, not a list
    label_names = res[1]

    indices, labels_keep = zip(*[(i, s) for i, s in enumerate(label_names) if s.lower() in ref_lst_lower]) \
            or ([], [])

    if not indices:
        raise ValueError(f'No labels found in {ltc_file} match those specified in the include_labels')

    if res[0].ndim == 2:    # Non-epoched data
        label_tcs = res[0][indices,:]
    else:
        label_tcs = res[0][:,indices,:]

    W = res[4][:,indices]

    return label_tcs, labels_keep, res[2], res[3], W, res[5], res[6], res[7] 


