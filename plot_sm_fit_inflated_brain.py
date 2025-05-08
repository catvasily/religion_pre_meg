"""
**Plot stat model fits results on an inflated brain surface** 
"""

import numpy as np
import pickle
import mne
from pathlib import Path
from pls_analysis import set_config_for_job, get_sm_results_file, get_eff_sizes_and_pvals
from view_inflated_brain_data import view_inflated_brain_data, expand_data_to_rois
from plot_pls_inflated_brain import construct_label_objects

def plot_sm_fit_inflated_brain(ss):
    """
    Display statistical model fit parameters distributions over the inflated brain surface.

    Args:
        ss(obj): reference to this app object

    """
    STEP = 'plot_sm_fit_inflated_brain'
    cfg = ss.args[STEP]
    cfg_pls = ss.args['pls_analysis']
    task = cfg_pls['task']
    job_parms = cfg_pls['array_job_parms']      # {'band': [], 'event_id': id} or
                                                # {'band': [], 'img_events': []} 
    njobs = len(job_parms)                      # Number of fit settings to plot
    row_labels = 'group', 'rel', 'group*rel'
    nparms = len(row_labels)

    hemi = cfg['hemi']
    fs_dir = ss.data_host.get_fsaverage_dir()

    # This is because construct_label_objects() uses args from
    # 'plot_pls_inflated_brain step':
    ss.args['plot_pls_inflated_brain']['hemi'] = hemi

    # Verify args
    if not cfg['expand_values_to_roi']:
        raise NotImplementedError('Processing of the case expand_values_to_roi = False not implemented')

    # Load data to plot
    ijob = cfg['ijob']
    set_config_for_job(cfg_pls, ijob)
    sm_pkl_file = get_sm_results_file(ss)

    with open(sm_pkl_file, "rb") as f:
        sm_fit_results = pickle.load(f)

    # sm_fit_results is a dictionary {roi: <OLS-results object>}
    labels = construct_label_objects(ss, list(sm_fit_results.keys()))

    # Returned:
    # eff_sizes(ndarray): shape(nparms, nrois) effect sizes for each ROI
    # pvalues(ndarray): shape(nparms, nrois) t-test p-values for each ROI
    _, eff_sizes, pvalues = get_eff_sizes_and_pvals(task, sm_fit_results)

    # Get data to plot and Expand single value to the whole ROI
    iparm = row_labels.index(cfg['what'])
    data = eff_sizes[iparm,:] 

    if cfg['apply_thresholds']:
        middle_val = np.mean(cfg['cbar_lims'])
        dmin, dmax = cfg['clip_interval']
        idx = np.logical_and(data > dmin, data < dmax)
        data[idx] = middle_val

    # The data.shape will now be (nvertices,)
    data, vertno = expand_data_to_rois(data, labels)

    kwargs_data = None if cfg['colorbar'] else {"colorbar": False} 
    
    # Save plot where the heatmaps are saved
    plot_path = ss.data_host.root / ss.data_host.meg / ss.data_host.config["out_root"] / \
                    ss.data_host.pipeline_version / cfg_pls['heatmap']['out_dir']

    png = str(plot_path / (Path(sm_pkl_file).stem + f'_{cfg["what"]}.png'))

    if cfg['all_in_one']:
        # Place all views into a single figure
        kwargs_brain = {'views': [v[0] for v in cfg['views']],
                        'view_layout':'horizontal', 'size': cfg['figsize'][0]}
    else:
        # Display and save each view in a separate figure
        kwargs_brain = {'size': cfg['figsize'][1]}

    atlas_name = ss.args['src_rec']['atlas']
    atlas = ss.args['src_rec']['parcellations'][atlas_name]
    brain = view_inflated_brain_data(
            atlas = atlas,
            show_atlas = cfg['show_atlas'],
            hemi = hemi,
            title = cfg['what'],                   # This is a window bar title
            data = data,
            cbar_lims = cfg['cbar_lims'],
            colormap = cfg['colormap'],
            alpha_data = cfg['alpha_data'],
            smoothing_steps = cfg['smoothing_steps'],
            rois_to_mark = cfg['rois_to_mark'],
            vertno = vertno,
            show_vertices = cfg['show_vertices'],
            subjects_dir = fs_dir,
            scale_factor = cfg['scale_factor'],
            color_dots = cfg['color_dots'],
            alpha_cortex = cfg['alpha_cortex'],
            resolution = cfg['resolution'],
            show = cfg['show_plots'],
            block = False,
            inflated = True,
            kwargs_brain = kwargs_brain,
            kwargs_data = kwargs_data,
            verbose = ss.args['verbose']
        )

    #fig = plt.gcf()
    #fig.suptitle('Very long title')    # THIS DOES NOT WORK

    # Also, there is no way to get to the colobar and change its font

    if cfg['show_plots']:
        row = 0
        col = 0
        nviews = len(cfg['views'])

        for view,hemi in cfg['views']:
            brain.show_view(view=view, hemi=hemi, row = row, col = col)

            if cfg['all_in_one']:
                col += 1

                if col == len(cfg['views']):
                    input(f"Press ENTER to continue...")
                    brain.save_image(png.replace('.png',f'_{nviews}_views.png'))
            else:
                input(f"View: {view}, hemi: {hemi}. Press ENTER to continue...")
                brain.save_image(png.replace('.png',f'_{view}_{hemi}.png'))

