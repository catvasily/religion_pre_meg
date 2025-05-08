"""
**Plot PLS results on an inflated brain surface** 
"""

import mne
import matplotlib.pyplot as plt
from pls_analysis import get_pls_results_hdf5_file, read_pls_results, construct_titles
from view_inflated_brain_data import view_inflated_brain_data, expand_data_to_rois

def plot_pls_inflated_brain(ss):
    """
    Display PLS results on top of the inflated brain surface.

    Args:
        ss(obj): reference to this app object

    """
    STEP = 'plot_pls_inflated_brain'
    verbose = ss.args['verbose']
    config = ss.args[STEP]

    # Output folder is defined via construct_titles(),  which use heatmap
    # settings. Change heatmap settings with this step output dir:
    ss.args['pls_analysis']['heatmap']['out_dir'] = config['out_dir']

    fs_dir = ss.data_host.get_fsaverage_dir()
    hemi = config['hemi']

    if ss.data_host.cluster_job:
        config['show_plots'] = False

    # Get the PLS scores data
    res_hdf5 = get_pls_results_hdf5_file(ss)
    res = read_pls_results(res_hdf5)
    zscores = res['z_scores']
    SR = res['SR']
    label_names = res['label_names']

    # Verify args
    if not config['expand_values_to_roi']:
        raise NotImplementedError('Processing of the case expand_values_to_roi = False not implemented')

    nlv, nlabels, ntimes = zscores.shape
    lv = ss.args['pls_analysis']['heatmap']['latent_var']

    if (lv < 0) or (lv >= nlv):
        raise ValueError(f'Latent variable number {lv} is out of range')

    time = config['time']
    pls_interval = ss.args['pls_analysis']['pls_interval']

    if (time < pls_interval[0]) or (time >= pls_interval[1]):
        raise ValueError(f'Specified time = {time} does not belong to PLS interval {pls_interval}')

    itime = int((time - pls_interval[0])*SR)

    # Get the data to display
    data = zscores[lv,:,itime]
    
    # Create a set of Label objects corresponding to label_names
    labels = construct_label_objects(ss, label_names)

    # Expand ROI single value to the whole ROI
    # The data shape becomes (nvertices,)
    data, vertno = expand_data_to_rois(data, labels)

    kwargs_data = None if config['colorbar'] else {"colorbar": False} 
    title, png = construct_titles(ss)
    png = str(png)

    if config['all_in_one']:
        # Place all views into a single figure
        kwargs_brain = {'views': [v[0] for v in config['views']],
                        'view_layout':'horizontal', 'size': config['figsize'][0]}
    else:
        # Display and save each view in a separate figure
        kwargs_brain = {'size': config['figsize'][1]}

    atlas_name = ss.args['src_rec']['atlas']
    atlas = ss.args['src_rec']['parcellations'][atlas_name]
    brain = view_inflated_brain_data(
            atlas = atlas,
            show_atlas = config['show_atlas'],
            hemi = hemi,
            title = title,
            data = data,
            cbar_lims = config['cbar_lims'],
            colormap = config['colormap'],
            alpha_data = config['alpha_data'],
            smoothing_steps = config['smoothing_steps'],
            rois_to_mark = config['rois_to_mark'],
            vertno = vertno,
            show_vertices = config['show_vertices'],
            subjects_dir = fs_dir,
            scale_factor = config['scale_factor'],
            color_dots = config['color_dots'],
            alpha_cortex = config['alpha_cortex'],
            resolution = config['resolution'],
            show = config['show_plots'],
            block = False,
            inflated = True,
            kwargs_brain = kwargs_brain,
            kwargs_data = kwargs_data,
            verbose = verbose
        )

    #fig = plt.gcf()
    #fig.suptitle('Very long title')    # THIS DOES NOT WORK

    # Also, there is no way to get to the colobar and change its font

    if config['show_plots']:
        row = 0
        col = 0
        nviews = len(config['views'])

        for view,hemi in config['views']:
            brain.show_view(view=view, hemi=hemi, row = row, col = col)

            if config['all_in_one']:
                col += 1

                if col == len(config['views']):
                    input(f"Press ENTER to continue...")
                    brain.save_image(png.replace('.png',f'_{nviews}_views_t{config["time"]}s.png'))
            else:
                input(f"View: {view}, hemi: {hemi}. Press ENTER to continue...")
                brain.save_image(png.replace('.png',f'_{view}_{hemi}_t{config["time"]}s.png'))

def construct_label_objects(ss, label_names):
    """
    Given the names of labels from specified atlas, construct corresponding
    MNE Python `Label` objects.

    Args:
        ss(Object): ref to this app object
        label_names(list of str): list of label names; each name should belong
            to selected atlas

    Returns:


    """
    # Create a set of Label objects corresponding to label_names
    verbose = ss.args['verbose']
    config = ss.args['plot_pls_inflated_brain']
    atlas_name = ss.args['src_rec']['atlas']
    atlas = ss.args['src_rec']['parcellations'][atlas_name]
    fs_dir = ss.data_host.get_fsaverage_dir()
    hemi = config['hemi']

    atlas_labels = mne.read_labels_from_annot(
        "fsaverage",
        parc = atlas,
        hemi = hemi if hemi != 'split' else 'both',
        surf_name='white', 
        subjects_dir=fs_dir,
        sort=True,                       # Sort labels in alphabetical order
        verbose=verbose
    )

    # atlas_labels may in principle be ordered differently than those in .hdf5 file,
    # and contain extra (unused) labels.
    dd = dict()
    for l in atlas_labels:
        dd[l.name] = l

    # Create a label list in accordance with label_names
    labels = [dd[name] for name in label_names]
    return labels

