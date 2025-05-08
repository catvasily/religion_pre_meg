"""
**Plot statistics of stat model fitting results distributions
over ROIs**

The specific SM fitting task used is determined by the setting
for the 'pls_analysis' step.
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from pls_analysis import set_config_for_job, get_sm_results_file, get_eff_sizes_and_pvals

def plot_sm_fit_stats(ss):
    """
    Create plots of statistics of distributions SM fittnig results
    (means, STDs, etc) over the ROIs, for various parameter
    combinations (band, image or a pair of images).

    Args:
        ss(obj): reference to this app object

    """
    cfg = ss.args['plot_sm_fit_stats']
    cfg_pls = ss.args['pls_analysis']
    task = cfg_pls['task']
    job_parms = cfg_pls['array_job_parms']      # {'band': [], 'event_id': id} or
                                                # {'band': [], 'img_events': []} 
    njobs = len(job_parms)                      # Number of fit settings to plot
    row_labels = 'group', 'rel', 'group*rel'
    nparms = len(row_labels)
    alpha = cfg['alpha']                        # Significance level
    multiplier = cfg['multiplier']              # Multiplier for plotting AVG, STD
    bar_labels = cfg['bar_labels']
    nrows = cfg['n_plot_rows']
    figsize = cfg['figsize']

    if '2img' in task:
        figsize[0]*=1.5     # because there will be 6 columns instead of 4

    dfs = []
    pool = {}               # {band -> pooled eff sizes for a band}
    minmax = np.zeros((njobs,nparms,2)) # min and max values over ROIs for each parameter
                                        # in each setting

    for ijob in range(njobs):
        set_config_for_job(cfg_pls, ijob)
        sm_pkl_file = get_sm_results_file(ss)

        with open(sm_pkl_file, "rb") as f:
            sm_fit_results = pickle.load(f)

        # Returned:
        # eff_sizes(ndarray): shape(nparms, nrois) effect sizes for each ROI
        # pvalues(ndarray): shape(nparms, nrois) t-test p-values for each ROI
        _, eff_sizes, pvalues = get_eff_sizes_and_pvals(task, sm_fit_results)

        band = tuple(job_parms[ijob]['band'])

        if band in pool:
            pool[band]=np.hstack((pool[band],eff_sizes))
        else:
            pool[band]=eff_sizes.copy()

        means = multiplier*np.mean(eff_sizes, axis = 1)
        stds = multiplier*np.std(eff_sizes, axis = 1)
        ratios = np.abs(means / stds)
        nsig = np.sum(pvalues < alpha, axis = 1)

        # calculate extrema for each parameter
        minmax[ijob,:,0] = np.min(eff_sizes, axis = 1)
        minmax[ijob,:,1] = np.max(eff_sizes, axis = 1)

        dfs.append(pd.DataFrame(dict(zip(bar_labels, (means,stds,ratios,nsig)))))

    # Calculate quantiles per band for each parameter
    # pool[band].shape = (nparms, nroi x <nimg | npairs>)
    thr = cfg['global_alpha']/2

    # Uncomment code below to calculate quantiles per each (band,parm) pair
    """
    band_quantiles = {}
    for band in pool:
        arr = pool[band]
        band_quantiles[band] = [np.quantile(arr,thr,axis = 1),np.quantile(arr,1-thr,axis = 1)]

    # Now band_quantiles is {band -> [lst_low_thrs, lst_high_thrs]}, where len of each list
    # is equal to nparm
    # At this point, one can find significant (band,parm, img|pair) combinations
    # separately for lower and upper thresholds:
    # Find which jobs produce significant ROIs using per-band thresholds
    for ijob in range(njobs):
        band = tuple(job_parms[ijob]['band'])

        for iparm in range(nparms):
            if minmax[ijob,iparm,0] <= band_quantiles[band][0][iparm]:
                print(f'min threshold hit for {job_parms[ijob]}, {row_labels[iparm]}')

            if minmax[ijob,iparm,1] >= band_quantiles[band][1][iparm]:
                print(f'max threshold hit for {job_parms[ijob]}, {row_labels[iparm]}')

    # Typically there will be a lot of "significant" (band, parm, extremum) combinations
    """

    # Now calculate thresholds after pooling 
    pooled_es = np.hstack(list(pool.values()))  # shape (nparms, nband*nroi*<nimg|npairs>)
    pooled_quantiles = np.zeros((nparms,2))
    pooled_quantiles[:,0] = np.quantile(pooled_es,thr,axis = 1)
    pooled_quantiles[:,1] = np.quantile(pooled_es,1-thr,axis = 1)

    # Use dataframe for pretty printing
    pd.options.display.float_format = '{:.2f}'.format
    df_quant = pd.DataFrame(pooled_quantiles, index=row_labels, columns=['lower', 'upper'])
    print(f'Pooled quantiles:\n{df_quant}')

    # Find significant (parm, extremum) combinations
    print_cols = 'ijob','fit_parm','band','image(s)'
    df_min = pd.DataFrame(columns=print_cols)
    df_max = pd.DataFrame(columns=print_cols)
    for ijob in range(njobs):
        for iparm in range(nparms):
            if minmax[ijob,iparm,0] <= pooled_quantiles[iparm][0]:
                vals = list(job_parms[ijob].values())
                df_min.loc[len(df_min)] = ijob,row_labels[iparm],vals[0],vals[1]

            if minmax[ijob,iparm,1] >= pooled_quantiles[iparm][1]:
                vals = list(job_parms[ijob].values())
                df_max.loc[len(df_max)] = ijob, row_labels[iparm],vals[0],vals[1]

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    for parm in row_labels:
        print('\n---------------------------------')
        print(f'Lower threshold hits for {parm}:')
        print('---------------------------------')
        print(df_min[df_min['fit_parm']==parm])
        print('\n---------------------')
        print(f'Upper threshold hits for {parm}:')
        print('---------------------')
        print(df_max[df_max['fit_parm']==parm])

    if cfg['stats_only']:
        return

    # Create subplots
    ncols = int(njobs/nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Flatten axes for easy iteration
    axes = axes.flatten()   # Flatten in row-wise way by default
    ylabel = ''
    ylim = cfg['ylim']

    # Plot each DataFrame in a subplot
    for i, df in enumerate(dfs):
        plot_legend = True if i == 0 else False
        df.T.plot(kind='bar', ax=axes[i], width=0.6, legend=plot_legend)

        title = construct_title(job_parms[i], cfg_pls['img_types'])
        axes[i].set_title(title)

        if i % nrows == 0:
            axes[i].set_ylabel(ylabel)
        else:
            axes[i].set_ylabel('')

        if i < njobs - ncols:
            axes[i].set_xticklabels([])
        else:
            axes[i].set_xticklabels(df.columns, rotation=0)  # Keep labels horizontal

        if plot_legend:
            legend = axes[i].legend(labels=row_labels, loc='lower right',
                                    fontsize = cfg['legend_fontsize'])

        axes[i].set_ylim(ylim)

    # Adjust layout
    plt.tight_layout()

    # Save plot where the heatmaps are saved
    plot_path = ss.data_host.root / ss.data_host.meg / ss.data_host.config["out_root"] / \
                    ss.data_host.pipeline_version / cfg_pls['heatmap']['out_dir']
    pngname = plot_path / (task + '_' + cfg['png_name_suffix'] + '.png')
    plt.savefig(pngname, dpi=cfg['dpi'])
    plt.show()

def construct_title(dict_parms, img_types):
    # {'band': [], 'event_id': id} or {'band': [], 'img_events': []}
    # img_types: {'<eID>': <type>}
    f1,f2 = dict_parms['band']

    if 'event_id' in dict_parms:
        title = f'{f1}-{f2} Hz, {img_types[str(dict_parms["event_id"])]}'
    elif 'img_events' in dict_parms:
        im1,im2 = [img_types[str(eID)] for eID in dict_parms['img_events']]
        title = f'{f1}-{f2} Hz, {im2}-{im1}'
    else:
        raise ValueError('Invalid dict_parms argument')

    return title

