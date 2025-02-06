import numpy as np
import matplotlib.pyplot as plt

def plot_channel_heatmap(signals, labels, *, fig = None, ax = None, t0 = 0, SR = 1, color_min=None, color_max=None, 
                    xlabel='Time (seconds)',
                    ylabel='Channels',
                    cbar = True,
                    cbar_label='Signal Value',
                    title='Signals',
                    cmap='jet',
                    figsize = (12,8),
                    fontsize_y = 10,
                    save_file = None,
                    dpi = 300,
                    show = True):
    """
    Plot a heat map of channel signals.

    Args:
        signals (ndarray): `shape(nchan, ntime)` signal data to plot
        labels (list of str): a list of `nchan `channel names (labels).
            May be empty or `None` - then labels will be generated.
        fig (Figure): if supplied, an existing matplotlib `Figure` object
            to use. Current axes of this figure will be used for plotting.
        ax(Axes): if supplied, the heatmat will be drawn inside these
            axes. The `fig` parameter will be ignored in this case. 
        t0 (float): time origin, s
        SR (float): sampling rate, 1/s
        color_min (float or None): min value for the color scale
        color_max (float or None): max value for the color scale
        xlabel (str): X-axis title
        ylabel (str): Y-axis title
        cbar (bool): flag to show the color bar
        cbar_label (str): Z-axis (color bar) title
        title (str): plot title
        cmap (str): name of matplotlib color map to use
        figsize (tuple w,h): figure `(width, height)` in inches
        fontsize_y (float): y-ticks labels font size
        save_file (Pathlike): if given - a pathname for the file to save
            the figure. Its extension will determine the image type.
        dpi (int): saved figure resolution
        show (bool): flag to display the plot (pauses code execution)

    Returns:
        ax(Axes): matplotlib axes object. IMPORTANTLY, if `show = True`,
            then the associated figure `ax.figure` `will be empty. Therefore
            for the figure object to be useful, set `show = False` in this call

    """
    if labels is None:
        labels = []
    
    nchan, ntime = signals.shape

    if len(labels) < nchan:
        ich = len(labels) + 1
        for i in range(nchan - len(labels)):
            labels.append(f'ch{ich + i}')
    
    # Determine color scale min/max values
    if color_min is None:
        color_min = signals.min()
    if color_max is None:
        color_max = signals.max()
    
    # Generate time vector in seconds
    time_vector = np.arange(t0, t0 + ntime / SR, 1 / SR)[:ntime]
    
    # Plotting the heat map
    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=figsize)
            ax = plt.gca()
        else:
            ax = fig.gca()
    else:
        fig = ax.figure

    im = ax.imshow(signals, aspect='auto', cmap=cmap, 
               extent=[time_vector[0], time_vector[-1], nchan-0.5, -0.5], 
               vmin=color_min, vmax=color_max)
    
    ax.set_yticks(np.arange(nchan))
    ax.set_yticklabels(labels, fontsize = fontsize_y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Adding labels and color scale
    #fig.colorbar(im, orientation='vertical', label=cbar_label, fraction=0.046, pad=0.04)
    if cbar:
        fig.colorbar(im, orientation='vertical', label=cbar_label)

    if save_file is not None:
        plt.savefig(save_file, dpi=dpi)

    if show:
        plt.show()

    return ax

if __name__ == '__main__':
    # Test configuration
    n_channels = 10          # Number of channels
    ntime = 100              # Number of time points in each signal
    t0 = 0                   # Starting time in seconds
    SR = 10                  # Sampling rate in Hz (10 samples per second)

    # Generate mock data for testing
    signals = np.zeros((n_channels, ntime))
    labels = []

    for i in range(n_channels):
        signals[i,:] = np.sin(2 * np.pi * 0.1 * np.arange(ntime)) * (i + 1)  # Sinusoidal signal with increasing amplitude
        labels.append(f'Channel_{i+1}')

    # Test color limits
    color_min = -10
    color_max = 10

    # Call the function with test data
    plot_channel_heatmap(signals, labels, t0 = t0, SR = SR, color_min=None, color_max=None)

    # Test specifying figure
    fig = plt.figure()
    plot_channel_heatmap(signals, labels, fig = fig, t0 = t0, SR = SR, color_min=None, color_max=None)

    # Test specifying axes
    fig = plt.figure()
    ax1 = fig.add_axes([0.15, 0.1, 0.2, 0.2])  # (x=0.1, y=0.1, width=0.2, height=0.2)
    ax2 = fig.add_axes([0.65, 0.1, 0.2, 0.2])
    plot_channel_heatmap(signals, labels, ax=ax1, t0 = t0, SR = SR, color_min=None, color_max=None, show = False)
    plot_channel_heatmap(signals, labels, ax=ax2, t0 = t0, SR = SR, color_min=None, color_max=None)

