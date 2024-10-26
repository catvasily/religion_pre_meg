import numpy as np
import matplotlib.pyplot as plt
import math

def plot_array(data_arrays, x_values=None, x_limits=None, y_limits=None, colors=None, 
               chnames=None, cond_names= None, xlabel="Time", ylabel="Signal",
               suptitle="Experiment Time Courses",
               ncols=1, cmap = 'tab10', show=True, dpi = 300, fname=None):
    """
    Plots several sets of multi-channel data for different experiments in a grid of subplots,
    one for each channel. All data should share the same set of values along X-axis.
    
    Args:
        data_arrays(list of ndarray): List of NumPy arrays of shape `(nchan, nx)`,
            one for each experiment.
        x_values(vector): NumPy array of shape `(nx,)`, specifying the X values. If
            `None`, sample indecies will be used
        x_limits(tuple): `(xmin, xmax)` the X-axis limits. If `None`, determined automatically.
        y_limits(tuple): `(ymin, ymax)` the Y-axis limits. If `None`, determined automatically.
        colors(list): List of colors (one peri experiment). Any matplotlib color specification
            is allowed. If `None`, automatically assigned in accordance with `cmap`.
        chnames(list of str): channel names. If None, `ch N` will be used where `N`is a 
            1-based channel number
        cond_names(list of str): names of conditions (experiments) that will be used as
            curves labels in the legend. If `None`, 'cond N' will be used where `N` is
            1-based condition (experiment) number
        xlabel(str): Label for the X-axis.
        ylabel(str): Label for the Y-axis.
        suptitle(str): Super title for the entire figure.
        ncols(int): Number of columns in the subplot grid.
        cmap (str): a name of the matplotlib color map to be used if colors are not
            specified
        show(bool): flag to display the plot
        dpi(int): an image resolution to use when saving image
        fname(Path | str | None): full pathname of the file to save the image, If `None`
            the image will not be saved
    """
    n_experiments = len(data_arrays)  # Number of experiments
    nchan, nx = data_arrays[0].shape  # Assuming all data arrays have the same shape
    
    # Calculate the number of rows based on ncols
    nrows = math.ceil(nchan / ncols)

    if x_values is None:
        x_values = np.arange(nx)
    
    # If colors are not provided, use a default colormap
    if colors is None:
        cmap = plt.get_cmap(cmap)
        colors = [cmap(i) for i in range(n_experiments)]

    if chnames is None:
        chnames = [f'ch {i+1}' for i in range(nchan)]
    
    if cond_names is None:
        cond_names = [f'cond {i+1}' for i in range(n_experiments)]
    
    # Create subplots grid (nrows x ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 2.5 * nrows), sharex=True, sharey=True)
    
    # Flatten the axes array for easy iteration (handles case of single row/column subplots)
    axs = axs.flatten()
    i_legend = min(nchan,ncols-1)    # Plot number where a legend will be shown

    for i in range(nchan):
        ax = axs[i]
        for j, data in enumerate(data_arrays):
            ax.plot(x_values, data[i, :], label=cond_names[j], color=colors[j])
        
        # Set X-axis limits if provided
        if x_limits is not None:
            ax.set_xlim(x_limits)
        
        # Set Y-axis limits if provided
        if y_limits is not None:
            ax.set_ylim(y_limits)

        ax.set_title(chnames[i])
        
        ax.set_ylabel(ylabel if (i % ncols == 0) else "")  # Label Y-axis for first column only

        if i == i_legend:
            ax.legend(loc='upper right')
    
    # Set X-axis label for the last row of subplots
    for ax in axs[-ncols:]:
        ax.set_xlabel(xlabel)
    
    # Remove any unused subplots if nchan < nrows * ncols
    for i in range(nchan, len(axs)):
        fig.delaxes(axs[i])
    
    # Automatically determine axis limits if not provided
    # using the 1st plot
    if x_limits is None:
        axs[0].autoscale(axis='x')
    if y_limits is None:
        axs[0].autoscale(axis='y')
    
    # Add a super title for the entire figure
    fig.suptitle(suptitle, fontsize=16)
    
    # Tight layout for avoiding overlap
    fig.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for the super title
    
    if fname is not None:
        plt.savefig(fname, dpi=dpi)

    # Show the plot
    if show:
        plt.show()

# ------------
# Test program
# ------------

if __name__ == "__main__":
    # Inputs
    nconds = 3      # Number of conditions (experiments)
    nchans = 9
    ntimes = 100
    seed = 12345

    rng = np.random.default_rng(seed = seed)
    data_arrays = rng.normal(loc=0, scale=1, size=(nconds, nchans, ntimes))

    plot_array(data_arrays, x_values=None, x_limits=None, y_limits=None, colors=None, 
               chnames = None, xlabel="Time", ylabel="Signal", suptitle="Experiment Time Courses",
               ncols=3, cmap = 'tab10', fname = 'qq.png')

