import math
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt, cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from imgstats.plot.utils import inset_axes


def polar_histogram(x, ax=None, **kwargs):
    """ Plot a histogram on polar coordinates (radius is relative frequency)

    Args:
        x: np array containing the data
        ax: pyplot axis to be plotted on. if not given, a new axis is created via subplots
        kwargs: arguments passed to https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html,
                after setting default color
    Returns:
        r: frequency / density
        theta: bin edges
        patches: matplotlib patches objects
    """

    # set default arguments for histogram
    kwargs = dict({'color': 'white', 'edgecolor': 'C0'}, **kwargs)

    # if no ax is given
    if not ax:
        f, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # do the plotting
    r, theta, patches = ax.hist(np.append(x, x + np.pi), range=(0, 2 * np.pi), **kwargs)

    return r, theta, patches


def polar_plotgrid(eccentricities, polars, figsize=(8.27, 8.27), subplot_polar=False, inset_size=0.11):
    """ Create a grid of subplots at the specified eccentricities and polar angles

    Args:
        eccentricities (List[int]): eccentricities
        polars (List[int]): polar angles
        figsize (Tuple[float]): figure size in inches
        subplot_polar: make the subplots in polar coordinates

    Returns:
        f: matplotlib Figure
        axes: dict[(ecc, pol)] containing the individual inset axe
    """
    f, main_ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"))

    maxr = max(eccentricities) * 1.2

    main_ax.set_rmin(0)
    main_ax.set_rmax(maxr)

    axes = {}
    for ecc in eccentricities:
        for pol in polars:

            # compute x and y values
            x = ecc * np.cos(np.deg2rad(pol))
            y = ecc * np.sin(np.deg2rad(pol))

            # normalize -> between 0 and 1
            x, y = (x / maxr) * 0.5 + 0.5, (y / maxr) * 0.5 + 0.5

            # get bottom and left coordinates instead of center
            x, y = (x - inset_size / 2), (y - inset_size / 2)

            # create inset axis at the location
            axin = inset_axes(main_ax, [x, y, inset_size, inset_size], polar=subplot_polar)
            axin.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                left=False,
                labelbottom=False,
                labelleft=False)

            axes[(ecc, pol)] = axin

            # we only need eccentricity = 0 once
            if ecc == 0:
                break

    main_ax.set_rticks(eccentricities)
    # rticklabels = [r"$\varphi = {{{}}}$°".format(ecc) for ecc in eccentricities]
    rticklabels = [r"${{{}}}$°".format(ecc) for ecc in eccentricities]
    rticklabels[0] = ""
    main_ax.set_yticklabels(rticklabels)

    main_ax.spines['polar'].set_visible(False)
    main_ax.set_rlabel_position(20.5)

    return f, main_ax, axes


def plot_color_wheel(cmap="hsv"):
    """ Plot a color wheel from 0 to 180 degrees (in order to illustrate the mapping from angles to colors)

    Args:
        cmap (str): matplotlib.cm colormap

    Returns:
        fig, ax: matplotlib Figure and Axes
    """
    azimuths = np.arange(0, 181, 1)
    zeniths = np.arange(40, 70, 1)
    values = azimuths * np.ones((30, 181))
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    ax.pcolormesh(azimuths * np.pi / 180.0, zeniths, values, cmap=cmap)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_yticks([])
    return fig, ax


def plot_params(params, param_names, cmaps=None, limits=None, display_names=None, fig_kw=None, nrow=1, colorbar=True,
                cb_orientation="horizontal", eccentricities=None, labelsize=8, title=True, labels=True):
    """ Plot the parameters of the power spectrum fits across the visual field
    Creates a plot similar to the one in Pamplona et al., 2013

    Args:
        params (pd.DataFrame): dataframe containing the parameters
        param_names (list of str): columns of the df to be plotted
        cmaps (str, pyplot colormap or list of those): colormap
                        (can be assigned individually per parameter using a list the same length as the keys)
        limits (dict of tuples of float): lower and upper limits for the parameters. keys have to be same as param_names
        display_names (list of str): alternative names for plotting
        fig_kw (dict): kwargs passed to plt.subplots()
        nrow (int): number of rows for the subplot grid

    Returns:
        fig, axes: matplotlib figure and axes
    """
    if not eccentricities:
        eccentricities = [0, 30, 50]

    if not limits:
        limits = dict()

    if not fig_kw:
        fig_kw = dict()

    default_cmap = "cool"
    if not cmaps:
        cmaps = dict(zip(param_names, [default_cmap for _ in param_names]))
    elif type(cmaps) == str:
        cmaps = dict(zip(param_names, [cmaps for _ in param_names]))
    elif type(cmaps) == list:
        cmaps = dict(zip(param_names, cmaps))
    else:
        raise ValueError("Please give a valid cmap or a list of those")

    if display_names is None:
        display_names = param_names

    ncol = math.ceil(len(param_names) / nrow)

    fig, axes = plt.subplots(nrow, ncol, **fig_kw)

    for i, ax in enumerate(axes.flat):
        group_names = np.arange(0, 360, step=45)
        group_names = [str(name) + "°" for name in group_names]
        group_size = np.ones(8)

        name = param_names[i]
        param0 = params[name][params.ecc == eccentricities[0]]
        param30 = params[name][params.ecc == eccentricities[1]]
        param50 = params[name][params.ecc == eccentricities[2]]

        cmap = cmaps[name]
        colormap = plt.get_cmap(cmap)

        if name in limits:
            pmin, pmax = limits[name]
        else:
            pmin, pmax = params[name].min(), params[name].max()

        color_norm = mpl.colors.Normalize(pmin, pmax)  # maps your data to the range [0, 1]

        # First Ring (outside)

        ax.axis('equal')
        plot50, label_names = ax.pie(group_size, radius=1., startangle=-22.5, labels=group_names if labels else None,
                                     labeldistance=1.225,
                                     colors=colormap(color_norm(param50.values)))
        plt.setp(plot50, width=0.4, edgecolor='white')

        # Second Ring (Inside)
        plot30, _ = ax.pie(group_size, radius=1.0 - 0.4, startangle=-22.5, labels=None,
                           colors=colormap(color_norm(param30.values)))
        plt.setp(plot30, width=0.4, edgecolor='white')

        plot0, _ = ax.pie(np.array([1]), radius=1. - 0.8, labels=None, colors=colormap(color_norm(param0.values)))
        plt.setp(plot0, width=0.2, edgecolor='white')

        # ax_cb = fig.add_axes([.8,.25,.03,.5])
        # cb =
        if colorbar:
            divider = make_axes_locatable(ax)
            pos = "bottom" if cb_orientation == "horizontal" else "right"
            cax = divider.append_axes(pos, size='5%', pad=0.1)
            cbar = fig.colorbar(cm.ScalarMappable(cmap=colormap, norm=color_norm), cax=cax, orientation=cb_orientation)
            cbar.ax.tick_params(labelsize=labelsize)
            if cb_orientation == "horizontal":
                cbar.ax.set_ylabel(display_names[i], rotation=0)
            else:
                cbar.ax.set_title(display_names[i])
        else:
            pass
            # fig.subplots_adjust(bottom=0.08)

        for t in label_names:
            t.set_horizontalalignment("center")
            t.set_fontsize(labelsize)
        if title:
            ax.set_title(display_names[i], y=1.05)
        # plt.margins(0,0)

    # fig.tight_layout()
    return fig, axes
