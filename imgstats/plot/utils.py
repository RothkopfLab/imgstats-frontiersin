from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.axes._axes import _make_inset_locator
from matplotlib.projections import PolarAxes

DEFAULT_FIGSIZE = 3.44556 * 2
GOLDEN_RATIO = 1.61803


def set_default_params(fontsize=10, labelsize=10, titlesize=12, ticksize=10):
    """ Set the parameters of the plots so that they look nice in my thesis.
        3.44556 inches is the width of one column (in 2-column layout) in my Latex document.

    Returns:
        None
    """
    params = {'text.usetex': True,
              'font.size': fontsize,
              'font.family': 'serif',
              'axes.spines.top': False,
              'axes.spines.right': False,
              'axes.labelsize': labelsize,
              'axes.titlesize': titlesize,
              'xtick.labelsize': ticksize,
              'ytick.labelsize': ticksize,
              'figure.figsize': (3.44556, 3.44556 / 1.61803),
              }
    plt.rcParams.update(params)


def get_figsize(width=1/2, ratio=GOLDEN_RATIO, base_figsize=DEFAULT_FIGSIZE):
    """ Get the figure size for a Latex document in inches

    Args:
        width (float): width in terms of textwidth in a Latex doc that the figure should occupy
        base_figsize (float): width of the page
        ratio (float): height = width / ratio

    Returns:
        (width, height) tuple of floats
    """
    return base_figsize * width, base_figsize * width / ratio


def inset_axes(self, bounds, *, polar=True, transform=None, zorder=5,
               **kwargs):
    """ Serves as a stand-in for Axes.inset_axes, but also allows polar projection

    Args:
        self (Axes): Axes object
        bounds (list): (x, y, xsize, ysize)
        polar (bool): polar projection
        transform (Transfomr): Defaults to `ax.transAxes`, i.e. the units of *rect* are in
            axes-relative coordinates.
        zorder (int): Defaults to 5 (same as `.Axes.legend`).  Adjust higher or lower
            to change whether it is above or below data plotted on the
            parent axes.
        **kwargs: Other *kwargs* are passed on to the `axes.Axes` child axes.

    Returns:
        Axes - The created `.axes.Axes` instance.
    """
    if transform is None:
        transform = self.transAxes
    label = kwargs.pop('label', 'inset_axes')

    # This puts the rectangle into figure-relative coordinates.
    inset_locator = _make_inset_locator(bounds, transform, self)
    bb = inset_locator(None, None)

    if polar:
        inset_ax = PolarAxes(self.figure, bb.bounds, zorder=zorder,
                             label=label, **kwargs)
    else:
        inset_ax = Axes(self.figure, bb.bounds, zorder=zorder,
                        label=label, **kwargs)

    # this locator lets the axes move if in data coordinates.
    # it gets called in `ax.apply_aspect() (of all places)
    inset_ax.set_axes_locator(inset_locator)

    self.add_child_axes(inset_ax)

    return inset_ax
