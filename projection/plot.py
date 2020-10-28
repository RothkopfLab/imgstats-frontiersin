import numpy as np
from matplotlib import pyplot as plt, cm
from mpl_toolkits.mplot3d.art3d import Path3DCollection, Line3DCollection, Poly3DCollection
from math import sqrt
import matplotlib

from projection import angles2sphere, spherical_image
from projection.tangential_plane import angles2tangential


DEFAULT_FIGSIZE = 3.44556 * 2
GOLDEN_RATIO = 1.61803


def set_default_params():
    """ Set the parameters of the plots so that they look nice in my thesis.
        3.44556 inches is the width of one column (in 2-column layout) in my Latex document.

    Returns:
        None
    """
    params = {'text.usetex': True,
              'font.size': 10,
              'font.family': 'serif',
              'axes.spines.top': False,
              'axes.spines.right': False,
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


def plot_plane(point, normal, ax=None, xrange=(-8, 8), yrange=(-8, 8), **kwargs):
    """ Plot a plane in 3d

    Args:
        point: 3x1 np.array, point on the plane (3d)
        normal: 3x1 np.array, normal vector of the plane (3d)
        ax: matplotlib ax to plot on
        xrange, yrange: tuples for x and y range of plane to plot

    Returns:
        fig, ax: matplotlib figure and axes objects
    """
    if not ax:
        f, ax = plt.subplots()
    else:
        f = ax.figure

    d = - point.T @ normal

    # create x,y
    xx, yy = np.meshgrid(np.linspace(*xrange, 100), np.linspace(*yrange, 100))

    # calculate corresponding z
    zplane = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    ax.plot_wireframe(zplane, xx, yy, **kwargs)

    return f, ax


def plot_spherical_img(img, r, philim=(0.0, np.deg2rad(60)), phicount=50,
                       thetalim=(0.0, 2.0 * np.pi), thetacount=50, alpha=1., view_init=None, ax=None, figsize=(3.8456,3.8456)):
    """ Plot the image in 3d on the sphere

    Args:
        img: (2d np.array) input image
        r: (int) radius of the eyeball (sphere will be centered at [0, 0, -r]
        philim: (float, float) eccentricity limits in radians
        phicount: (int) number of points for meshgrid
        thetalim: (float, float) polar angle limits in radians
        thetacount: (int) number of points for meshgrid
        view_init: e.g. dict(azim=-30, elev=0), view direction

    Returns:
        fig, ax: matplotlib figure and axes objects
    """
    if not view_init:
        view_init = {}

    # setup meshgrid
    phi = np.linspace(philim[0], philim[1], phicount)
    theta = np.linspace(thetalim[0], thetalim[1], thetacount)
    phigrid, thetagrid = np.meshgrid(phi, theta)

    # convert to spherical coordinates
    x, y, z = angles2sphere(r, phigrid, thetagrid)

    # compute the spherical image
    spherical_img = spherical_image(img, r, phigrid, thetagrid)

    if ax is None:
        # initialize figure and 3d axis
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    # plot the image
    ax.plot_surface(
        z, x, y, facecolors=cm.gray(spherical_img), rstride=1, cstride=1, alpha=alpha, zorder=-1)

    # adjust axis
    ax.set_xlim([0, -2 * r])
    ax.set_ylim([r, -r])
    ax.set_zlim([-r, r])
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_zlabel("y")

    ax.view_init(**view_init)

    return fig, ax


def plot_tangential_img(img, r, ecc0, polar0, philim=(0, np.deg2rad(60)), phicount=100,
                        thetalim=(0, 2 * np.pi), thetacount=100, view_init=None, figsize=(3.8456,3.8456)):
    """ Plot the image on a tangential plane (in 3d)

    Args:
        img: (2d np.array) input image
        r: (float) radius of the eyeball (sphere will be centered at [0, 0, -r]
        ecc0: (float) eccentricity of the tangential plane center
        polar0: (float) polar angle of the tangential plane center
        philim: (float, float) eccentricity limits in radians
        phicount: (int) number of points for meshgrid
        thetalim: (float, float) polar angle limits in radians
        thetacount: (int) number of points for meshgrid
        view_init: e.g. dict(azim=-30, elev=0), view direction

    Returns:
        fig, ax: matplotlib figure and axes objects
    """
    if not view_init:
        view_init = {}

    # setup meshgrid
    phi = np.linspace(philim[0], philim[1], phicount)
    theta = np.linspace(thetalim[0], thetalim[1], thetacount)
    phigrid, thetagrid = np.meshgrid(phi, theta)

    # convert to coordinates on tangential plane
    tangential = np.zeros((3, *phigrid.shape))
    for (i, j), _ in np.ndenumerate(phigrid):
        tangential[:, i, j] = angles2tangential(phigrid[i, j], thetagrid[i, j], r, ecc0, polar0).flatten()
        # tangential[:,i,j] = angles2sphere(r, phigrid[i,j], thetagrid[i,j]).flatten()

    x, y, z = tangential

    # compute the spherical image
    spherical_img = spherical_image(img, r, phigrid, thetagrid)

    # initialize figure and 3d axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # plot the image
    # ax.plot_surface(
    #    z, x, y, facecolors=cm.gray(spherical_img), rstride=1, cstride=1, alpha=1., zorder=-1)

    # xs, ys, zs = angles2sphere(r, phigrid, thetagrid)
    # ax.plot_wireframe(zs, xs, ys, zorder=10)

    ax.plot_surface(
        z, x, y, facecolors=cm.gray(spherical_img), rstride=1, cstride=1, alpha=0.5)

    # adjust axis
    ax.set_xlim([0, -2 * r])
    ax.set_ylim([r, -r])
    ax.set_zlim([-r, r])
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_zlabel("y")

    ax.view_init(**view_init)

    return fig, ax


class FixZorderScatter(Path3DCollection):
    _zorder = 2000

    @property
    def zorder(self):
        return self._zorder

    @zorder.setter
    def zorder(self, value):
        pass
    
class FixZorderLine(Line3DCollection):
    _zorder = 2000

    @property
    def zorder(self):
        return self._zorder

    @zorder.setter
    def zorder(self, value):
        pass

class FixZorderPoly(Poly3DCollection):
    _zorder = 2000

    @property
    def zorder(self):
        return self._zorder

    @zorder.setter
    def zorder(self, value):
        pass


def fix_zorder(ax, index=-1):
    """ Fix the z order of a Patch3DCollection (e.g. scatter plot)

    Args:
        ax: axis that contains the Patch3DCollection
        index: index to ax.collections[index]

    Returns:
        ax: the fixed axis
    """
    if type(ax.collections[index]) == Path3DCollection:
        ax.collections[index].__class__ = FixZorderScatter
    elif type(ax.collections[index]) == Line3DCollection:
        ax.collections[index].__class__ = FixZorderLine
    elif type(ax.collections[index]) == Poly3DCollection:
        ax.collections[index].__class__ = FixZorderPoly
    else:
        raise TypeError("ax.collections[index] must be a Path3DCollection")

    return ax