import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from projection import hom2img_matrix, angles2projective, angles2sphere


def spherical_image(img, r, phi, theta, fov=np.deg2rad(120)):
    """ Transform a planar image to spherical coordinates

    Args:
        img: (2d np.array) input image (squre image is assumed)
        r: radius of the eyeball in mm
        phi, theta: np.meshgrid from eccentricity and polar angle ranges
        fov: total field of view in radians

    Returns:
        spherical_img (np.array, same shape as phi and theta)
    """

    # get size of image (we assume a square image)
    total_size = img.shape[0]

    # get the matrix that transforms from world to pixel coordinates
    K = hom2img_matrix(r, total_size, fov)

    # get the coordinates of the input angles
    # on the projective plane at the back of the eyeball
    px = angles2projective(r * np.ones_like(phi), phi, theta)
    # set the third coordinate to 1
    px[2] = 1.

    # compute the x and y pixel coordinates of the input angles
    x, y, depth = np.tensordot(K, px, axes=1)

    # x and y index of the actual image
    idx = np.arange(total_size)
    # get an interpolatable version of the image
    f = interpolate.interp2d(idx, idx, img)

    # initialize empty array
    spherical_img = np.zeros_like(phi)
    # for each coordinate
    for i in range(spherical_img.shape[0]):
        for j in range(spherical_img.shape[1]):
            # get the interpolated pixel value
            # IMPORTANT: scipy has the indices the other way around compared to numpy arrays in this case!
            spherical_img[i, j] = f(y[i, j], x[i, j])[0]

    return spherical_img


if __name__ == "__main__":
    import os
    import skimage
    from projection.plot import plot_spherical_img

    # Here, we are testing the projection onto the sphere
    # First we generate dots on an image with known eccentricities using angles2projective -> px
    # We can also compute the 3d coordinates on the sphere using angles2sphere -> ps
    # We then transform the image with the known coordinates using spherical_image()
    # Finally we plot the true points on the sphere (ps) and compare

    # initialize figure (for test image)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 8)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    r = 16.67 / 2
    # build a grid in polar coordinates (with test values at different eccentricities and polar angles)
    phi = np.linspace(0.0, np.deg2rad(50), 6)
    theta = np.linspace(0.0, 2.0 * np.pi, 9)
    phigrid, thetagrid = np.meshgrid(phi, theta)

    # get points on image plane
    px = angles2projective(r * np.ones_like(phigrid), phigrid, thetagrid)
    px[2] = 1.

    # get points on the sphere (plot these for comparison later)
    # slightly larger radius so the points are visible over the actual sphere
    ps = angles2sphere(1.01 * r * np.ones_like(phigrid), phigrid, thetagrid)

    # half the fov
    ecc_max = np.deg2rad(120) / 2

    # get maximal coordinates on image plane
    x_max, _, _ = angles2projective(r, ecc_max, 0.)

    # adapt axes and save image
    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-x_max, x_max)

    # scatter points on the image plane
    plt.scatter(px[0], px[1], marker="x", s=200)

    # save, load and delete the file (TODO: we should be able to skip this intermediate step)
    fig.savefig("grid.png")
    img = skimage.io.imread("grid.png", as_gray=True)
    os.remove("grid.png")

    # coordinate grid
    phi = np.linspace(0.0, np.deg2rad(60), 200)
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    phigrid, thetagrid = np.meshgrid(phi, theta)
    # phigrid, thetagrid = np.mgrid[0.0:np.rad2deg(50):100j, 0.0:2.0*pi:100j]

    # grid into spherical coordinates
    x, y, z = angles2sphere(r, phigrid, thetagrid)

    # compute the spherical image
    spherical_img = spherical_image(img, r, phigrid, thetagrid)

    fig, ax = plot_spherical_img(img, r, phicount=100, thetacount=100)

    # plot the spherical points (should be same points as image)
    ax.scatter(ps[0], ps[2], ps[1], s=100)
    plt.show()
