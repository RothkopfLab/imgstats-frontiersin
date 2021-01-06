import numpy as np
import scipy
import scipy.ndimage


def radial_average(f, rmax=None):
    """ Compute the average along the radius of an image (or power spectrum)

    Args:
        f: np.array, input image or power spectrum

    Returns:
        radial_avg: 1d array with average values along the radius (shape depends on input image diagonal)
        idx: 1d array containing indices at which the average was evaluated
    """
    # make a meshgrid
    sx, sy = f.shape
    X, Y = np.ogrid[0:sx, 0:sy]

    # compute distance from center for every point in the grid
    r = np.hypot(X - sx / 2, Y - sy / 2)

    # bin into integer values (labels over which we compute the average)
    rbin = r.astype(np.int)

    # compute the radial average
    idx = np.arange(1, rbin.max() + 1)
    radial_avg = scipy.ndimage.mean(f, labels=rbin, index=idx)

    if not rmax:
        # use the size of the image divided by 4 as the maximum frequency
        rmax = np.minimum(sx, sy) / 4
    return radial_avg[idx < rmax], idx[idx < rmax]


def polar_average(f, dtheta=5, rmin=10, rmax=None):
    """ Compute the average of a power spectrum (or image) as a function of the polar angle.
    The average is computed over a range of radii specified by rmin and rmax.
    Adapted from adapted from https://medium.com/tangibit-studios/2d-spectrum-characterization-e288f255cc59

    Args:
        f (np.array): input image
        dtheta: size of angular bins
        rmin: minimal radius
        rmax: maximal radius

    Returns:
        polar_power: power as a function of the polar angle
        idx: angles (indices)
    """
    if f.ndim == 2:
        f = f[np.newaxis,:,:]

    h = f.shape[1]
    w = f.shape[2]
    wc = w // 2
    hc = h // 2

    if not rmax:
        rmax = int(np.minimum(wc, hc) / 2) + 1

    # note that displaying PSD as image inverts Y axis
    # create an array of integer angular slices of dTheta
    Y, X = np.ogrid[0:h, 0:w]
    theta = np.rad2deg(np.arctan2(-(Y - hc), (X - wc)))
    theta = np.mod(theta + dtheta / 2 + 360, 360)
    theta = dtheta * (theta // dtheta)
    theta = theta.astype(np.int)

    # mask below rmin and above rmax by setting to -100
    R = np.hypot(-(Y - hc), (X - wc))
    mask = np.logical_and(R > rmin, R < rmax)
    theta = theta + 100
    theta = np.multiply(mask, theta)
    theta = theta - 100

    idx = np.arange(0, 360, int(dtheta))

    output = np.zeros((f.shape[0], idx.size))

    for i, img in enumerate(f):
        # SUM all psd2D pixels with label 'theta' for 0<=theta❤60 between rMin and rMax
        output[i] = scipy.ndimage.mean(img, theta, index=idx)

    # normalize each sector to the total sector power
    # total_power = np.sum(polar_power)
    # polar_power = polar_power / total_power

    return output, idx


def weighted_average(images, weights):
    """ Compute the weighted average of a set of images

    Args:
        images (np.array): array of images (last two dimensions are image size dimensions)
        weights (np.array): array of the same shape as the images

    Returns:
        weighted average of the images (shape is maintained)
    """
    return (images * weights).sum(axis=(1, 2), keepdims=True) / weights.sum()


def normalize(images, weights=None):
    """

    Args:
        images (np.array, n x h x w): a set of images

    Returns:
        np.array, same shape as input: normalized images
    """

    # compute mean
    if weights is not None:
        mu = weighted_average(images, weights=weights)
    else:
        mu = np.mean(images, axis=(1, 2), keepdims=True)
    # subtract the average, normalize
    images = (images - mu) / mu

    return images