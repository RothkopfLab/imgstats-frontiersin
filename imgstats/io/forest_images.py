import os
import itertools

import numpy as np
import skimage
import skimage.io

from imgstats.utils import rgb2luminance, divisive_normalize, zscore, minmax_normalize


def angle_pairs(eccentricities=None, polars=None):
    """ Get a list of tuples containing eccentricity and polar angle pairs.
        Computes the product of the two lists, but removes all pairs with zero eccentricity but nonzero polar angle
        (to avoid duplicates, since when the eccentricity is zero, the polar angle does not matter)

    Args:
        eccentricities (list of int): list of eccentricities
        polars (list of int): list of polar angles

    Returns:
        angles (list of tuples)
    """
    if not eccentricities:
        eccentricities = [0, 30, 50]
    if not polars:
        polars = [0, 45, 90, 135, 180, 225, 270, 315]

    angles = itertools.product(eccentricities, polars)
    angles = [(ecc, pol) for (ecc, pol) in angles if not (ecc == 0 and pol != 0)]

    return angles


def load_forest_images(dataset, ecc, polar, root_path="/data/forest-scenes", normalize=True, size=None):
    """ Load a set of forest images from the specified folder (subfolder determined by ecc and polar)

    Args:
        dataset (str): path relative to root_path
        ecc (int): eccentricity
        polar (int): polar angle
        root_path (str): base directory
        normalize (bool): normalize gray scale images by mean luminance
        size (int): only take a central square patch of size size

    Returns:
        np.array (n_images, img_size, img_size)
    """
    full_path = os.path.join(root_path, dataset, 'ecc{}_polar{}'.format(ecc, polar))

    files = [f for f in os.listdir(full_path) if f.endswith('.png')]
    images = np.array([skimage.io.imread(os.path.join(full_path, f)) for f in files])

    # convert to grayscale
    if images.ndim == 4:
        images = np.array([rgb2luminance(img, normalize=normalize) for img in images])
    elif images.ndim == 3:
        if normalize:
            images = np.array([divisive_normalize(img) for img in images])

    if size:
        n, sx, sy = images.shape
        midx, midy = (int(s / 2) for s in [sx, sy])
        images = images[:, midx - size / 2:midx + size / 2, midy - size / 2:midy + size / 2]

    return images
