import array

import numpy as np
import skimage
from PIL import Image


def read_vanhateren(filename, normalize=False):
    """ Read an image with the specified filename from the van Hateren database

    Args:
        filename (str): whole path to file

    Returns:
        np.array
    """
    with open(filename, 'rb') as handle:
        s = handle.read()

    arr = array.array('H', s)
    arr.byteswap()
    img = np.array(arr, dtype='uint16').reshape(1024, 1536)
    if normalize:
        img = img / img.mean()
    return img


def read_hyvarinen(filename):
    """ Read an image with the specified filename from the database used in Hyvärinen, Hurri & Hoyer

    Args:
        filename (str): whole path to file

    Returns:
        np.array
    """
    # read image
    img = np.array(Image.open(filename)) / 255
    return img


def read_image(filename):
    """ Read an image from the forest dataset
    
    Args:
        filename (str): whole path to file
        
    Returns:
        np.array
        
    """
    img = skimage.io.imread(filename, as_gray=True)
    return img
