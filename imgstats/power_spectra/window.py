import numpy as np


def kaiser2d(size, beta=2):
    """ Generate 2d square kaiser window

    Args:
        size (int): window size

    Returns:
        k2d (np.array): 2d kaiser window
    """
    k = np.kaiser(size, beta=beta)
    k2d = np.outer(k, k)[np.newaxis, :, :]
    k2d = k2d / k2d.sum()
    return k2d


def hamming2d(size):
    """ Generate 2d square kaiser window

    Args:
        size (int): window size

    Returns:
        k2d (np.array): 2d kaiser window
    """
    k = np.hamming(size)
    k2d = np.outer(k, k)[np.newaxis, :, :]
    k2d = k2d / k2d.sum()
    return k2d
