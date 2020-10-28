import numpy as np


def make_grid(size=512, linewidth=4, spacing=64):
    """ Make a square grid with white background and black lines

    Args:
        size: (int) size in pixels
        linewidth: (int) width of black lines in pixels
        spacing: (int) space between lines in pixels

    Returns:
        2d np array
    """
    img = np.ones((size, size))
    for i, x in enumerate(img):
        if i % spacing == 0:
            img[i - int(linewidth / 2):i + int(linewidth / 2), :] = 0
            img[:, i - int(linewidth / 2):i + int(linewidth / 2)] = 0

    return img
