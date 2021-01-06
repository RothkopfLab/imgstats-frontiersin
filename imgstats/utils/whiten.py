import numpy as np


def whiten(img):
    """ Whiten an image using the procedure described by Olshausen & Field (1996)

    Args:
        img (np.array): square image patch

    Returns:
        Whitened image
    """
    N, _ = img.shape

    fx, fy = np.mgrid[-N / 2:N / 2, -N / 2:N / 2]
    rho = np.sqrt(fx ** 2 + fy ** 2)
    f_0 = 0.4 * N
    filt = rho * np.exp(-(rho / f_0) ** 4)

    If = np.fft.fft2(img)
    imagew = np.real(np.fft.ifft2(If * np.fft.fftshift(filt)))

    return imagew


if __name__ == "__main__":
    # TODO: write script that whitens a dataset
    pass
