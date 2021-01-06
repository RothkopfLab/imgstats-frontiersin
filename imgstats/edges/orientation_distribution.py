import os

import numpy as np
from astropy.stats import vonmisesmle
from scipy.ndimage import convolve
# from spherecluster import VonMisesFisherMixture

from skimage.filters.edges import farid_v, farid_h
from spherecluster import VonMisesFisherMixture

from imgstats.power_spectra.average import normalize
from imgstats.utils import rgb2luminance


def orientation_tensor(img, k=5):
    """ Compute the edges tensor (i.e. the local covariance matrix of the gradient vectors
        computed by averaging their outer products over square regions)

    Args:
        img (np.array): input image
        k (int): size of the kernel for averaging

    Returns:
        Hxx, Hxy, Hyy: edges tensor
    """

    # compute horizontal and vertical derivatives
    dh = farid_h(img)
    dv = farid_v(img)

    # box kernel for averaging
    mean_kernel = np.ones((k, k)) / 9

    # outer product of image derivatives, averaged within k by k regions
    Hxx = convolve(dh * dh, mean_kernel)
    Hxy = convolve(dh * dv, mean_kernel)
    Hyy = convolve(dv * dv, mean_kernel)

    return Hxx, Hxy, Hyy


def orientation_properties(img, k=5):
    """ Compute the energy (the sum of eigenvalues), the orientedness (the eigenvalue difference divided by the sum)
        and the dominant edges (the angle of the leading eigenvector), based on an image's structur tensor

    Args:
        image (np.array): input image
        k (int): size of the kernel for averaging (when computing edges tensor)

    Returns:
        energy, orientedness, angle
    """
    # compute and reshape edges tensor
    Hxx, Hxy, Hyy = orientation_tensor(img=img, k=k)
    H = np.array([[Hxx, Hxy], [Hxy, Hyy]])
    H = H.transpose(2, 3, 0, 1)

    # eigenvalue decomposition
    W, V = np.linalg.eigh(H[2:-2, 2:-2])

    # compute the three properties as described above
    energy = W.sum(axis=2)
    orientedness = (W[:, :, 1] - W[:, :, 0]) / (W[:, :, 1] + W[:, :, 0])
    angle = np.arctan2(V[:, :, 1, 1], V[:, :, 0, 1])

    return energy, orientedness, angle


def dominant_orientations(img, energy_percentile=68, min_orientedness=0.8, normalize=True):
    """ This function implements the procedure for computing edges histograms
        from Girshick, Landy & Simoncelli (2011).

    Args:
        img (np.nd_array): input image, can be either gray value (2d) or RGB (3d)
        energy_percentile (int): only pixels with energy higher than this percentile (0 - 100) are used
        min_orientedness (float): only pixels with orientedness higher than this (0 - 1) are used
        normalize (bool): normalize gray scale images by mean luminance

    Returns:
        (list of 1d np.arrays): all edge angles above the specified thresholds at each Gaussian Pyramid level
    """

    if len(img.shape) == 3:
        # convert from rgb to xyz
        img = rgb2luminance(img, normalize=normalize)

    # compute energy, orientedness and angle from the structur tensor
    energy, orientedness, angle = orientation_properties(img)

    # choose only those pixels where the energy is above some percentile and the orientedness above some threshold
    theta = np.where((energy > np.percentile(energy, energy_percentile)) & (orientedness > min_orientedness), angle,
                     np.nan)
    return theta[~np.isnan(theta)] % np.pi


def fit_movmf(theta, **kwargs):
    """ Fit a mixture of 2 von-Mises-Fisher distributions

    Args:
        theta (np.array): angles
        **kwargs: keyword arguments to spherecluster.VonMisesFisherMixture()

    Returns:
        pi, mu, kappa
    """
    X = np.array([np.cos(theta * 2), np.sin(theta * 2)]).T

    vmf_soft = VonMisesFisherMixture(n_clusters=2, **kwargs)
    vmf_soft.fit(X)

    mu = np.arctan2(vmf_soft.cluster_centers_[:, 1], vmf_soft.cluster_centers_[:, 0])
    pi = vmf_soft.weights_
    kappa = vmf_soft.concentrations_

    return pi, mu, kappa


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy import stats
    from imgstats.io.load import read_vanhateren
    from imgstats.io import load_forest_images
    from imgstats.plot import set_default_params, get_figsize, polar_histogram

    set_default_params()

    base_path = "/home/dominik/HESSENBOX-DA/new-scenes"
    forest_dataset = f"{base_path}/human/output-Human"

    fig, axes = plt.subplots(ncols=2, figsize=get_figsize(), subplot_kw=dict(projection="polar"))
    # angle_fig, angle_ax = plt.subplots(1, 2, subplot_kw=dict(projection="polar"))

    # TODO: bring this into main script
    for i, dataset in enumerate(["forest", "vanhateren"]):

        if dataset == "forest":
            name = "Virtual"
            images = load_forest_images(forest_dataset, 0, 0, normalize=False)
            # images = images[0]
        elif dataset == "vanhateren":
            name = "Van Hateren"
            images = np.stack(
                [read_vanhateren(f"/home/dominik/HESSENBOX-DA/vanhateren/imk{str(i).zfill(5)}.iml") for i in
                 range(1, 277)])
            images = images[:, 256:768, 1024:1536]

        n_images, sx, sy = images.shape

        recompute_edges = True
        if recompute_edges:
            theta = np.concatenate([dominant_orientations(img, min_orientedness=0.2, energy_percentile=68) for img in images])
            np.save(os.path.join(forest_dataset, "{}_edges.npy".format(dataset)), theta)
        else:
            theta = np.load(os.path.join(forest_dataset, "{}_edges.npy".format(dataset)))


        ax = axes[i]

        n, bins, patches = polar_histogram(theta, ax=ax, bins=30, density=True)

        # do fitting only on part of the data
        theta = np.random.choice(theta, size=100_000)
        do_fit = True
        if do_fit:
            # fit von Mises distribution
            pi, mu, kappa = fit_movmf(theta, n_init=1, init="spherical-k-means")
            with open(os.path.join(forest_dataset, "{}_movmf_params.txt".format(dataset)), "w+") as outfile:
                outfile.write("{}, {}, {}\n".format(pi[0], mu[0], kappa[0]))
                outfile.write("{}, {}, {}".format(pi[1], mu[1], kappa[1]))

            # plot von Mises fit to orientation histogram
            x = np.linspace(0, 2 * np.pi, num=51)
            y = pi[0] * stats.vonmises.pdf(x * 2, kappa=kappa[0], loc=mu[0]) + pi[1] * stats.vonmises.pdf(x * 2,
                                                                                                          kappa=kappa[
                                                                                                              1],
                                                                                                          loc=mu[1])
            ax.plot(x, y, color="C2")

    fig.show()
