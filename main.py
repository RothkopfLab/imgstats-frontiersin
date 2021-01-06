import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.stats import vonmisesmle

from imgstats.io import load_forest_images, angle_pairs
from imgstats.edges import dominant_orientations
from imgstats.plot import polar_histogram, polar_plotgrid

if __name__ == '__main__':

    base_path = "data/images/"
    dataset = "human-transformed"

    # angular positions in the visual field
    eccentricities = [0, 30, 50]
    polars = [0, 45, 90, 135, 180, 225, 270, 315]

    # generate pairs of eccentricity and polar angles
    angles = angle_pairs(eccentricities=eccentricities, polars=polars)

    # setup plot grid
    edge_fig, _, edge_axes = polar_plotgrid(eccentricities=eccentricities, polars=polars,
                                            subplot_polar=True)

    # initialize dicts for parameters to save
    vm_params = {}
    for ecc, pol in angles:
        print(f"Computing statistics for phi = {ecc}, chi = {pol}")
        # load image dataset
        images = load_forest_images(root_path=base_path, dataset=dataset, ecc=ecc, polar=pol, normalize=False)

        # compute edge histograms
        theta = np.concatenate(
            [dominant_orientations(img, min_orientedness=0.2, energy_percentile=68) for img in images])

        # fit von Mises distribution
        mu, kappa = vonmisesmle(theta * 2)
        vm_params[(ecc, pol)] = mu, kappa

        # plot histogram
        n, bins, patches = polar_histogram(theta, bins=20, density=True, ax=edge_axes[(ecc, pol)])

    # save von Mises fit parameters
    vm_params_df = pd.DataFrame(vm_params).T
    vm_params_df.columns = ["mu", "kappa"]
    vm_params_df.to_csv(f"{base_path}/{dataset}/vm_fits.csv")

    edge_fig.suptitle("Edge orientations")
    plt.show()
