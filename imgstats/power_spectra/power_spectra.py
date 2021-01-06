import numpy as np

from imgstats.power_spectra.average import radial_average, normalize, polar_average


def power_spectrum(image):
    """ Compute the power spectrum of an image

    Args:
        image: np.array, the last two dimension are the image dimensions, other dimensions are treated as batches

    Returns:
        np.array, same shape as the input: the power spectrum of the image (centered)
    """

    # squared absolute of the shifted fourier transform
    return np.abs(np.fft.fftshift(np.fft.fft2(image))) ** 2


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import skimage
    from imgstats.io.load import read_vanhateren
    from imgstats.io import load_forest_images
    from imgstats.power_spectra import window
    from imgstats.plot import set_default_params, get_figsize

    poster = False
    if poster:
        import seaborn as sns
        set_default_params()
        sns.set_context("poster")
    else:
        set_default_params(fontsize=8, labelsize=8, titlesize=10)

    tud3d = [c / 255 for c in [0, 113, 94]]

    colors = [tud3d, "C1"]

    forest_dataset = "human/output-Human"

    freq_fig, freq_ax = plt.subplots(figsize=get_figsize(width=.33,  base_figsize=15.199 if poster else 6.97522))
    spec_fig, spec_ax = plt.subplots(ncols=2, figsize=get_figsize(width=.33, base_figsize=15.199 if poster else 6.97522,
                                                                  ratio=3))
    angle_fig, angle_ax = plt.subplots(1, 2, subplot_kw=dict(projection="polar"))

    for i, dataset in enumerate(["forest", "vanhateren"]):

        if dataset == "forest":
            name = "Virtual"
            images = load_forest_images(forest_dataset, 0, 0, normalize=True,
                                        root_path="/home/dominik/HESSENBOX-DA/forest-scenes")
            # images = images[0]
            print(images.shape)
        elif dataset == "vanhateren":
            name = "Van Hateren"
            images = np.stack(
                [read_vanhateren(f"/home/dominik/HESSENBOX-DA/vanhateren/imk{str(i).zfill(5)}.iml") for i in
                 range(1, 277)])
            images = images[:, 256:768, 1024:1536]

        rotate = False
        if rotate:
            # rotate image by 45 degrees to check for grid artifacts
            images = np.array([skimage.transform.rotate(image, 45) for image in images])

        n_images, sx, sy = images.shape

        k2d = window.hamming2d(sx)

        images = normalize(images, weights=k2d)
        # apply the window
        images = images * k2d

        power = power_spectrum(image=images)

        avg_power = np.mean(power, axis=0)
        log_avg_power = np.log10(avg_power)

        windowed_spectrum = log_avg_power[128:385, 128:385]
        levels = [50, 70, 90]
        cs = spec_ax[i].contour(windowed_spectrum.T, color=colors[i], levels=np.percentile(windowed_spectrum, levels))

        spec_ax[i].axis("off")
        spec_ax[i].axis("equal")
        spec_ax[i].set_title(name)

        rotmean, r = radial_average(avg_power, rmax=75)

        logr = np.log10(r)
        logrotmean = np.log10(rotmean)

        m, b = np.polyfit(logr, logrotmean, deg=1)

        xx = np.linspace(0, 2., 100)

        freq_ax.scatter(r, rotmean, s=6, marker="o", label=name, color=colors[i])
        freq_ax.plot(np.power(10, xx), np.power(10, m * xx + b), label=r"$\alpha = {}$".format(np.round(-m, 2)),
                     color=colors[i])
        freq_ax.loglog(basex=10, basey=10)
        # freq_ax.set_ylim((10**(-7), 10 ** 1))
        freq_ax.set_xlim((0.5, 10**2.5))
        freq_ax.set_xticks([10**0, 10**1, 10**2])
        freq_ax.set_yticklabels([])
        freq_ax.set_xlabel("Radial Frequency")
        freq_ax.set_ylabel("Average Power")
        lgd = freq_ax.legend(frameon=False, ncol=1, loc="center left", bbox_to_anchor=(0.5, 0.8))


    if not poster:
        labels = [r"{}$\%$".format(p) for p in levels]
        for j in range(len(labels)):
            cs.collections[j].set_label(labels[j])
        spec_fig.legend(frameon=False, loc="lower center")
    spec_fig.tight_layout()
    if poster:
        spec_fig.savefig("/home/dominik/repos/msc-thesis/poster/figures/power-contour-central.eps", dpi=300)
    else:
        spec_fig.savefig("/home/dominik/repos/msc-thesis/figures/results/power_contour.eps", dpi=300)
        spec_fig.savefig("/home/dominik/repos/msc-thesis/figures/results/power_contour.pdf", dpi=300)
    spec_fig.show()

    freq_fig.tight_layout()
    if poster:
        pass
    else:
        freq_fig.savefig("/home/dominik/repos/msc-thesis/vss/radial_avg_power.eps", dpi=300)
        freq_fig.savefig("/home/dominik/repos/msc-thesis/vss/radial_avg_power.pdf", dpi=300)

    freq_fig.show()