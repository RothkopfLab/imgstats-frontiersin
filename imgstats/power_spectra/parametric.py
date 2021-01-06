import numpy as np
from scipy.optimize import least_squares, minimize, Bounds

#PARAS_LO = dict(A=1e2, a=1e-3, theta=0., alpha=0.9, B=1e-8, beta=0.9)
#PARAS_HI = dict(A=1e4, a=1, theta=np.pi, alpha=3., B=0.01, beta=5)

PARAS_LO = dict(A=1e2, a=1e-3, theta=0., alpha=0.9)
PARAS_HI = dict(A=1e4, a=1, theta=np.pi, alpha=3.)


def parametric_ps(size, A=5200, a=0.7, theta=0, alpha=1.4, B=0., beta=0.):
    """ Parametric function for fitting power spectra from Pamplona et al.
        (sum of scaled hyperbola and elliptical power law)

    Args:
        size (int or tuple): size of quadratic image or (height, width) for rectangular image
        A (float): overall magnitude
        a (float): shape of the ellipse
        theta (float): angle of ellipse (in radians)
        alpha (float): 1/f exponent
        B (float): weight of hyperbola relative to elliptical component
        beta (float): exponent of hyperbola

    Returns:
        PS(x, y) (np.array): parametric power law, shape determined by size parameter
    """
    if isinstance(size, tuple) and len(size) == 2:
        h, w = size
    elif isinstance(size, int):
        h, w = size, size
    else:
        raise ValueError("Size must be either a tuple of size 2 or an int.")

    half_h = int(h / 2)
    half_w = int(w / 2)
    x, y = np.mgrid[-half_h:half_h, -half_w:half_w]

    fxr = x * np.cos(theta) + y * np.sin(theta)
    fyr = -x * np.sin(theta) + y * np.cos(theta)
    ellipse = (fxr ** 2 + fyr ** 2 / a) ** (-alpha)
    hyperbola = np.abs(x * y) ** (-beta)
    return A * ((1 - B) * ellipse + B * hyperbola)


def residuals(params, B, beta, power, fx, fy, indices):
    """ Element-wise differences between a power spectrum and a parametric fit

    Args:
        params (list): parameters [A, a, theta, alpha, B, beta] for a call to parametric_ps()
        power (np.array): power spectrum

    Returns:
        residuals (np.array): 1d array of shape (power.size,)
    """

    # A, a, theta, alpha, B, beta = params
    A, a, theta, alpha = params
    fxr = fx * np.cos(theta) + fy * np.sin(theta)
    fyr = -fx * np.sin(theta) + fy * np.cos(theta)
    ellipse = (fxr ** 2 + fyr ** 2 / a) ** (-alpha)
    hyperbola = np.abs(fx * fy) ** (-beta)
    f = A * ((1 - B) * ellipse + B * hyperbola)

    residuals = np.log10(f[indices]) - np.log10(power[indices])
    # residuals = np.log10(f) - np.log10(power)

    return residuals.flatten()


def squared_diff(params, power):
    return np.sum(residuals(params, power) ** 2)


def fit_power_spectrum(ps, x0=None, bounds=None):
    """ Fit a parametric function to the power spectra

    Args:
        ps (np.array): 2d power spectrum
        x0 (dict): dictionary of intial values, keys [A, a, theta, alpha, B, beta]
        bounds (tuple): tuple of two dicts (lower and upper bounds), keys [A, a, theta, alpha, B, beta]

    Returns:
        scipy OptimizeResult (cf https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
    """
    # x_init = dict(A=1000, a=0.99, theta=np.deg2rad(90), alpha=1., B=2e-6, beta=1.1)
    x_init = dict(A=1000, a=0.99, theta=np.deg2rad(90), alpha=1.)
    if x0:
        x_init.update(x0)

    x_lo = PARAS_LO
    x_hi = PARAS_HI
    if bounds:
        x_lo.update(bounds[0])
        x_hi.update(bounds[1])

    size = ps.shape
    h, w = size
    half_h = int(h / 2)
    half_w = int(w / 2)

    fx, fy = np.mgrid[-half_h:half_h, -half_w:half_w]
    ffrs = np.sqrt(fx ** 2 + fy ** 2)

    # f = np.delete(np.delete(f, half_h, axis=0), half_w, axis=1)
    # power = np.delete(np.delete(power, half_h, axis=0), half_w, axis=1)
    # ffrs = np.delete(np.delete(ffrs, half_h, axis=0), half_w, axis=1)

    indices = np.nonzero(np.logical_and.reduce([ffrs > 5, ffrs < min(half_h / 2, half_w / 2), fx != 0, fy != 0]))

    sol = least_squares(residuals, x0=np.array(list(x_init.values())), args=(0, 0, ps, fx, fy, indices), ftol=1e-12,
                        bounds=(np.array(list(x_lo.values())), np.array(list(x_hi.values()))))
    # sol = minimize(squared_diff, x0=np.array(list(x_init.values())), args=(ps,),
    #               bounds=Bounds(list(x_lo.values()), list(x_hi.values())))

    return sol


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # parameter bounds
    #paras_lo = dict(A=1e2, a=1e-3, theta=0., alpha=0.9, B=1e-8, beta=0.9)
    #paras_hi = dict(A=1e4, a=1, theta=np.pi, alpha=3., B=0.01, beta=5)

    paras_lo = dict(A=1e2, a=1e-3, theta=0., alpha=0.9)
    paras_hi = dict(A=1e4, a=1, theta=np.pi, alpha=3.)

    # draw a random set of parameters within the bounds
    params = {key: np.random.uniform(paras_lo[key], paras_hi[key]) for key in paras_lo.keys()}
    # params = {A=200, B=0.01, a=0.3, theta=np.deg2rad(45), alpha=0.9, beta=1.4)
    ps = parametric_ps(512, **params)
    log_ps = np.log10(ps)

    f, axes = plt.subplots(1, 2)
    axes[0].imshow(log_ps, cmap="cool", vmin=log_ps.min(), vmax=log_ps[np.isfinite(log_ps)].max())

    sol = fit_power_spectrum(ps)
    print("True values: A = {:.2f}, a = {:.2f}, theta = {:.2f}, alpha = {:.2f}".format(
        *params.values()))
    print("   Estimate: A = {:.2f}, a = {:.2f}, theta = {:.2f}, alpha = {:.2f}".format(
        *sol["x"]))

    optpower = parametric_ps(512, *sol["x"])

    axes[1].imshow(np.log10(optpower), cmap="cool", vmin=log_ps.min(), vmax=log_ps[np.isfinite(log_ps)].max())
    # axes[1].colorbar()
    plt.show()
