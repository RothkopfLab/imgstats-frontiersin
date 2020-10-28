import numpy as np


def angles2sphere(r, ecc, polar):
    """ Convert from eccentricity and polar angle to spherical coordinates

    Args:
        r: eye radius
        ecc: eccentricity (radians)
        polar: polar angle (radians

    Returns:
        [x, y, z] coordinates on the sphere
    """
    z = -2 * r * np.cos(ecc) ** 2
    x = -z * np.tan(ecc) * np.cos(polar)
    y = x * np.tan(polar)

    if type(x) is not np.ndarray:
        x = np.array([x])
    if type(y) is not np.ndarray:
        y = np.array([y])
    if type(z) is not np.ndarray:
        z = np.array([z])

    return np.array([x, y, z])


def angles2projective(r, ecc, polar):
    """ Convert from eccentricity and polar angle to coordinates on the image plane

    Args:
        r: eye radius
        ecc: eccentricity (radians)
        polar: polar angle (radians

    Returns:
        [x, y, z] coordinates on the sphere
    """

    z = -2 * r
    x = -z * np.tan(ecc) * np.cos(polar)
    y = x * np.tan(polar)

    if type(x) is not np.ndarray:
        x = np.array([x])
    if type(y) is not np.ndarray:
        y = np.array([y])
    if type(z) is not np.ndarray:
        z = np.array([z])

    return np.array([x, y, z])


def hom2img_matrix(r, total_size, fov=np.deg2rad(120)):
    """ Compute the matrix that converts from homogeneous [x, y, 1]' coordinates to pixel coordinates

    Args:
        r: radius of the eye ball
        total_size: overall size of the image
        fov: overall field of view (matching the overall size)

    Returns:
        3x3 homogeneous transform matrix
    """
    ecc_max = fov / 2

    # get maximal coordinates on image plane
    x_max, y_max, z_max = angles2projective(r, ecc_max, 0.)

    # compute pixel size
    pixel_size = abs(2 * x_max / total_size)[0]

    mx = 1 / pixel_size

    K = np.array([[0, -mx, (1 + total_size) / 2],
                  [mx, 0, (1 + total_size) / 2],
                  [0, 0, 1]])

    return K


def sphere2projective(x):
    """ Convert a point on the sphere to a point on the projective plane

    Args:
        x: point on the sphere (3d vector or matrix consisting of 3d column vectors)

    Returns:
        point(s) on the projective plane
    """
    r_sqrt = np.sum(x ** 2, axis=0)
    norm = r_sqrt / x[2] ** 2

    p = x * norm
    return p


def cartesian2spherical(x):
    """ convert from cartesian coordinates to spherical coordinates

    Args:
        x: 3d vector, point (x, y, z)

    Returns:
        r, ecc, polar (spherical coordinates)
    """
    r = np.linalg.norm(x, axis=0)

    ecc = np.arccos(-x[2] / r)

    polar = np.arctan2(x[1], x[0])

    return r, ecc, polar


if __name__ == "__main__":
    r = np.array([100])
    ecc = np.array([0])
    polar = np.array([0])
    print(angles2projective(r=10, ecc=ecc, polar=polar))
