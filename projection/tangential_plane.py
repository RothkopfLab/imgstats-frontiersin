import numpy as np

from projection import angles2sphere
from projection.aux_functions import angles2projective, cartesian2spherical


def tangential_plane(r, ecc, polar):
    """ Compute the central point and normal vector of a tangential plane to a sphere
        at (r, ecc, polar)

    Args:
        r: radius of the sphere (mm)
        ecc: eccentricity (radians)
        polar: polar angle (radians)

    Returns:
        2d 3d vectors
        s: central point of tangential plane
        normal: normal vector of tangential plane
    """
    # center of the eyeball
    c = np.array([[0], [0], [-r]])

    # point at which the tangential plane is centered
    s = angles2sphere(r, ecc, polar)

    # radius vector
    normal = s - c

    return s, normal


def angles2tangential(ecc, polar, r0, ecc0, polar0):
    """ Convert from eccentricity and polar angle to xyz coordinates on the tangential plane

    Args:
        ecc: eccentricity (radians)
        polar: polar angle (radians)
        r0: radius of the sphere
        ecc0: eccentricity of tangential plane's center point
        polar0: polar angle of tangential plane's center point

    Returns:
        u: point on the tangential plane
    """
    s, n = tangential_plane(r0, ecc0, polar0)

    l = angles2sphere(r0, ecc, polar)

    # compute intersection of this vector with the tangential plane
    u = (s.T @ n) / (l.T @ n) * l

    return u


def tangential2projective(x, r):
    """ Convert from xyz coordinates (e.g. on a tangential plane) to xyz coordinates on the projective plane

    Args:
        x: 3d vector, point (x, y, z)
        r: radius of the eyeball

    Returns:
        xp: point on the projective plane
    """
    _, ecc, polar = cartesian2spherical(x)

    xp = angles2projective(r * np.ones_like(ecc), ecc, polar)

    return xp
