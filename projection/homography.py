import numpy as np

from projection.aux_functions import angles2projective, cartesian2spherical
from projection.tangential_plane import tangential_plane, angles2tangential


def image_plane_matrix(r, ecc, polar):
    """ Compute the projective matrix for the image plane with an image center at the location
        specified by eccentricity and polar angle.

    Args:
        r: eye radius
        ecc: eccentricity (radians)
        polar: polar angle (radians)

    Returns:
        3x3 matrix B
    """

    # projection matrix for original image plane
    B = np.hstack([np.array([[1, 0, 0]]).T,
                   np.array([[0, 1, 0]]).T,
                   angles2projective(r, ecc, polar)])
    return B


def tangential_plane_matrix(r, ecc, polar):
    """ Compute the projective matrix for the tangential plane at the retina
        at a particular eccentricity and polar angle.

    Args:
        r: eye radius
        ecc: eccentricity (radians)
        polar: polar angle (radians)

    Returns:
        3x3 matrix A
    """
    s, n = tangential_plane(r, ecc, polar)

    # get central point in image plane coordinates
    pimg = angles2projective(r, ecc, polar)

    # vectors from origin to basis vectors in image plane
    l1 = pimg + np.array([[1], [0], [0]])
    l2 = pimg + np.array([[0], [1], [0]])

    # alternative way to do it (convert basis vectors to spherical coordinates and then to coordinates on the plane)
    _, ecc1, pol1 = cartesian2spherical(l1)
    _, ecc2, pol2 = cartesian2spherical(l2)

    u1 = angles2tangential(ecc1, pol1, r0=r, ecc0=ecc, polar0=polar)
    u2 = angles2tangential(ecc2, pol2, r0=r, ecc0=ecc, polar0=polar)

    # compute intersection of these vectors with the tangential plane
    #u1 = (s.T @ n) / (l1.T @ n) * l1
    #u2 = (s.T @ n) / (l2.T @ n) * l2

    # basis vectors in the tangential plane
    v1 = (u1 - s) / np.linalg.norm(u1 - s)
    v2 = (u2 - s) / np.linalg.norm(u2 - s)

    A = np.hstack([v1, v2, s])
    return A


def homography_matrix(r, ecc, polar, fov=np.deg2rad(120), projective_patch_size=512, tangential_patch_size=128,
                      total_img_size=5954, c=1):
    """ Compute the homography that relates image coordinates on the image plane to coordinates on the tangential plane.

    Args:
        r: eye radius
        ecc: eccentricity (radians)
        polar: polar angle (radians)
        fov: total field of view (radians)
        patch_size: size of the image patch on the image plane in pixels
        total_img_size: overall size of the image (whole fov) in pixels
        c: arbitrary scaling constant for the homography matrix

    Returns:
        3x3 homography matrix H
    """

    # maximal eccentricity is half the fov (symmetric visual field)
    ecc_max = fov / 2
    # get maximal coordinates on image plane
    x_max, y_max, z_max = angles2projective(r, np.array([ecc_max]), np.array([0]))

    # compute pixel size
    pixel_size = abs(2 * x_max / total_img_size)[0]

    # get transformation matrices
    B = image_plane_matrix(r, ecc, polar)
    A = tangential_plane_matrix(r, ecc, polar)

    mx = 1 / pixel_size

    Kb = np.array([[mx, 0, (1 + projective_patch_size) / 2],
                   [0, -mx, (1 + projective_patch_size) / 2],
                   [0, 0, 1]])

    Ka = np.array([[mx, 0, (1 + tangential_patch_size) / 2],
                   [0, -mx, (1 + tangential_patch_size) / 2],
                   [0, 0, 1]])

    H = c * Ka @ np.linalg.inv(A) @ B @ np.linalg.inv(Kb)

    return H
