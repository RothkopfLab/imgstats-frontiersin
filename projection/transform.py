import json
import os

import numpy as np
import skimage
import skimage.io
import skimage.transform

from projection import homography_matrix


def transform_samples(dataset, patch_size=128):
    """ Transform a folder of samples from the virtual environment.
    The folder needs to have subfolders with the following name structure:
        ecc30_polar45 (for samples taken at 30 deg eccentricity, 45 deg polar angle)

    The output is then in os.path.join(root_dir, dataset)+"-transformed"

    Args:
        dataset: (str) path to the dataset relative to the root directory
        patch_size: (int) patch size on tangential plane

    Returns:
        None
    """
    input_dir = dataset
    output_dir = dataset + "-transformed"

    with open(os.path.join(input_dir, "info.txt")) as info_file:
        params = json.load(info_file)

    r = params["eyeRadius"]
    fov = np.deg2rad(params["fieldOfView"])
    img_size = params["imageSize"]

    directories = next(os.walk(input_dir))[1]
    for current_dir in directories:
        if "full" in current_dir:
            continue
        ecc, polar = [
            float("".join(x for x in name if x.isdigit())) for name in current_dir.split("_")
        ]
        filenames = next(os.walk(os.path.join(input_dir, current_dir)))[2]
        filenames = [filename for filename in filenames if ".png" in filename]
        for filename in filenames:
            filepath = os.path.join(input_dir, current_dir, filename)

            # print("Loading {} at ecc = {}, polar = {}".format(filename, ecc, polar))
            img = skimage.io.imread(filepath, as_gray=False)
            img = img[:, :, 1]  # use only luminance channel

            H = homography_matrix(r=r, ecc=np.deg2rad(ecc), polar=np.deg2rad(polar), fov=fov, total_img_size=img_size,
                                  projective_patch_size=img.shape[0], tangential_patch_size=patch_size)

            tform = skimage.transform.ProjectiveTransform(H)

            img_warped = skimage.transform.warp(img, tform.inverse, output_shape=(patch_size, patch_size))
            img_warped = 255 * img_warped  # Now scale by 255
            img_warped = img_warped.astype(np.uint8)

            output_path = os.path.join(output_dir, current_dir, filename)
            if not os.path.exists(os.path.join(output_dir, current_dir)):
                os.makedirs(os.path.join(output_dir, current_dir))
            if not skimage.exposure.is_low_contrast(img_warped):
                skimage.io.imsave(output_path, img_warped)
