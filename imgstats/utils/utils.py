import numpy as np
import skimage
import pandas as pd


def divisive_normalize(img):
    """ Normalize an image, dividing it by its mean intentsity

    Args:
        img (np.array): an image

    Returns:
        normalized image
    """

    norm_img = img / img.mean()
    return norm_img


def zscore(img):
    norm_img = (img - img.mean()) / img.std()
    return norm_img


def minmax_normalize(img):
    norm_img = (img - img.min()) / (img.max() - img.min())
    return norm_img


def rgb2luminance(img, normalize=False):
    """ Convert from rbg images to xyz colorspace and keep only luminance (y component)

    Args:
        img (np.array): input image
        normalize (bool): normalize the luminance channel

    Returns:
        grayscale image
    """
    # convert from rgb to xyz
    img = skimage.color.rgb2xyz(img)
    # normalize by mean luminance
    if normalize:
        img /= img[:, :, [1]].mean()
    # use only luminance
    img = img[:, :, 1]
    if normalize:
        img = (img - img.min()) / (img.max() - img.min())
    return img


def prep_movmf_df(df):
    movmf = df.copy()

    movmf[[("i", 0), ("i", 1)]] = pd.DataFrame(movmf["pi"].apply(np.argsort).tolist(), index=movmf.index)

    movmf[("mu", 0)] = pd.concat([movmf["mu"].apply(lambda x: x[i])[movmf[("i", 0)] == i] for i in range(2)])
    movmf[("mu", 1)] = pd.concat([movmf["mu"].apply(lambda x: x[i])[movmf[("i", 1)] == i] for i in range(2)])

    movmf[("kappa", 0)] = pd.concat(
        [movmf["kappa"].apply(lambda x: x[i])[movmf[("i", 0)] == i] for i in range(2)])
    movmf[("kappa", 1)] = pd.concat(
        [movmf["kappa"].apply(lambda x: x[i])[movmf[("i", 1)] == i] for i in range(2)])

    movmf[("pi", 0)] = pd.concat([movmf["pi"].apply(lambda x: x[i])[movmf[("i", 0)] == i] for i in range(2)])
    movmf[("pi", 1)] = pd.concat([movmf["pi"].apply(lambda x: x[i])[movmf[("i", 1)] == i] for i in range(2)])

    movmf.drop(["pi", "mu", "kappa", ("i", 0), ("i", 1)], axis=1, inplace=True)

    movmf.columns = pd.MultiIndex.from_tuples(movmf.columns, names=['Parameter', 'Index'])

    movmf.index.names = ["ecc", "pol"]
    movmf.reset_index(inplace=True)

    return movmf
