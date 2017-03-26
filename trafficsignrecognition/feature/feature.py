import numpy as np

from menpo.image import Image
from menpo.feature import ndfeature, igo, fast_dsift, no_op

from ..normalisation import image_normalisation


@ndfeature
def rgb2hsi(pixels):
    r"""
    Converts an RGB image to HSI.

    Parameters
    ----------
    pixels : `menpo.image.Image` or subclass or ``(3, X, Y)`` `ndarray`
        Either the menpo image object itself or an array where the first
        dimension is interpreted as the 3 RGB channels.

    Returns
    -------
    hsi : `menpo.image.Image` or subclass or ``(3, X, Y)`` `ndarray`
        The 3-channels HSI image.
    """
    r = pixels[0, ...]
    g = pixels[1, ...]
    b = pixels[2, ...]

    # Implement the conversion equations
    eps = 2 ** -50
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    theta = np.arccos(num / (den + eps))

    # Compute hue channel
    h = theta
    h[b > g] = 2 * np.pi - h[b > g]
    h /= 2 * np.pi

    # Compute Saturation channel
    num = np.minimum(np.minimum(r, g), b)
    den = r + g + b
    den[den == 0] = eps
    s = 1 - 3 * num / den

    h[s == 0] = 0

    # Compute Intensity channel
    i = (r + g + b) / 3

    hsi = np.zeros((3,) + pixels.shape[1:])
    hsi[0] = h
    hsi[1] = s
    hsi[2] = i

    return hsi


@ndfeature
def rgb_hsi(pixels):
    r"""
    Concatenates the RGB and HSI channels.

    Parameters
    ----------
    pixels : `menpo.image.Image` or subclass or ``(3, X, Y)`` `ndarray`
        Either the menpo image object itself or an array where the first
        dimension is interpreted as the 3 RGB channels.

    Returns
    -------
    hsi : `menpo.image.Image` or subclass or ``(6, X, Y)`` `ndarray`
        The 6-channels image that occurs by concatenating RGB and HSI.
    """
    return np.concatenate((pixels, rgb2hsi(pixels)))


@ndfeature
def igo_hsi(pixels):
    r"""
    Concatenates the HSI and IGO channels.

    Parameters
    ----------
    pixels : `menpo.image.Image` or subclass or ``(3, X, Y)`` `ndarray`
        Either the menpo image object itself or an array where the first
        dimension is interpreted as the 3 RGB channels.

    Returns
    -------
    hsi : `menpo.image.Image` or subclass or ``(5, X, Y)`` `ndarray`
        The 5-channels image that occurs by concatenating IGO and HSI.
    """
    igo_pixels = igo(Image(pixels).as_greyscale()).pixels
    return np.concatenate((igo_pixels, rgb2hsi(pixels)))


@ndfeature
def fast_dsift_hsi(pixels):
    r"""
    Concatenates the HSI and (fast) Dense SIFT channels.

    Parameters
    ----------
    pixels : `menpo.image.Image` or subclass or ``(3, X, Y)`` `ndarray`
        Either the menpo image object itself or an array where the first
        dimension is interpreted as the 3 RGB channels.

    Returns
    -------
    hsi : `menpo.image.Image` or subclass or ``(11, X, Y)`` `ndarray`
        The 11-channels image that occurs by concatenating Dense SIFT and HSI.
    """
    return np.concatenate((fast_dsift(pixels), rgb2hsi(pixels)))


def image_pyramid(image, scales, features=no_op, normalisation=no_op):
    r"""
    Create a feature-based pyramid for the provided image. Note that the features
    are computed only on the initial resolution and then the pyramid is created
    by rescaling the features image.

    Parameters
    ----------
    image : `menpo.image.Image` or subclass
        The input menpo image object.
    scales : `tuple` of `float`
        The scales to be used. They must be defined with respect to the image's
        original resolution.
    features : `callable`, optional
        The features to be used. The must be densely sampled features. It must
        accept and return a `menpo.image.Image`.
    normalisation : `callable`, optional
        A method that performs some kind of normalisation. It must accept and
        return a `menpo.image.Image`.

    Returns
    -------
    image_pyramid : `list` of ``(C, X, Y)`` `ndarray`
        The `list` of feature-based images per pyramid scale.
    """
    images = []
    for i in range(len(scales)):
        # If this is the first pass through the loop, then extract features
        if i == 0:
            feature_image = features(image)

        # Rescale image
        if scales[i] != 1:
            # Scale feature image only if scale is different than 1
            scaled_image = feature_image.rescale(scales[i])
        else:
            # Otherwise the image remains the same
            scaled_image = feature_image

        # Normalise the scaled image. Do not use cosine mask.
        scaled_image = image_normalisation(scaled_image,
                                           normalisation=normalisation,
                                           cosine_mask=None)

        # Add scaled image to list
        images.append(scaled_image.pixels)

    return images
