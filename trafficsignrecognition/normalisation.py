import numpy as np

from menpo.feature import ndfeature, no_op


def center_array(array):
    r"""
    Method that centers a given array.

    Parameters
    ----------
    array : `ndarray`
        The input array.

    Returns
    -------
    centered_array : `ndarray`
        The centered (zero mean) array.
    """
    return array - np.mean(array)


def normalise_norm_array(array):
    r"""
    Method that normalises a given array so that it has zero mean and unit norm.

    Parameters
    ----------
    array : `ndarray`
        The input array.

    Returns
    -------
    centered_array : `ndarray`
        The normalised array.
    """
    centered_arr = center_array(array)
    return centered_arr / np.linalg.norm(centered_arr)


def create_cosine_mask(shape):
    r"""
    Method that creates a cosine mask (Hanning function).

    Parameters
    ----------
    shape : (`int`, `int`)
        The mask's shape.

    Returns
    -------
    cosine_mask : `ndarray`
        The cosine mask with the specified shape.
    """
    cy = np.hanning(shape[0])
    cx = np.hanning(shape[1])
    return cy[..., None].dot(cx[None, ...])


@ndfeature
def image_normalisation(pixels, normalisation=normalise_norm_array,
                        cosine_mask=None):
    r"""
    Method that normalises an image by:

        * Applying a normalisation method (e.g. zero mean, unit norm)
        * Applying a cosine mask (Hanning window), if specified.

    Parameters
    ----------
    pixels : `menpo.image.Image` or subclass or ``(C, X, Y)`` `ndarray`
        Either the menpo image object itself or an array where the first
        dimension is interpreted as the channels.
    normalisation : `callable`, optional
        A method that performs some kind of normalisation. It must accept and
        return either a `menpo.image.Image` or a ``(C, X, Y)`` `ndarray`.
    cosine_mask : `ndarray` or ``None``, optional
        The cosine mask (Hanning window) to be applied on the image. If ``None``,
        then no mask is applied.

    Returns
    -------
    normalised_image : `menpo.image.Image` or subclass or ``(C, X, Y)`` `ndarray`
        The normalised image.
    """
    # Normalise image
    pixels = normalisation(pixels)
    # Apply cosine mask if specified
    if cosine_mask is not None:
        pixels = cosine_mask * pixels
    return pixels
