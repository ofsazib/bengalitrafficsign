import numpy as np
from scipy.stats import multivariate_normal

from menpofit.math.fft_utils import pad, crop


def centered_meshgrid(shape):
    r"""
    Method that generates a centered meshgrid.

    Parameters
    ----------
    shape : (`int`, `int`)
        The desired meshgrid shape.

    Returns
    -------
    grid : ``shape + (2,)`` `ndarray`
        The centered meshgrid.
    """
    # Compute rounded half of provided shape
    shape = np.asarray(shape)
    half_shape = np.floor(shape / 2)
    half_shape = np.require(half_shape, dtype=int)
    # Find start and end values for meshgrid
    start = -half_shape
    end = half_shape + shape % 2
    # Create grid
    sampling_grid = np.mgrid[start[0]:end[0], start[1]:end[1]]
    return np.rollaxis(sampling_grid, 0, 3)


def gaussian_response(shape, cov=2):
    r"""
    Method that returns a 2D gaussian response centered in the middle with the
    specified shape.

    Parameters
    ----------
    shape : (`int`, `int`)
        The desired shape.
    cov : `int`, optional
        The covariance of the normal distribution.

    Returns
    -------
    response : ``(1,) + shape`` `ndarray`
        The Gaussian response.
    """
    grid = centered_meshgrid(shape)
    return multivariate_normal(mean=np.zeros(2), cov=cov).pdf(grid)[None, ...]


def conv2d(image, f, mode='same', boundary='symmetric'):
    r"""
    Performs fast 2D convolution in the frequency domain. Note that if the input
    is multi-channel, then the convolution is performed per channel.

    Parameters
    ----------
    image : ``(C, Y, X)`` `ndarray`
        The input image.
    f : ``(C, Y, X)`` `ndarray`
        The filter to convolve with defined on the spatial domain.
    mode : {``full``, ``same``, ``valid``}, optional
        Determines the shape of the resulting convolution.
    boundary: str {``constant``, ``symmetric``}, optional
        Determines the padding applied on the image.

    Returns
    -------
    response : ``(C, Y, X)`` `ndarray`
        Result of convolving each image channel with its corresponding
        filter channel.
    """
    # Compute the extended shape
    image_shape = np.asarray(image.shape[-2:])
    filter_shape = np.asarray(f.shape[-2:])
    ext_shape = image_shape + filter_shape - 1

    # Pad image and f
    ext_image = pad(image, ext_shape, boundary=boundary)
    ext_filter = pad(f, ext_shape)

    # Compute ffts of extended image and extended f
    fft_ext_image = np.fft.fft2(ext_image)
    fft_ext_filter = np.fft.fft2(ext_filter)

    # Compute extended convolution in Fourier domain
    fft_ext_c = fft_ext_filter * fft_ext_image

    # Compute ifft of extended convolution
    ext_c = np.real(np.fft.ifftshift(np.fft.ifft2(fft_ext_c), axes=(-2, -1)))

    # Fix response shape.
    if mode is 'full':
        return ext_c
    elif mode is 'same':
        return crop(ext_c, image_shape)
    elif mode is 'valid':
        return crop(ext_c, image_shape - filter_shape + 1)
    else:
        raise ValueError("mode must be 'full', 'same' or 'valid'")
