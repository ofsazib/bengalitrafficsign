import numpy as np

from menpo.shape import PointCloud


def rescale_wrt_min_dim(image, min_size):
    r"""
    Method that rescales a given image, so that its minimum dimension equals a
    given value.

    Parameters
    ----------
    image : `menpo.image.Image`
        The input menpo image object.
    min_size : `int`
        The desired size of the minimum dimension.

    Returns
    -------
    rescaled_image : `menpo.image.Image`
        The rescaled image.
    """
    # Compute the scale factor by enforcing that the minimum image
    # direction equals to the provided size.
    scale_factor = min_size / min(image.shape)
    # Rescale the image
    return image.rescale(scale_factor)


def random_centered_crops(image, crop_shape, n_crops):
    r"""
    Method that generates centered crops that are randomly sampled from a normal
    distribution. The method performs the sampling only over the maximum
    dimension, because it is assumed that the minimum dimension of the image
    equals to the crop shape of that dimension. This method can be used for data
    augmentation.

    Parameters
    ----------
    image : `menpo.image.Image`
        The input menpo image object.
    crop_shape : (`int`, `int`)
        The desired crop shape.
    n_crops : `int`
        The number of randomly sampled cropped images to generate. Note that the
        number of returned images is ``n_crops + 1``, as it also includes the
        region cropped around the image's center.

    Returns
    -------
    cropped_images : `list` of `menpo.image.Image`
        The `list` of cropped images.
    """
    # Check whether the image has the same shape as the desired crop shape,
    # in which case no crop should be performed.
    if image.shape == crop_shape:
        cropped_images = [image]
    else:
        # Get image centre
        centre = image.centre()
        # Get dimension over which to randomly sample
        max_dim = np.argmax(image.shape)
        # Sample from a normal distribution.
        # The mean of the distribution is the image center and the std the
        # margin that is left for sampling.
        gau_m = centre[max_dim]
        gau_std = np.sqrt(np.abs(image.shape[max_dim] - crop_shape[max_dim]))
        sample_centres = np.random.normal(gau_m, gau_std, n_crops)
        # Create array of offsets from the image center.
        sample_offsets = np.zeros((n_crops + 1, 2))
        sample_offsets[1:, max_dim] = centre[max_dim] - sample_centres
        # Extract cropped images
        cropped_images = image.extract_patches(
            PointCloud([centre]), patch_shape=crop_shape,
            sample_offsets=sample_offsets, as_single_array=False)
    return cropped_images


def data_augmentation(image, crop_shape, n_crops=2):
    r"""
    Method that generates augmented data from a given image. Specifically, it:

        1. Rescales the image so that the minimum dimension equals to the
           provided shape.
        2. Generates centered crops that are randomly sampled from a normal
           distribution. The sampling is performed only over the maximum
           dimension.

    Parameters
    ----------
    image : `menpo.image.Image`
        The input menpo image object.
    crop_shape : (`int`, `int`)
        The desired crop shape. It must be rectangular.
    n_crops : `int`, optional
        The number of randomly sampled cropped images to generate. Note that the
        number of returned images is ``n_crops + 1``, as it also includes the
        region cropped around the image's center.

    Returns
    -------
    augmented_data : `list` of `menpo.image.Image`
        The `list` of generated images.
    """
    if crop_shape[0] != crop_shape[1]:
        raise ValueError('Only rectangular crop shapes are supported.')
    image = rescale_wrt_min_dim(image, crop_shape[0])
    return random_centered_crops(image, crop_shape, n_crops)
