import os
import numpy as np
from functools import partial

from menpo.base import name_of_callable
from menpo.shape import bounding_box
from menpofit.visualize import print_progress
from menpo.visualize import print_dynamic
from menpo.transform import Scale
from menpo.image import Image

from .correlationfilter import CorrelationFilter
from .normalisation import (normalise_norm_array, image_normalisation,
                            create_cosine_mask)
from .feature import fast_dsift_hsi
from .result import DetectionResult, print_str, ClassificationResult


def data_dir_path():
    r"""
    The path to the data folder.

    :type: `pathlib.Path`
    """
    from pathlib import Path  # to avoid cluttering the menpo.base namespace
    return Path(os.path.abspath(__file__)).parent / 'data'


def load_pretrained_model():
    r"""
    Method that loads a pretrained classification model.

    :type: `Classification`
    """
    import menpo.io as mio
    return mio.import_pickle(data_dir_path() / 'pretrained_model.pkl')


def get_bounding_box(center, shape):
    r"""
    Method that returns a bounding box PointDirectedGraph, given the box center
    and shape.

    Parameters
    ----------
    center : (`float`, `float`)
        The box center.
    shape : (`int`, `int`)
        The box shape.

    Returns
    -------
    bbox : `menpo.shape.PointDirectedGraph`
        The bounding box
    """
    half_size = np.asarray(shape) / 2
    return bounding_box((center[0] - half_size[0], center[1] - half_size[1]),
                        (center[0] + half_size[0], center[1] + half_size[1]))


def response_thresholding(response, threshold, filter_shape, scale,
                          correction_transform):
    r"""
    Method for selecting candidate detections by thresholding the response map.
    The bounding boxes of these detections are transformed back to the original
    image resolution.

    Parameters
    ----------
    response : `ndarray`
        The response map.
    threshold : `float`
        The score threshold to use selecting candidate locations.
    filter_shape : (`int`, `int`)
        The shape of the filter.
    scale : `float`
        The current scale factor.
    correction_transform : `menpo.transform.AffineTransform`
        The transform object to go back to the original image resolution.

    Returns
    -------
    bboxes : `list` of `menpo.shape.PointDirectedGraph`
        The list of selected bounding boxes in the original image resolution.
    scores : `list`
        The corresponding scores.
    """
    # Find all response values abave threshold
    all_x, all_y = np.nonzero(response >= threshold)
    # Find corresponding scores
    scores = response[response >= threshold]
    # Create bounding boxes for the above candidate detections. Note that the
    # bounding boxes need to be transformed to the original image resolution.
    bboxes = []
    for x, y in zip(all_x, all_y):
        # Get bounding box at current scale
        bbox = get_bounding_box((x, y), filter_shape)
        # Transform bounding box to original scale (scale = 1)
        bbox = Scale(1 / scale, n_dims=2).apply(bbox)
        # Apply the correction affine transform to go to original image
        # resolution
        if correction_transform is not None:
            bbox = correction_transform.apply(bbox)
        bboxes.append(bbox)
    return bboxes, list(scores)


def non_max_suppression(bboxes, scores, overlap_thresh):
    r"""
    Faster Non-Maximum Suppression by Malisiewicz et al.

    Parameters
    ----------
    bboxes : `list` of `menpo.shape.PointDirectedGraph`
        The candidate bounding boxes.
    scores : `list` of `float`
        The corresponding scores per bounding box.
    overlap_thresh : `float`
        The overlapping threshold.

    Returns
    -------
    bboxes : `list` of `menpo.shape.PointDirectedGraph`
        The list of final bounding boxes in the original image resolution.
    scores : `list`
        The corresponding scores.
    """
    # Malisiewicz et al. method.
    # if there are no boxes, return an empty list
    if len(bboxes) == 0:
        return [], []

    # grab the coordinates of the bounding boxes
    x1 = np.empty((len(bboxes), 1))
    y1 = np.empty((len(bboxes), 1))
    x2 = np.empty((len(bboxes), 1))
    y2 = np.empty((len(bboxes), 1))
    for i, b in enumerate(bboxes):
        x1[i] = np.min(b.points[:, 0])
        y1[i] = np.min(b.points[:, 1])
        x2[i] = np.max(b.points[:, 0])
        y2[i] = np.max(b.points[:, 1])
    sc = np.asarray(scores)   # score confidence

    # initialize the list of picked indexes
    pick = []

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(sc)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return [bboxes[i] for i in pick], [scores[i] for i in pick]


def attach_bboxes_to_image(image, bboxes):
    r"""
    Method that attaches the given bounding boxes to the landmark manager of the
    provided image.
    """
    for i, bbox in enumerate(bboxes):
        image.landmarks['bbox_{:0{}d}'.format(i, len(str(len(bboxes))))] = bbox


class Detector(object):
    r"""
    Class for training a multi-channel correlation filter object detector.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The training images from which to learn the detector.
    algorithm : ``{'mosse', 'mccf'}``, optional
        If 'mosse', then the Minimum Output Sum of Squared Errors (MOSSE)
        filter [1] will be used. If 'mccf', then the Multi-Channel Correlation
        (MCCF) filter [2] will be used.
    filter_shape : (`int`, `int`), optional
        The shape of the filter.
    features : `callable`, optional
        The holistic dense features to be extracted from the images.
    normalisation : `callable`, optional
        The callable to be used for normalising the images.
    cosine_mask : `bool`, optional
        If ``True``, then a cosine mask (Hanning window) will be applied on the
        images.
    response_covariance : `int`, optional
        The covariance of the Gaussian desired response that will be used during
        training of the correlation filter.
    l : `float`, optional
        Regularization parameter of the correlation filter.
    boundary : ``{'constant', 'symmetric'}``, optional
        Determines the type of padding that will be applied on the images.
    prefix : `str`, optional
        The prefix of the progress bar.
    verbose : `bool`, optional
        If ``True``, then a progress bar is printed.

    References
    ----------
    .. [1] D. S. Bolme, J. R. Beveridge, B. A. Draper, and Y. M. Lui. "Visual
        Object Tracking using Adaptive Correlation Filters", IEEE Proceedings
        of International Conference on Computer Vision and Pattern Recognition
        (CVPR), 2010.
    .. [2] H. K. Galoogahi, T. Sim, and Simon Lucey. "Multi-Channel
        Correlation Filters". IEEE Proceedings of International Conference on
        Computer Vision (ICCV), 2013.
    """
    def __init__(self, images, algorithm='mosse', filter_shape=(25, 25),
                 features=fast_dsift_hsi, normalisation=normalise_norm_array,
                 cosine_mask=False, response_covariance=2, l=0.01,
                 boundary='symmetric', prefix='', verbose=True):
        # Assign properties
        self.algorithm = algorithm
        self.features = features
        self.filter_shape = filter_shape
        self.normalisation = normalisation
        self.cosine_mask = cosine_mask
        self.boundary = boundary

        # Create cosine mask if asked
        cosine_mask = None
        if cosine_mask:
            cosine_mask = create_cosine_mask(filter_shape)

        # Prepare data
        wrap = partial(print_progress, prefix=prefix + 'Pre-processing data',
                       verbose=verbose, end_with_newline=False)
        normalized_data = []
        for im in wrap(images):
            im = features(im)
            im = image_normalisation(im, normalisation=normalisation,
                                     cosine_mask=cosine_mask)
            normalized_data.append(im.pixels)

        # Create data array
        normalized_data = np.asarray(normalized_data)

        # Train correlation filter
        self.model = CorrelationFilter(
            normalized_data, algorithm=algorithm, filter_shape=filter_shape,
            response_covariance=response_covariance, l=l, boundary=boundary,
            prefix=prefix, verbose=verbose)

    @property
    def n_channels(self):
        r"""
        Returns the model's number of channels.

        :type: `int`
        """
        return self.model.n_channels

    def detect(self, image, scales='all', diagonal=400, score_thresh=0.025,
               overlap_thresh=0.1, return_responses=False, prefix='Detecting ',
               verbose=True):
        r"""
        Perform detection in a test image.

        Parameters
        ----------
        image : `menpo.image.Image`
            The test image.
        scales : `list` of `float` or ``'all'`` or None, optional
            The scales on which to apply the detection. The scales must be
            defined with respect to the original image resolution (after the
            diagonal normalisation). If ``None``, then no pyramid is used. If
            ``'all'``, then ``scales = np.arange(0.05, 1.05, 0.05)``.
        diagonal : `float` or ``None``, optional
            The diagonal to which the input image will be rescaled before the
            detection.
        score_thresh : `float`, optional
            The threshold to use for the response map (scores).
        overlap_thresh: `float`, optional,
            The overlapping threshold of non-maximum suppression.
        return_responses : `bool`, optional
            If ``True``, then the response maps per scale will be stored and
            returned.
        prefix : `str`, optional
            The prefix of the progress bar.
        verbose : `bool`, optional
            If ``True``, a progress bar is printed.

        Returns
        -------
        result : `DetectionResult`
            A detection result object.
        """
        # Normalise the input image size with respect to diagonal.
        # Keep the transform object, because we need to transform it back after
        # the detection is done.
        if diagonal is not None:
            tmp_image, correction_transform = image.rescale_to_diagonal(
                diagonal, return_transform=True)
        else:
            tmp_image = image
            correction_transform = None

        # Parse scales argument
        if scales == 'all':
            scales = tuple(np.arange(0.05, 1.05, 0.05))
        elif scales is None:
            scales = [1.]

        # Compute features of the original image
        feat_image = self.features(tmp_image)

        # Initialize lists
        selected_bboxes = []
        selected_scores = []
        responses = None
        if return_responses:
            responses = []

        # Get response and candidate bounding boxes at each scale
        wrap = partial(print_progress, prefix=prefix, verbose=verbose,
                       end_with_newline=False, show_count=False)
        for scale in wrap(list(scales)[::-1]):
            # Scale image
            if scale != 1:
                # Scale feature image only if scale is different than 1
                scaled_image = feat_image.rescale(scale)
            else:
                # Otherwise the image remains the same
                scaled_image = feat_image

            # Normalise the scaled image. Do not use cosine mask.
            scaled_image = image_normalisation(
                scaled_image, normalisation=self.normalisation, cosine_mask=None)

            # Convolve image with filter
            response = self.model.convolve(scaled_image, as_sum=True)
            if return_responses:
                responses.append(Image(response))

            # Threshold the response and transform resulting bounding boxes to
            # original image resolution
            bboxes, scores = response_thresholding(
                response, score_thresh, self.filter_shape, scale,
                correction_transform)

            # Updated selected bboxes and scores lists
            selected_bboxes += bboxes
            selected_scores += scores

        # Perform non-maximum suppression
        bboxes, scores = non_max_suppression(selected_bboxes, selected_scores,
                                             overlap_thresh)

        if verbose:
            print_dynamic(print_str(bboxes, len(scales)))

        # Return detection result object
        return DetectionResult(image, bboxes, scores, scales, responses)

    def view_spatial_filter(self, figure_id=None, new_figure=False,
                            channels='all', interpolation='bilinear',
                            cmap_name='afmhot', alpha=1., render_axes=False,
                            axes_font_name='sans-serif', axes_font_size=10,
                            axes_font_style='normal', axes_font_weight='normal',
                            axes_x_limits=None, axes_y_limits=None,
                            axes_x_ticks=None, axes_y_ticks=None,
                            figure_size=(10, 8)):
        r"""
        View the multi-channel filter on the spatial domain.

        Parameters
        ----------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        channels : `int` or `list` of `int` or ``all`` or ``None``
            If `int` or `list` of `int`, the specified channel(s) will be
            rendered. If ``all``, all the channels will be rendered in subplots.
            If ``None`` and the image is RGB, it will be rendered in RGB mode.
            If ``None`` and the image is not RGB, it is equivalent to ``all``.
        interpolation : See Below, optional
            The interpolation used to render the image. For example, if
            ``bilinear``, the image will be smooth and if ``nearest``, the
            image will be pixelated.
            Example options ::

                {none, nearest, bilinear, bicubic, spline16, spline36,
                hanning, hamming, hermite, kaiser, quadric, catrom, gaussian,
                bessel, mitchell, sinc, lanczos}
        cmap_name: `str`, optional,
            If ``None``, single channel and three channel images default
            to greyscale and rgb colormaps respectively.
        alpha : `float`, optional
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : See Below, optional
            The font of the axes.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : {``normal``, ``italic``, ``oblique``}, optional
            The font style of the axes.
        axes_font_weight : See Below, optional
            The font weight of the axes.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                semibold, demibold, demi, bold, heavy, extra bold, black}

        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the Image as a percentage of the Image's width. If
            `tuple` or `list`, then it defines the axis limits. If ``None``, then
            the limits are set automatically.
        axes_y_limits : (`float`, `float`) `tuple` or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the Image as a percentage of the Image's height. If
            `tuple` or `list`, then it defines the axis limits. If ``None``, then
            the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) `tuple` or ``None``, optional
            The size of the figure in inches.

        Returns
        -------
        viewer : `ImageViewer`
            The image viewing object.
        """
        return self.model.view_spatial_filter(
            figure_id=figure_id, new_figure=new_figure, channels=channels,
            interpolation=interpolation, cmap_name=cmap_name, alpha=alpha,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
            axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
            axes_y_ticks=axes_y_ticks, figure_size=figure_size)

    def view_frequency_filter(self, figure_id=None, new_figure=False,
                              channels='all', interpolation='bilinear',
                              cmap_name='afmhot', alpha=1., render_axes=False,
                              axes_font_name='sans-serif', axes_font_size=10,
                              axes_font_style='normal', axes_font_weight='normal',
                              axes_x_limits=None, axes_y_limits=None,
                              axes_x_ticks=None, axes_y_ticks=None,
                              figure_size=(10, 8)):
        r"""
        View the multi-channel filter on the frequency domain.

        Parameters
        ----------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        channels : `int` or `list` of `int` or ``all`` or ``None``
            If `int` or `list` of `int`, the specified channel(s) will be
            rendered. If ``all``, all the channels will be rendered in subplots.
            If ``None`` and the image is RGB, it will be rendered in RGB mode.
            If ``None`` and the image is not RGB, it is equivalent to ``all``.
        interpolation : See Below, optional
            The interpolation used to render the image. For example, if
            ``bilinear``, the image will be smooth and if ``nearest``, the
            image will be pixelated.
            Example options ::

                {none, nearest, bilinear, bicubic, spline16, spline36,
                hanning, hamming, hermite, kaiser, quadric, catrom, gaussian,
                bessel, mitchell, sinc, lanczos}
        cmap_name: `str`, optional,
            If ``None``, single channel and three channel images default
            to greyscale and rgb colormaps respectively.
        alpha : `float`, optional
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : See Below, optional
            The font of the axes.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : {``normal``, ``italic``, ``oblique``}, optional
            The font style of the axes.
        axes_font_weight : See Below, optional
            The font weight of the axes.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                semibold, demibold, demi, bold, heavy, extra bold, black}

        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the Image as a percentage of the Image's width. If
            `tuple` or `list`, then it defines the axis limits. If ``None``, then
            the limits are set automatically.
        axes_y_limits : (`float`, `float`) `tuple` or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the Image as a percentage of the Image's height. If
            `tuple` or `list`, then it defines the axis limits. If ``None``, then
            the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) `tuple` or ``None``, optional
            The size of the figure in inches.

        Returns
        -------
        viewer : `ImageViewer`
            The image viewing object.
        """
        return self.model.view_frequency_filter(
            figure_id=figure_id, new_figure=new_figure, channels=channels,
            interpolation=interpolation, cmap_name=cmap_name, alpha=alpha,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
            axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
            axes_y_ticks=axes_y_ticks, figure_size=figure_size)

    def __str__(self):
        output_str = r"""Correlation Filter Detector
 - Features: {}
 - Channels: {}
 """.format(name_of_callable(self.features), self.n_channels)
        return output_str + self.model.__str__()


class Classification(object):
    r"""
    Class for training a filter-bank of multi-channel correlation filters for
    object classification.

    Parameters
    ----------
    images : `list` of `list` of `menpo.image.Image`
        The training images per class.
    labels : `list` of `str`
        The label per class.
    algorithm : ``{'mosse', 'mccf'}``, optional
        If 'mosse', then the Minimum Output Sum of Squared Errors (MOSSE)
        filter [1] will be used. If 'mccf', then the Multi-Channel Correlation
        (MCCF) filter [2] will be used.
    filter_shape : (`int`, `int`), optional
        The shape of the filter.
    features : `callable`, optional
        The holistic dense features to be extracted from the images.
    normalisation : `callable`, optional
        The callable to be used for normalising the images.
    cosine_mask : `bool`, optional
        If ``True``, then a cosine mask (Hanning window) will be applied on the
        images.
    response_covariance : `int`, optional
        The covariance of the Gaussian desired response that will be used during
        training of the correlation filter.
    l : `float`, optional
        Regularization parameter of the correlation filter.
    boundary : ``{'constant', 'symmetric'}``, optional
        Determines the type of padding that will be applied on the images.
    prefix : `str`, optional
        The prefix of the progress bar.
    verbose : `bool`, optional
        If ``True``, then a progress bar is printed.

    References
    ----------
    .. [1] D. S. Bolme, J. R. Beveridge, B. A. Draper, and Y. M. Lui. "Visual
        Object Tracking using Adaptive Correlation Filters", IEEE Proceedings
        of International Conference on Computer Vision and Pattern Recognition
        (CVPR), 2010.
    .. [2] H. K. Galoogahi, T. Sim, and Simon Lucey. "Multi-Channel
        Correlation Filters". IEEE Proceedings of International Conference on
        Computer Vision (ICCV), 2013.
    """
    def __init__(self, images, labels, algorithm='mosse',
                 filter_shape=(29, 29), features=fast_dsift_hsi,
                 normalisation=normalise_norm_array, cosine_mask=False,
                 response_covariance=2, l=0.01, boundary='symmetric',
                 verbose=True):
        # Check images
        if len(images) != len(labels):
            raise ValueError('The provided images and labels have different '
                             'number of classes.')

        # Assign properties
        self.algorithm = algorithm
        self.features = features
        self.filter_shape = filter_shape
        self.normalisation = normalisation
        self.cosine_mask = cosine_mask
        self.boundary = boundary
        self.labels = labels
        self.n_classes = len(labels)

        # Train filters
        self.models = []
        for cl in range(self.n_classes):
            class_str = 'Class {}: '.format(cl)
            detector = Detector(
                images[cl], algorithm=algorithm, filter_shape=filter_shape,
                features=features, normalisation=normalisation,
                cosine_mask=cosine_mask, response_covariance=response_covariance,
                l=l, boundary=boundary, prefix=class_str, verbose=verbose)
            self.models.append(detector)

    def fit(self, image, scales='all', diagonal=400, score_thresh=0.025,
            overlap_thresh=0.1, return_all_detections=True, verbose=True):
        r"""
        Fit a test image.

        Parameters
        ----------
        image : `menpo.image.Image`
            The test image.
        scales : `list` of `float` or ``'all'`` or None, optional
            The scales on which to apply the detection. The scales must be
            defined with respect to the original image resolution (after the
            diagonal normalisation). If ``None``, then no pyramid is used. If
            ``'all'``, then ``scales = np.arange(0.05, 1.05, 0.05)``.
        diagonal : `float` or ``None``, optional
            The diagonal to which the input image will be rescaled before the
            detection.
        score_thresh : `float`, optional
            The threshold to use for the response map (scores).
        overlap_thresh: `float`, optional,
            The overlapping threshold of non-maximum suppression.
        return_all_detections : `bool`, optional
            If ``True``, then all the detections from all filters will be
            returned.
        verbose : `bool`, optional
            If ``True``, a progress bar is printed.

        Returns
        -------
        result : `DetectionResult`
            A detection result object.
        """
        # Initialize lists
        all_bboxes = []
        all_scores = []
        all_classnames = []
        # initialize final result
        classname = None
        bbox = None
        max_score = -np.inf
        results = []
        # For each class filter
        for cl in range(self.n_classes):
            # Perform detection
            result = self.models[cl].detect(
                image, scales=scales, diagonal=diagonal,
                return_responses=False, score_thresh=score_thresh,
                overlap_thresh=overlap_thresh,
                prefix="Filter '{}'".format(self.labels[cl]), verbose=verbose)
            # If at least one bounding box was returned, then check if there is
            # a score larger than the current maximum.
            if len(result.scores) > 0:
                if np.max(result.scores) > max_score:
                    max_score = np.max(result.scores)
                    classname = self.labels[cl]
                    idx = np.argmax(result.scores)
                    bbox = result.bboxes[idx]
                if return_all_detections:
                    all_bboxes += result.bboxes
                    all_scores += result.scores
                    all_classnames += [self.labels[cl]] * len(result.bboxes)
            results.append(result)

        if verbose:
            if classname is not None:
                print_dynamic("Detected class: '{}'".format(classname))
            else:
                print_dynamic('No detections.')

        # Return all detected results, if required
        all_detections = None
        if return_all_detections:
            all_detections = (all_bboxes, all_scores, all_classnames)

        # Return a classification result object
        return ClassificationResult(image, bbox, classname, scales, self.labels,
                                    all_detections=all_detections)

    def view_spatial_filters_widget(self, browser_style='buttons',
                                    figure_size=(10, 8), style='coloured'):
        r"""
        Visualize the spatial filters using an interactive widget.

        Parameters
        ----------
        browser_style : {``'buttons'``, ``'slider'``}, optional
            It defines whether the selector of the images will have the form of
            plus/minus buttons or a slider.
        figure_size : (`int`, `int`), optional
            The initial size of the rendered figure.
        style : {``'coloured'``, ``'minimal'``}, optional
            If ``'coloured'``, then the style of the widget will be coloured. If
            ``minimal``, then the style is simple using black and white colours.
        """
        filters = [Image(m.model.correlation_filter) for m in self.models]
        try:
            from menpowidgets import visualize_images
            visualize_images(filters, figure_size=figure_size,
                             style=style, browser_style=browser_style)
        except ImportError:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def view_frequency_filters_widget(self, browser_style='buttons',
                                      figure_size=(10, 8), style='coloured'):
        r"""
        Visualize the frequency filters using an interactive widget.

        Parameters
        ----------
        browser_style : {``'buttons'``, ``'slider'``}, optional
            It defines whether the selector of the images will have the form of
            plus/minus buttons or a slider.
        figure_size : (`int`, `int`), optional
            The initial size of the rendered figure.
        style : {``'coloured'``, ``'minimal'``}, optional
            If ``'coloured'``, then the style of the widget will be coloured. If
            ``minimal``, then the style is simple using black and white colours.
        """
        filters = []
        for m in self.models:
            freq_f = np.abs(np.fft.fftshift(np.fft.fft2(m.model.correlation_filter)))
            filters.append(Image(freq_f))
        try:
            from menpowidgets import visualize_images
            visualize_images(filters, figure_size=figure_size,
                             style=style, browser_style=browser_style)
        except ImportError:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def __str__(self):
        output_str = r"""Filter-bank of Correlation Filters Classification
 - Classes: {}
   - {}
 - Features: {}
 """.format(self.n_classes, self.labels, name_of_callable(self.features))
        return output_str + self.models[0].__str__()
