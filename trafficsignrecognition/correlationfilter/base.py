import numpy as np

from menpo.image import Image

from .correlationfilter import train_mosse, train_mccf
from .utils import gaussian_response, conv2d


class CorrelationFilter(object):
    r"""
    Class for training a multi-channel correlation filter.

    Parameters
    ----------
    images : ``(n_images, channels, height, width)`` `ndarray` or `list` of ``(channels, height, width)`` `ndarray`
        The training images from which to learn the filter.
    algorithm : ``{'mosse', 'mccf'}``, optional
        If 'mosse', then the Minimum Output Sum of Squared Errors (MOSSE)
        filter [1] will be used. If 'mccf', then the Multi-Channel Correlation
        (MCCF) filter [2] will be used.
    filter_shape : (`int`, `int`), optional
        The shape of the filter.
    response_covariance : `int`, optional
        The covariance of the Gaussian desired response that will be used during
        training.
    l : `float`, optional
        Regularization parameter.
    boundary : ``{'constant', 'symmetric'}``, optional
        Determines the type of padding that will be applied on the images.
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
    def __init__(self, images, algorithm='mosse', filter_shape=(64, 64),
                 response_covariance=2, l=0.01, boundary='symmetric',
                 prefix='', verbose=True):
        # Assign properties
        self.algorithm = algorithm
        self.response_covariance = response_covariance
        self.l = l
        self.boundary = boundary

        # Convert images to correct form
        if isinstance(images, list):
            images = np.asarray(images)
        self.n_training_images = images.shape[0]

        # Create desired Gaussian response
        self.desired_response = gaussian_response(filter_shape,
                                                  cov=response_covariance)

        # Train filter
        if algorithm == 'mosse':
            self.correlation_filter = train_mosse(
                images, self.desired_response, l=l, boundary=boundary,
                crop_filter=True, prefix=prefix, verbose=verbose)
        elif algorithm == 'mccf':
            self.correlation_filter = train_mccf(
                images, self.desired_response, l=l, boundary=boundary,
                crop_filter=True, prefix=prefix, verbose=verbose)
        else:
            raise ValueError("Algorithm can be either 'mosse' or 'mccf'.")

    @property
    def filter_shape(self):
        r"""
        Returns the filter' shape.

        :type: (`int`, `int`)
        """
        return self.correlation_filter.shape[1:]

    @property
    def n_channels(self):
        r"""
        Returns the filter' number of channels.

        :type: `int`
        """
        return self.correlation_filter.shape[0]

    def convolve(self, image, as_sum=True):
        r"""
        Method for convolving the correlation filter on a given image. The
        convolution is happening per channel and the method can optionally return
        the sum over the channels' convolutions.

        Returns
        -------
        image : `menpo.image.Image`
            The input image.
        as_sum : `bool`, optional
            If ``True``, then the sum of the convolution result over the channels
            is returned.

        Returns
        -------
        response : `ndarray`
            The response.
        """
        m_conv = conv2d(image.pixels, self.correlation_filter,
                        boundary=self.boundary)
        if as_sum:
            m_conv = np.sum(m_conv, axis=0)
        #m_conv = probability_map(m_conv, axes=(-2, -1))
        return m_conv

    def view_gaussian_response(self, figure_id=None, new_figure=False,
                               interpolation='bilinear', cmap_name='jet',
                               alpha=1., render_axes=True,
                               axes_font_name='sans-serif', axes_font_size=10,
                               axes_font_style='normal',
                               axes_font_weight='normal', axes_x_limits=None,
                               axes_y_limits=None, axes_x_ticks=None,
                               axes_y_ticks=None, figure_size=(10, 8)):
        r"""
        View the desired Gaussian response that was used during training.

        Returns
        -------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
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
        viewer : `menpo.visualize.ImageViewer`
            The image viewing object.
        """
        img = Image(self.desired_response)
        return img.view(
            figure_id=figure_id, new_figure=new_figure,
            interpolation=interpolation, cmap_name=cmap_name, alpha=alpha,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
            axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
            axes_y_ticks=axes_y_ticks, figure_size=figure_size)

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

        Returns
        -------
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
        return Image(self.correlation_filter).view(
            figure_id=figure_id, new_figure=new_figure,
            channels=channels, interpolation=interpolation, cmap_name=cmap_name,
            alpha=alpha, render_axes=render_axes, axes_font_name=axes_font_name,
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

        Returns
        -------
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
        freq_f = np.abs(np.fft.fftshift(np.fft.fft2(self.correlation_filter)))
        return Image(freq_f).view(
            figure_id=figure_id, new_figure=new_figure,
            channels=channels, interpolation=interpolation, cmap_name=cmap_name,
            alpha=alpha, render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
            axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
            axes_y_ticks=axes_y_ticks, figure_size=figure_size)

    def __str__(self):
        method_name = 'Minimum Output Sum of Squared Errors (MOSSE)'
        if self.algorithm == 'mccf':
            method_name = 'Multi-Channel Correlation (MCCF)'
        output_str = r"""Correlation Filter
 - {}
 - Shape: {}x{}
 - Response covariance: {}
 - Regularization parameter: {}
 - Boundary: {}
 - {} training samples""".format(method_name, self.filter_shape[0],
                                 self.filter_shape[1], self.response_covariance,
                                 self.l, self.boundary, self.n_training_images)
        return output_str
