import matplotlib.pyplot as plt


def print_str(bboxes, n_scales):
    r"""
    Returns a detection result `str`.
    """
    msg = "No detections."
    sc_str = "after detecting in {} scales.".format(n_scales)
    if len(bboxes) == 1:
        msg = "1 detection " + sc_str
    elif len(bboxes) > 1:
        msg = "{} detections {}".format(len(bboxes), sc_str)
    return msg


class DetectionResult(object):
    r"""
    Class for storing a detection result.

    Parameters
    ----------
    image : `menpo.image.Image`
        The test image.
    bboxes : `list` of `menpo.shape.PointDirectedGraph`
        The detected bounding boxes.
    scores : `list` of `float`
        The scores that correspond to the bounding boxes.
    scales : `list` of `float`
        The pyramid's scale factors.
    responses : `list` of `menpo.image.Image` or ``None``, optional
        The responses per scale.
    """
    def __init__(self, image, bboxes, scores, scales, responses=None):
        self.image = image
        self.bboxes = bboxes
        self.scores = scores
        self.responses = responses
        self.scales = scales

    @property
    def n_scales(self):
        """
        Returns the number of scales.

        :type: `int`
        """
        return len(self.scales)

    def view(self, figure_id=None, new_figure=False, channels=None,
             interpolation='bilinear', cmap_name=None, alpha=1.,
             render_lines=True, line_colour='r', line_style='-', line_width=2,
             render_markers=True, marker_style='o', marker_size=5,
             marker_face_colour='r', marker_edge_colour='k',
             marker_edge_width=1., render_axes=False,
             axes_font_name='sans-serif', axes_font_size=10,
             axes_font_style='normal', axes_font_weight='normal',
             axes_x_limits=None, axes_y_limits=None,
             axes_x_ticks=None, axes_y_ticks=None,
             figure_size=(10, 8)):
        """
        Visualize the detection result. It renders the image and all the bounding
        boxes.

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
            image will be pixelated. Example options ::

                {none, nearest, bilinear, bicubic, spline16, spline36, hanning,
                hamming, hermite, kaiser, quadric, catrom, gaussian, bessel,
                mitchell, sinc, lanczos}

        cmap_name: `str`, optional,
            If ``None``, single channel and three channel images default
            to greyscale and rgb colormaps respectively.
        alpha : `float`, optional
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        render_lines : `bool`, optional
            If ``True``, the edges will be rendered.
        line_colour : See Below, optional
            The colour of the lines.
            Example options::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        line_style : ``{-, --, -., :}``, optional
            The style of the lines.
        line_width : `float`, optional
            The width of the lines.
        render_markers : `bool`, optional
            If ``True``, the markers will be rendered.
        marker_style : See Below, optional
            The style of the markers. Example options ::

                {., ,, o, v, ^, <, >, +, x, D, d, s, p, *, h, H, 1, 2, 3, 4, 8}

        marker_size : `int`, optional
            The size of the markers in points.
        marker_face_colour : See Below, optional
            The face (filling) colour of the markers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_colour : See Below, optional
            The edge colour of the markers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_width : `float`, optional
            The width of the markers' edge.
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : See Below, optional
            The font of the axes. Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : ``{normal, italic, oblique}``, optional
            The font style of the axes.
        axes_font_weight : See Below, optional
            The font weight of the axes.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                semibold,demibold, demi, bold, heavy, extra bold, black}

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
        figure_size : (`float`, `float`) `tuple` or ``None`` optional
            The size of the figure in inches.
        """
        viewer = self.image.view(
            figure_id=figure_id, new_figure=new_figure, channels=channels,
            interpolation=interpolation, cmap_name=cmap_name, alpha=alpha,
            render_axes=False)
        plt.title(print_str(self.bboxes, self.n_scales))
        if len(self.bboxes) == 0:
            return viewer
        if len(self.bboxes) > 2:
            for bb in self.bboxes[:-1]:
                bb.view(image_view=True, render_lines=render_lines,
                        line_colour=line_colour, line_style=line_style,
                        line_width=line_width, render_markers=render_markers,
                        marker_style=marker_style, marker_size=marker_size,
                        marker_face_colour=marker_face_colour,
                        marker_edge_colour=marker_edge_colour,
                        marker_edge_width=marker_edge_width,
                        render_numbering=False, render_axes=False)
        return self.bboxes[-1].view(
            image_view=True, render_lines=render_lines,
            line_colour=line_colour, line_style=line_style,
            line_width=line_width, render_markers=render_markers,
            marker_style=marker_style, marker_size=marker_size,
            marker_face_colour=marker_face_colour,
            marker_edge_colour=marker_edge_colour,
            marker_edge_width=marker_edge_width, render_numbering=False,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
            axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
            axes_y_ticks=axes_y_ticks, figure_size=figure_size)

    def view_response(self, scale_id=0, figure_id=None, new_figure=False,
                      channels=None, interpolation='bilinear', cmap_name='jet',
                      alpha=1., render_axes=False, axes_font_name='sans-serif',
                      axes_font_size=10, axes_font_style='normal',
                      axes_font_weight='normal', axes_x_limits=None,
                      axes_y_limits=None, axes_x_ticks=None, axes_y_ticks=None,
                      figure_size=(10, 8)):
        """
        Visualize responses, if they exist.

        Parameters
        ----------
        scale_id : `int`
            The scale index.
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
            image will be pixelated. Example options ::

                {none, nearest, bilinear, bicubic, spline16, spline36, hanning,
                hamming, hermite, kaiser, quadric, catrom, gaussian, bessel,
                mitchell, sinc, lanczos}

        cmap_name: `str`, optional,
            If ``None``, single channel and three channel images default
            to greyscale and rgb colormaps respectively.
        alpha : `float`, optional
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : See Below, optional
            The font of the axes. Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : ``{normal, italic, oblique}``, optional
            The font style of the axes.
        axes_font_weight : See Below, optional
            The font weight of the axes.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                semibold,demibold, demi, bold, heavy, extra bold, black}

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
        figure_size : (`float`, `float`) `tuple` or ``None`` optional
            The size of the figure in inches.
        """
        if self.responses is None:
            raise ValueError('Responses were not saved')
        return self.responses[scale_id].view(
            figure_id=figure_id, new_figure=new_figure, channels=channels,
            interpolation=interpolation, cmap_name=cmap_name, alpha=alpha,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
            axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
            axes_y_ticks=axes_y_ticks, figure_size=figure_size)

    def view_responses_widget(self, browser_style='buttons', figure_size=(10, 8),
                              style='coloured'):
        r"""
        Visualize the responses using an interactive widget, if they exist.

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
        if self.responses is None:
            raise ValueError('Responses were not saved')
        try:
            from menpowidgets import visualize_images
            visualize_images(self.responses, figure_size=figure_size,
                             style=style, browser_style=browser_style)
        except ImportError:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def __str__(self):
        return print_str(self.bboxes, self.n_scales)


class ClassificationResult(object):
    r"""
    Class for storing a classification result.

    Parameters
    ----------
    image : `menpo.image.Image`
        The test image.
    bbox : `menpo.shape.PointDirectedGraph`
        The detected bounding box.
    classname : `str`
        The name of the class that the object was classified.
    scales : `list` of `float`
        The pyramid's scale factors.
    labels : `list` of `str`
        The label of each potential class.
    all_detections : (`list`, `list`, `list`) or ``None``, optional
        All the detections from the filterbank. The first list has the bboxes,
        the second has the scores and the third has the classnames.
    """
    def __init__(self, image, bbox, classname, scales, labels,
                 all_detections=None):
        self.image = image
        self.bbox = bbox
        self.classname = classname
        self.scales = scales
        self.labels = labels
        self.all_detections = all_detections

    @property
    def n_scales(self):
        """
        Returns the number of scales.

        :type: `int`
        """
        return len(self.scales)

    def view(self, figure_id=None, new_figure=False, channels=None,
             interpolation='bilinear', cmap_name=None, alpha=1.,
             render_lines=True, line_colour='r', line_style='-', line_width=2,
             render_markers=True, marker_style='o', marker_size=5,
             marker_face_colour='r', marker_edge_colour='k',
             marker_edge_width=1., render_axes=False,
             axes_font_name='sans-serif', axes_font_size=10,
             axes_font_style='normal', axes_font_weight='normal',
             axes_x_limits=None, axes_y_limits=None,
             axes_x_ticks=None, axes_y_ticks=None,
             figure_size=(10, 8)):
        """
        Visualize the detection result. It renders the image and all the bounding
        boxes.

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
            image will be pixelated. Example options ::

                {none, nearest, bilinear, bicubic, spline16, spline36, hanning,
                hamming, hermite, kaiser, quadric, catrom, gaussian, bessel,
                mitchell, sinc, lanczos}

        cmap_name: `str`, optional,
            If ``None``, single channel and three channel images default
            to greyscale and rgb colormaps respectively.
        alpha : `float`, optional
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        render_lines : `bool`, optional
            If ``True``, the edges will be rendered.
        line_colour : See Below, optional
            The colour of the lines.
            Example options::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        line_style : ``{-, --, -., :}``, optional
            The style of the lines.
        line_width : `float`, optional
            The width of the lines.
        render_markers : `bool`, optional
            If ``True``, the markers will be rendered.
        marker_style : See Below, optional
            The style of the markers. Example options ::

                {., ,, o, v, ^, <, >, +, x, D, d, s, p, *, h, H, 1, 2, 3, 4, 8}

        marker_size : `int`, optional
            The size of the markers in points.
        marker_face_colour : See Below, optional
            The face (filling) colour of the markers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_colour : See Below, optional
            The edge colour of the markers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_width : `float`, optional
            The width of the markers' edge.
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : See Below, optional
            The font of the axes. Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : ``{normal, italic, oblique}``, optional
            The font style of the axes.
        axes_font_weight : See Below, optional
            The font weight of the axes.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                semibold,demibold, demi, bold, heavy, extra bold, black}

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
        figure_size : (`float`, `float`) `tuple` or ``None`` optional
            The size of the figure in inches.
        """
        viewer = self.image.view(
            figure_id=figure_id, new_figure=new_figure, channels=channels,
            interpolation=interpolation, cmap_name=cmap_name, alpha=alpha,
            render_axes=False)
        if self.bbox is not None:
            viewer = self.bbox.view(
                image_view=True, render_lines=render_lines,
                line_colour=line_colour, line_style=line_style,
                line_width=line_width, render_markers=render_markers,
                marker_style=marker_style, marker_size=marker_size,
                marker_face_colour=marker_face_colour,
                marker_edge_colour=marker_edge_colour,
                marker_edge_width=marker_edge_width, render_numbering=False,
                render_axes=render_axes, axes_font_name=axes_font_name,
                axes_font_size=axes_font_size, axes_font_style=axes_font_style,
                axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
                axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
                axes_y_ticks=axes_y_ticks, figure_size=figure_size)
            plt.title("Class: '{}'".format(self.classname))
        else:
            plt.title("No classification.")
        return viewer

    def view_all_detections(self, figure_id=None, new_figure=False, channels=None,
                            interpolation='bilinear', cmap_name=None, alpha=1.,
                            figure_size=(10, 8)):
        """
        Visualize all the potential detections with their labels.

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
            image will be pixelated. Example options ::

                {none, nearest, bilinear, bicubic, spline16, spline36, hanning,
                hamming, hermite, kaiser, quadric, catrom, gaussian, bessel,
                mitchell, sinc, lanczos}

        cmap_name: `str`, optional,
            If ``None``, single channel and three channel images default
            to greyscale and rgb colormaps respectively.
        alpha : `float`, optional
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        figure_size : (`float`, `float`) `tuple` or ``None`` optional
            The size of the figure in inches.
        """
        from menpo.visualize.viewmatplotlib import (_set_legend,
                                                    sample_colours_from_colourmap)
        colours = sample_colours_from_colourmap(len(self.labels), 'jet')
        viewer = self.image.view(
            figure_id=figure_id, new_figure=new_figure, channels=channels,
            interpolation=interpolation, cmap_name=cmap_name, alpha=alpha,
            render_axes=False)
        if len(self.all_detections[0]) == 0:
            return viewer
        for bb, lb in zip(self.all_detections[0], self.all_detections[2]):
            idx = self.labels.index(lb)
            c = colours[idx]
            bb.view(line_colour=c, render_markers=False, line_width=2, label=lb,
                    figure_size=figure_size)
        _set_legend(plt.gca(), legend_handles=None)

    def __str__(self):
        if self.classname is not None:
            return "Detected class '{}'.".format(self.classname)
        else:
            return "No detections."
