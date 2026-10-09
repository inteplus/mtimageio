#!/usr/bin/python3

"""Viewing an image with metadata, on the terminal or in an OpenCV window.

The public function is :func:`immview`.
"""

from mt import tp, logg, cv, np


__all__ = ["immview"]


def get_image(imm):
    """Produces BGR image for display using OpenCV.

    Parameters
    ----------
    imm : mt.opencv.image.Image
        an image with metadata, whose pixel format is 'gray', 'bgr', 'rgb', 'rgba' or 'bgra'

    Returns
    -------
    numpy.ndarray
        a uint8 image of shape `(height, width)` for 'gray' or `(height, width, 3)` in BGR order for
        'bgr' and 'rgb'. For 'rgba' and 'bgra' the image has shape `(2*height, 2*width, 3)` and is
        a 2x2 mosaic: the top-left is the colour image pre-multiplied by alpha, the top-right is the
        alpha channel (in the first, blue, channel), the bottom-left is the colour image without
        alpha, and the bottom-right is black.

    Raises
    ------
    ValueError
        if the pixel format is not supported, for example 'argb' and 'abgr'

    Examples
    --------
    >>> import numpy as np
    >>> from mt import cv
    >>> from mt.imageio.immview import get_image
    >>> imm = cv.Image(np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8))
    >>> get_image(imm)  # RGB flipped into BGR
    array([[[30, 20, 10],
            [60, 50, 40]]], dtype=uint8)
    """
    if imm.pixel_format in ["gray", "bgr"]:
        return imm.image

    if imm.pixel_format == "rgb":
        return np.ascontiguousarray(np.flip(imm.image, axis=-1))

    if imm.pixel_format == "rgba":
        h, w = imm.image.shape[:2]
        image = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        image[h : h * 2, :w, 0] = imm.image[:, :, 2]
        image[h : h * 2, :w, 1] = imm.image[:, :, 1]
        image[h : h * 2, :w, 2] = imm.image[:, :, 0]

        image[:h, w : w * 2, 0] = imm.image[:, :, 3]

        image[:h, :w, 0] = np.round(
            imm.image[:, :, 2].astype(float) * imm.image[:, :, 3].astype(float) / 255
        ).astype(np.uint8)
        image[:h, :w, 1] = np.round(
            imm.image[:, :, 1].astype(float) * imm.image[:, :, 3].astype(float) / 255
        ).astype(np.uint8)
        image[:h, :w, 2] = np.round(
            imm.image[:, :, 0].astype(float) * imm.image[:, :, 3].astype(float) / 255
        ).astype(np.uint8)
        return image

    if imm.pixel_format == "bgra":
        h, w = imm.image.shape[:2]
        image = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        image[h : h * 2, :w, 0] = imm.image[:, :, 0]
        image[h : h * 2, :w, 1] = imm.image[:, :, 1]
        image[h : h * 2, :w, 2] = imm.image[:, :, 2]

        image[:h, w : w * 2, 0] = imm.image[:, :, 3]

        image[:h, :w, 0] = np.round(
            imm.image[:, :, 0].astype(float) * imm.image[:, :, 3].astype(float) / 255
        ).astype(np.uint8)
        image[:h, :w, 1] = np.round(
            imm.image[:, :, 1].astype(float) * imm.image[:, :, 3].astype(float) / 255
        ).astype(np.uint8)
        image[:h, :w, 2] = np.round(
            imm.image[:, :, 2].astype(float) * imm.image[:, :, 3].astype(float) / 255
        ).astype(np.uint8)
        return image

    raise ValueError(
        f"Imm with pixel format '{imm.pixel_format}' is not supported."
    )


def view(image, max_width=640, as_ansi=True):
    """Displays a BGR image.

    Parameters
    ----------
    image : numpy.ndarray
        a uint8 image in BGR order (or gray) of shape `(height, width[, 3])`
    max_width : int, optional
        the maximum width in pixels. A wider image is resized to this width, keeping the aspect
        ratio. Default is 640.
    as_ansi : bool, optional
        if True, draw the image on the terminal, using :mod:`term_image` if available or ANSI
        colours via :func:`mt.opencv.ansi.to_ansi` otherwise. If False, show it in an OpenCV
        highgui window and wait for a key press. Default is True.
    """
    if max_width < image.shape[1]:
        height = image.shape[0] * max_width // image.shape[1]
        image = cv.resize(image, dsize=(max_width, height))
    if as_ansi:
        img2 = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        try:
            from PIL import Image
            from term_image.image import AutoImage

            img3 = Image.fromarray(img2)
            img4 = AutoImage(img3)
            img4.draw(animate=False)
        except ImportError:
            print(cv.to_ansi(img2))
    else:
        cv.namedWindow("image")
        print("Press any key to exit.")
        cv.imshow("image", image)
        cv.waitKey(0)


def immview(
    imm: cv.Image,
    use_highgui: bool = False,
    max_width: int = 640,
    filepath: tp.Optional[str] = None,
    logger: tp.Optional[logg.IndentedLoggerAdapter] = None,
):
    """Views an image with metadata, either via OpenCV's highgui or on the terminal.

    The image is converted to BGR with :func:`get_image` and displayed with :func:`view`. If a
    logger is provided, the file path, pixel format, resolution and metadata are logged first.

    Parameters
    ----------
    imm : mt.opencv.image.Image
        an image with metadata
    use_highgui : bool, optional
        whether to use OpenCV's highgui window (True) or the terminal (False). Default is False.
    max_width : int, optional
        the maximum width in pixels. A wider image is resized to this width, keeping the aspect
        ratio, in both display modes. Default is 640.
    filepath : str, optional
        the filepath to the imm, only used for logging
    logger : mt.logg.IndentedLoggerAdapter, optional
        logger for printing purposes

    Raises
    ------
    ValueError
        if the pixel format of the image is not supported

    See Also
    --------
    mt.imageio.imread.immread
        loads an image with metadata from file
    """
    if logger:
        if filepath:
            logger.info(f"Image path: {filepath}")
        logger.info(f"Pixel format: {imm.pixel_format}")
        logger.info(f"Resolution: {imm.image.shape[1]}x{imm.image.shape[0]}")
        logger.info("Meta:")
        logger.info(imm.meta)
    view(get_image(imm), max_width=max_width, as_ansi=not use_highgui)
