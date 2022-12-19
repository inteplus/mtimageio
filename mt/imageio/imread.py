"""Extra imread functions."""

import typing as tp

from imageio import v3 as iio

from mt.base import aio, path
from mt import np, cv


__all__ = [
    "imread_asyn",
    "immread_asyn",
    "immread",
]


async def imread_asyn(
    filepath,
    plugin: tp.Optional[str] = None,
    extension: tp.Optional[str] = None,
    format_hint: tp.Optional[str] = None,
    plugin_kwargs: dict = {},
    context_vars: dict = {},
) -> np.ndarray:
    """An asyn function that loads an image file using :func:`imageio.v3.imread`.

    Parameters
    ----------
    filepath : str
        local filepath to the image
    plugin : str, optional
        The plugin in :func:`imageio.v3.imread` to use. If set to None (default) imread will
        perform a search for a matching plugin. If not None, this takes priority over the provided
        format hint (if present).
    extension : str, optional
        Passed as-is to :func:`imageio.v3.imread`. If not None, treat the provided ImageResource as
        if it had the given extension. This affects the order in which backends are considered.
    format_hint : str, optional
        A format hint for `func:`imageio.v3.imread` to help optimize plugin selection given as the
        format’s extension, e.g. '.png'. This can speed up the selection process for ImageResources
        that don’t have an explicit extension, e.g. streams, or for ImageResources where the
        extension does not match the resource’s content.
    plugin_kwargs : dict
        Additional keyword arguments to be passed as-is to the plugin's read call of
        :func:`imageio.v3.imread`.
    context_vars : dict
        a dictionary of context variables within which the function runs. It must include
        `context_vars['async']` to tell whether to invoke the function asynchronously or not.

    Returns
    -------
    numpy.ndarray
        the loaded image

    Notes
    -----
    This imread version differs from :func:`cv2.imread` in that by default the output color image
    has RGB channels instead of OpenCV's old style BGR channels since it uses imageio and pillow
    plugin by default.

    Raises
    ------
    ValueError
    OSError

    See Also
    --------
    imageio.v3.imread
        the underlying imread function
    """

    if not context_vars["async"]:
        data = filepath
    else:
        data = await aio.read_binary(filepath, context_vars=context_vars)

    return iio.imread(
        data,
        plugin=plugin,
        extension=extension,
        format_hint=format_hint,
        **plugin_kwargs
    )


async def immread_asyn(
    filepath,
    plugin: tp.Optional[str] = None,
    extension: tp.Optional[str] = None,
    format_hint: tp.Optional[str] = None,
    plugin_kwargs: dict = {},
    context_vars: dict = {},
) -> cv.Image:
    """An asyn function that loads an image file and its metadata using :module:`imageio.v3`.

    Parameters
    ----------
    filepath : str
        local filepath to the image
    plugin : str, optional
        The plugin in :func:`imageio.v3.imread` to use. If set to None (default) imread will
        perform a search for a matching plugin. If not None, this takes priority over the provided
        format hint (if present).
    extension : str, optional
        Passed as-is to :func:`imageio.v3.imread`. If not None, treat the provided ImageResource as
        if it had the given extension. This affects the order in which backends are considered.
    format_hint : str, optional
        A format hint for `func:`imageio.v3.imread` to help optimize plugin selection given as the
        format’s extension, e.g. '.png'. This can speed up the selection process for ImageResources
        that don’t have an explicit extension, e.g. streams, or for ImageResources where the
        extension does not match the resource’s content.
    plugin_kwargs : dict
        Additional keyword arguments to be passed as-is to the plugin's read call of
        :func:`imageio.v3.imread`.
    context_vars : dict
        a dictionary of context variables within which the function runs. It must include
        `context_vars['async']` to tell whether to invoke the function asynchronously or not.

    Returns
    -------
    mt.opencv.image.Image
        the loaded image with metadata

    Notes
    -----
    This immread version wraps mtopencv's immload version by providing code to load stardard image
    files that come with metadata using :func:`imageio.v3.immeta`. In any case, it uses
    :class:`mt.opencv.Image` to store the result.

    For speed reasons, files with extension '.imm' are loaded with  :func:`mt.opencv.immload_asyn`.
    Other files are loaded using :module:`imageio.v3`.

    Raises
    ------
    ValueError
    OSError

    See Also
    --------
    mt.opencv.image.immload
        the underlying immload function for json and h5 formats
    imageio.v3.imread
        the underlying imread function
    imageio.v3.immeta
        the underlying immeta function
    """

    ext = path.splitext(path.basename(filepath)).lower()
    if ext == ".imm":
        return await cv.immload_asyn(filepath, context_vars=context_vars)

    data = await aio.read_binary(filepath, context_vars=context_vars)
    meta = iio.immeta(data, plugin=plugin, extension=extension, **plugin_kwargs)
    image = iio.imread(
        data,
        plugin=plugin,
        extension=extension,
        format_hint=format_hint,
        **plugin_kwargs
    )

    pillow_mode2pixel_format = {
        "RGB": "rgb",
        "RGBA": "rgba",
        "L": "gray",
        "P": "gray",
    }
    pixel_format = pillow_mode2pixel_format[meta["mode"]]

    imm = cv.Image(image, pixel_format=pixel_format, meta=meta)
    return imm


def immread(
    filepath,
    plugin: tp.Optional[str] = None,
    extension: tp.Optional[str] = None,
    format_hint: tp.Optional[str] = None,
    plugin_kwargs: dict = {},
) -> cv.Image:
    """Loads an image file and its metadata using :module:`imageio.v3`.

    Parameters
    ----------
    filepath : str
        local filepath to the image
    plugin : str, optional
        The plugin in :func:`imageio.v3.imread` to use. If set to None (default) imread will
        perform a search for a matching plugin. If not None, this takes priority over the provided
        format hint (if present).
    extension : str, optional
        Passed as-is to :func:`imageio.v3.imread`. If not None, treat the provided ImageResource as
        if it had the given extension. This affects the order in which backends are considered.
    format_hint : str, optional
        A format hint for `func:`imageio.v3.imread` to help optimize plugin selection given as the
        format’s extension, e.g. '.png'. This can speed up the selection process for ImageResources
        that don’t have an explicit extension, e.g. streams, or for ImageResources where the
        extension does not match the resource’s content.
    plugin_kwargs : dict
        Additional keyword arguments to be passed as-is to the plugin's read call of
        :func:`imageio.v3.imread`.
    context_vars : dict
        a dictionary of context variables within which the function runs. It must include
        `context_vars['async']` to tell whether to invoke the function asynchronously or not.

    Returns
    -------
    mt.opencv.image.Image
        the loaded image with metadata

    Notes
    -----
    This immread version wraps mtopencv's immload version by providing code to load stardard image
    files that come with metadata using :func:`imageio.v3.immeta`. In any case, it uses
    :class:`mt.opencv.Image` to store the result.

    For speed reasons, files with extension '.imm' are loaded with  :func:`mt.opencv.immload_asyn`.
    Other files are loaded using :module:`imageio.v3`.

    Raises
    ------
    ValueError
    OSError

    See Also
    --------
    mt.opencv.image.immload
        the underlying immload function for json and h5 formats
    imageio.v3.imread
        the underlying imread function
    imageio.v3.immeta
        the underlying immeta function
    """

    return aio.srun(
        immread_asyn,
        filepath,
        plugin=plugin,
        extension=extension,
        format_hint=format_hint,
        plugin_kwargs=plugin_kwargs,
    )
