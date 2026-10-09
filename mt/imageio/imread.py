"""Extra imread functions.

The functions read image files with :mod:`imageio.v3` while following the conventions of
:mod:`mt.aio` (asynchronous file access via `context_vars`). Colour images are returned with RGB
channel order (not OpenCV's BGR). Functions with prefix `imm` read images together with their
metadata into :class:`mt.opencv.image.Image` instances.

Examples
--------
>>> import os, tempfile
>>> import numpy as np
>>> from mt import aio
>>> from mt.imageio.imwrite import imwrite_asyn
>>> from mt.imageio.imread import imread_asyn
>>> path = os.path.join(tempfile.mkdtemp(), "x.png")
>>> img = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
>>> _ = aio.srun(imwrite_asyn, path, img)
>>> np.array_equal(aio.srun(imread_asyn, path), img)
True
"""

import typing as tp
import json

from imageio import v3 as iio

from mt import np, cv, path, aio


__all__ = [
    "imread_asyn",
    "immeta2immmeta",
    "immdecode",
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
        A format hint for :func:`imageio.v3.imread` to help optimize plugin selection given as the
        format's extension, e.g. '.png'. This can speed up the selection process for ImageResources
        that don't have an explicit extension, e.g. streams, or for ImageResources where the
        extension does not match the resource's content.
    plugin_kwargs : dict, optional
        Additional keyword arguments to be passed as-is to the plugin's read call of
        :func:`imageio.v3.imread`. Default is an empty dictionary.
    context_vars : dict, optional
        context variables within which the function runs. It must include `context_vars['async']`
        (bool), telling whether to invoke the function asynchronously or not. A :class:`KeyError`
        is raised otherwise. When used via :func:`mt.aio.srun`, it is provided automatically.

    Returns
    -------
    numpy.ndarray
        the loaded image, of shape `(height, width)` or `(height, width, nchannels)`

    Raises
    ------
    ValueError
        if the content cannot be decoded
    OSError
        if the file cannot be read

    Notes
    -----
    This imread version differs from :func:`cv2.imread` in that by default the output color image
    has RGB channels instead of OpenCV's old style BGR channels since it uses imageio and pillow
    plugin by default.

    See Also
    --------
    imageio.v3.imread
        the underlying imread function

    Examples
    --------
    >>> import os, tempfile
    >>> import numpy as np
    >>> from mt import aio
    >>> from mt.imageio.imwrite import imwrite_asyn
    >>> from mt.imageio.imread import imread_asyn
    >>> path = os.path.join(tempfile.mkdtemp(), "x.png")
    >>> img = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    >>> _ = aio.srun(imwrite_asyn, path, img)
    >>> aio.srun(imread_asyn, path).shape
    (2, 4, 3)
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


def immeta2immmeta(
    meta: dict,
) -> dict:
    """Converts the metadata read by invoking :func:`imageio.v3.immeta` into the metadata of an imm
    file.

    If the metadata contains an 'xmp' entry of bytes type, as written by :func:`immencode` for the
    'webp' format, it is parsed as JSON and becomes the metadata, to which the 'mode' and 'shape'
    entries of the original metadata are added. Otherwise, the metadata is returned as-is. Note that
    the input dictionary is modified in place in the first case.

    Parameters
    ----------
    meta : dict
        the output of invoking :func:`imageio.v3.immeta`

    Returns
    -------
    dict
        the converted/adjusted metadata

    See Also
    --------
    imageio.v3.immeta
        the underlying immeta function

    Examples
    --------
    >>> from mt.imageio.imread import immeta2immmeta
    >>> meta = {"xmp": b'{"id": 1}', "mode": "RGB", "shape": (4, 2)}
    >>> immeta2immmeta(meta)
    {'id': 1, 'mode': 'RGB', 'shape': (4, 2)}
    >>> immeta2immmeta({"mode": "L", "shape": (4, 2)})
    {'mode': 'L', 'shape': (4, 2)}
    """

    if "xmp" in meta and isinstance(meta["xmp"], bytes):
        meta2 = json.loads(meta["xmp"])
        del meta["xmp"]
        for x in ["mode", "shape"]:
            meta2[x] = meta[x]
            del meta[x]
        # meta2["image_meta"] = meta # MT-TODO: expose later in future.
        meta = meta2

    return meta


def immdecode(
    data: bytes,
    plugin: tp.Optional[str] = None,
    extension: tp.Optional[str] = None,
    format_hint: tp.Optional[str] = None,
    plugin_kwargs: dict = {},
) -> cv.Image:
    """Decodes an image file content and its metadata using :mod:`imageio.v3`.

    Parameters
    ----------
    data : bytes
        the content of an image file that has been read into memory
    plugin : str, optional
        The plugin in :func:`imageio.v3.imread` to use. If set to None (default) imread will
        perform a search for a matching plugin. If not None, this takes priority over the provided
        format hint (if present).
    extension : str, optional
        Passed as-is to :func:`imageio.v3.imread`. If not None, treat the provided ImageResource as
        if it had the given extension. This affects the order in which backends are considered.
    format_hint : str, optional
        A format hint for :func:`imageio.v3.imread` to help optimize plugin selection given as the
        format's extension, e.g. '.png'. This can speed up the selection process for ImageResources
        that don't have an explicit extension, e.g. streams, or for ImageResources where the
        extension does not match the resource's content.
    plugin_kwargs : dict, optional
        Additional keyword arguments to be passed as-is to the plugin's read call of
        :func:`imageio.v3.imread`. Default is an empty dictionary.

    Returns
    -------
    mt.opencv.image.Image
        the loaded image with metadata. The pixel format is 'rgb', 'rgba' or 'gray', according to
        the mode of the image reported by imageio (modes 'L' and 'P' are both mapped to 'gray').
        The metadata contain the image mode and shape, plus the metadata stored by :func:`immencode`
        (for PNG files the metadata values are strings).

    Raises
    ------
    ValueError
        if the content cannot be decoded
    KeyError
        if the image mode is not one of 'RGB', 'RGBA', 'L' and 'P'

    Notes
    -----
    This immread version loads a standard image file that come with metadata using
    :mod:`imageio.v3`. However, it uses :class:`mt.opencv.image.Image` to store the result.

    See Also
    --------
    imageio.v3.imread
        the underlying imread function
    imageio.v3.immeta
        the underlying immeta function
    immencode
        the inverse operation

    Examples
    --------
    >>> import numpy as np
    >>> from mt import cv
    >>> from mt.imageio.imwrite import immencode
    >>> from mt.imageio.imread import immdecode
    >>> imm = cv.Image(np.zeros((2, 4, 3), dtype=np.uint8), meta={"id": 1})
    >>> immdecode(immencode(imm, encoding_format="webp")).meta
    {'id': 1, 'mode': 'RGB', 'shape': (4, 2)}
    """

    meta = iio.immeta(data, plugin=plugin, extension=extension, **plugin_kwargs)
    meta = immeta2immmeta(meta)

    image = iio.imread(
        data,
        plugin=plugin,
        extension=extension,
        format_hint=format_hint,
        **plugin_kwargs
    )

    iio_mode2pixel_format = {
        "RGB": "rgb",
        "RGBA": "rgba",
        "L": "gray",
        "P": "gray",
    }
    pixel_format = iio_mode2pixel_format[meta["mode"]]

    imm = cv.Image(image, pixel_format=pixel_format, meta=meta)
    return imm


async def immread_asyn(
    filepath,
    plugin: tp.Optional[str] = None,
    extension: tp.Optional[str] = None,
    format_hint: tp.Optional[str] = None,
    plugin_kwargs: dict = {},
    context_vars: dict = {},
) -> cv.Image:
    """An asyn function that loads an image file and its metadata using :mod:`imageio.v3`.

    Parameters
    ----------
    filepath : str
        local filepath to the image. If its extension is '.imm', the file is loaded with
        :func:`mt.opencv.image.immload_asyn` and the other arguments are ignored.
    plugin : str, optional
        The plugin in :func:`imageio.v3.imread` to use. If set to None (default) imread will
        perform a search for a matching plugin. If not None, this takes priority over the provided
        format hint (if present).
    extension : str, optional
        Passed as-is to :func:`imageio.v3.imread`. If not None, treat the provided ImageResource as
        if it had the given extension. This affects the order in which backends are considered.
    format_hint : str, optional
        A format hint for :func:`imageio.v3.imread` to help optimize plugin selection given as the
        format's extension, e.g. '.png'. This can speed up the selection process for ImageResources
        that don't have an explicit extension, e.g. streams, or for ImageResources where the
        extension does not match the resource's content.
    plugin_kwargs : dict, optional
        Additional keyword arguments to be passed as-is to the plugin's read call of
        :func:`imageio.v3.imread`. Default is an empty dictionary.
    context_vars : dict, optional
        context variables within which the function runs. It must include `context_vars['async']`
        (bool), telling whether to invoke the function asynchronously or not. A :class:`KeyError`
        is raised otherwise. When used via :func:`mt.aio.srun`, it is provided automatically.

    Returns
    -------
    mt.opencv.image.Image
        the loaded image with metadata

    Raises
    ------
    ValueError
        if the content cannot be decoded
    OSError
        if the file cannot be read

    Notes
    -----
    This immread version combines :func:`mt.opencv.image.immload_asyn` (if the extension is '.imm')
    and :func:`immdecode` (otherwise). In any case, it uses :class:`mt.opencv.image.Image` to store
    the result.

    See Also
    --------
    mt.opencv.image.immload_asyn
        the underlying immload function for json and h5 formats
    immdecode
        the underlying immdecode function
    immread
        the synchronous version
    """

    ext = path.splitext(path.basename(filepath))[1].lower()
    if ext == ".imm":
        return await cv.immload_asyn(filepath, context_vars=context_vars)

    data = await aio.read_binary(filepath, context_vars=context_vars)
    return immdecode(
        data,
        plugin=plugin,
        extension=extension,
        format_hint=format_hint,
        **plugin_kwargs
    )


def immread(
    filepath,
    plugin: tp.Optional[str] = None,
    extension: tp.Optional[str] = None,
    format_hint: tp.Optional[str] = None,
    plugin_kwargs: dict = {},
) -> cv.Image:
    """Loads an image file and its metadata using :mod:`imageio.v3`.

    Parameters
    ----------
    filepath : str
        local filepath to the image. If its extension is '.imm', the file is loaded with
        :func:`mt.opencv.image.immload` and the other arguments are ignored.
    plugin : str, optional
        The plugin in :func:`imageio.v3.imread` to use. If set to None (default) imread will
        perform a search for a matching plugin. If not None, this takes priority over the provided
        format hint (if present).
    extension : str, optional
        Passed as-is to :func:`imageio.v3.imread`. If not None, treat the provided ImageResource as
        if it had the given extension. This affects the order in which backends are considered.
    format_hint : str, optional
        A format hint for :func:`imageio.v3.imread` to help optimize plugin selection given as the
        format's extension, e.g. '.png'. This can speed up the selection process for ImageResources
        that don't have an explicit extension, e.g. streams, or for ImageResources where the
        extension does not match the resource's content.
    plugin_kwargs : dict, optional
        Additional keyword arguments to be passed as-is to the plugin's read call of
        :func:`imageio.v3.imread`. Default is an empty dictionary.

    Returns
    -------
    mt.opencv.image.Image
        the loaded image with metadata

    Raises
    ------
    ValueError
        if the content cannot be decoded
    OSError
        if the file cannot be read

    Notes
    -----
    This immread version combines :func:`mt.opencv.image.immload` (if the extension is '.imm') and
    :func:`immdecode` (otherwise). In any case, it uses :class:`mt.opencv.image.Image` to store the
    result. Unlike :func:`immread_asyn`, it has no `context_vars` argument.

    See Also
    --------
    mt.opencv.image.immload
        the underlying immload function for json and h5 formats
    immdecode
        the underlying immdecode function
    immwrite
        the inverse operation

    Examples
    --------
    >>> import os, tempfile
    >>> import numpy as np
    >>> from mt import cv
    >>> from mt.imageio.imwrite import immwrite
    >>> from mt.imageio.imread import immread
    >>> path = os.path.join(tempfile.mkdtemp(), "x.webp")
    >>> imm = cv.Image(np.zeros((2, 4, 3), dtype=np.uint8), meta={"id": 1})
    >>> _ = immwrite(path, imm)
    >>> immread(path).meta["id"]
    1
    """

    return aio.srun(
        immread_asyn,
        filepath,
        plugin=plugin,
        extension=extension,
        format_hint=format_hint,
        plugin_kwargs=plugin_kwargs,
    )
