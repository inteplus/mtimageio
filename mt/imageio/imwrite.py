r"""Extra imwrite functions.

The functions write images with :mod:`imageio.v3` while following the conventions of
:mod:`mt.aio` (asynchronous file access via `context_vars`). Functions with prefix `imm` write
:class:`mt.opencv.image.Image` instances, which are images together with their metadata, as
PNG or WEBP files that embed the metadata, or as 'imm' (JSON) or HDF5 files.

Examples
--------
>>> import numpy as np
>>> from mt import cv
>>> from mt.imageio.imwrite import immencode
>>> imm = cv.Image(np.zeros((2, 4, 3), dtype=np.uint8), meta={"id": 1})
>>> data = immencode(imm, encoding_format="png")
>>> data[:4]
b'\x89PNG'
"""

import typing as tp

import json
from imageio import v3 as iio
from PIL import PngImagePlugin

from mt import np, cv, path, aio


__all__ = [
    "imwrite_asyn",
    "immencode",
    "immwrite_asyn",
    "immwrite",
]


async def imwrite_asyn(
    fname: str,
    image: np.ndarray,
    plugin: tp.Optional[str] = None,
    extension: tp.Optional[str] = None,
    format_hint: tp.Optional[str] = None,
    plugin_kwargs: dict = {},
    make_dirs: bool = False,
    context_vars: dict = {},
    **kwargs,
):
    r"""An asyn function that saves an image file using :func:`imageio.v3.imwrite`.

    The file format is determined by the extension of `fname`, unless `extension` is provided.

    Parameters
    ----------
    fname : str
        local filepath where the image will be saved. If "<bytes>" is provided, the function
        returns bytes instead of writes to a file. In that case `extension` or `plugin` must be
        provided so that imageio knows the format.
    image : numpy.ndarray
        the image to write to, in gray, RGB or RGBA pixel format, of shape `(height, width)` or
        `(height, width, nchannels)`
    plugin : str, optional
        The plugin to use. Passed directly to imageio's imwrite function.
    extension : str, optional
        File extension, for example '.png'. Passed directly to imageio's imwrite function. If not
        provided and `fname` is a filepath, it is the lower-cased extension of `fname`.
    format_hint : str, optional
        A format hint to help optimise plugin selection. Passed directly to imageio's imwrite
        function.
    plugin_kwargs : dict, optional
        Additional keyword arguments to be passed to the plugin write call. Default is an empty
        dictionary.
    make_dirs : bool, optional
        Whether or not to make the folders containing the path before writing to the file. Only
        valid when `fname` is a local filepath. Default is False.
    context_vars : dict, optional
        context variables within which the function runs. It must include `context_vars['async']`
        (bool), telling whether to invoke the function asynchronously or not. A :class:`KeyError`
        is raised otherwise (unless `fname` is "<bytes>"). When used via :func:`mt.aio.srun`, it is
        provided automatically.
    **kwargs
        ignored

    Returns
    -------
    int or bytes
        If "<bytes>" is provided for argument `fname`, a bytes object is returned. Otherwise, it
        returns whatever :func:`mt.aio.write_binary` returns, for example the number of bytes
        written.

    See Also
    --------
    imageio.v3.imwrite
        The underlying function for all the hard work.
    mt.imageio.imread.imread_asyn
        the inverse operation

    Examples
    --------
    >>> import numpy as np
    >>> from mt import aio
    >>> from mt.imageio.imwrite import imwrite_asyn
    >>> img = np.zeros((2, 4, 3), dtype=np.uint8)
    >>> data = aio.srun(imwrite_asyn, "<bytes>", img, extension=".png")
    >>> data[:4]
    b'\x89PNG'
    """

    if fname == "<bytes>":
        return iio.imwrite(
            fname,
            image,
            plugin=plugin,
            extension=extension,
            format_hint=format_hint,
            **plugin_kwargs,
        )

    if extension is None:
        extension = path.splitext(fname.lower())[1]

    data = iio.imwrite(
        "<bytes>",
        image,
        plugin=plugin,
        extension=extension,
        format_hint=format_hint,
        **plugin_kwargs,
    )

    return await aio.write_binary(
        fname, data, context_vars=context_vars, make_dirs=make_dirs
    )


def immencode_png(imm: cv.Image) -> bytes:
    """Encodes an image with metadata as PNG bytes, with metadata stored as PNG text chunks.

    The pixel format is stored under key 'pixel_format'. Every metadata value that is not a string
    is JSON-encoded, so it is read back as a string. Only the pixel formats 'gray', 'rgb' and
    'rgba' are supported, otherwise a :class:`KeyError` is raised.

    Parameters
    ----------
    imm : mt.opencv.image.Image
        an image with metadata

    Returns
    -------
    bytes
        the content of the PNG file
    """
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("pixel_format", imm.pixel_format)
    for k, v in imm.meta.items():
        if not isinstance(v, str):
            v = json.dumps(v)
        pnginfo.add_text(k, v)

    pixel_format2iio_mode = {
        "gray": "L",
        "rgba": "RGBA",
        "rgb": "RGB",
    }
    mode = pixel_format2iio_mode[imm.pixel_format]

    image = imm.image
    if mode == "L" and len(image.shape) == 3:
        image = image[:, :, 0]

    data = iio.imwrite(
        "<bytes>", image, plugin="pillow", extension=".png", mode=mode, pnginfo=pnginfo
    )
    return data


def immencode_webp(
    imm: cv.Image, lossless: bool = True, quality: tp.Optional[int] = None
) -> bytes:
    """Encodes an image with metadata as WEBP bytes, with metadata stored as a JSON XMP chunk.

    Parameters
    ----------
    imm : mt.opencv.image.Image
        an image with metadata. Its metadata must be JSON-serialisable.
    lossless : bool, optional
        whether or not to compress with lossless mode. Default is True.
    quality : int, optional
        a number between 0 and 100. If not provided, 90 for lossless and 80 for lossy.

    Returns
    -------
    bytes
        the content of the WEBP file
    """
    xmp = json.dumps(imm.meta)
    if lossless:  # mtimageio default, not pillow default
        if quality is None:
            quality = 90  # pillow default is 80
    else:
        if quality is None:
            quality = 80  # pillow default is 80
    data = iio.imwrite(
        "<bytes>",
        imm.image,
        plugin="pillow",
        extension=".webp",
        lossless=lossless,
        exact=True,  # pillow default is False
        quality=quality,
        method=5,  # pillow default is 4
        xmp=xmp,
    )
    return data


def immencode(
    imm: cv.Image,
    encoding_format: str = "png",
    lossless: bool = True,
    quality: tp.Optional[int] = None,
) -> bytes:
    """Encodes a :class:`mt.opencv.image.Image` instance as a PNG or WEBP image with metadata.

    Parameters
    ----------
    imm : mt.opencv.image.Image
        an image with metadata
    encoding_format : {'png', 'webp'}, optional
        the encoding format. Default is 'png'.
    lossless : bool, optional
        whether or not to compress with lossless mode. Only valid for 'webp' format. For 'png'
        format, it is always lossless. Default is True.
    quality : int, optional
        a number between 0 and 100. Only valid for 'webp'. If not provided, 90 for lossless and 80
        for lossy.

    Returns
    -------
    data : bytes
        the encoded image, ready for writing to file

    Raises
    ------
    NotImplementedError
        if the encoding format is neither 'png' nor 'webp'

    Notes
    -----
    For 'png', all metadata values are converted to json strings if they are not strings, and
    they are read back as strings by :func:`mt.imageio.imread.immdecode`. For 'webp', the metadata
    are stored as a JSON document, so JSON types are preserved.

    See Also
    --------
    imageio.v3.imwrite
        the underlying imwrite function
    mt.imageio.imread.immdecode
        the inverse operation

    Examples
    --------
    >>> import numpy as np
    >>> from mt import cv
    >>> from mt.imageio.imwrite import immencode
    >>> from mt.imageio.imread import immdecode
    >>> imm = cv.Image(np.zeros((2, 4, 3), dtype=np.uint8), meta={"id": 1})
    >>> immdecode(immencode(imm, "png")).meta["id"]  # PNG keeps metadata as strings
    '1'
    >>> immdecode(immencode(imm, "webp")).meta["id"]
    1
    """

    if encoding_format == "png":
        return immencode_png(imm)

    if encoding_format == "webp":
        return immencode_webp(imm, lossless=lossless, quality=quality)

    raise NotImplementedError(f"Unknown encoding format '{encoding_format}'.")


async def immwrite_asyn(
    filepath: str,
    imm: cv.Image,
    file_format: tp.Optional[str] = None,
    file_mode: int = 0o664,
    file_write_delayed: bool = False,
    make_dirs: bool = False,
    lossless: bool = True,
    quality: tp.Optional[int] = None,
    context_vars: dict = {},
    logger=None,
):
    """An asyn function that saves an image with metadata to file.

    Parameters
    ----------
    filepath : str
        local filepath to save the content to.
    imm : mt.opencv.image.Image
        an image with metadata
    file_format : {'imm', 'json', 'hdf5', 'png', 'webp'}, optional
        format to be used for saving the content. If not provided, it will be figured out from the
        file extension (without the dot). Formats 'imm' and 'json' both mean a JSON file with
        jpg-encoded (hence lossy) pixels, 'hdf5' requires :mod:`h5py`, and 'png' and 'webp' produce
        standard image files with the metadata embedded.
    file_mode : int, optional
        file mode to be set to using :func:`os.chmod`. If None is given, no setting of file mode
        will happen. Default is 0o664.
    file_write_delayed : bool, optional
        Only valid in asynchronous mode. If True, wraps the file write task into a future and
        returns the future. In all other cases, proceeds as usual. Default is False.
    make_dirs : bool, optional
        Whether or not to make the folders containing the path before writing to the file. Default
        is False.
    lossless : bool, optional
        whether or not to compress with lossless mode. Only valid for 'webp' format. For 'png'
        format, it is always lossless. Default is True.
    quality : int, optional
        a number between 0 and 100. Only valid for 'webp'. If not provided, 90 for lossless and 80
        for lossy.
    context_vars : dict, optional
        context variables within which the function runs. It must include `context_vars['async']`
        (bool), telling whether to invoke the function asynchronously or not. A :class:`KeyError`
        is raised otherwise. When used via :func:`mt.aio.srun`, it is provided automatically.
    logger : logging.Logger, optional
        logger for debugging purposes. Only used for the 'imm', 'json' and 'hdf5' formats.

    Returns
    -------
    int
        the number of bytes written to file (or a future if `file_write_delayed` is True)

    Raises
    ------
    NotImplementedError
        if the file format is not supported

    See Also
    --------
    immwrite
        the synchronous version
    mt.imageio.imread.immread_asyn
        the inverse operation
    """

    if file_format is None:
        ext = path.splitext(filepath)[1]
        file_format = ext[1:]

    if file_format in ("imm", "json"):
        return await cv.immsave_asyn(
            imm,
            filepath,
            file_mode=file_mode,
            file_write_delayed=file_write_delayed,
            make_dirs=make_dirs,
            image_codec="jpg",
            file_format="json",
            context_vars=context_vars,
            logger=logger,
        )

    if file_format == "hdf5":
        return await cv.immsave_asyn(
            imm,
            filepath,
            file_format=file_format,
            file_mode=file_mode,
            file_write_delayed=file_write_delayed,
            make_dirs=make_dirs,
            image_codec="png",
            context_vars=context_vars,
            logger=logger,
        )

    data = immencode(
        imm, encoding_format=file_format, lossless=lossless, quality=quality
    )
    return await aio.write_binary(
        filepath,
        data,
        file_mode=file_mode,
        file_write_delayed=file_write_delayed,
        make_dirs=make_dirs,
        context_vars=context_vars,
    )


def immwrite(
    filepath: str,
    imm: cv.Image,
    file_format: tp.Optional[str] = None,
    file_mode: int = 0o664,
    file_write_delayed: bool = False,
    lossless: bool = True,
    quality: tp.Optional[int] = None,
    logger=None,
):
    """Saves an image with metadata to file.

    Parameters
    ----------
    filepath : str
        local filepath to save the content to.
    imm : mt.opencv.image.Image
        an image with metadata
    file_format : {'imm', 'json', 'hdf5', 'png', 'webp'}, optional
        format to be used for saving the content. If not provided, it will be figured out from the
        file extension (without the dot). Formats 'imm' and 'json' both mean a JSON file with
        jpg-encoded (hence lossy) pixels, 'hdf5' requires :mod:`h5py`, and 'png' and 'webp' produce
        standard image files with the metadata embedded.
    file_mode : int, optional
        file mode to be set to using :func:`os.chmod`. If None is given, no setting of file mode
        will happen. Default is 0o664.
    file_write_delayed : bool, optional
        Only valid in asynchronous mode. If True, wraps the file write task into a future and
        returns the future. In all other cases, proceeds as usual. Default is False.
    lossless : bool, optional
        whether or not to compress with lossless mode. Only valid for 'webp' format. For 'png'
        format, it is always lossless. Default is True.
    quality : int, optional
        a number between 0 and 100. Only valid for 'webp'. If not provided, 90 for lossless and 80
        for lossy.
    logger : logging.Logger, optional
        logger for debugging purposes. Only used for the 'imm', 'json' and 'hdf5' formats.

    Returns
    -------
    int
        the number of bytes written to file

    Raises
    ------
    NotImplementedError
        if the file format is not supported

    Notes
    -----
    Unlike :func:`immwrite_asyn`, this function has no `make_dirs` or `context_vars` arguments.

    See Also
    --------
    immwrite_asyn
        the asynchronous version
    mt.imageio.imread.immread
        the inverse operation

    Examples
    --------
    >>> import os, tempfile
    >>> import numpy as np
    >>> from mt import cv
    >>> from mt.imageio.imwrite import immwrite
    >>> from mt.imageio.imread import immread
    >>> path = os.path.join(tempfile.mkdtemp(), "x.png")
    >>> imm = cv.Image(np.zeros((2, 4, 3), dtype=np.uint8), meta={"id": 1})
    >>> nbytes = immwrite(path, imm)
    >>> immread(path).image.shape
    (2, 4, 3)
    """

    return aio.srun(
        immwrite_asyn,
        filepath,
        imm,
        file_format=file_format,
        file_mode=file_mode,
        file_write_delayed=file_write_delayed,
        lossless=lossless,
        quality=quality,
        logger=logger,
    )
