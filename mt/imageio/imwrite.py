"""Extra imwrite functions."""

import typing as tp

import json
from imageio import v3 as iio
from PIL import PngImagePlugin

from mt.base import aio, path
from mt import np, cv


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
    context_vars: dict = {},
    **kwargs
):
    """An asyn function that saves an image file using :func:`imageio.v3.imwrite`.

    Parameters
    ----------
    fname : str
        local filepath where the image will be saved. If "<bytes>" is provided, the function
        returns bytes instead of writes to a file.
    image : numpy.ndarray
        the image to write to, in A, RGB or RGBA pixel format
    plugin : str, optional
        The plugin to use. Passed directly to imageio's imwrite function.
    extension : str, optional
        File extension. Passed directly to imageio's imwrite function.
    format_hint : str, optional
        A format hint to help optimise plugin selection. Passed directly to imageio's imwrite
        function.
    plugin_kwargs : dict
        Additional keyword arguments to be passed to the plugin write call.
    context_vars : dict
        a dictionary of context variables within which the function runs. It must include
        `context_vars['async']` to tell whether to invoke the function asynchronously or not.

    Returns
    -------
    int or bytes
        If "<bytes>" is provided for argument `fname`, a bytes object is returned. Otherwise, it
        returns whatever :func:`mt.base.aio.write_binary` returns.

    See Also
    --------
    imageio.v3.imwrite
        The underlying function for all the hard work.
    """

    if fname == "<bytes>":
        return iio.imwrite(
            fname,
            image,
            plugin=plugin,
            extension=extension,
            format_hint=format_hint,
            **plugin_kwargs
        )

    if extension is None:
        extension = path.splitext(fname.lower())[1]

    data = iio.imwrite(
        "<bytes>",
        image,
        plugin=plugin,
        extension=extension,
        format_hint=format_hint,
        **plugin_kwargs
    )

    return await aio.write_binary(fname, data, context_vars=context_vars)


def immencode(imm: cv.Image) -> bytes:
    """Encodes a :class:`mt.cv.opencv.Image` instance as a PNG image with metadata.

    Parameters
    ----------
    imm : cv.Image
        an image with metadata

    Returns
    -------
    data : bytes
        the image encoded into PNG content, ready for writing to file

    Notes
    -----
    All metadata values are converted to json strings if they are not strings.

    See Also
    --------
    imageio.v3.imwrite
        the underlying imwrite function
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

    data = iio.imwrite(
        "<bytes>",
        imm.image,
        plugin="pillow",
        extension=".png",
        mode=mode,
        pnginfo=pnginfo,
    )
    return data


async def immwrite_asyn(
    filepath: str,
    imm: cv.Image,
    file_format: str = "hdf5",
    file_mode: int = 0o664,
    file_write_delayed: bool = False,
    context_vars: dict = {},
    logger=None,
):
    """An asyn function that saves an image with metadata to file.

    Parameters
    ----------
    filepath : str
        local filepath to save the content to.
    imm : Image
        an image with metadata
    file_format : {'hdf5', 'png'}
        format to be used for saving the content.
    file_mode : int
        file mode to be set to using :func:`os.chmod`. If None is given, no setting of file mode
        will happen.
    file_write_delayed : bool
        Only valid in asynchronous mode and the format is 'json'. If True, wraps the file write
        task into a future and returns the future. In all other cases, proceeds as usual.
    context_vars : dict
        a dictionary of context variables within which the function runs. It must include
        `context_vars['async']` to tell whether to invoke the function asynchronously or not.
    logger : logging.Logger, optional
        logger for debugging purposes

    Returns
    -------
    int
        the number of bytes written to file
    """

    if file_format == "hdf5":
        return await cv.immsave_asyn(
            imm,
            filepath,
            file_format=file_format,
            file_mode=file_mode,
            file_write_delayed=file_write_delayed,
            image_codec="png",
            context_vars=context_vars,
            logger=logger,
        )

    data = immencode(imm)
    return await aio.write_binary(
        filepath,
        data,
        file_mode=file_mode,
        file_write_delayed=file_write_delayed,
        context_vars=context_vars,
    )


def immwrite(
    filepath: str,
    imm: cv.Image,
    file_format: str = "hdf5",
    file_mode: int = 0o664,
    file_write_delayed: bool = False,
    context_vars: dict = {},
    logger=None,
):
    """Saves an image with metadata to file.

    Parameters
    ----------
    filepath : str
        local filepath to save the content to.
    imm : Image
        an image with metadata
    file_format : {'hdf5', 'png'}
        format to be used for saving the content.
    file_mode : int
        file mode to be set to using :func:`os.chmod`. If None is given, no setting of file mode
        will happen.
    file_write_delayed : bool
        Only valid in asynchronous mode and the format is 'json'. If True, wraps the file write
        task into a future and returns the future. In all other cases, proceeds as usual.
    context_vars : dict
        a dictionary of context variables within which the function runs. It must include
        `context_vars['async']` to tell whether to invoke the function asynchronously or not.
    logger : logging.Logger, optional
        logger for debugging purposes

    Returns
    -------
    int
        the number of bytes written to file
    """

    return aio.srun(
        immwrite_asyn,
        filepath,
        imm,
        file_format=file_format,
        file_mode=file_mode,
        file_write_delayed=file_write_delayed,
        logger=logger,
    )
