"""Extra imwrite functions."""

import typing as tp

from imageio import v3 as iio

from mt.base import aio, path
from mt import np, cv


__all__ = [
    "imwrite_asyn",
]


async def imwrite_asyn(
    fname: str,
    image: np.ndarray,
    plugin: tp.Optional[str] = None,
    extension: tp.Optional[str] = None,
    format_hint: tp.Optional[str] = None,
    plugin_kwargs: dict = {},
    context_vars: dict = {},
    logger=None,
    **kwargs
):
    """An asyn function that saves an image file for ML using :func:`imageio.v3.imwrite`.

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
    logger : logging.Logger or equivalent, optional
        logger for debugging purposes

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
