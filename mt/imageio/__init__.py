"""Minh-Tri Pham's extra modules using imageio.

The package provides asynchronous-friendly image reading and writing functions with metadata
support, in :mod:`mt.imageio.imread`, :mod:`mt.imageio.imwrite` and :mod:`mt.imageio.immview`.
The recommended way of using them is via the :mod:`mt.iio` namespace, which also re-exports
everything from :mod:`imageio.v3`.

Examples
--------
>>> from mt import iio
"""

import imageio.v3 as iio

from mt.base import logger

from .version import version as __version__
