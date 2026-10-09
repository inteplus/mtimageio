"""The `mt.iio` namespace: :mod:`imageio.v3` plus Minh-Tri Pham's extra functions.

Instead of `import imageio.v3 as iio`, do:

.. code-block:: python

   from mt import iio

It re-exports everything in :mod:`imageio.v3` (e.g. `iio.imread`) and adds the functions of
:mod:`mt.imageio.imread` (e.g. `imread_asyn`, `immread`), :mod:`mt.imageio.imwrite` (e.g.
`imwrite_asyn`, `immwrite`) and :func:`mt.imageio.immview.immview`.

Examples
--------
>>> from mt import iio
>>> callable(iio.immread) and callable(iio.imread)
True
"""

from imageio import __version__
from imageio.v3 import *

from mt.imageio.imread import *
from mt.imageio.imwrite import *
from mt.imageio.immview import immview
