#!/usr/bin/env python3

from setuptools import setup, find_namespace_packages
from mt.imageio.version import version

setup(
    name="mtimageio",
    version=version,
    description="Minh-Tri Pham's extra modules using imageio",
    author=["Minh-Tri Pham"],
    packages=find_namespace_packages(include=["mt.*"]),
    # scripts=[
    #    "scripts/immview",
    # ],
    install_requires=[
        # 'h5py>=3', # for pdh5 file format. Lazy import because TX2 may not need it.
        "imageio>=2.15",  # for loading image files in a modern way
        "mtbase>=3.1",  # to have mt.np.dequantise_images
        "mtgeo>=1.8",  # to have the latest mt.opencv.image.Image implementation
    ],
)
