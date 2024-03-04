#!/usr/bin/env python3

import os
from setuptools import setup, find_namespace_packages

VERSION_FILE = os.path.join(os.path.dirname(__file__), "VERSION.txt")

setup(
    name="mtimageio",
    description="Minh-Tri Pham's extra modules using imageio",
    author=["Minh-Tri Pham"],
    packages=find_namespace_packages(include=["mt.*"]),
    scripts=[
        "scripts/immview",
    ],
    install_requires=[
        # 'h5py>=3', # for pdh5 file format. Lazy import because TX2 may not need it.
        "Pillow>=9.0",  # for processing PNG images
        "imageio>=2.15",  # for loading image files in a modern way
        "mtbase>=3.1",  # to have mt.np.dequantise_images
        "mtopencv>=1.9",  # to have mtopencv's immview script removed
    ],
    setup_requires=["setuptools-git-versioning<2"],
    setuptools_git_versioning={
        "enabled": True,
        "version_file": VERSION_FILE,
        "count_commits_from_version_file": True,
        "template": "{tag}",
        "dev_template": "{tag}.dev{ccount}+{branch}",
        "dirty_template": "{tag}.post{ccount}",
    },
)
