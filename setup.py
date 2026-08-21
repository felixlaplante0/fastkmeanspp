"""Builds the Google Highway distance extension."""

import sys
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parent
HIGHWAY = ROOT / "highway"

COMPILE_ARGS = ["/O2"] if sys.platform == "win32" else ["-O3"]

setup(
    ext_modules=cythonize(
        [
            Extension(
                "fastkmeanspp._highway",
                [
                    "fastkmeanspp/_highway.pyx",
                    "fastkmeanspp/_highway_kernel.cc",
                    "highway/hwy/targets.cc",
                ],
                include_dirs=[np.get_include(), str(HIGHWAY)],
                extra_compile_args=COMPILE_ARGS,
                language="c++",
            )
        ],
        language_level=3,
    )
)
