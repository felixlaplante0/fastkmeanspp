"""Build the Highway distance extension."""

import sys
from pathlib import Path

import numpy as np
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

ROOT = Path(__file__).resolve().parent
HIGHWAY = ROOT / "highway"
WINDOWS = sys.platform == "win32"
COMPILE_ARGS = ["-O3", "-pthread"]
if WINDOWS:
    COMPILE_ARGS += ["-std=c++17", "-DHWY_DISABLE_FUTEX"]

EXTENSION = Pybind11Extension(
    "fastkmeanspp._highway",
    [
        "fastkmeanspp/_highway_bindings.cpp",
        "fastkmeanspp/_highway_kernel.cc",
        "highway/hwy/abort.cc",
        "highway/hwy/aligned_allocator.cc",
        "highway/hwy/contrib/sort/vqsort.cc",
        "highway/hwy/contrib/sort/vqsort_have.cc",
        "highway/hwy/contrib/sort/vqsort_f64a.cc",
        "highway/hwy/contrib/thread_pool/thread_pool.cc",
        "highway/hwy/contrib/thread_pool/topology.cc",
        "highway/hwy/profiler.cc",
        "highway/hwy/targets.cc",
        "highway/hwy/timer.cc",
    ],
    include_dirs=[np.get_include(), str(HIGHWAY), str(ROOT / "fastkmeanspp")],
    extra_compile_args=COMPILE_ARGS,
    extra_link_args=["-pthread"],
    cxx_std=0 if WINDOWS else 17,
)
if WINDOWS:
    EXTENSION.extra_compile_args = [
        arg for arg in EXTENSION.extra_compile_args if not arg.startswith("/")
    ]


setup(
    options={"build_ext": {"compiler": "mingw32"}} if WINDOWS else {},
    ext_modules=[EXTENSION],
)
