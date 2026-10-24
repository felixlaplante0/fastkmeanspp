"""Build the Highway distance extension."""

import os
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

WINDOWS = sys.platform == "win32"
COMPILE_ARGS = ["-O3", "-pthread"]
LINK_ARGS = ["-pthread"]
if WINDOWS:
    COMPILE_ARGS.append("-DHWY_DISABLE_FUTEX")

cxxflags = os.environ.get("CXXFLAGS", "")
if not {"-std=c++17", "-std=gnu++17"}.intersection(cxxflags.split()):
    os.environ["CXXFLAGS"] = f"{cxxflags} -std=c++17".strip()

HIGHWAY_SOURCES = [
    "highway/hwy/abort.cc",
    "highway/hwy/aligned_allocator.cc",
    "highway/hwy/per_target.cc",
    "highway/hwy/contrib/sort/vqsort.cc",
    "highway/hwy/contrib/sort/vqsort_f64a.cc",
    "highway/hwy/contrib/thread_pool/thread_pool.cc",
    "highway/hwy/contrib/thread_pool/topology.cc",
    "highway/hwy/profiler.cc",
    "highway/hwy/targets.cc",
    "highway/hwy/timer.cc",
]
# Highway after 1.4.0 splits these out of the sources above.
HIGHWAY_SOURCES += [
    path
    for path in ("highway/hwy/contrib/sort/vqsort_have.cc", "highway/hwy/os_rng.cc")
    if Path(path).exists()
]

EXTENSION = Pybind11Extension(
    "fastkmeanspp._highway",
    ["fastkmeanspp/_highway.cpp", *HIGHWAY_SOURCES],
    include_dirs=["highway", "fastkmeanspp"],
    extra_compile_args=COMPILE_ARGS,
    extra_link_args=LINK_ARGS,
)
if WINDOWS:
    EXTENSION.extra_compile_args = [
        arg for arg in EXTENSION.extra_compile_args if not arg.startswith("/")
    ]

setup(
    options={"build_ext": {"compiler": "mingw32"}} if WINDOWS else {},
    ext_modules=[EXTENSION],
)
