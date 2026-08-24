"""Builds the Google Highway distance extension."""

import os
import platform
import sys
from pathlib import Path

import numpy as np
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ROOT = Path(__file__).resolve().parent
HIGHWAY = ROOT / "highway"

COMPILE_ARGS = ["/O2"] if sys.platform == "win32" else [
    "-O3",
    "-pthread",
]
LINK_ARGS = [] if sys.platform == "win32" else ["-pthread"]
CLANG_CL = os.environ.get("CLANG_CL", "").strip('"')
if CLANG_CL and platform.machine().lower() == "arm64":
    COMPILE_ARGS.append("--target=arm64-pc-windows-msvc")


class ClangBuildExt(build_ext):
    def build_extensions(self):
        if CLANG_CL:
            self.compiler.initialize()
            self.compiler.cc = CLANG_CL
        super().build_extensions()


setup(
    ext_modules=[
        Pybind11Extension(
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
            extra_link_args=LINK_ARGS,
            cxx_std=17,
        )
    ],
    cmdclass={"build_ext": ClangBuildExt},
)
