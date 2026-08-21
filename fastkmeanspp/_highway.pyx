# distutils: language = c++

import numpy as np
cimport numpy as cnp


cdef extern from "_highway_kernel.h" namespace "fastkmeanspp":
    void highway_dists(
        const float* x, const float* y, float* out,
        size_t n, size_t m, size_t dim
    ) noexcept nogil


def dists(
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] X,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] y,
) -> np.ndarray:
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] out = np.empty(
        (X.shape[0], y.shape[0]), dtype=np.float32
    )
    with nogil:
        highway_dists(
            &X[0, 0], &y[0, 0], &out[0, 0],
            X.shape[0], y.shape[0], X.shape[1]
        )
    return out
