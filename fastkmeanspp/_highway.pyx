# distutils: language = c++

import numpy as np
cimport numpy as cnp


cdef extern from "_highway_kernel.h" namespace "fastkmeanspp":
    void* highway_cdist_pool_create(size_t n_jobs, size_t m) except +
    void highway_cdist_pool_destroy(void* pool) noexcept nogil
    void highway_cdist(
        const float* x, const float* y, float* out,
        size_t n, size_t m, size_t d,
        const float* minimums, size_t minimum_stride,
        float* inertias, void* pool
    ) noexcept nogil


def cdist(X, y, size_t n_jobs=0) -> np.ndarray:
    return _Cdist(n_jobs, y.shape[0])(X, y)


cdef class _Cdist:
    cdef void* pool

    def __cinit__(self, size_t n_jobs, size_t m):
        self.pool = highway_cdist_pool_create(n_jobs, m)

    def __dealloc__(self):
        with nogil:
            highway_cdist_pool_destroy(self.pool)

    def __call__(
        self,
        cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] X,
        cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] y,
    ) -> np.ndarray:
        cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] out = np.empty(
            (X.shape[0], y.shape[0]), dtype=np.float32
        )
        with nogil:
            highway_cdist(
                &X[0, 0], &y[0, 0], &out[0, 0],
                X.shape[0], y.shape[0], X.shape[1], NULL, 0, NULL, self.pool
            )
        return out

    def minimum(
        self,
        cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] X,
        cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] y,
        cnp.ndarray[cnp.float32_t, ndim=1, mode="strided"] minimums,
    ) -> tuple[np.ndarray, np.ndarray]:
        cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] out = np.empty(
            (X.shape[0], y.shape[0]), dtype=np.float32
        )
        cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] inertias = np.empty(
            y.shape[0], dtype=np.float32
        )
        with nogil:
            highway_cdist(
                &X[0, 0], &y[0, 0], &out[0, 0],
                X.shape[0], y.shape[0], X.shape[1],
                &minimums[0], minimums.strides[0] // sizeof(float),
                &inertias[0], self.pool
            )
        return out, inertias
