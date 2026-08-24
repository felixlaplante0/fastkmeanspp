# distutils: language = c++

import numpy as np
cimport numpy as cnp


cdef extern from "_highway_kernel.h" namespace "fastkmeanspp":
    void* create_pool(size_t n_jobs) except +
    void destroy_pool(void* pool) noexcept nogil
    void dispatch_cdist(
        const float* x, const float* y, float* out,
        size_t n, size_t m, size_t d,
        const float* minimums, size_t minimum_stride,
        float* inertias, void* pool
    ) noexcept nogil
    void dispatch_lloyd(
        const float* x, float* centers, cnp.int64_t* labels,
        size_t n, size_t k, size_t d, void* pool
    ) noexcept nogil
    void dispatch_assign(
        const float* x, const float* centers, cnp.int64_t* labels,
        size_t n, size_t k, size_t d, void* pool
    ) noexcept nogil


def cdist(X, y, size_t n_jobs=0) -> np.ndarray:
    return KMeansWorker(n_jobs)(X, y)


cdef class KMeansWorker:
    cdef void* pool

    def __cinit__(self, size_t n_jobs):
        self.pool = create_pool(n_jobs)

    def __dealloc__(self):
        with nogil:
            destroy_pool(self.pool)

    def __call__(
        self,
        cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] X,
        cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] y,
    ) -> np.ndarray:
        cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] out = np.empty(
            (X.shape[0], y.shape[0]), dtype=np.float32
        )
        with nogil:
            dispatch_cdist(
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
            dispatch_cdist(
                &X[0, 0], &y[0, 0], &out[0, 0],
                X.shape[0], y.shape[0], X.shape[1],
                &minimums[0], minimums.strides[0] // sizeof(float),
                &inertias[0], self.pool
            )
        return out, inertias

    def lloyd(
        self,
        cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] X,
        cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] centers,
        cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] labels,
    ) -> None:
        with nogil:
            dispatch_lloyd(
                &X[0, 0], &centers[0, 0], &labels[0],
                X.shape[0], centers.shape[0], X.shape[1], self.pool
            )

    def assign(
        self,
        cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] X,
        cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] centers,
        cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] labels,
    ) -> None:
        with nogil:
            dispatch_assign(
                &X[0, 0], &centers[0, 0], &labels[0],
                X.shape[0], centers.shape[0], X.shape[1], self.pool
            )
