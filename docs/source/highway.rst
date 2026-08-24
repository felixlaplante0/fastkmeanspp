Highway
=======

FastKMeans++ uses `Google Highway <https://github.com/google/highway>`__ for
the distance kernel used during fast KMeans++ initialization.

Google Highway provides a C++ interface that expresses vector operations without
locking the implementation to one instruction set. FastKMeans++ defines its
distance kernel with these operations. Highway compiles the kernel for the
targets enabled on the current platform, then selects the strongest supported
target when the code path is first used.

This approach keeps one implementation readable while allowing the same source
to use SSE, AVX2, AVX-512, NEON, or another supported target when available. A
scalar fallback remains available on processors without vector extensions.

SIMD distance calculations
--------------------------

Highway dispatches portable SIMD implementations at runtime. The distance
kernel loads several feature values at a time, subtracts centroid coordinates,
and uses fused multiply-add operations to calculate squared distances. A scalar
remainder handles dimensions that do not fill a complete vector.

Parallel distance calculations
------------------------------

The Highway thread pool divides independent sample rows across workers. The
results are combined before the next centroid-selection step. Set ``n_jobs=1``
for serial execution or ``n_jobs=-1`` to use all available threads. The
initialized centroids are then handed to
`FAISS <https://github.com/facebookresearch/faiss>`__ for clustering.

Runtime dispatch
----------------

The native source uses Highway's target iteration and dynamic dispatch helpers.
The public Python layer passes ordinary contiguous NumPy arrays into the native
entry points. Target-specific vector types stay inside the C++ implementation.
This keeps the Python API independent of the processor used to run the model.

Highway is a separate open-source project. See its
`repository <https://github.com/google/highway>`__ for the library and its
supported targets.
