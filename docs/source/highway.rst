Highway
=======

FastKMeans++ uses `Google Highway <https://github.com/google/highway>`__ for
portable SIMD distance and clustering kernels.

Google Highway provides a C++ interface that expresses vector operations without
locking the implementation to one instruction set. FastKMeans++ defines its
distance and Lloyd kernels with these operations. Highway compiles the kernels
for the targets enabled on the current platform, then selects the strongest
supported target when the code path is first used.

This approach keeps one implementation readable while allowing the same source
to use SSE, AVX2, AVX-512, NEON, or another supported target when available. A
scalar fallback remains available on processors without vector extensions.

SIMD distance calculations
--------------------------

Highway dispatches portable SIMD implementations at runtime. The distance
kernel loads several feature values at a time, subtracts centroid coordinates,
and uses fused multiply-add operations to calculate squared distances. A scalar
remainder handles dimensions that do not fill a complete vector.

The same vectorized distance calculation drives the Lloyd phase. Each sample is
assigned to its nearest centroid. The selected cluster receives the sample's
feature values in a native accumulation buffer. After all rows are processed,
the accumulated sums are divided by cluster counts to form the next centroids.

Parallel distance calculations
------------------------------

The Highway thread pool divides independent sample rows across workers. Each
worker writes to a private accumulation buffer, which avoids contention while
clusters are updated. The buffers are combined before the next iteration. Set
``n_jobs=1`` for serial execution or ``n_jobs=-1`` to use all available threads.

The thread pool is also used for prediction. Prediction assigns each input row
to its nearest fitted centroid and returns the same integer label format as the
estimator's ``labels_`` attribute.

Runtime dispatch
----------------

The native source uses Highway's target iteration and dynamic dispatch helpers.
The public Python layer passes ordinary contiguous NumPy arrays into the native
entry points. Target-specific vector types stay inside the C++ implementation.
This keeps the Python API independent of the processor used to run the model.

Highway is a separate open-source project. See its
`repository <https://github.com/google/highway>`__ for the library and its
supported targets.
