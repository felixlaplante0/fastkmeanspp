Highway
=======

FastKMeans++ uses `Google Highway <https://github.com/google/highway>`__ for
the distance kernel used during fast KMeans++ initialization.

SIMD distance calculations
--------------------------

Highway dispatches portable SIMD implementations at runtime. The kernel uses
fused multiply-add operations to calculate squared distances several values at
a time, with a scalar remainder for dimensions that do not fill a vector.

Parallel distance calculations
------------------------------

The Highway thread pool divides independent sample rows across workers. The
results are combined before the next centroid-selection step. Set
``n_jobs=1`` for serial execution or ``n_jobs=-1`` to use all available
threads.

Highway is a separate open-source project. See its
`repository <https://github.com/google/highway>`__ for the library and its
supported targets.
