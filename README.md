<p align="center">
  <img src="https://raw.githubusercontent.com/felixlaplante0/fastkmeanspp/main/docs/source/_static/fastkmeanspp-logo.svg" alt="fastkmeanspp logo" width="128">
</p>

<h1 align="center">K-Means++</h1>

<p align="center"><strong>Fast KMeans++ initialization.</strong><br>
A scikit-learn-compatible KMeans implementation with fast SIMD distance computations and parallel centroid initialization.</p>

<p align="center">
  <a href="https://fastkmeanspp.readthedocs.io/en/latest/">Documentation</a> ·
  <a href="https://pypi.org/project/fastkmeanspp/">PyPI</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/fastkmeanspp/"><img src="https://img.shields.io/pypi/v/fastkmeanspp?logo=pypi&logoColor=white" alt="PyPI version"></a>
  <a href="https://pypi.org/project/fastkmeanspp/"><img src="https://img.shields.io/badge/python-3.11--3.14-blue?logo=python&logoColor=white" alt="Supported Python versions: 3.11–3.14"></a>
  <a href="https://github.com/felixlaplante0/fastkmeanspp/actions/workflows/lint.yml"><img src="https://github.com/felixlaplante0/fastkmeanspp/actions/workflows/lint.yml/badge.svg" alt="Lint status"></a>
  <a href="https://codecov.io/gh/felixlaplante0/fastkmeanspp"><img src="https://codecov.io/gh/felixlaplante0/fastkmeanspp/graph/badge.svg" alt="Coverage"></a>
  <a href="https://fastkmeanspp.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/fastkmeanspp/badge/?version=latest" alt="Documentation status"></a>
  <a href="https://github.com/felixlaplante0/fastkmeanspp/blob/main/LICENSE"><img src="https://img.shields.io/github/license/felixlaplante0/fastkmeanspp" alt="License"></a>
</p>

**fastkmeanspp** is a Python package that implements a KMeans clone from
[scikit-learn](https://scikit-learn.org/) with a faster KMeans++ centroid
initialization. It is designed to be a drop-in replacement for
scikit-learn's `KMeans` when initialization is the bottleneck.

---

## ✨ Features

- **Fast KMeans++ initialization**: Uses optimized squared-distance computations
  while selecting candidate centroids.
- **Portable SIMD kernels**: Uses [Google Highway](https://github.com/google/highway)
  to compile vectorized kernels for supported CPU targets and select the best
  implementation at runtime.
- **Parallel initialization**: Computes distance rows in parallel with Highway's
  thread pool.
- **scikit-learn compatibility**: Provides familiar `fit`, `predict`, `labels_`,
  `cluster_centers_`, and `inertia_` interfaces.
- **FAISS clustering**: Uses [FAISS](https://github.com/facebookresearch/faiss)
  for the Lloyd iterations after initialization.

### How Google Highway fits in

Google Highway is a C++ library for portable SIMD programming. The distance
kernel is written once with Highway's vector operations. Highway then builds
target-specific versions for the available instruction sets, such as SSE,
AVX2, and AVX-512 on x86 CPUs. A small runtime dispatch layer selects the
strongest target supported by the current processor and keeps a scalar fallback
for portability.

The distance kernel loads several feature values at a time, subtracts the
corresponding centroid values, and uses fused multiply-add operations to build
the squared distance. Feature dimensions that do not fill a complete vector
are handled by a short scalar tail.

During KMeans++ initialization, Highway's thread pool divides independent data
rows between workers. The initialized centroids are then handed to FAISS for
the Lloyd iterations.

---

## 🚀 Installation

```bash
python -m pip install fastkmeanspp
```

## 🔧 Usage

```python
import numpy as np
from fastkmeanspp import KMeans

X = np.array([[0.0, 0.0], [0.1, 0.2], [4.0, 4.0], [4.2, 3.9]])
model = KMeans(n_clusters=2, random_state=42)
model.fit(X)

print(model.labels_)
print(model.cluster_centers_)
```

Set `n_jobs=1` for serial centroid initialization or `n_jobs=-1` to use all
available threads.

