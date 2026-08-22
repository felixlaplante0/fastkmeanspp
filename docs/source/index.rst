Fast KMeans++
=============

.. raw:: html

   <section class="hero">
     <img class="hero-logo" src="_static/fastkmeanspp-logo.svg" alt="Fast KMeans++ logo">
     <p class="eyebrow">K-MEANS++, BUILT FOR SPEED</p>
     <h1>Fast centroid initialization for KMeans.</h1>
     <p class="hero-copy">FastKMeans++ combines Highway SIMD distance kernels,
     parallel initialization, and FAISS clustering behind a scikit-learn-compatible API.</p>
     <div class="hero-actions">
       <a class="primary" href="getting-started.html">Get started</a>
       <a class="secondary" href="https://github.com/felixlaplante0/fastkmeanspp/blob/main/examples/tutorial.ipynb">Run the MNIST example</a>
     </div>
   </section>

.. raw:: html

   <aside class="pypi-card">
     <div>
       <span class="pypi-kicker">PYTHON PACKAGE</span>
       <strong>Available on PyPI</strong>
       <p>Install FastKMeans++ in one command and keep the familiar KMeans workflow.</p>
     </div>
     <a href="https://pypi.org/project/fastkmeanspp/">View package&nbsp;→</a>
   </aside>

Why FastKMeans++?
-----------------

KMeans++ initialization repeatedly computes distances between every sample and a
small set of candidate centroids. FastKMeans++ moves this work into a
Highway-powered native kernel, using fused SIMD operations and parallel row
processing before handing the initialized centroids to FAISS.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Fast initialization
      :class-card: feature-card

      Spend less time selecting KMeans++ centroids on large, high-dimensional data.

   .. grid-item-card:: Portable SIMD
      :class-card: feature-card

      Google Highway dispatches vectorized fused operations for the available CPU.

   .. grid-item-card:: Familiar workflow
      :class-card: feature-card

      Fit, predict, inspect labels, and read cluster centers with a familiar estimator API.

Get started
-----------

Install ``fastkmeanspp`` from PyPI:

.. code-block:: bash

   python -m pip install fastkmeanspp

.. code-block:: python

   import numpy as np
   from fastkmeanspp import KMeans

   X = np.array([[0.0, 0.0], [0.1, 0.2], [4.0, 4.0], [4.2, 3.9]])
   model = KMeans(n_clusters=2, random_state=42)
   model.fit(X)

   labels = model.predict(X)

Highway and parallel distance computation
-------------------------------------------

The distance kernel uses `Google Highway <https://github.com/google/highway>`__
to dispatch portable SIMD implementations at runtime. Fused multiply-add
operations calculate squared distances several values at a time. Highway's
thread pool also divides independent sample rows across workers during
initialization. Set ``n_jobs=1`` for serial execution or ``n_jobs=-1`` to use
all available threads.

Tutorial
--------

The :doc:`tutorial` notebook compares FastKMeans++ with scikit-learn on MNIST.
It reports adjusted Rand index before timing both ``K=10`` and ``K=100``.

API Reference
-------------

.. autoclass:: fastkmeanspp.KMeans
   :members:
   :undoc-members:
   :show-inheritance:

.. toctree::
   :hidden:
   :maxdepth: 2

   getting-started
   tutorial
