Fast KMeans++
=============

**fastkmeanspp** is a Python package that provides a ``KMeans`` estimator with a
faster KMeans++ centroid initialization. It follows the scikit-learn estimator API
and targets workloads where initialization is the bottleneck.

.. code-block:: bash

   pip install fastkmeanspp

See :doc:`getting-started` for a first example, or the :doc:`tutorial`. The package is
available on `PyPI <https://pypi.org/project/fastkmeanspp/>`_.

Why FastKMeans++?
-----------------

KMeans++ initialization repeatedly computes distances between every sample and a
small set of candidate centroids. FastKMeans++ moves this work into a
Highway-powered native kernel, using fused SIMD operations and parallel row
processing. The Lloyd iterations that follow reuse the same native kernels for
nearest-centroid assignment and centroid updates. Google Highway picks the
vectorized kernel for the available CPU at runtime, and the estimator exposes the
familiar ``fit``, ``predict``, ``labels_``, ``cluster_centers_`` and ``inertia_``
interface.

Quick example
-------------

.. code-block:: python

   import numpy as np
   from fastkmeanspp import KMeans

   X = np.array([[0.0, 0.0], [0.1, 0.2], [4.0, 4.0], [4.2, 3.9]])
   model = KMeans(n_clusters=2, random_state=42)
   model.fit(X)
   labels = model.predict(X)

The :doc:`getting-started` page covers threading options, and the :doc:`tutorial`
walks through a full example.

Learn more
----------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Get started
      :link: getting-started
      :link-type: doc

      Install FastKMeans++ and run your first clustering model.

   .. grid-item-card:: Highway
      :link: highway
      :link-type: doc

      See how SIMD distance calculations and parallel rows speed KMeans++ initialization.

   .. grid-item-card:: API reference
      :link: modules
      :link-type: doc

      Browse the scikit-learn-style ``KMeans`` estimator API.

.. toctree::
   :hidden:

   getting-started
   highway
   tutorial
   modules
