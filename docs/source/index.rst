Fast KMeans++
=============

.. raw:: html

   <section class="hero">
     <img class="hero-logo" src="_static/fastkmeanspp-logo.svg" alt="Fast KMeans++ logo">
     <p class="eyebrow">K-MEANS++, BUILT FOR SPEED</p>
     <h1>Fast KMeans++ initialization.</h1>
     <p class="hero-copy">FastKMeans++ combines Highway SIMD distance kernels,
     parallel initialization, and FAISS clustering behind a scikit-learn-compatible API.</p>
     <div class="hero-actions">
       <a class="primary" href="getting-started.html">Get started</a>
       <a class="secondary" href="tutorial.html">See the tutorial</a>
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
   :maxdepth: 2

   getting-started
   highway
   tutorial
   modules
