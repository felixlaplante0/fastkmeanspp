Fast KMeans++
=============

**fastkmeanspp** is a Python package that implements a KMeans clone from `scikit-learn` but with a much faster centroid initialization and optimized for speed with FAISS. It is designed to be a drop-in replacement for `scikit-learn`'s KMeans implementation.

Installation
------------

You can install the package via pip:

.. code-block:: bash

   pip install fastkmeanspp

API Reference
-------------

.. autoclass:: fastkmeanspp.KMeans
   :members:
   :undoc-members:
   :show-inheritance:
