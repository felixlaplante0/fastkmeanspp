Getting started
===============

Install ``fastkmeanspp`` from PyPI:

.. code-block:: bash

   python -m pip install fastkmeanspp

Use the estimator like ``scikit-learn``'s ``KMeans``:

.. code-block:: python

   import numpy as np
   from fastkmeanspp import KMeans

   X = np.array([[0.0, 0.0], [0.1, 0.2], [4.0, 4.0], [4.2, 3.9]])
   model = KMeans(n_clusters=2, random_state=42)
   model.fit(X)
   labels = model.predict(X)

Set ``n_jobs=1`` for serial centroid initialization or ``n_jobs=-1`` to use
all available threads.
