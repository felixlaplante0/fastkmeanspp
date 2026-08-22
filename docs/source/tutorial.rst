MNIST tutorial
==============

The executable notebook is available at
`examples/tutorial.ipynb <https://github.com/felixlaplante0/fastkmeanspp/blob/main/examples/tutorial.ipynb>`__.
It loads MNIST, compares adjusted Rand index first, and then measures runtime
for ``K=10`` and ``K=100``.

.. note::

   The notebook downloads MNIST from OpenML the first time it runs. Timings
   depend on the machine, CPU, thread settings, and installed FAISS build.
