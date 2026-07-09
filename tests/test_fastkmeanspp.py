"""Tests for the fastkmeanspp package."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from fastkmeanspp import KMeans


def _data() -> np.ndarray:
    return np.array(
        [
            [-1.2, -1.0],
            [-1.0, -0.8],
            [-0.8, -1.2],
            [0.9, 1.1],
            [1.1, 0.8],
            [1.3, 1.2],
        ]
    )


def test_fit():
    """Exercises fitting and fitted attributes."""
    X = _data()
    n_clusters = 2
    estimator = KMeans(n_clusters=n_clusters, n_iter=2, random_state=42)

    assert not hasattr(estimator, "cluster_centers_")
    assert clone(estimator).get_params()["n_clusters"] == n_clusters
    assert estimator.fit(X) is estimator

    assert estimator.cluster_centers_.shape == (2, 2)
    assert estimator.labels_.shape == (X.shape[0],)
    assert estimator.X_.dtype == np.float32
    assert np.isfinite(estimator.inertia_)


def test_predict():
    """Checks prediction and fitted feature validation."""
    X = _data()
    estimator = KMeans(n_clusters=2, n_iter=2, random_state=42)

    with pytest.raises(NotFittedError):
        estimator.predict(X)

    estimator.fit(X)
    labels = estimator.predict(X)

    assert labels.shape == (X.shape[0],)

    with pytest.raises(ValueError, match="X has 3 features"):
        estimator.predict(np.column_stack([X, np.ones(X.shape[0])]))


def test_params():
    """Checks deferred constructor parameter validation."""
    estimator = KMeans(n_clusters=0)

    assert estimator.n_clusters == 0
    with pytest.raises(ValueError, match="n_clusters"):
        estimator.fit(_data())


def test_too_many_clusters():
    """Checks fit rejects more clusters than samples."""
    with pytest.raises(ValueError, match="n_clusters must be less than or equal"):
        KMeans(n_clusters=7).fit(_data())


def test_trials():
    """Checks default local trials without mutating constructor parameters."""
    estimator = KMeans(n_clusters=3, n_iter=2, random_state=42)

    estimator.fit(_data())

    assert estimator.n_local_trials is None
