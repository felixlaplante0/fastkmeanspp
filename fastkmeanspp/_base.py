from functools import cached_property
from numbers import Integral
from typing import ClassVar, Self, cast

import faiss
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils._param_validation import Interval, Options, validate_params
from sklearn.utils.validation import check_is_fitted, validate_data

from ._highway import _CdistWorker


class KMeans(ClusterMixin, BaseEstimator):
    """K-means clustering using FAISS.

    Attributes:
        n_clusters (int): The number of clusters to form.
        n_iter (int): The number of iterations to run the k-means algorithm.
        n_local_trials  (int | None): The number of seeding trials for centroids
            initialization.
        n_jobs (int | None): Number of threads used for local trials. ``None`` uses
            one thread, ``-1`` uses all available threads.
        X_ (np.ndarray | None): The input data matrix.
        random_state (int | None) Determines random number generation for centroid
            initialization.
        cluster_centers_ (np.ndarray | None): Coordinates of cluster centers.
        labels_ (np.ndarray | None): Labels of each point (index) in X.
    """

    n_clusters: int
    n_iter: int
    n_local_trials: int | None
    n_jobs: int | None
    random_state: int | None
    X_: np.ndarray
    cluster_centers_: np.ndarray
    labels_: np.ndarray

    _parameter_constraints: ClassVar[dict] = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "n_iter": [Interval(Integral, 1, None, closed="left")],
        "n_local_trials": [Interval(Integral, 1, None, closed="left"), None],
        "n_jobs": [
            Options(Integral, {-1}),
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_clusters: int = 8,
        n_iter: int = 20,
        n_local_trials: int | None = None,
        random_state: int | None = None,
        n_jobs: int | None = -1,
    ):
        """Initializes the KMeans class.

        Args:
            n_clusters (int, optional): The number of clusters to form. Defaults to 8.
            n_iter (int, optional): The number of iterations to run the k-means
                algorithm. Defaults to 20.
            n_local_trials (int | None, optional): The number of seeding trials for
                centroids initialization. Defaults to None.
            random_state (int | None, optional) Determines random number generation for
                centroid initialization. Defaults to None.
            n_jobs (int | None, optional): Number of threads used for local trials.
                ``None`` uses one thread and ``-1`` uses all available threads.
                Defaults to -1.
        """
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.n_local_trials = n_local_trials
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initializes the centroids in a K-means++ fashion.

        Args:
            X (np.ndarray): The fixed data matrix.

        Returns:
            np.ndarray: The initialized centroids.
        """
        rng = np.random.default_rng(self.random_state)

        centroids = np.empty((self.n_clusters, X.shape[1]), dtype=X.dtype)
        centroids[0] = X[rng.integers(X.shape[0])]

        n_jobs = 1 if self.n_jobs is None else 0 if self.n_jobs == -1 else self.n_jobs
        n_local_trials = self.n_local_trials
        if n_local_trials is None:
            n_local_trials = 2 + int(np.log(self.n_clusters))

        cdist = _CdistWorker(n_jobs)
        distances = cdist(X, centroids[:1]).ravel()

        for i in range(1, self.n_clusters):
            probabilities = np.asarray(distances, dtype=np.float64)
            probabilities /= probabilities.sum()
            candidate_ids = rng.choice(X.shape[0], size=n_local_trials, p=probabilities)
            candidates = X[candidate_ids]

            candidate_distances, inertias = cdist.minimum(X, candidates, distances)
            best_inertia = inertias.argmin()
            best_candidate = candidate_ids[best_inertia]
            distances = candidate_distances[:, best_inertia]
            centroids[i] = X[best_candidate]

        return centroids

    @validate_params(
        {
            "X": ["array-like"],
            "y": [None],
        },
        prefer_skip_nested_validation=True,
    )
    def fit(self, X: npt.ArrayLike, y: None = None) -> Self:  # noqa: ARG002
        """Run k-means clustering on the input data X.

        Args:
            X (npt.ArrayLike): Input data matrix to cluster.
            y (None, optional): Placeholder for y.

        Raises:
            ValueError: If ``X`` contains inf or NaN values.

        Returns:
            Self: The fitted model.
        """
        self._validate_params()

        X_f32 = np.ascontiguousarray(
            cast(np.ndarray, validate_data(self, X, dtype=np.float32))
        )
        n, d = X_f32.shape
        if self.n_clusters > n:
            raise ValueError("n_clusters must be less than or equal to n_samples.")

        index = faiss.IndexFlatL2(d)
        kmeans = faiss.Clustering(d, self.n_clusters)
        init_centroids = self._init_centroids(X_f32)

        kmeans.centroids.resize(init_centroids.size)
        faiss.copy_array_to_vector(init_centroids.ravel(), kmeans.centroids)  # type: ignore
        kmeans.niter = self.n_iter
        kmeans.min_points_per_centroid = 0
        kmeans.max_points_per_centroid = -1
        kmeans.train(X_f32, index)  # type: ignore

        self.X_ = X_f32
        self.cluster_centers_ = cast(
            np.ndarray,
            faiss.vector_to_array(kmeans.centroids).reshape(self.n_clusters, d),  # type: ignore
        )
        self.labels_ = cast(np.ndarray, index.search(X_f32, 1)[1].ravel())  # type: ignore

        return self

    @validate_params(
        {
            "X": ["array-like"],
        },
        prefer_skip_nested_validation=True,
    )
    def predict(self, X: npt.ArrayLike) -> np.ndarray:
        """Predict the nearest cluster index for each input data point.

        Args:
            X (npt.ArrayLike): The input data.

        Raises:
            ValueError: If ``X`` contains inf or NaN values.
            ValueError: If ``self.cluster_centers_`` is not set.

        Returns:
            np.ndarray The predicted cluster indices.
        """
        check_is_fitted(self, "cluster_centers_")

        X_f32 = cast(
            np.ndarray,
            validate_data(self, X, dtype=np.float32, reset=False),
        )
        index = faiss.IndexFlatL2(X_f32.shape[1])
        index.add(self.cluster_centers_)  # type: ignore

        return cast(np.ndarray, index.search(X_f32, 1)[1]).ravel()  # type: ignore

    @cached_property
    def inertia_(self) -> float:
        """Get the inertia of the fitted model.

        Args:
            X (npt.ArrayLike): The input data.

        Raises:
            ValueError: If ``self.X_``, ``self.labels_`` and ``self.cluster_centers_``
                are not all set.

        Returns:
            float: The inertia of the fitted model.
        """
        check_is_fitted(self, ["X_", "labels_", "cluster_centers_"])

        return float(
            np.sum(
                (
                    cast(np.ndarray, self.X_)
                    - cast(np.ndarray, self.cluster_centers_)[
                        cast(np.ndarray, self.labels_)
                    ]
                )
                ** 2
            )
        )
