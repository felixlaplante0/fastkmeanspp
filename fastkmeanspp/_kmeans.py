from functools import cached_property
from numbers import Integral
from typing import Self, cast

import faiss
import numpy as np
import numpy.typing as npt
from scipy.linalg.blas import sgemm
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils._param_validation import Interval, validate_params
from sklearn.utils.validation import check_is_fitted


class KMeans(BaseEstimator, ClusterMixin):
    """K-means clustering using FAISS.

    Attributes:
        n_clusters (int): The number of clusters to form.
        n_iter (int): The number of iterations to run the k-means
            algorithm.
        n_local_trials  (int | None): The number of seeding trials for
            centroids initialization.
        X_ (np.ndarray | None): The input data matrix.
        random_state (int | None) Determines random number generation for
            centroid initialization.
        cluster_centers_ (np.ndarray | None): Coordinates of cluster centers.
        labels_ (np.ndarray | None): Labels of each point (index) in X.
    """

    n_clusters: int
    n_iter: int
    n_local_trials: int | None
    random_state: int | None
    X_: np.ndarray | None
    cluster_centers_: np.ndarray | None
    labels_: np.ndarray | None

    @validate_params(
        {
            "n_clusters": [Interval(Integral, 1, None, closed="left")],
            "n_iter": [Interval(Integral, 1, None, closed="left")],
            "n_local_trials": [Interval(Integral, 1, None, closed="left"), None],
            "random_state": ["random_state"],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        n_clusters: int = 8,
        n_iter: int = 20,
        n_local_trials: int | None = None,
        random_state: int | None = None,
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
        """
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.n_local_trials = n_local_trials
        self.random_state = random_state
        self.X_ = None
        self.cluster_centers_ = None
        self.labels_ = None

    @staticmethod
    def _dists(X: np.ndarray, y: np.ndarray, XX: np.ndarray) -> np.ndarray:
        """Computes the pairwise distances between a fixed data matrix and some points.

        Args:
            X (np.ndarray): The fixed data matrix.
            y (np.ndarray): The non fixed points.
            XX (np.ndarray): The fixed matrix squared norm.

        Returns:
            np.ndarray: The computed pairwise distances.
        """
        yy = np.einsum("ij,ij->i", y, y)
        dists = XX - cast(np.ndarray, sgemm(2.0, X, y, trans_b=True)) + yy
        np.clip(dists, 0, None, out=dists)

        return dists

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

        XX = np.einsum("ij,ij->i", X, X)[:, None]

        dists = self._dists(X, centroids[:1], XX).ravel()
        inertia = dists.sum()

        if self.n_local_trials is None:
            self.n_local_trials = 2 + int(np.log(self.n_clusters))

        for i in range(1, self.n_clusters):
            candidate_ids = rng.choice(
                X.shape[0], size=self.n_local_trials, p=dists / inertia
            )
            candidates = np.asfortranarray(X[candidate_ids])

            current_candidates_dists = self._dists(X, candidates, XX)
            candidates_dists = np.minimum(current_candidates_dists, dists[:, None])

            inertias = candidates_dists.sum(axis=0)
            best_inertia = inertias.argmin()
            best_candidate = candidate_ids[best_inertia]
            dists = candidates_dists[:, best_inertia]
            inertia = inertias[best_inertia]

            centroids[i] = X[best_candidate]

        return centroids

    @staticmethod
    def _validate_X(X: npt.ArrayLike, order: str) -> np.ndarray:
        """Validates and converts the data matrix.

        Args:
            X (npt.ArrayLike): The fixed data matrix.
            order (str): The order of the array.

        Raises:
            ValueError: If `X` is not 2D array.
            ValueError: If `X` contains inf values.
            ValueError: If `X` contains NaN values.

        Returns:
            np.ndarray: The validated and converted data matrix.
        """
        X_f32 = cast(np.ndarray, np.asarray(X, dtype=np.float32, order=order))  # type: ignore

        if X.ndim != 2:  # noqa: PLR2004
            raise ValueError("X must be a 2D array")

        if np.isinf(X_f32).any():
            raise ValueError("X must not contain inf values")
        if np.isnan(X_f32).any():
            raise ValueError("X must not contain NaN values")

        return X_f32

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
            ValueError: If `X` contains inf or NaN values.

        Returns:
            Self: The fitted model.
        """
        X_f32 = self._validate_X(X, order="F")
        index = faiss.IndexFlatL2(X_f32.shape[1])
        kmeans = faiss.Clustering(X_f32.shape[1], self.n_clusters)

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
            faiss.vector_to_array(kmeans.centroids).reshape(  # type: ignore
                self.n_clusters, X_f32.shape[1]
            ),
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
            ValueError: If `X` contains inf or NaN values.
            ValueError: If `self.cluster_centers_` is not set.

        Returns:
            np.ndarray The predicted cluster indices.
        """
        check_is_fitted(self, "cluster_centers_")

        X_f32 = self._validate_X(X, order="K")
        index = faiss.IndexFlatL2(X_f32.shape[1])
        index.add(self.cluster_centers_)  # type: ignore

        return cast(np.ndarray, index.search(X_f32, 1)[1]).ravel()  # type: ignore

    @cached_property
    def inertia_(self) -> float:
        """Get the inertia of the fitted model.

        Args:
            X (npt.ArrayLike): The input data.

        Raises:
            ValueError: If `self.X_`, `self.labels_` and `self.cluster_centers_` are
                not all set.

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
