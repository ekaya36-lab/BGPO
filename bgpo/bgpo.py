"""Boundary-Guided Composite Pool Oversampling (BGPO).

This module provides a compact, reusable implementation of the proposed
Boundary-Guided composite Pool Oversampling method for binary imbalanced
classification.

BGPO separates oversampling into two stages:
1. Build a composite synthetic candidate pool using SMOTE, ADASYN, MWMOTE,
   and G-SMOTE.
2. Select the most informative candidates according to a boundary-guided
   score computed from class-conditional neighborhood distances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_X_y

try:
    import smote_variants as sv
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "BGPO requires the 'smote-variants' package. Install it with: "
        "pip install smote-variants"
    ) from exc


GeneratorMap = Dict[str, object]


@dataclass
class BGPO:
    """Boundary-Guided composite Pool Oversampling.

    Parameters
    ----------
    augmentation_rate : float, default=1.0
        Fraction of the imbalance gap to close. ``1.0`` aims to make the
        minority class as large as the majority class.
    k_neighbors : int, default=5
        Maximum neighborhood size used by the oversampling generators and by
        the boundary-guided distance calculations. The effective value is
        automatically reduced when the minority class is small.
    random_state : int, default=42
        Random seed used by compatible generators and fallback sampling steps.
    generators : iterable of str or None, default=None
        Generator names used to construct the composite pool. If ``None``,
        ``("SMOTE", "ADASYN", "MWMOTE", "G_SMOTE")`` is used.
    epsilon : float, default=1e-9
        Numerical stability constant in the boundary score.

    Notes
    -----
    The boundary score for a synthetic candidate z is:

    ``score(z) = 1 / (|d_min(z) - d_maj(z)| + eps) *
                1 / (d_min(z) + d_maj(z) + eps)``

    where ``d_min`` and ``d_maj`` are average distances from z to its nearest
    minority and majority neighbors in the original training set.
    """

    augmentation_rate: float = 1.0
    k_neighbors: int = 5
    random_state: int = 42
    generators: Optional[Iterable[str]] = None
    epsilon: float = 1e-9

    def __post_init__(self) -> None:
        if self.augmentation_rate < 0:
            raise ValueError("augmentation_rate must be non-negative.")
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be at least 1.")

        default_generators = ("SMOTE", "ADASYN", "MWMOTE", "G_SMOTE")
        self.generators = tuple(self.generators or default_generators)
        self.generator_classes_: GeneratorMap = {
            "SMOTE": sv.SMOTE,
            "ADASYN": sv.ADASYN,
            "MWMOTE": sv.MWMOTE,
            "G_SMOTE": sv.G_SMOTE,
        }

    def fit_resample(self, X, y) -> Tuple[np.ndarray, np.ndarray]:
        """Return a resampled dataset using BGPO.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.
        y : array-like of shape (n_samples,)
            Binary class labels.

        Returns
        -------
        X_resampled : ndarray
            Original samples plus selected synthetic minority samples.
        y_resampled : ndarray
            Labels corresponding to ``X_resampled``.
        """
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            raise ValueError("BGPO currently supports binary classification only.")

        minority_label = classes[np.argmin(counts)]
        majority_label = classes[np.argmax(counts)]
        n_minority = int(np.min(counts))
        n_majority = int(np.max(counts))

        n_new = int(round((n_majority - n_minority) * self.augmentation_rate))
        if n_new <= 0 or n_minority < 2:
            return X.copy(), y.copy()

        pool_X = self._build_composite_pool(X, y, minority_label)
        if len(pool_X) == 0:
            return X.copy(), y.copy()

        selected_X = self._select_boundary_guided(
            pool_X=pool_X,
            X_min=X[y == minority_label],
            X_maj=X[y == majority_label],
            n_new=n_new,
        )

        selected_y = np.full(len(selected_X), minority_label, dtype=y.dtype)
        X_resampled = np.vstack([X, selected_X])
        y_resampled = np.concatenate([y, selected_y])
        return X_resampled, y_resampled

    def _safe_k(self, y: np.ndarray, minority_label) -> Optional[int]:
        n_minority = int(np.sum(y == minority_label))
        if n_minority < 2:
            return None
        return max(1, min(self.k_neighbors, n_minority - 1))

    def _build_composite_pool(self, X: np.ndarray, y: np.ndarray, minority_label) -> np.ndarray:
        safe_k = self._safe_k(y, minority_label)
        if safe_k is None:
            return np.empty((0, X.shape[1]), dtype=X.dtype)

        synthetic_parts = []
        for name in self.generators:
            if name not in self.generator_classes_:
                raise ValueError(
                    f"Unknown generator '{name}'. Available generators are: "
                    f"{', '.join(self.generator_classes_.keys())}."
                )
            X_res, y_res = self._fit_generator(
                self.generator_classes_[name], X, y, safe_k=safe_k
            )
            if X_res is None:
                continue

            new_X = X_res[len(X) :]
            new_y = y_res[len(y) :]
            if len(new_X) == 0:
                continue

            synthetic_minority = new_X[new_y == minority_label]
            if len(synthetic_minority) > 0:
                synthetic_parts.append(synthetic_minority)

        if not synthetic_parts:
            return np.empty((0, X.shape[1]), dtype=X.dtype)
        return np.vstack(synthetic_parts)

    def _fit_generator(self, generator_class, X: np.ndarray, y: np.ndarray, safe_k: int):
        """Fit a smote-variants generator with tolerant parameter handling."""
        candidate_param_sets = [
            {
                "proportion": self.augmentation_rate,
                "random_state": self.random_state,
                "n_neighbors": safe_k,
                "k_neighbors": safe_k,
            },
            {
                "proportion": self.augmentation_rate,
                "random_state": self.random_state,
                "k_neighbors": safe_k,
            },
            {
                "proportion": self.augmentation_rate,
                "random_state": self.random_state,
                "n_neighbors": safe_k,
            },
            {"proportion": self.augmentation_rate, "random_state": self.random_state},
            {"random_state": self.random_state, "n_neighbors": safe_k},
            {"random_state": self.random_state, "k_neighbors": safe_k},
            {"random_state": self.random_state},
            {},
        ]

        tried = set()
        for params in candidate_param_sets:
            key = tuple(sorted(params.items()))
            if key in tried:
                continue
            tried.add(key)
            try:
                generator = generator_class(**params)
                return generator.sample(X, y)
            except Exception:
                continue
        return None, None

    def _select_boundary_guided(
        self,
        pool_X: np.ndarray,
        X_min: np.ndarray,
        X_maj: np.ndarray,
        n_new: int,
    ) -> np.ndarray:
        if len(pool_X) <= n_new:
            return pool_X.copy()

        k_min = min(self.k_neighbors, len(X_min))
        k_maj = min(self.k_neighbors, len(X_maj))

        nn_min = NearestNeighbors(n_neighbors=k_min)
        nn_maj = NearestNeighbors(n_neighbors=k_maj)
        nn_min.fit(X_min)
        nn_maj.fit(X_maj)

        d_min, _ = nn_min.kneighbors(pool_X)
        d_maj, _ = nn_maj.kneighbors(pool_X)
        d_min_mean = d_min.mean(axis=1)
        d_maj_mean = d_maj.mean(axis=1)

        boundary_balance = 1.0 / (np.abs(d_min_mean - d_maj_mean) + self.epsilon)
        common_proximity = 1.0 / (d_min_mean + d_maj_mean + self.epsilon)
        scores = boundary_balance * common_proximity

        finite_scores = np.isfinite(scores)
        if not finite_scores.all():
            scores = np.where(finite_scores, scores, -np.inf)

        top_idx = np.argsort(scores)[-n_new:][::-1]
        return pool_X[top_idx]
