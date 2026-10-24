"""M07: Shared-within-class Mahalanobis detection and threshold selection."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf

from .common import hash_array, stable_hash


class MahalanobisDetector:
    """Fit class means and a Ledoit-Wolf covariance on within-class residuals."""

    def __init__(self, num_intents: int = 150) -> None:
        if not isinstance(num_intents, (int, np.integer)) or num_intents <= 0:
            raise ValueError("num_intents must be a positive integer")
        self.num_intents = int(num_intents)

    def fit(self, features: Any, labels: Any, sample_ids: Any | None = None) -> "MahalanobisDetector":
        values = np.asarray(features, dtype=np.float64)
        intent_ids = np.asarray(labels)
        if values.ndim != 2 or len(values) == 0 or values.shape[1] == 0 or not np.isfinite(values).all():
            raise ValueError("features must be a finite, non-empty two-dimensional array")
        if intent_ids.ndim != 1 or len(intent_ids) != len(values):
            raise ValueError("labels must be one-dimensional and match features")
        try:
            converted_ids = intent_ids.astype(np.int64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("labels must contain integer IDs") from exc
        if not np.all(intent_ids == converted_ids):
            raise ValueError("labels must contain integer IDs")
        intent_ids = converted_ids
        if np.any(intent_ids < 0) or np.any(intent_ids >= self.num_intents):
            raise ValueError("labels must be in [0, num_intents)")
        if sample_ids is not None and len(sample_ids) != len(values):
            raise ValueError("sample_ids must match features")

        means = np.empty((self.num_intents, values.shape[1]), dtype=np.float64)
        residuals = np.empty_like(values)
        for intent in range(self.num_intents):
            mask = intent_ids == intent
            if not mask.any():
                raise ValueError(f"intent {intent} has no fitting samples")
            means[intent] = values[mask].mean(axis=0, dtype=np.float64)
            residuals[mask] = values[mask] - means[intent]
        estimator = LedoitWolf(assume_centered=True).fit(residuals)
        covariance = np.asarray(estimator.covariance_, dtype=np.float64)
        precision = np.linalg.solve(covariance, np.eye(covariance.shape[0], dtype=np.float64))
        precision = (precision + precision.T) / 2.0
        if not np.isfinite(means).all() or not np.isfinite(covariance).all() or not np.isfinite(precision).all():
            raise ValueError("non-finite fitted Mahalanobis parameters")

        self.class_means_ = means
        self.covariance_ = covariance
        self.precision_ = precision
        self.n_features_in_ = values.shape[1]
        self.n_samples_fit_ = len(values)
        self.dtype_ = "float64"
        self.sample_hash_ = stable_hash({
            "features": hash_array(values),
            "labels": hash_array(np.ascontiguousarray(intent_ids)),
            "sample_ids": None if sample_ids is None else [str(value) for value in sample_ids],
        })
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "precision_"):
            raise ValueError("detector has not been fitted")

    def mahalanobis_scores(self, features: Any, batch_size: int = 1024) -> np.ndarray:
        """Return the nearest class's squared Mahalanobis distance for each row."""
        self._check_fitted()
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or len(values) == 0 or values.shape[1] != self.n_features_in_ or not np.isfinite(values).all():
            raise ValueError("features must be finite and have the fitted feature dimension")
        if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        mean_quadratic = np.einsum("ij,ij->i", self.class_means_ @ self.precision_, self.class_means_)
        scores = np.empty(len(values), dtype=np.float64)
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            batch = values[start:stop]
            projected = batch @ self.precision_
            distances = (
                np.einsum("ij,ij->i", projected, batch)[:, None]
                + mean_quadratic[None, :]
                - 2.0 * (projected @ self.class_means_.T)
            )
            scores[start:stop] = np.maximum(distances.min(axis=1), 0.0)
        if not np.isfinite(scores).all():
            raise ValueError("Mahalanobis scores are not finite")
        return scores

    def score(self, features: Any, batch_size: int = 1024) -> np.ndarray:
        return self.mahalanobis_scores(features, batch_size=batch_size)

    @property
    def metadata(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            "num_intents": self.num_intents,
            "n_features_in": self.n_features_in_,
            "n_samples_fit": self.n_samples_fit_,
            "dtype": self.dtype_,
            "sample_hash": self.sample_hash_,
            "covariance_estimator": "LedoitWolf(assume_centered=True) on within-class residuals",
        }

    def save(self, path: str | PathLike[str]) -> None:
        self._check_fitted()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            np.savez(
                handle,
                class_means=self.class_means_,
                covariance=self.covariance_,
                precision=self.precision_,
                metadata=np.asarray(json.dumps(self.metadata, sort_keys=True, allow_nan=False)),
            )

    @classmethod
    def load(cls, path: str | PathLike[str]) -> "MahalanobisDetector":
        with np.load(Path(path), allow_pickle=False) as archive:
            required = {"class_means", "covariance", "precision", "metadata"}
            if not required.issubset(archive.files):
                raise ValueError("invalid Mahalanobis detector archive")
            metadata = json.loads(str(archive["metadata"].item()))
            means = np.asarray(archive["class_means"], dtype=np.float64)
            covariance = np.asarray(archive["covariance"], dtype=np.float64)
            precision = np.asarray(archive["precision"], dtype=np.float64)
        detector = cls(num_intents=int(metadata["num_intents"]))
        expected = (detector.num_intents, int(metadata["n_features_in"]))
        if means.shape != expected or covariance.shape != (expected[1], expected[1]) or precision.shape != covariance.shape:
            raise ValueError("inconsistent detector array shapes")
        if not np.isfinite(means).all() or not np.isfinite(covariance).all() or not np.isfinite(precision).all():
            raise ValueError("non-finite detector archive")
        detector.class_means_ = means
        detector.covariance_ = covariance
        detector.precision_ = precision
        detector.n_features_in_ = expected[1]
        detector.n_samples_fit_ = int(metadata["n_samples_fit"])
        detector.dtype_ = str(metadata["dtype"])
        detector.sample_hash_ = str(metadata["sample_hash"])
        if detector.dtype_ != "float64" or not detector.sample_hash_:
            raise ValueError("invalid detector metadata")
        return detector


def select_threshold(
    is_scores: Any,
    oos_scores: Any,
    min_is_acceptance: float = 0.95,
) -> dict[str, Any]:
    """Maximize validation OOS recall under the IS constraint.

    Ties in OOS recall keep the threshold with the highest IS acceptance, so the
    rule does not reject extra in-scope samples for no OOS gain.
    """
    in_scope = np.asarray(is_scores, dtype=np.float64)
    out_of_scope = np.asarray(oos_scores, dtype=np.float64)
    if in_scope.ndim != 1 or out_of_scope.ndim != 1 or len(in_scope) == 0 or len(out_of_scope) == 0:
        raise ValueError("threshold selection requires non-empty one-dimensional IS and OOS scores")
    if not np.isfinite(in_scope).all() or not np.isfinite(out_of_scope).all():
        raise ValueError("scores must be finite")
    if not 0 <= min_is_acceptance <= 1:
        raise ValueError("min_is_acceptance must be in [0, 1]")

    unique = np.unique(np.concatenate([in_scope, out_of_scope]))
    candidates = np.concatenate([
        np.atleast_1d(np.nextafter(unique.min(), -np.inf, dtype=np.float64)),
        unique,
        np.nextafter(unique, np.inf, dtype=np.float64),
    ])
    candidates = np.unique(candidates)
    feasible: list[tuple[float, float, float, float]] = []
    for tau in candidates:
        is_acceptance = float(np.mean(in_scope < tau))
        if is_acceptance < min_is_acceptance:
            continue
        oos_recall = float(np.mean(out_of_scope >= tau))
        feasible.append((float(tau), is_acceptance, oos_recall, 1.0 - oos_recall))
    if not feasible:
        raise ValueError("no threshold satisfies the IS acceptance constraint")
    best_recall = max(row[2] for row in feasible)
    tied = [row for row in feasible if row[2] == best_recall]
    best_acceptance = max(row[1] for row in tied)
    tau, is_acceptance, oos_recall, oos_fa = min(row for row in tied if row[1] == best_acceptance)
    return {
        "threshold": tau,
        "tau": tau,
        "is_acceptance": is_acceptance,
        "oos_recall": oos_recall,
        "oos_false_acceptance": oos_fa,
        "min_is_acceptance": float(min_is_acceptance),
        "n_is": len(in_scope),
        "n_oos": len(out_of_scope),
        "n_candidates": len(candidates),
        "n_tied_optimal_thresholds": len(tied),
        "threshold_rule": "reject iff score >= tau; accept iff score < tau",
        "selection_rule": "maximize OOS recall, then maximize IS acceptance, then choose smallest tau",
    }
