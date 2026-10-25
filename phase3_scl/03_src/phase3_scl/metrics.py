"""P3-M3: Calibration and representation-geometry metrics used to explain *why* SCL helps or not."""

from __future__ import annotations

from typing import Any

import numpy as np


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> dict[str, Any]:
    """Equal-width ECE over max-probability confidence; also returns the reliability-diagram bins."""
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if confidence.shape != correct.shape or confidence.ndim != 1 or len(confidence) == 0:
        raise ValueError("confidence and correct must be equal-length, non-empty 1-D arrays")
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        mask = which == b
        if not mask.any():
            continue
        accuracy, mean_confidence = float(correct[mask].mean()), float(confidence[mask].mean())
        ece += mask.mean() * abs(accuracy - mean_confidence)
        table.append({"bin": b, "count": int(mask.sum()), "accuracy": accuracy, "confidence": mean_confidence})
    return {"ece": float(ece), "bins": table}


def representation_geometry(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    oos_features: np.ndarray | None = None,
) -> dict[str, float]:
    """Compactness and separation of L2-normalized CLS features.

    * ``within_between_ratio``: mean squared distance to the own class centroid divided by the mean
      squared distance between class centroids (lower = tighter, better separated classes).
    * ``min_centroid_cosine_distance``: closest pair of class centroids (separation of the nearest intents).
    * ``test_*`` / ``oos_*``: cosine distance from each test (or OOS) example to its nearest train centroid.
    """
    def unit(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)

    train, labels = unit(train_features), np.asarray(train_labels)
    classes = np.unique(labels)
    centroids = np.stack([train[labels == c].mean(axis=0) for c in classes])
    within = np.mean([np.sum((train[labels == c] - centroids[i]) ** 2, axis=1).mean() for i, c in enumerate(classes)])
    differences = centroids[:, None, :] - centroids[None, :, :]
    pairwise = np.sum(differences ** 2, axis=2)[np.triu_indices(len(classes), k=1)]
    unit_centroids = unit(centroids)
    cosine = unit_centroids @ unit_centroids.T
    np.fill_diagonal(cosine, -np.inf)
    result = {
        "within_between_ratio": float(within / pairwise.mean()),
        "min_centroid_cosine_distance": float(1.0 - cosine.max()),
    }
    test_distance = 1.0 - (unit(test_features) @ unit_centroids.T).max(axis=1)
    result["test_nearest_centroid_distance_mean"] = float(test_distance.mean())
    if oos_features is not None and len(oos_features):
        oos_distance = 1.0 - (unit(oos_features) @ unit_centroids.T).max(axis=1)
        result["oos_nearest_centroid_distance_mean"] = float(oos_distance.mean())
        result["oos_minus_test_distance"] = float(oos_distance.mean() - test_distance.mean())
    return result
