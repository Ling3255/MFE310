"""P4-M3: Domain-level geometry of intent centroids (mechanism check for domain-aware negatives)."""

from __future__ import annotations

import numpy as np


def domain_centroid_separation(features: np.ndarray, labels: np.ndarray, intent_domain: np.ndarray) -> dict[str, float]:
    """Mean cosine distance from each intent centroid to its nearest centroid in the SAME domain and in
    OTHER domains (L2-normalized CLS). Domain-aware negatives are predicted to enlarge the former."""
    unit = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    intents = np.unique(labels)
    centroids = np.stack([unit[labels == intent].mean(axis=0) for intent in intents])
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    distance = 1.0 - centroids @ centroids.T
    np.fill_diagonal(distance, np.inf)
    domains = intent_domain[intents]
    same = np.where(domains[:, None] == domains[None, :], distance, np.inf).min(axis=1)
    other = np.where(domains[:, None] != domains[None, :], distance, np.inf).min(axis=1)
    return {
        "same_domain_nearest_centroid_distance": float(np.mean(same[np.isfinite(same)])),
        "other_domain_nearest_centroid_distance": float(np.mean(other[np.isfinite(other)])),
    }
