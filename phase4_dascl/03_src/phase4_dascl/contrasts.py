"""P4-M2: Linear contrasts over the four paired arms (e.g. the 2x2 interaction) with two-level bootstrap.

A contrast c = sum_a w_a * metric(arm a). Per seed s this gives c_s; seeds are resampled as whole
quadruples (the four arms share the seed's initialization), evaluation rows are resampled within intent
(and OOS rows uniformly), and the mean contrast over resampled seeds forms the bootstrap distribution.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from clinc_domain_aux.confirmatory import intent_groups, stratified_indices


def _interval(distribution: np.ndarray, level: float) -> list[float]:
    tail = round((1.0 - level) / 2.0, 12)
    low, high = np.quantile(distribution, [tail, 1.0 - tail])
    return [float(low), float(high)]


def seed_contrasts(values: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    """``values[arm]`` is the per-seed metric vector; returns the per-seed contrast."""
    return sum(float(weight) * np.asarray(values[arm], dtype=np.float64) for arm, weight in weights.items())


def contrast_bootstrap(
    gold: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_boot: int,
    seed: int,
    level: float,
) -> dict[str, Any]:
    """Classification metric contrast; ``predictions[arm]`` has shape (n_seeds, n_rows)."""
    n_seeds = next(iter(predictions.values())).shape[0]
    rng = np.random.default_rng(seed)
    groups = intent_groups(gold)
    distribution = np.empty(n_boot)
    for b in range(n_boot):
        seeds = rng.integers(0, n_seeds, size=n_seeds)
        rows = stratified_indices(groups, rng)
        sampled = gold[rows]
        distribution[b] = np.mean([
            sum(float(w) * metric(sampled, predictions[arm][s, rows]) for arm, w in weights.items()) for s in seeds
        ])
    return {"ci": _interval(distribution, level), "level": level, "n_boot": n_boot, "seed": seed, "distribution": distribution}


def contrast_bootstrap_oos(
    gold: np.ndarray,
    scores: Mapping[str, tuple[np.ndarray, np.ndarray]],
    weights: Mapping[str, float],
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_boot: int,
    seed: int,
    level: float,
) -> dict[str, Any]:
    """OOS metric contrast; ``scores[arm] = (in_scope (n_seeds, n), oos (n_seeds, m))``."""
    n_seeds, n_oos = next(iter(scores.values()))[1].shape
    rng = np.random.default_rng(seed)
    groups = intent_groups(gold)
    distribution = np.empty(n_boot)
    for b in range(n_boot):
        seeds = rng.integers(0, n_seeds, size=n_seeds)
        rows = stratified_indices(groups, rng)
        oos_rows = rng.integers(0, n_oos, size=n_oos)
        distribution[b] = np.mean([
            sum(float(w) * metric(scores[arm][0][s, rows], scores[arm][1][s, oos_rows]) for arm, w in weights.items())
            for s in seeds
        ])
    return {"ci": _interval(distribution, level), "level": level, "n_boot": n_boot, "seed": seed, "distribution": distribution}
