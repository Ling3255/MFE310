"""M09: Seed summaries, paired stratified bootstrap, and overlap sensitivity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .evaluation import classification_metrics

BOOTSTRAP_CONDITION = (
    "Conditional on the fitted models and selected lambda; the interval reflects test-sample "
    "sampling variability, not retraining, lambda selection, or historical test exposure."
)


def summarize_seeds(values: Mapping[Any, float] | Sequence[float]) -> dict[str, Any]:
    """Report seed values, their mean, and sample standard deviation (ddof=1)."""
    if isinstance(values, Mapping):
        if not values:
            raise ValueError("at least one seed is required")
        seeds = list(values)
        numbers = np.asarray([values[seed] for seed in seeds], dtype=np.float64)
    else:
        numbers = np.asarray(values, dtype=np.float64)
        if numbers.ndim != 1 or len(numbers) == 0:
            raise ValueError("values must be a non-empty one-dimensional sequence")
        seeds = list(range(len(numbers)))
    if not np.isfinite(numbers).all():
        raise ValueError("seed metrics must be finite")
    if len(numbers) < 2:
        raise ValueError("sample standard deviation requires at least two seeds")
    return {
        "values": {seed: float(value) for seed, value in zip(seeds, numbers)},
        "mean": float(numbers.mean()),
        "std": float(numbers.std(ddof=1)),
        "n_seeds": len(numbers),
        "ddof": 1,
    }


def paired_seed_differences(
    ce_values: Mapping[Any, float] | Sequence[float],
    dom_values: Mapping[Any, float] | Sequence[float],
) -> dict[str, Any]:
    """Calculate Dom_s - CE_s without pairing different seeds."""
    if isinstance(ce_values, Mapping) or isinstance(dom_values, Mapping):
        if not isinstance(ce_values, Mapping) or not isinstance(dom_values, Mapping):
            raise ValueError("CE and Dom inputs must both be seed mappings")
        if set(ce_values) != set(dom_values):
            raise ValueError("CE and Dom seed sets differ")
        ce = np.asarray([ce_values[seed] for seed in sorted(ce_values)], dtype=np.float64)
        dom = np.asarray([dom_values[seed] for seed in sorted(dom_values)], dtype=np.float64)
        seeds = sorted(ce_values)
    else:
        ce = np.asarray(ce_values, dtype=np.float64)
        dom = np.asarray(dom_values, dtype=np.float64)
        if ce.ndim != 1 or dom.shape != ce.shape or len(ce) == 0:
            raise ValueError("CE and Dom sequences must have the same non-empty shape")
        seeds = list(range(len(ce)))
    differences = dom - ce
    summary = summarize_seeds(differences)
    return {
        "seed_deltas": {seed: float(value) for seed, value in zip(seeds, differences)},
        "mean_delta": summary["mean"],
        "std_delta": summary["std"],
        "n_seeds": len(seeds),
    }


def _field(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row:
            return row[name]
    raise ValueError(f"prediction row lacks all of {list(names)}")


def _arrays(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str] | None, np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("prediction groups must not be empty")
    sample_ids = [str(_field(row, ["sample_id"])) for row in rows] if "sample_id" in rows[0] else None
    if sample_ids is not None and len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_id values must be unique within a seed")
    gold = np.asarray([int(_field(row, ["gold_intent", "gold_id", "gold_intent_id"])) for row in rows], dtype=np.int64)
    pred = np.asarray([int(_field(row, ["pred_intent", "pred_id", "pred_intent_id"])) for row in rows], dtype=np.int64)
    return sample_ids, gold, pred


def _prediction_groups(predictions: Any) -> dict[Any, list[Mapping[str, Any]]]:
    if isinstance(predictions, Mapping):
        return {seed: list(rows) for seed, rows in predictions.items()}
    rows = list(predictions)
    if not rows:
        raise ValueError("predictions must not be empty")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("predictions must contain row mappings")
    if "seed" not in rows[0]:
        return {0: rows}
    groups: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["seed"], []).append(row)
    return groups


def _align_groups(ce_predictions: Any, dom_predictions: Any, num_intents: int) -> dict[Any, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    ce_groups = _prediction_groups(ce_predictions)
    dom_groups = _prediction_groups(dom_predictions)
    if set(ce_groups) != set(dom_groups):
        raise ValueError("CE and Dom seed sets differ")
    aligned = {}
    for seed in sorted(ce_groups):
        ce_ids, gold, ce_pred = _arrays(ce_groups[seed])
        dom_ids, dom_gold, dom_pred = _arrays(dom_groups[seed])
        if ce_ids is not None or dom_ids is not None:
            if ce_ids is None or dom_ids is None:
                raise ValueError("both models must provide sample_id for pairing")
            positions = {sample_id: index for index, sample_id in enumerate(dom_ids)}
            if set(ce_ids) != set(dom_ids):
                raise ValueError("CE and Dom test sample sets differ")
            order = np.asarray([positions[sample_id] for sample_id in ce_ids], dtype=np.int64)
            dom_gold = dom_gold[order]
            dom_pred = dom_pred[order]
        elif len(gold) != len(dom_gold):
            raise ValueError("CE and Dom prediction lengths differ")
        if not np.array_equal(gold, dom_gold):
            raise ValueError("CE and Dom gold labels differ")
        if np.any(gold < 0) or np.any(gold >= num_intents) or np.any(ce_pred < 0) or np.any(ce_pred >= num_intents) or np.any(dom_pred < 0) or np.any(dom_pred >= num_intents):
            raise ValueError("intent IDs must be in [0, num_intents)")
        aligned[seed] = (gold, ce_pred, dom_pred)
    return aligned


def _full_macro_f1(gold: np.ndarray, pred: np.ndarray, num_intents: int) -> float:
    confusion = np.bincount(gold * num_intents + pred, minlength=num_intents**2).reshape(num_intents, num_intents)
    true_positive = np.diag(confusion).astype(np.float64)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2.0 * true_positive, denominator, out=np.zeros_like(true_positive), where=denominator > 0)
    return float(f1.mean())


def _bootstrap_macro_f1(gold: np.ndarray, pred: np.ndarray, indices: np.ndarray, num_intents: int) -> np.ndarray:
    n_bootstrap, _, samples_per_intent = indices.shape
    sampled_pred = pred[indices]
    true_classes = np.arange(num_intents)[None, :, None]
    true_positive = (sampled_pred == true_classes).sum(axis=2)
    offsets = np.arange(n_bootstrap, dtype=np.int64)[:, None, None] * num_intents
    predicted_count = np.bincount(
        (sampled_pred + offsets).ravel(),
        minlength=n_bootstrap * num_intents,
    ).reshape(n_bootstrap, num_intents)
    denominator = predicted_count + samples_per_intent
    f1 = np.divide(2.0 * true_positive, denominator, out=np.zeros_like(true_positive, dtype=np.float64), where=denominator > 0)
    return f1.mean(axis=1)


def intent_stratified_paired_bootstrap(
    ce_predictions: Any,
    dom_predictions: Any,
    *,
    num_intents: int = 150,
    n_bootstrap: int = 1000,
    samples_per_intent: int = 30,
    seed: int = 20261024,
    return_indices: bool = False,
) -> dict[str, Any]:
    """Paired intent-stratified bootstrap; all models and seeds share every index draw."""
    if not isinstance(num_intents, (int, np.integer)) or num_intents <= 0:
        raise ValueError("num_intents must be a positive integer")
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not isinstance(samples_per_intent, (int, np.integer)) or samples_per_intent <= 0:
        raise ValueError("samples_per_intent must be positive")
    aligned = _align_groups(ce_predictions, dom_predictions, num_intents)
    reference_gold = next(iter(aligned.values()))[0]
    groups = []
    for intent in range(num_intents):
        positions = np.flatnonzero(reference_gold == intent)
        if len(positions) == 0:
            raise ValueError(f"intent {intent} has no test rows")
        groups.append(positions)

    rng = np.random.default_rng(seed)
    indices = np.empty((n_bootstrap, num_intents, samples_per_intent), dtype=np.int64)
    for intent, positions in enumerate(groups):
        local = rng.integers(0, len(positions), size=(n_bootstrap, samples_per_intent))
        indices[:, intent, :] = positions[local]

    observed: dict[Any, float] = {}
    bootstrap_deltas: dict[Any, np.ndarray] = {}
    for model_seed, (gold, ce_pred, dom_pred) in aligned.items():
        if not np.array_equal(gold, reference_gold):
            raise ValueError("all seeds must have the same aligned gold labels")
        ce_full = _full_macro_f1(gold, ce_pred, num_intents)
        dom_full = _full_macro_f1(gold, dom_pred, num_intents)
        observed[model_seed] = dom_full - ce_full
        ce_bootstrap = _bootstrap_macro_f1(gold, ce_pred, indices, num_intents)
        dom_bootstrap = _bootstrap_macro_f1(gold, dom_pred, indices, num_intents)
        bootstrap_deltas[model_seed] = dom_bootstrap - ce_bootstrap
    delta_matrix = np.column_stack([bootstrap_deltas[model_seed] for model_seed in sorted(aligned)])
    mean_deltas = delta_matrix.mean(axis=1)
    interval = np.quantile(mean_deltas, [0.025, 0.975])
    result = {
        "observed_seed_deltas": observed,
        "observed_mean_delta": float(np.mean(list(observed.values()))),
        "bootstrap_seed_deltas": bootstrap_deltas,
        "bootstrap_mean_deltas": mean_deltas,
        "confidence_interval": (float(interval[0]), float(interval[1])),
        "confidence_level": 0.95,
        "quantiles": (0.025, 0.975),
        "n_bootstrap": int(n_bootstrap),
        "samples_per_intent": int(samples_per_intent),
        "seed": int(seed),
        "condition": BOOTSTRAP_CONDITION,
    }
    if return_indices:
        result["bootstrap_indices"] = indices
    return result


def exact_train_test_overlap(
    train_rows: Sequence[Mapping[str, Any]] | Sequence[str],
    test_rows: Sequence[Mapping[str, Any]] | Sequence[str],
    text_key: str = "text",
) -> dict[str, Any]:
    """Find exact raw-text train-test overlap without normalization or mutation."""
    def texts(rows: Sequence[Any]) -> list[str]:
        return [str(row[text_key]) if isinstance(row, Mapping) else str(row) for row in rows]

    train_texts = set(texts(train_rows))
    test_texts = texts(test_rows)
    indices = [index for index, text in enumerate(test_texts) if text in train_texts]
    sample_ids = [str(test_rows[index]["sample_id"]) for index in indices if isinstance(test_rows[index], Mapping) and "sample_id" in test_rows[index]]
    return {
        "test_indices": indices,
        "test_sample_ids": sample_ids,
        "overlap_count": len(indices),
        "unique_overlap_texts": sorted({test_texts[index] for index in indices}),
        "normalization": "none (exact raw-text equality)",
    }


def train_test_overlap_sensitivity(
    prediction_rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]] | Sequence[str],
    intent_to_domain: Any,
    *,
    test_rows: Sequence[Mapping[str, Any]] | Sequence[str] | None = None,
    num_intents: int = 150,
    text_key: str = "text",
) -> dict[str, Any]:
    """Recompute after overlap removal; separate test_rows can pair by sample_id."""
    rows = list(prediction_rows)
    if test_rows is None:
        overlap = exact_train_test_overlap(train_rows, rows, text_key=text_key)
        excluded = set(overlap["test_indices"])
    else:
        overlap = exact_train_test_overlap(train_rows, test_rows, text_key=text_key)
        if len(test_rows) != len(rows):
            raise ValueError("test_rows and prediction_rows must have the same length")
        excluded_ids = set(overlap["test_sample_ids"])
        if len(excluded_ids) != overlap["overlap_count"]:
            raise ValueError("separate test_rows must provide sample_id for every overlap")
        excluded = {index for index, row in enumerate(rows) if str(row["sample_id"]) in excluded_ids}
    retained = [row for index, row in enumerate(rows) if index not in excluded]
    if not retained:
        raise ValueError("overlap sensitivity has no remaining test rows")
    metrics = classification_metrics(
        [row["gold_intent"] for row in retained],
        [row["pred_intent"] for row in retained],
        intent_to_domain,
        num_intents=num_intents,
    )
    return {
        "metrics": metrics,
        "n_total": len(rows),
        "n_excluded": len(excluded),
        "n_remaining": len(retained),
        "overlap": overlap,
        "interpretation": "Sensitivity to exact train-test text overlap only; not a decontaminated benchmark claim.",
    }
