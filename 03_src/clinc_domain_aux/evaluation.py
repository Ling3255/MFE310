"""M08: Pure classification/OOS metrics and prediction-row export."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _id_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.issubdtype(array.dtype, np.integer):
        try:
            converted = array.astype(np.int64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must contain integer IDs") from exc
        if not np.all(array == converted):
            raise ValueError(f"{name} must contain integer IDs")
        array = converted
    return array.astype(np.int64, copy=False)


def _domain_mapping(intent_to_domain: Any, num_intents: int) -> np.ndarray:
    if not isinstance(num_intents, (int, np.integer)) or num_intents <= 0:
        raise ValueError("num_intents must be a positive integer")
    if isinstance(intent_to_domain, Mapping):
        try:
            mapping = np.asarray([intent_to_domain[index] for index in range(num_intents)])
        except KeyError as exc:
            raise ValueError("intent_to_domain must cover every intent ID") from exc
    else:
        mapping = np.asarray(intent_to_domain)
        if mapping.ndim != 1 or len(mapping) < num_intents:
            raise ValueError("intent_to_domain must cover every intent ID")
        mapping = mapping[:num_intents]
    mapping = _id_array(mapping, "intent_to_domain")
    if np.any(mapping < 0):
        raise ValueError("domain IDs must be non-negative")
    return mapping


def _confusion(gold_ids: Any, pred_ids: Any, num_intents: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gold = _id_array(gold_ids, "gold_ids")
    pred = _id_array(pred_ids, "pred_ids")
    if len(gold) != len(pred):
        raise ValueError("gold_ids and pred_ids must have the same length")
    if np.any(gold < 0) or np.any(gold >= num_intents) or np.any(pred < 0) or np.any(pred >= num_intents):
        raise ValueError("intent IDs must be in [0, num_intents)")
    confusion = np.bincount(gold * num_intents + pred, minlength=num_intents**2).reshape(num_intents, num_intents)
    return gold, pred, confusion


def _macro_f1(confusion: np.ndarray) -> float:
    true_positive = np.diag(confusion).astype(np.float64)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2.0 * true_positive, denominator, out=np.zeros_like(true_positive), where=denominator > 0)
    return float(f1.mean())


def classification_metrics(
    gold_ids: Any,
    pred_ids: Any,
    intent_to_domain: Any,
    num_intents: int = 150,
) -> dict[str, float]:
    """Compute pre-rejection IS metrics over the fixed complete intent set."""
    gold, pred, confusion = _confusion(gold_ids, pred_ids, num_intents)
    domains = _domain_mapping(intent_to_domain, num_intents)
    gold_domain = domains[gold]
    pred_domain = domains[pred]
    correct = gold == pred
    cross = gold_domain != pred_domain
    within = ~correct & ~cross
    total = len(gold)
    accuracy = float(correct.sum() / total)
    cross_rate = float(cross.sum() / total)
    if int(correct.sum() + cross.sum() + within.sum()) != total:
        raise ValueError("classification outcomes are not mutually exclusive")
    within_rate = 1.0 - accuracy - cross_rate
    domain_labels = np.unique(domains)
    lookup = np.full(int(domains.max()) + 1, -1, dtype=np.int64)
    lookup[domain_labels] = np.arange(len(domain_labels))
    domain_confusion = np.bincount(
        lookup[gold_domain] * len(domain_labels) + lookup[pred_domain],
        minlength=len(domain_labels) ** 2,
    ).reshape(len(domain_labels), len(domain_labels))
    return {
        "macro_f1": _macro_f1(confusion),
        "accuracy": accuracy,
        "cross_domain_error_rate": cross_rate,
        "within_domain_error_rate": within_rate,
        "domain_macro_f1": _macro_f1(domain_confusion),
    }


def per_domain_intent_f1(
    gold_ids: Any,
    pred_ids: Any,
    intent_to_domain: Any,
    num_intents: int = 150,
) -> dict[int, float]:
    """Average intent F1 within each domain without discarding cross-domain false positives."""
    _, _, confusion = _confusion(gold_ids, pred_ids, num_intents)
    domains = _domain_mapping(intent_to_domain, num_intents)
    true_positive = np.diag(confusion).astype(np.float64)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2.0 * true_positive, denominator, out=np.zeros_like(true_positive), where=denominator > 0)
    return {int(domain): float(f1[domains == domain].mean()) for domain in np.unique(domains)}


def msp_scores(values: Any, input_type: str) -> np.ndarray:
    """Return 1-max probability; input_type must be exactly 'logits' or 'probabilities'."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0 or not np.isfinite(array).all():
        raise ValueError("values must be a finite, non-empty two-dimensional array")
    if input_type == "logits":
        row_max = array.max(axis=1, keepdims=True)
        log_normalizer = row_max + np.log(np.exp(array - row_max).sum(axis=1, keepdims=True))
        maximum_probability = np.exp(row_max[:, 0] - log_normalizer[:, 0])
    elif input_type == "probabilities":
        if np.any(array < 0) or not np.allclose(array.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9):
            raise ValueError("probabilities must be non-negative and sum to one per row")
        maximum_probability = array.max(axis=1)
    else:
        raise ValueError("input_type must be 'logits' or 'probabilities'")
    scores = 1.0 - maximum_probability
    if not np.isfinite(scores).all():
        raise ValueError("MSP scores are not finite")
    return scores


def _score_pair(is_scores: Any, oos_scores: Any) -> tuple[np.ndarray, np.ndarray]:
    in_scope = np.asarray(is_scores, dtype=np.float64)
    out_of_scope = np.asarray(oos_scores, dtype=np.float64)
    if in_scope.ndim != 1 or out_of_scope.ndim != 1 or len(in_scope) == 0 or len(out_of_scope) == 0:
        raise ValueError("OOS metrics require non-empty one-dimensional IS and OOS scores")
    if not np.isfinite(in_scope).all() or not np.isfinite(out_of_scope).all():
        raise ValueError("scores must be finite")
    return in_scope, out_of_scope


def oos_metrics(is_scores: Any, oos_scores: Any, target_tpr: float = 0.95) -> dict[str, Any]:
    """AUROC uses OOS-positive s; FPR95 uses IS-positive acceptance -s."""
    in_scope, out_of_scope = _score_pair(is_scores, oos_scores)
    if not 0 < target_tpr <= 1:
        raise ValueError("target_tpr must be in (0, 1]")
    scores = np.concatenate([in_scope, out_of_scope])
    oos_labels = np.concatenate([np.zeros(len(in_scope), dtype=np.int64), np.ones(len(out_of_scope), dtype=np.int64)])
    is_labels = 1 - oos_labels
    acceptance = -scores
    fpr, tpr, thresholds = roc_curve(is_labels, acceptance, drop_intermediate=False)
    eligible = np.flatnonzero(tpr >= target_tpr)
    if len(eligible) == 0:
        raise ValueError("ROC has no point reaching target_tpr")
    index = int(eligible[0])
    return {
        "auroc_oos": float(roc_auc_score(oos_labels, scores)),
        "fpr95_is_positive": float(fpr[index]),
        "fpr95_actual_is_acceptance": float(tpr[index]),
        "fpr95_acceptance_threshold": float(thresholds[index]),
        "fpr95_score_upper_bound": float(-thresholds[index]),
        "positive_class": "OOS for AUROC; IS for FPR95",
        "score_direction": "OOS score increases with OOS likelihood; ROC acceptance=-score",
        "target_tpr": float(target_tpr),
    }


def frozen_threshold_metrics(is_scores: Any, oos_scores: Any, tau: float) -> dict[str, float]:
    """Evaluate a frozen validation threshold with strict score >= tau rejection."""
    in_scope, out_of_scope = _score_pair(is_scores, oos_scores)
    if not np.isfinite(tau):
        raise ValueError("tau must be finite")
    is_accepted = in_scope < tau
    oos_accepted = out_of_scope < tau
    return {
        "oos_fa_at_val95": float(oos_accepted.mean()),
        "is_acceptance_at_val95": float(is_accepted.mean()),
        "oos_recall_at_val95": float((~oos_accepted).mean()),
    }


def combine_classification_and_oos_metrics(
    classification: Mapping[str, Any],
    oos: Mapping[str, Any],
    frozen: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge metric mappings while rejecting ambiguous duplicate fields."""
    result = dict(classification)
    for source in (oos, frozen):
        if source is None:
            continue
        overlap = result.keys() & source.keys()
        if overlap:
            raise ValueError(f"duplicate metric fields: {sorted(overlap)}")
        result.update(source)
    return result


def export_predictions(
    sample_ids: Sequence[Any],
    gold_ids: Any,
    pred_ids: Any,
    intent_to_domain: Any,
    *,
    split: str,
    model: str,
    seed: int,
    msp_score: Any,
    mahalanobis_score: Any,
    tau_msp: float,
    tau_mahalanobis: float,
    checkpoint_hash: str,
    num_intents: int = 150,
) -> list[dict[str, Any]]:
    """Assemble batch outputs; pass gold_ids=None for OOS rows with null gold fields."""
    domains = _domain_mapping(intent_to_domain, num_intents)
    pred = _id_array(pred_ids, "pred_ids")
    if np.any(pred < 0) or np.any(pred >= num_intents):
        raise ValueError("pred_ids must be in [0, num_intents)")
    if gold_ids is None:
        gold = None
    else:
        gold = _id_array(gold_ids, "gold_ids")
        if len(gold) != len(pred) or np.any(gold < 0) or np.any(gold >= num_intents):
            raise ValueError("gold_ids must match pred_ids and be in [0, num_intents)")
    count = len(pred)
    msp = np.asarray(msp_score, dtype=np.float64)
    mahalanobis = np.asarray(mahalanobis_score, dtype=np.float64)
    if len(sample_ids) != count or msp.shape != (count,) or mahalanobis.shape != (count,):
        raise ValueError("all prediction outputs must have the same sample count")
    if not np.isfinite(msp).all() or not np.isfinite(mahalanobis).all():
        raise ValueError("prediction scores must be finite")
    return [
        {
            "sample_id": str(sample_ids[index]),
            "split": split,
            "gold_intent": None if gold is None else int(gold[index]),
            "gold_domain": None if gold is None else int(domains[gold[index]]),
            "pred_intent": int(pred[index]),
            "pred_domain_from_intent": int(domains[pred[index]]),
            "msp_score": float(msp[index]),
            "mahalanobis_score": float(mahalanobis[index]),
            "accepted_msp": bool(msp[index] < tau_msp),
            "accepted_mahalanobis": bool(mahalanobis[index] < tau_mahalanobis),
            "seed": int(seed),
            "model": model,
            "checkpoint_hash": checkpoint_hash,
        }
        for index in range(count)
    ]
