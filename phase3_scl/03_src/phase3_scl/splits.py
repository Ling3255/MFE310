"""P3-M1: Fixed phase-3 split built only from CLINC150 data that was never used for evaluation.

Layout written for each split (same format as CLINC ``data_full.json``, so the existing loaders work):

* ``train``     : CLINC ``train`` minus the internal test (T3, 80 per intent)
* ``val``       : CLINC ``val`` (unchanged)
* ``test``      : internal held-out test (IHT, 20 per intent drawn from CLINC ``train``)
* ``oos_train`` : empty (no OOS is used for training)
* ``oos_val``   : CLINC ``oos_val`` (unchanged)
* ``oos_test``  : CLINC ``data_full`` ``oos_train`` + ``data_oos_plus``-only OOS utterances (251)

Low-resource training sets (R2) are nested per-intent subsets of T3; val and test stay identical.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

SPLIT_NAMES = ("train", "val", "test", "oos_train", "oos_val", "oos_test")


def normalize(text: str) -> str:
    """Case- and whitespace-insensitive key used for duplicate and leakage checks."""
    return " ".join(str(text).lower().split())


def _excluded_train_texts(full: Mapping[str, Sequence[Sequence[str]]]) -> set[str]:
    """Train texts that must not enter the internal test: any text that also occurs in another
    official split, or that carries more than one label anywhere in CLINC."""
    other = {normalize(text) for split in SPLIT_NAMES if split != "train" for text, _ in full[split]}
    labels: dict[str, set[str]] = defaultdict(set)
    for split in SPLIT_NAMES:
        for text, intent in full[split]:
            labels[normalize(text)].add(intent)
    conflicts = {key for key, values in labels.items() if len(values) > 1}
    return {normalize(text) for text, _ in full["train"]} & (other | conflicts)


def build_main_split(
    full: Mapping[str, Sequence[Sequence[str]]],
    oos_plus: Mapping[str, Sequence[Sequence[str]]],
    *,
    seed: int,
    test_per_intent: int,
) -> tuple[dict[str, list], dict[str, Any]]:
    """Return the main split and a record mapping each new sample id to its CLINC origin."""
    rng = np.random.default_rng(seed)
    excluded = _excluded_train_texts(full)
    by_intent: dict[str, list[int]] = defaultdict(list)
    for index, (_, intent) in enumerate(full["train"]):
        by_intent[intent].append(index)
    test_indices: set[int] = set()
    for intent in sorted(by_intent):
        candidates = [index for index in by_intent[intent] if normalize(full["train"][index][0]) not in excluded]
        if len(candidates) < test_per_intent:
            raise ValueError(f"Intent {intent} has too few eligible examples for the internal test")
        chosen = rng.choice(len(candidates), size=test_per_intent, replace=False)
        test_indices.update(candidates[position] for position in chosen)

    train_rows = [(index, row) for index, row in enumerate(full["train"]) if index not in test_indices]
    test_rows = [(index, row) for index, row in enumerate(full["train"]) if index in test_indices]
    seen_oos: set[str] = set()
    oos_rows: list[tuple[str, list[str]]] = []
    for source, rows in (("data_full.oos_train", full["oos_train"]), ("data_oos_plus.oos_train", oos_plus["oos_train"])):
        for index, (text, intent) in enumerate(rows):
            if normalize(text) not in seen_oos:
                seen_oos.add(normalize(text))
                oos_rows.append((f"{source}:{index:05d}", [text, intent]))

    split = {
        "train": [list(row) for _, row in train_rows],
        "val": [list(row) for row in full["val"]],
        "test": [list(row) for _, row in test_rows],
        "oos_train": [],
        "oos_val": [list(row) for row in full["oos_val"]],
        "oos_test": [row for _, row in oos_rows],
    }
    origin = {
        **{f"train:{new:05d}": f"data_full.train:{old:05d}" for new, (old, _) in enumerate(train_rows)},
        **{f"test:{new:05d}": f"data_full.train:{old:05d}" for new, (old, _) in enumerate(test_rows)},
        **{f"oos_test:{new:05d}": source for new, (source, _) in enumerate(oos_rows)},
    }
    return split, {"origin": origin, "excluded_from_test": sorted(excluded)}


def build_low_resource(main: Mapping[str, Sequence[Sequence[str]]], *, seed: int, sizes: Sequence[int]) -> dict[int, dict[str, list]]:
    """Nested per-intent subsets of T3 (the smaller size is contained in the larger one)."""
    rng = np.random.default_rng(seed)
    by_intent: dict[str, list[int]] = defaultdict(list)
    for index, (_, intent) in enumerate(main["train"]):
        by_intent[intent].append(index)
    order = {intent: [indices[position] for position in rng.permutation(len(indices))] for intent, indices in sorted(by_intent.items())}
    subsets = {}
    for size in sorted(sizes):
        keep = sorted(index for indices in order.values() for index in indices[:size])
        subsets[size] = {**{key: list(value) for key, value in main.items()}, "train": [list(main["train"][index]) for index in keep]}
    return subsets


def leakage_report(split: Mapping[str, Sequence[Sequence[str]]]) -> dict[str, int]:
    """Count normalized text overlaps between every pair of splits."""
    keys = {name: {normalize(text) for text, _ in split[name]} for name in SPLIT_NAMES}
    return {
        f"{a}__{b}": len(keys[a] & keys[b])
        for position, a in enumerate(SPLIT_NAMES) for b in SPLIT_NAMES[position + 1:]
    }


def check_no_test_leakage(split: Mapping[str, Sequence[Sequence[str]]]) -> None:
    """The evaluation splits (test, oos_test) must share no text with training or validation data."""
    report = leakage_report(split)
    evaluation = {"test", "oos_test"}
    bad = {pair: count for pair, count in report.items() if count and evaluation & set(pair.split("__"))}
    if bad:
        raise ValueError(f"Test leakage detected: {bad}")


def describe(split: Mapping[str, Sequence[Sequence[str]]]) -> dict[str, Any]:
    """Counts, per-intent ranges and word-length statistics for the audit table."""
    summary = {}
    for name in SPLIT_NAMES:
        rows = split[name]
        lengths = np.asarray([len(str(text).split()) for text, _ in rows]) if rows else np.zeros(0)
        per_intent = Counter(intent for _, intent in rows)
        summary[name] = {
            "count": len(rows),
            "intents": len(per_intent),
            "per_intent_min": min(per_intent.values()) if per_intent else 0,
            "per_intent_max": max(per_intent.values()) if per_intent else 0,
            "words_mean": float(lengths.mean()) if len(lengths) else None,
            "words_p95": float(np.percentile(lengths, 95)) if len(lengths) else None,
        }
    return summary
