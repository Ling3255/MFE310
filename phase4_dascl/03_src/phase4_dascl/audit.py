"""P4-M4: Batch-composition audit for the intent-balanced and domain-aware samplers (no training)."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def batch_composition(batches: Iterable[Sequence[int]], labels: Sequence[int], domains: Sequence[int]) -> dict[str, float]:
    """Per-anchor counts of positives, same-domain negatives and other-domain negatives, plus coverage."""
    labels, domains = np.asarray(labels), np.asarray(domains)
    positives, same_negatives, other_negatives, sizes, domains_per_batch, seen = [], [], [], [], [], []
    for batch in batches:
        index = np.asarray(batch)
        seen.extend(batch)
        sizes.append(len(index))
        domains_per_batch.append(len(set(domains[index].tolist())))
        for anchor in index:
            others = index[index != anchor]
            same_label = labels[others] == labels[anchor]
            same_domain = domains[others] == domains[anchor]
            positives.append(int(same_label.sum()))
            same_negatives.append(int((~same_label & same_domain).sum()))
            other_negatives.append(int((~same_domain).sum()))
    return {
        "batches": len(sizes),
        "mean_batch_size": float(np.mean(sizes)),
        "mean_domains_per_batch": float(np.mean(domains_per_batch)),
        "mean_positives_per_anchor": float(np.mean(positives)),
        "mean_same_domain_negatives_per_anchor": float(np.mean(same_negatives)),
        "mean_other_domain_negatives_per_anchor": float(np.mean(other_negatives)),
        "every_example_once": bool(sorted(seen) == list(range(len(labels)))),
    }
