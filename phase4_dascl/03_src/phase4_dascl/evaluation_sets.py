"""P4-M1: The unseen evaluation pool U (primary phase-4 evaluation set).

U = phase-3 T3 (CLINC train minus the internal test) minus the nested n25 subset. Phase-4 models are
trained on n10 or n25 only, so they never see U, and U was never used for any evaluation before.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from phase3_scl.splits import normalize


def build_unseen(
    main: Mapping[str, Sequence[Mapping[str, Any]]],
    n25: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    per_intent: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Return U as loader-style records (split 'unseen') and a map to the phase-3 T3 sample ids.

    ``main``/``n25`` are datasets returned by ``load_datasets`` for the phase-3 main and n25 splits.
    """
    used = {normalize(row["text"]) for row in n25["train"]}
    evaluated = {normalize(row["text"]) for split in ("val", "test", "oos_val", "oos_test") for row in main[split]}
    rows = [row for row in main["train"] if normalize(row["text"]) not in used]
    if any(normalize(row["text"]) in evaluated for row in rows):
        raise ValueError("U overlaps an evaluation split")  # cannot happen for CLINC train (checked in phase 3)
    counts = Counter(row["intent"] for row in rows)
    if set(counts.values()) != {per_intent} or len(counts) != len({row["intent"] for row in main["train"]}):
        raise ValueError(f"U must contain exactly {per_intent} examples per intent, got {sorted(set(counts.values()))}")
    records, origin = [], {}
    for position, row in enumerate(rows):
        sample_id = f"unseen:{position:05d}"
        records.append({**row, "sample_id": sample_id, "split": "unseen"})
        origin[sample_id] = row["sample_id"]
    return records, origin
