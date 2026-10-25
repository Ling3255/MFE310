"""P4-M1: The unseen evaluation pool U (primary phase-4 evaluation set).

U = phase-3 T3 (CLINC train minus the internal test) minus the nested n25 subset, minus the texts that
phase 3 already excluded from its internal test (CLINC train texts that also occur in another official
split or carry conflicting labels; three of them occur in CLINC val). Phase-4 models are trained on n10
or n25 only, so they never see U, and U was never used for any evaluation before.

Revision 3 (2026-10-25, before any phase-4 training): the exclusion rule was added after the sampler
audit stopped on three train/val duplicates; U is therefore 54-55 utterances per intent, not exactly 55.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from clinc_domain_aux.common import read_json, resolve_path
from clinc_domain_aux.data import load_datasets
from phase3_scl.splits import normalize


def build_unseen(
    main: Mapping[str, Sequence[Mapping[str, Any]]],
    n25: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    excluded_texts: Iterable[str],
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, int]]:
    """Return U as loader-style records (split 'unseen'), a map to phase-3 T3 sample ids, and per-intent counts.

    ``main``/``n25`` are datasets returned by ``load_datasets`` for the phase-3 main and n25 splits;
    ``excluded_texts`` are removed (compared after normalisation) exactly as for the phase-3 internal test.
    """
    excluded = {normalize(text) for text in excluded_texts}
    used = {normalize(row["text"]) for row in n25["train"]}
    evaluated = {normalize(row["text"]) for split in ("val", "test", "oos_val", "oos_test") for row in main[split]}
    rows = [row for row in main["train"] if normalize(row["text"]) not in used | excluded]
    overlap = sorted({normalize(row["text"]) for row in rows} & evaluated)
    if overlap:  # invariant kept from revision 2: U never shares text with val or any test split
        raise ValueError(f"U overlaps an evaluation split: {overlap}")
    counts = Counter(row["intent"] for row in rows)
    if set(counts) != {row["intent"] for row in main["train"]}:
        raise ValueError("Every intent must remain in U")
    records, origin = [], {}
    for position, row in enumerate(rows):
        sample_id = f"unseen:{position:05d}"
        records.append({**row, "sample_id": sample_id, "split": "unseen"})
        origin[sample_id] = row["sample_id"]
    return records, origin, dict(sorted(counts.items()))


def load_unseen(study: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, int]]:
    """Build U from the study's phase-3 split directories and the phase-3 exclusion list (single source)."""
    phase3 = study["phase3_split"]
    excluded = read_json(resolve_path(phase3["exclusions"]))["excluded_from_internal_test"]
    return build_unseen(
        load_datasets(resolve_path(phase3["main"]), strict_counts=False),
        load_datasets(resolve_path(phase3["n25"]), strict_counts=False),
        excluded_texts=excluded,
    )
