"""M02: CLINC150 six-split loading, stable IDs, label maps, and loaders."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .common import content_sha256, git_commit, resolve_path, sha256_file, utc_now, write_csv, write_json

SPLITS = ("train", "val", "test", "oos_train", "oos_val", "oos_test")
IN_SCOPE_SPLITS = ("train", "val", "test")
OOS_SPLITS = ("oos_train", "oos_val", "oos_test")
EXPECTED_COUNTS = {
    "train": 15000,
    "val": 3000,
    "test": 4500,
    "oos_train": 100,
    "oos_val": 100,
    "oos_test": 1000,
}
EXPECTED_PER_INTENT = {"train": 100, "val": 20, "test": 30}
OOS_ID = -1

Record = dict[str, Any]
Datasets = dict[str, list[Record]]


def load_datasets(
    data_root: str | Path,
    metadata_dir: str | Path | None = None,
    strict_counts: bool = True,
) -> Datasets:
    """Load and validate CLINC150 without changing any input text.

    IDs follow original row order within each official split. When
    ``metadata_dir`` is supplied, fixed sorted maps and a versioned manifest
    are written there. ``strict_counts=False`` is intended only for small
    fixtures; all structural and label-validation checks remain enabled.
    """
    root = resolve_path(data_root)
    data_path = root / "data_full.json"
    domains_path = root / "domains.json"
    with data_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    with domains_path.open("r", encoding="utf-8") as handle:
        raw_domains = json.load(handle)
    if not isinstance(raw, dict) or not isinstance(raw_domains, dict):
        raise ValueError("data_full.json and domains.json must contain JSON objects")
    if set(raw) != set(SPLITS):
        raise ValueError(f"Expected exactly the six splits {SPLITS}; got {sorted(raw)}")

    intent_to_domain, domain_to_intents = _parse_domains(raw_domains)
    datasets: Datasets = {}
    for split in SPLITS:
        rows = raw[split]
        if not isinstance(rows, list):
            raise ValueError(f"Split {split!r} must be a list")
        split_records: list[Record] = []
        for index, row in enumerate(rows):
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                raise ValueError(f"{split}[{index}] must be [text, intent]")
            text, intent = row
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{split}[{index}] has empty or non-string text")
            if not isinstance(intent, str) or not intent:
                raise ValueError(f"{split}[{index}] has an invalid intent")
            is_oos = split in OOS_SPLITS
            if is_oos != (intent == "oos"):
                raise ValueError(f"{split}[{index}] has an intent inconsistent with its split")
            if not is_oos and intent not in intent_to_domain:
                raise ValueError(f"{split}[{index}] has unknown intent {intent!r}")
            domain = None if is_oos else intent_to_domain[intent]
            split_records.append(
                {
                    "sample_id": f"{split}:{index:05d}",
                    "split": split,
                    "text": text,
                    "intent": intent,
                    "domain": domain,
                    "is_oos": is_oos,
                }
            )
        datasets[split] = split_records

    _validate_counts_and_labels(datasets, domain_to_intents, strict_counts)
    label_to_id = {label: index for index, label in enumerate(sorted(intent_to_domain))}
    domain_to_id = {domain: index for index, domain in enumerate(sorted(domain_to_intents))}
    id_to_label = {index: label for label, index in label_to_id.items()}
    id_to_domain = {index: domain for domain, index in domain_to_id.items()}
    if len(id_to_label) != len(label_to_id) or len(id_to_domain) != len(domain_to_id):
        raise AssertionError("Label and domain IDs must be reversible")
    for records in datasets.values():
        for record in records:
            record["intent_id"] = OOS_ID if record["is_oos"] else label_to_id[record["intent"]]
            record["domain_id"] = OOS_ID if record["is_oos"] else domain_to_id[record["domain"]]

    if metadata_dir is not None:
        metadata = resolve_path(metadata_dir)
        _write_metadata(
            metadata,
            data_path,
            domains_path,
            datasets,
            label_to_id,
            domain_to_id,
            intent_to_domain,
            strict_counts,
        )
    return datasets


def load_label_maps(metadata_dir: str | Path) -> tuple[dict[str, int], dict[str, int], dict[str, str]]:
    """Return ``(label_to_id, domain_to_id, intent_to_domain)`` from M02 files."""
    metadata = resolve_path(metadata_dir)
    with (metadata / "label_to_id.json").open("r", encoding="utf-8") as handle:
        label_to_id = json.load(handle)
    with (metadata / "domain_to_id.json").open("r", encoding="utf-8") as handle:
        domain_to_id = json.load(handle)
    with (metadata / "intent_domain_map.csv").open("r", encoding="utf-8", newline="") as handle:
        import csv

        intent_to_domain = {row["intent"]: row["domain"] for row in csv.DictReader(handle)}
    if set(label_to_id) != set(intent_to_domain):
        raise ValueError("Saved intent map and intent-domain map disagree")
    return label_to_id, domain_to_id, intent_to_domain


class ClincDataset:
    """PyTorch-compatible dataset returning original-text token encodings.

    Set ``in_scope_only=True`` for intent/domain training to exclude every OOS
    row. OOS rows retained for evaluation have intent and domain targets -1.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        tokenizer: Any,
        max_length: int = 64,
        in_scope_only: bool = False,
        label_to_id: Mapping[str, int] | None = None,
        domain_to_id: Mapping[str, int] | None = None,
        intent_to_domain: Mapping[str, str] | None = None,
    ) -> None:
        if max_length < 2:
            raise ValueError("max_length must include at least two special-token positions")
        self.records = [dict(row) for row in records if not in_scope_only or not row.get("is_oos", row["intent"] == "oos")]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_id = dict(label_to_id) if label_to_id is not None else None
        self.domain_to_id = dict(domain_to_id) if domain_to_id is not None else None
        self.intent_to_domain = dict(intent_to_domain) if intent_to_domain is not None else None
        for row in self.records:
            if "sample_id" not in row or "text" not in row or "intent" not in row:
                raise ValueError("Every dataset record needs sample_id, text, and intent")
            self._targets(row)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        row = self.records[index]
        encoded = self.tokenizer(
            row["text"],
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        intent_id, domain_id = self._targets(row)
        return {
            "input_ids": torch.as_tensor(encoded["input_ids"][0], dtype=torch.long),
            "attention_mask": torch.as_tensor(encoded["attention_mask"][0], dtype=torch.long),
            "intent_id": torch.tensor(intent_id, dtype=torch.long),
            "domain_id": torch.tensor(domain_id, dtype=torch.long),
            "sample_id": row["sample_id"],
        }

    def _targets(self, row: Mapping[str, Any]) -> tuple[int, int]:
        if row.get("is_oos", row["intent"] == "oos"):
            return OOS_ID, OOS_ID
        intent_id = row.get("intent_id")
        domain_id = row.get("domain_id")
        if intent_id is None:
            if self.label_to_id is None:
                raise ValueError("label_to_id is required when records have no intent_id")
            intent_id = self.label_to_id[row["intent"]]
        if domain_id is None:
            if self.domain_to_id is not None and "domain" in row:
                domain_id = self.domain_to_id[row["domain"]]
            elif self.domain_to_id is not None and self.intent_to_domain is not None:
                domain_id = self.domain_to_id[self.intent_to_domain[row["intent"]]]
            else:
                raise ValueError("domain maps are required when records have no domain_id")
        return int(intent_id), int(domain_id)


def make_dataloader(
    records: Sequence[Mapping[str, Any]] | ClincDataset,
    tokenizer: Any = None,
    batch_size: int = 32,
    max_length: int = 64,
    shuffle: bool = False,
    in_scope_only: bool = False,
    num_workers: int = 0,
    generator: Any = None,
    **dataset_kwargs: Any,
) -> Any:
    """Build a DataLoader; use ``in_scope_only=True`` for 150-class training."""
    from torch.utils.data import DataLoader

    dataset = records if isinstance(records, ClincDataset) else ClincDataset(
        records,
        tokenizer,
        max_length=max_length,
        in_scope_only=in_scope_only,
        **dataset_kwargs,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def iter_records(datasets: Mapping[str, Sequence[Record]], splits: Sequence[str] = SPLITS) -> Iterator[Record]:
    """Iterate records in the fixed official split and original row order."""
    for split in splits:
        yield from datasets[split]


def _parse_domains(raw_domains: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, list[str]]]:
    intent_to_domain: dict[str, str] = {}
    domain_to_intents: dict[str, list[str]] = {}
    for domain, intents in raw_domains.items():
        if not isinstance(domain, str) or not domain or not isinstance(intents, list):
            raise ValueError("Each domain must have a list of intent names")
        domain_to_intents[domain] = list(intents)
        for intent in intents:
            if not isinstance(intent, str) or not intent or intent == "oos":
                raise ValueError(f"Invalid in-scope intent in domain {domain!r}: {intent!r}")
            if intent in intent_to_domain:
                raise ValueError(f"Intent {intent!r} occurs in more than one domain")
            intent_to_domain[intent] = domain
    return intent_to_domain, domain_to_intents


def _validate_counts_and_labels(
    datasets: Mapping[str, Sequence[Record]],
    domain_to_intents: Mapping[str, Sequence[str]],
    strict_counts: bool,
) -> None:
    counts = {split: len(datasets[split]) for split in SPLITS}
    if strict_counts:
        mismatches = {split: (counts[split], expected) for split, expected in EXPECTED_COUNTS.items() if counts[split] != expected}
        if mismatches:
            raise ValueError(f"Official split counts failed (actual, expected): {mismatches}")
        if len(domain_to_intents) != 10 or any(len(intents) != 15 for intents in domain_to_intents.values()):
            raise ValueError("Official domain map must contain 10 domains with 15 intents each")
        if sum(map(len, domain_to_intents.values())) != 150:
            raise ValueError("Official domain map must contain exactly 150 unique intents")
    expected_labels = {
        intent for intents in domain_to_intents.values() for intent in intents
    }
    observed_labels = {
        split: {record["intent"] for record in datasets[split]} for split in IN_SCOPE_SPLITS
    }
    if strict_counts:
        for split in IN_SCOPE_SPLITS:
            if observed_labels[split] != expected_labels:
                raise ValueError(f"Split {split!r} does not contain exactly the 150 mapped intents")
            per_intent = Counter(record["intent"] for record in datasets[split])
            bad = {
                intent: (per_intent[intent], EXPECTED_PER_INTENT[split])
                for intent in sorted(expected_labels)
                if per_intent[intent] != EXPECTED_PER_INTENT[split]
            }
            if bad:
                raise ValueError(f"Per-intent counts failed for {split} (actual, expected): {bad}")


def _write_metadata(
    metadata: Path,
    data_path: Path,
    domains_path: Path,
    datasets: Mapping[str, Sequence[Record]],
    label_to_id: Mapping[str, int],
    domain_to_id: Mapping[str, int],
    intent_to_domain: Mapping[str, str],
    strict_counts: bool,
) -> None:
    write_json(metadata / "label_to_id.json", dict(label_to_id))
    write_json(metadata / "domain_to_id.json", dict(domain_to_id))
    write_csv(
        metadata / "intent_domain_map.csv",
        (
            {
                "intent": intent,
                "intent_id": label_to_id[intent],
                "domain": intent_to_domain[intent],
                "domain_id": domain_to_id[intent_to_domain[intent]],
            }
            for intent in sorted(intent_to_domain)
        ),
        fieldnames=("intent", "intent_id", "domain", "domain_id"),
    )
    license_path = data_path.parent.parent / "LICENSE"
    manifest = {
        "schema_version": 1,
        "dataset": "CLINC150 (CLINC OOS-Eval data_full)",
        "source": "https://github.com/clinc/oos-eval",
        "git_commit": git_commit(data_path.parent),
        "generated_at": utc_now(),
        "strict_counts": strict_counts,
        "files": {
            "data_full.json": {"sha256": sha256_file(data_path), "content_sha256": content_sha256(data_path)},
            "domains.json": {"sha256": sha256_file(domains_path), "content_sha256": content_sha256(domains_path)},
        },
        "license": {
            "name": "Creative Commons Attribution 3.0 Unported (CC BY 3.0)",
            "url": "https://creativecommons.org/licenses/by/3.0/",
            "file": "LICENSE",
            "sha256": sha256_file(license_path) if license_path.is_file() else None,
        },
        "counts": {split: len(datasets[split]) for split in SPLITS},
        "per_intent_counts": {
            split: dict(sorted(Counter(record["intent"] for record in datasets[split]).items()))
            for split in IN_SCOPE_SPLITS
        },
        "oos": {"intent_id": OOS_ID, "domain_id": OOS_ID, "domain": None},
        "sample_id_rule": "split + zero-padded original row index, for example train:00000",
        "text_policy": "Original text is preserved; normalization is used only by M03 audits.",
        "mappings": {
            name: {"sha256": sha256_file(metadata / name)}
            for name in ("label_to_id.json", "domain_to_id.json", "intent_domain_map.csv")
        },
    }
    write_json(metadata / "data_manifest.json", manifest)
