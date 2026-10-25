"""P3-M4: Shared helpers for the phase-3 step scripts (study config, runs, predictions)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np

from clinc_domain_aux.common import load_config, read_json, resolve_path, sha256_file
from clinc_domain_aux.confirmatory import verify_run
from clinc_domain_aux.data import load_datasets, load_label_maps
from clinc_domain_aux.evaluation import msp_scores

STUDY_CONFIG = "phase3_scl/01_configs/study.yaml"


def load_study(path: str = STUDY_CONFIG) -> dict[str, Any]:
    return load_config(path)


def results_dir(study: dict[str, Any], *parts: str) -> Path:
    target = resolve_path(study["output_dir"]).joinpath(*parts)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_split(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Load the split a run config points to (phase-3 configs set strict_counts: false)."""
    return load_datasets(config["data"]["root"], strict_counts=bool(config["data"].get("strict_counts", True)))


def intent_domain_array(config: dict[str, Any]) -> np.ndarray:
    label_to_id, domain_to_id, intent_to_domain = load_label_maps(config["metadata_dir"])
    mapping = np.empty(len(label_to_id), dtype=np.int64)
    for intent, index in label_to_id.items():
        mapping[index] = domain_to_id[intent_to_domain[intent]]
    return mapping


def iter_arm_runs(study: dict[str, Any], section: str = "main") -> Iterator[tuple[str, str, int, dict[str, Any]]]:
    """Yield (group, arm, seed, config) for the main arms or for every low-resource size."""
    if section == "main":
        for arm, path in study["main"]["arms"].items():
            config = load_config(path)
            for seed in study["main"]["seeds"]:
                yield "n80", arm, int(seed), config
    elif section == "low_resource":
        for size, arms in study["low_resource"]["sizes"].items():
            for arm, path in arms.items():
                config = load_config(path)
                for seed in study["low_resource"]["seeds"]:
                    yield size, arm, int(seed), config
    else:
        raise ValueError(f"Unknown section {section!r}")


def verified_run(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Completed, non-smoke run trained with the unchanged config (lambda_domain is always 0 here)."""
    return verify_run(config, 0.0, seed)


def load_features(run_dir: Path, name: str) -> dict[str, np.ndarray]:
    with np.load(run_dir / name, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def msp(logits: np.ndarray) -> np.ndarray:
    return msp_scores(logits, "logits")


def file_record(path: Path) -> dict[str, str]:
    return {"sha256": sha256_file(path)}


def selection_record(study: dict[str, Any]) -> dict[str, Any]:
    path = resolve_path(study["output_dir"]) / "dev" / "selection.json"
    if not path.exists():
        raise ValueError("Run 02_select_scl.py first: dev selection record is missing")
    return read_json(path)
