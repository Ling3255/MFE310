"""S05: Fit validation-only OOS policies and create an approval-gated freeze."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "03_src"))

from clinc_domain_aux.common import (
    load_config, read_json, resolve_path, run_name, save_yaml, sha256_file,
    stable_hash, utc_now, write_json,
)
from clinc_domain_aux.data import load_datasets
from clinc_domain_aux.detectors import MahalanobisDetector, select_threshold
from clinc_domain_aux.evaluation import msp_scores
from clinc_domain_aux.training import predict_records


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def save_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def fit_bert_detector(base: dict, datasets: dict, tokenizer, run_dir: Path, output: Path) -> dict:
    train = load_npz(run_dir / "train_features.npz")
    validation = load_npz(run_dir / "val_features.npz")
    oos_validation = predict_records(base, run_dir, datasets["oos_val"], tokenizer)
    save_npz(output / "oos_val_features.npz", oos_validation)
    if not np.array_equal(train["intent_id"], np.asarray([row["intent_id"] for row in datasets["train"]])):
        raise ValueError(f"Training feature order/labels differ from train: {run_dir}")
    detector = MahalanobisDetector(int(base["model"]["num_intents"]))
    detector.fit(train["cls"], train["intent_id"], train["sample_id"])
    detector_path = output / "mahalanobis.npz"
    detector.save(detector_path)
    msp_val = msp_scores(validation["logits"], "logits")
    msp_oos = msp_scores(oos_validation["logits"], "logits")
    msp = select_threshold(msp_val, msp_oos)
    mahal_val = detector.mahalanobis_scores(validation["cls"])
    mahal_oos = detector.mahalanobis_scores(oos_validation["cls"])
    mahalanobis = select_threshold(mahal_val, mahal_oos)
    write_json(output / "thresholds.json", {
        "source": "IS val + OOS val only",
        "msp": msp,
        "mahalanobis": mahalanobis,
    })
    return {
        "run_dir": relative(run_dir),
        "detector_path": relative(detector_path),
        "detector_sha256": sha256_file(detector_path),
        "detector_fit_sample_hash": detector.sample_hash_,
        "oos_val_features_sha256": sha256_file(output / "oos_val_features.npz"),
        "thresholds": {"msp": msp, "mahalanobis": mahalanobis},
    }


def fit_baseline_msp(run_dir: Path, output: Path) -> dict:
    validation = load_npz(run_dir / "val_scores.npz")
    oos_validation = load_npz(run_dir / "oos_val_scores.npz")
    msp = select_threshold(
        msp_scores(validation["probabilities"], "probabilities"),
        msp_scores(oos_validation["probabilities"], "probabilities"),
    )
    result = {"source": "IS val + OOS val only", "msp": msp, "mahalanobis": None}
    write_json(output / "thresholds.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="01_configs/03_search.yaml")
    args = parser.parse_args()
    search = load_config(args.config)
    frozen_path = resolve_path(search["frozen_config"])
    if frozen_path.exists() and load_config(frozen_path).get("frozen", False):
        raise ValueError("A frozen configuration already exists; preserve it and create a reviewed revision instead")
    base = load_config(search["base_config"])
    datasets = load_datasets(base["data"]["root"], strict_counts=True)
    validation_dir = resolve_path(search["validation_results_dir"])
    selected = read_json(validation_dir / "selected_lambda.json")
    lambda_domain = float(selected["selected_lambda"])
    runs_dir = resolve_path(search["runs_dir"])
    model_specs = [(0.0, seed, "bert_ce") for seed in search["seeds"]]
    model_specs += [(lambda_domain, seed, "bert_dom") for seed in search["seeds"]]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        base["model"]["name"], revision=base["model"]["revision"], local_files_only=True,
    )
    models = []
    for value, seed, model_name in model_specs:
        run_dir = runs_dir / run_name(float(value), int(seed))
        completed = read_json(run_dir / "completed.json")
        if completed.get("status") != "completed" or completed.get("smoke", False):
            raise ValueError(f"Invalid formal run: {run_dir}")
        for artifact, expected in completed["artifacts"].items():
            if sha256_file(run_dir / artifact) != expected:
                raise ValueError(f"Run artifact changed after training: {run_dir / artifact}")
        detector_dir = validation_dir / "detectors" / run_dir.name
        fitted = fit_bert_detector(base, datasets, tokenizer, run_dir, detector_dir)
        models.append({
            "model": model_name,
            "lambda_domain": float(value),
            "seed": int(seed),
            **fitted,
            "best_checkpoint_sha256": sha256_file(run_dir / "best.pt"),
            "best_checkpoint_state_hash": completed["best_checkpoint_hash"],
            "config_hash": completed["config_hash"],
            "initialization_hash": completed["initialization_hash"],
        })
        print(f"Fitted detectors: {run_dir.name}")

    baseline_dir = resolve_path("08_runs/baseline_tfidf_lr")
    baseline_completed = read_json(baseline_dir / "completed.json")
    if baseline_completed.get("status") != "completed" or any(baseline_completed["convergence"].values()):
        raise ValueError("Baseline is incomplete or has unexplained convergence warnings")
    baseline_thresholds = fit_baseline_msp(baseline_dir, validation_dir / "detectors" / "baseline_tfidf_lr")
    manifest_path = resolve_path(base["metadata_dir"]) / "data_manifest.json"
    approval = dict(search.get("supervisor_approval", {}))
    approval["approved"] = False
    frozen = {
        "schema_version": 1,
        "status": "awaiting_supervisor_approval",
        "frozen": True,
        "created_at": utc_now(),
        "base_config": search["base_config"],
        "base_config_content_hash": stable_hash(base),
        "data_manifest": relative(manifest_path),
        "data_manifest_sha256": sha256_file(manifest_path),
        "selected_lambda": lambda_domain,
        "selected_lambda_record": relative(validation_dir / "selected_lambda.json"),
        "baseline": {
            "run_dir": relative(baseline_dir),
            "checkpoint_sha256": sha256_file(baseline_dir / "pipeline.joblib"),
            "thresholds": baseline_thresholds,
        },
        "models": models,
        "evaluation": dict(base["selection"]),
        "optional_analyses": dict(search["optional_analyses"]),
        "supervisor_approval": approval,
        "test_history": {
            "domain_aux_design_used_test_results": False,
            "other_group_members_previously_inspected_test": True,
            "evaluation_wording": "one-time evaluation under a frozen protocol; not fully independent confirmation",
        },
        "metric_protocol": "00_docs/03_metric_definitions.md",
    }
    save_yaml(frozen_path, frozen)
    record = resolve_path(search["freeze_record"])
    lines = [
        "# E5 freeze record", "", f"Created: {frozen['created_at']}", "",
        "Status: **awaiting supervisor approval; test evaluation is blocked**.", "",
        f"Selected lambda: `{lambda_domain:g}` (validation only).", "",
        "The domain-auxiliary design did not use test results. Other group members had previously inspected this benchmark's test set. Final results must be described as a one-time evaluation under a frozen protocol, not fully independent confirmation.", "",
        "## Model and detector hashes", "",
    ]
    lines += [
        f"- `{item['model']}` seed `{item['seed']}`: checkpoint SHA-256 `{item['best_checkpoint_sha256']}`, detector SHA-256 `{item['detector_sha256']}`."
        for item in models
    ]
    lines += [
        "", "## Supervisor confirmation", "",
        "Set `supervisor_approval.approved: true` in `01_configs/04_frozen.yaml` only after actual confirmation, and fill in `confirmed_by`, ISO `confirmed_at`, and `evidence`. S06 checks all four fields and all artifact hashes.", "",
        "Any protocol change after test inspection requires a new recorded revision and disclosure; do not overwrite this record.",
    ]
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Freeze created but NOT approved: {frozen_path}")
    print("E6 remains blocked until explicit supervisor confirmation is recorded.")


if __name__ == "__main__":
    main()
