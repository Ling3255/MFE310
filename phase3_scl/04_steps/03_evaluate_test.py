"""P3-S03: The single approved test evaluation of phase 3 (E15).

Gate: study.yaml must be frozen with complete supervisor approval, the dev selection must exist and
match the generated configs, and no earlier test evaluation may exist. For every run (B0, main arms
A / A' / B over seeds 0-9, R2 low-resource arms over seeds 0-4) this writes per-example predictions on
the internal test (IHT + OOS-251) — plus R3 perturbed copies for the main arms — together with
validation-selected thresholds and representation-geometry statistics.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT / "03_src"), str(PROJECT_ROOT / "phase3_scl" / "03_src")]

from clinc_domain_aux.baseline import load_baseline, predict_scores
from clinc_domain_aux.common import (
    content_sha256, environment_info, load_config, make_run_id, read_json, resolve_path, stable_hash, utc_now,
    write_csv, write_json,
)
from clinc_domain_aux.confirmatory import require_approval
from clinc_domain_aux.detectors import KNNDetector, MahalanobisDetector, select_threshold
from clinc_domain_aux.evaluation import msp_scores
from clinc_domain_aux.training import _data_hashes, predict_records
from phase3_scl.metrics import representation_geometry
from phase3_scl.perturb import make_variants, variant_of
from phase3_scl.study import (
    STUDY_CONFIG, iter_arm_runs, load_features, load_split, load_study, results_dir, selection_record, softmax, verified_run,
)

FIELDS = ["sample_id", "variant", "split", "gold_intent", "pred_intent", "confidence",
          "msp_score", "mahalanobis_score", "knn_score"]


def check_gate(study: dict) -> dict:
    """Approval, frozen selection and single use; returns the selection record."""
    require_approval(study)
    selection = selection_record(study)
    if not selection.get("selected"):
        raise ValueError("Dev selection found no eligible SCL configuration; H7/H8 are not evaluated")
    for name, digest in selection["selected"]["generated_configs"].items():
        if content_sha256(resolve_path(f"phase3_scl/01_configs/generated/{name}")) != digest:
            raise ValueError(f"Generated config changed after selection: {name}")
    for manifest in results_dir(study, "test").glob("*/evaluation_manifest.json"):
        raise ValueError(f"Phase-3 test set was already evaluated: {manifest.parent}")
    return selection


def rows_for(records: list[dict], logits: np.ndarray, scores: dict[str, np.ndarray]) -> list[dict]:
    probabilities = softmax(logits)
    rows = []
    for index, record in enumerate(records):
        base_id = record["sample_id"].split("#", 1)[0]
        rows.append({
            "sample_id": base_id,
            "variant": variant_of(record["sample_id"]),
            "split": record["split"],
            "gold_intent": "" if record["is_oos"] else int(record["intent_id"]),
            "pred_intent": int(probabilities[index].argmax()),
            "confidence": float(probabilities[index].max()),
            **{f"{name}_score": ("" if values is None else float(values[index])) for name, values in scores.items()},
        })
    return rows


def evaluate_bert(group: str, arm: str, seed: int, config: dict, tokenizer, study: dict, knn_k: int, output: Path) -> dict:
    run = verified_run(config, seed)
    datasets = load_split(config)
    if _data_hashes(datasets)["combined"] != run["data_hash"]:
        raise ValueError(f"Current split differs from the data used in training: {run['run_dir']}")
    clean = [*datasets["test"], *datasets["oos_test"]]
    robustness = study["robustness"]
    variants = make_variants(clean, seed=int(robustness["seed"]), noise_rates=robustness["noise_rates"],
                             punctuation=bool(robustness["strip_punctuation"])) if group == "n80" else []
    evaluated = [*clean, *variants]
    features = predict_records(config, run["run_dir"], [*evaluated, *datasets["oos_val"]], tokenizer)
    count = len(evaluated)
    train = load_features(run["run_dir"], "train_features.npz")
    val = load_features(run["run_dir"], "val_features.npz")
    num_labels = int(config["model"]["num_intents"])
    mahalanobis = MahalanobisDetector(num_labels).fit(train["cls"], train["intent_id"], train["sample_id"])
    knn = KNNDetector(knn_k).fit(train["cls"])
    scores = {
        "msp": msp_scores(features["logits"], "logits"),
        "mahalanobis": mahalanobis.mahalanobis_scores(features["cls"]),
        "knn": knn.score(features["cls"]),
    }
    thresholds = {
        "msp": select_threshold(msp_scores(val["logits"], "logits"), scores["msp"][count:]),
        "mahalanobis": select_threshold(mahalanobis.mahalanobis_scores(val["cls"]), scores["mahalanobis"][count:]),
    }
    name = f"{group}__{arm}__seed{seed}.csv"
    write_csv(output / "predictions" / name, rows_for(evaluated, features["logits"][:count],
                                                     {key: value[:count] for key, value in scores.items()}), fieldnames=FIELDS)
    n_test = len(datasets["test"])
    geometry = representation_geometry(train["cls"], train["intent_id"], features["cls"][:n_test],
                                       features["cls"][n_test:len(clean)])
    if group == "n80" and seed == 0 and arm in ("ce_balanced", "scl"):
        mahalanobis.save(output / "detectors" / f"{arm}_seed0_mahalanobis.npz")  # used by the demo step
    return {"file": name, "content_sha256": content_sha256(output / "predictions" / name),
            "run_dir": run["run_dir"].relative_to(PROJECT_ROOT).as_posix(), "best_epoch": int(run["best_epoch"]),
            "thresholds": {key: value["threshold"] for key, value in thresholds.items()},
            "val_operating_point": {key: {k: value[k] for k in ("is_acceptance", "oos_recall")} for key, value in thresholds.items()},
            "geometry": geometry}


def evaluate_baseline(study: dict, output: Path) -> dict:
    """B0 on the same split: MSP is its only OOS score; R3 perturbations are applied as well."""
    config = load_config(study["baseline"])
    base = load_config(config["base_config"])
    run_dir = resolve_path(config["output_dir"])
    completed = read_json(run_dir / "completed.json")
    datasets = load_split(base)
    expected = stable_hash({split: [[row["sample_id"], row["text"], row["intent_id"], row["domain_id"]] for row in datasets[split]]
                            for split in ("train", "val", "oos_val")})
    if completed.get("status") != "completed" or completed["data_hash"] != expected:
        raise ValueError("B0 was not trained on the current phase-3 split")
    if any(completed["convergence"].values()):
        raise ValueError("B0 has unexplained convergence warnings")
    clean = [*datasets["test"], *datasets["oos_test"]]
    robustness = study["robustness"]
    evaluated = [*clean, *make_variants(clean, seed=int(robustness["seed"]), noise_rates=robustness["noise_rates"],
                                        punctuation=bool(robustness["strip_punctuation"]))]
    _, probabilities = predict_scores(load_baseline(run_dir), [row["text"] for row in evaluated])
    logits = np.log(np.clip(probabilities, 1e-12, 1.0))  # softmax(log p) == p, so rows_for reuses the same code
    with np.load(run_dir / "val_scores.npz") as val, np.load(run_dir / "oos_val_scores.npz") as oos_val:
        threshold = select_threshold(msp_scores(val["probabilities"], "probabilities"),
                                     msp_scores(oos_val["probabilities"], "probabilities"))
    name = "n80__b0__seed0.csv"
    write_csv(output / "predictions" / name, rows_for(evaluated, logits, {
        "msp": msp_scores(probabilities, "probabilities"), "mahalanobis": None, "knn": None}), fieldnames=FIELDS)
    return {"file": name, "content_sha256": content_sha256(output / "predictions" / name),
            "selected_c": completed["selected_c"], "thresholds": {"msp": threshold["threshold"]}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default=STUDY_CONFIG)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    study = load_study(args.study)
    selection = check_gate(study)
    knn_k = int(selection["knn"]["selected_k"])
    output = results_dir(study, "test") / make_run_id("phase3_test")
    (output / "predictions").mkdir(parents=True)
    started = time.time()
    reference = load_config(study["main"]["arms"]["ce_balanced"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(reference["model"]["name"], revision=reference["model"]["revision"], local_files_only=True)
    runs = {"n80__b0__seed0": evaluate_baseline(study, output)}
    for section in ("main", "low_resource"):
        for group, arm, seed, config in iter_arm_runs(study, section):
            runs[f"{group}__{arm}__seed{seed}"] = evaluate_bert(group, arm, seed, config, tokenizer, study, knn_k, output)
            print(f"Evaluated {group} {arm} seed {seed}")
    write_json(output / "evaluation_manifest.json", {
        "created_at": utc_now(), "status": "completed", "elapsed_seconds": round(time.time() - started, 1),
        "study_config_content_sha256": content_sha256(resolve_path(args.study)),
        "selection_content_sha256": content_sha256(results_dir(study, "dev") / "selection.json"),
        "supervisor_approval": study["supervisor_approval"], "knn_k": knn_k,
        "robustness": study["robustness"], "environment": environment_info(), "runs": runs,
    })
    print(f"Phase-3 test predictions: {output}")


if __name__ == "__main__":
    main()
