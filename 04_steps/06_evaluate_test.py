"""S06: Evaluate only explicitly approved frozen artifacts on the test split."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "03_src"))

from clinc_domain_aux.baseline import load_baseline, predict_scores
from clinc_domain_aux.common import (
    content_sha256, load_config, make_run_id, read_json, resolve_path, sha256_file, stable_hash,
    utc_now, write_csv, write_json,
)
from clinc_domain_aux.data import load_datasets, load_label_maps
from clinc_domain_aux.detectors import MahalanobisDetector
from clinc_domain_aux.evaluation import (
    classification_metrics, export_predictions, frozen_threshold_metrics,
    msp_scores, oos_metrics,
)
from clinc_domain_aux.training import predict_records


def require_approved_freeze(frozen: dict, base: dict) -> None:
    if not frozen.get("frozen") or frozen.get("status") == "awaiting_training_and_freeze":
        raise ValueError("E5 has not produced a frozen configuration")
    approval = frozen.get("supervisor_approval", {})
    missing = [key for key in ("confirmed_by", "confirmed_at", "evidence") if not approval.get(key)]
    if not approval.get("approved") or missing:
        raise ValueError(f"E6 is blocked: explicit supervisor approval is missing ({', '.join(missing) or 'approved=false'})")
    try:
        datetime.fromisoformat(str(approval["confirmed_at"]).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("supervisor_approval.confirmed_at must be an ISO timestamp") from error
    if frozen.get("base_config_content_hash") != stable_hash(base):
        raise ValueError("Base configuration changed after freezing")
    manifest = resolve_path(frozen["data_manifest"])
    if sha256_file(manifest) != frozen["data_manifest_sha256"]:
        raise ValueError("Data manifest changed after freezing")
    data_root = resolve_path(base["data"]["root"])
    for name, entry in read_json(manifest)["files"].items():
        if "content_sha256" not in entry:
            raise ValueError("Data manifest lacks content_sha256; rerun S01 and S05 before E6")
        if content_sha256(data_root / name) != entry["content_sha256"]:
            raise ValueError(f"Data file differs from the frozen manifest: {name}")


def refuse_repeat_evaluation(frozen_path: Path) -> None:
    """E6 runs once per frozen configuration; a second run must be a recorded revision."""
    digest = sha256_file(frozen_path)
    for manifest in resolve_path("09_results/02_test").glob("*/evaluation_manifest.json"):
        if read_json(manifest).get("frozen_config_sha256") == digest:
            raise ValueError(f"This frozen configuration was already evaluated: {manifest.parent}")


def prediction_fields() -> list[str]:
    return [
        "sample_id", "split", "gold_intent", "gold_domain", "pred_intent",
        "pred_domain_from_intent", "msp_score", "mahalanobis_score",
        "accepted_msp", "accepted_mahalanobis", "seed", "model", "checkpoint_hash",
    ]


def metric_bundle(gold, pred, intent_domain, is_scores, oos_scores, tau, num_intents) -> dict:
    return {
        **classification_metrics(gold, pred, intent_domain, num_intents=num_intents),
        **oos_metrics(is_scores, oos_scores),
        **frozen_threshold_metrics(is_scores, oos_scores, float(tau)),
        "is_false_rejection_rate": float(1.0 - frozen_threshold_metrics(is_scores, oos_scores, float(tau))["is_acceptance_at_val95"]),
        "threshold_source": "validation_frozen",
    }


def evaluate_baseline(frozen: dict, datasets: dict, intent_domain: np.ndarray, num_intents: int) -> tuple[list[dict], dict]:
    entry = frozen["baseline"]
    run_dir = resolve_path(entry["run_dir"])
    if sha256_file(run_dir / "pipeline.joblib") != entry["checkpoint_sha256"]:
        raise ValueError("Baseline checkpoint hash mismatch")
    pipeline = load_baseline(run_dir)
    all_records = [*datasets["test"], *datasets["oos_test"]]
    _, probabilities = predict_scores(pipeline, [row["text"] for row in all_records])
    scores = msp_scores(probabilities, "probabilities")
    predictions = probabilities.argmax(axis=1)
    tau = float(entry["thresholds"]["msp"]["threshold"])
    test_count = len(datasets["test"])
    rows = []
    for index, record in enumerate(all_records):
        is_oos = index >= test_count
        rows.append({
            "sample_id": record["sample_id"], "split": record["split"],
            "gold_intent": None if is_oos else int(record["intent_id"]),
            "gold_domain": None if is_oos else int(record["domain_id"]),
            "pred_intent": int(predictions[index]),
            "pred_domain_from_intent": int(intent_domain[predictions[index]]),
            "msp_score": float(scores[index]), "mahalanobis_score": None,
            "accepted_msp": bool(scores[index] < tau), "accepted_mahalanobis": None,
            "seed": 0, "model": "B0", "checkpoint_hash": entry["checkpoint_sha256"],
        })
    gold = np.asarray([row["intent_id"] for row in datasets["test"]], dtype=np.int64)
    metrics = {"msp": metric_bundle(
        gold, predictions[:test_count], intent_domain, scores[:test_count], scores[test_count:], tau, num_intents,
    )}
    return rows, metrics


def evaluate_bert(
    entry: dict, base: dict, datasets: dict, tokenizer, intent_domain: np.ndarray,
) -> tuple[list[dict], dict]:
    run_dir = resolve_path(entry["run_dir"])
    if sha256_file(run_dir / "best.pt") != entry["best_checkpoint_sha256"]:
        raise ValueError(f"BERT checkpoint hash mismatch: {run_dir}")
    detector_path = resolve_path(entry["detector_path"])
    if sha256_file(detector_path) != entry["detector_sha256"]:
        raise ValueError(f"Detector hash mismatch: {detector_path}")
    detector = MahalanobisDetector.load(detector_path)
    if detector.sample_hash_ != entry["detector_fit_sample_hash"]:
        raise ValueError(f"Detector fit sample hash mismatch: {detector_path}")
    records = [*datasets["test"], *datasets["oos_test"]]
    features = predict_records(base, run_dir, records, tokenizer)
    expected_ids = np.asarray([row["sample_id"] for row in records], dtype=np.str_)
    if not np.array_equal(features["sample_id"], expected_ids):
        raise ValueError(f"Test sample order/set mismatch: {run_dir}")
    msp = msp_scores(features["logits"], "logits")
    mahalanobis = detector.mahalanobis_scores(features["cls"])
    tau_msp = float(entry["thresholds"]["msp"]["threshold"])
    tau_mahal = float(entry["thresholds"]["mahalanobis"]["threshold"])
    count = len(datasets["test"])
    common = {
        "model": entry["model"], "seed": int(entry["seed"]),
        "tau_msp": tau_msp, "tau_mahalanobis": tau_mahal,
        "checkpoint_hash": entry["best_checkpoint_sha256"],
        "num_intents": len(intent_domain),
    }
    is_rows = export_predictions(
        features["sample_id"][:count], features["intent_id"][:count],
        features["logits"][:count].argmax(axis=1), intent_domain, split="test",
        msp_score=msp[:count], mahalanobis_score=mahalanobis[:count], **common,
    )
    oos_rows = export_predictions(
        features["sample_id"][count:], None,
        features["logits"][count:].argmax(axis=1), intent_domain, split="oos_test",
        msp_score=msp[count:], mahalanobis_score=mahalanobis[count:], **common,
    )
    gold = features["intent_id"][:count]
    pred = features["logits"][:count].argmax(axis=1)
    metrics = {
        "msp": metric_bundle(gold, pred, intent_domain, msp[:count], msp[count:], tau_msp, len(intent_domain)),
        "mahalanobis": metric_bundle(gold, pred, intent_domain, mahalanobis[:count], mahalanobis[count:], tau_mahal, len(intent_domain)),
    }
    return [*is_rows, *oos_rows], metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="01_configs/04_frozen.yaml")
    args = parser.parse_args()
    frozen_path = resolve_path(args.config)
    frozen = load_config(frozen_path)
    base = load_config(frozen["base_config"])
    require_approved_freeze(frozen, base)
    refuse_repeat_evaluation(frozen_path)
    datasets = load_datasets(base["data"]["root"], strict_counts=True)
    label_to_id, _, intent_to_domain_names = load_label_maps(base["metadata_dir"])
    intent_domain = np.empty(len(label_to_id), dtype=np.int64)
    domain_to_id = {domain: index for index, domain in enumerate(sorted(set(intent_to_domain_names.values())))}
    for intent, identifier in label_to_id.items():
        intent_domain[identifier] = domain_to_id[intent_to_domain_names[intent]]
    expected = {("bert_ce", seed) for seed in (0, 1, 2)} | {("bert_dom", seed) for seed in (0, 1, 2)}
    actual = {(entry["model"], int(entry["seed"])) for entry in frozen["models"]}
    if actual != expected or len(frozen["models"]) != 6:
        raise ValueError("Frozen evaluation must contain exactly three CE and three selected-Dom models")
    if any(float(entry["lambda_domain"]) != float(frozen["selected_lambda"]) for entry in frozen["models"] if entry["model"] == "bert_dom"):
        raise ValueError("A Dom model does not use the frozen selected lambda")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base["model"]["name"], revision=base["model"]["revision"], local_files_only=True)
    output = resolve_path("09_results/02_test") / make_run_id("frozen_test")
    output.mkdir(parents=True, exist_ok=False)
    all_rows = []
    metrics = {}
    baseline_rows, baseline_metrics = evaluate_baseline(frozen, datasets, intent_domain, len(label_to_id))
    all_rows.extend(baseline_rows)
    metrics["B0"] = baseline_metrics
    write_csv(output / "baseline_predictions.csv", baseline_rows, fieldnames=prediction_fields())
    for entry in frozen["models"]:
        rows, model_metrics = evaluate_bert(entry, base, datasets, tokenizer, intent_domain)
        key = f"{entry['model']}_seed{entry['seed']}"
        all_rows.extend(rows)
        metrics[key] = model_metrics
        write_csv(output / f"{key}_predictions.csv", rows, fieldnames=prediction_fields())
        print(f"Evaluated {key}")
    write_csv(output / "predictions.csv", all_rows, fieldnames=prediction_fields())
    write_json(output / "metrics.json", metrics)
    write_json(output / "evaluation_manifest.json", {
        "created_at": utc_now(), "status": "completed",
        "frozen_config_sha256": sha256_file(frozen_path),
        "supervisor_approval": frozen["supervisor_approval"],
        "sample_counts": {"test": len(datasets["test"]), "oos_test": len(datasets["oos_test"])},
        "sample_set_hash": stable_hash([row["sample_id"] for row in [*datasets["test"], *datasets["oos_test"]]]),
        "predictions_sha256": sha256_file(output / "predictions.csv"),
        "protocol": "Frozen artifacts and thresholds only; no test-based selection.",
    })
    print(f"Frozen test predictions and metrics: {output}")


if __name__ == "__main__":
    main()
