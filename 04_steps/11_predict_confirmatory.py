"""S11: One approved test-set prediction pass for a frozen confirmatory study."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "03_src"))

from clinc_domain_aux.common import content_sha256, make_run_id, resolve_path, sha256_file, utc_now, write_csv, write_json
from clinc_domain_aux.confirmatory import load_study, output_root, require_approval, study_arms, verify_run
from clinc_domain_aux.data import load_datasets
from clinc_domain_aux.detectors import MahalanobisDetector
from clinc_domain_aux.evaluation import msp_scores
from clinc_domain_aux.training import _data_hashes, predict_records

FIELDS = ["sample_id", "split", "gold_intent", "pred_intent", "msp_score", "mahalanobis_score"]


def refuse_repeat(root: Path) -> None:
    """One test evaluation per study output directory, even if the study config is edited afterwards."""
    for manifest in (root / "test").glob("*/evaluation_manifest.json"):
        raise ValueError(f"This study was already evaluated on test: {manifest.parent}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Frozen study config, e.g. 01_configs/06_confirm_massive.yaml")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    study_path = resolve_path(args.config)
    study, base = load_study(args.config)
    require_approval(study)
    root = output_root(study)
    refuse_repeat(root)

    seeds = [int(seed) for seed in study["seeds"]]
    arms = study_arms(study)
    runs = {(arm["name"], seed): verify_run(base, float(arm["lambda_domain"]), seed) for arm in arms for seed in seeds}
    data_hashes = {run["data_hash"] for run in runs.values()}
    if len(data_hashes) != 1:
        raise ValueError("Runs in one study must share the same data hash")

    datasets = load_datasets(base["data"]["root"], strict_counts=bool(base["data"].get("strict_counts", True)))
    if _data_hashes(datasets)["combined"] != data_hashes.pop():
        raise ValueError("Current data files differ from the data used in training")
    has_oos = len(datasets["oos_test"]) > 0
    records = [*datasets["test"], *datasets["oos_test"]]
    train_labels = np.asarray([row["intent_id"] for row in datasets["train"]])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base["model"]["name"], revision=base["model"]["revision"], local_files_only=True)
    output = root / "test" / make_run_id("confirmatory_test")
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for (arm_name, seed), run in runs.items():
        features = predict_records(base, run["run_dir"], records, tokenizer)
        if list(features["sample_id"]) != [row["sample_id"] for row in records]:
            raise ValueError(f"Test sample order mismatch: {run['run_dir']}")
        msp = msp_scores(features["logits"], "logits")
        mahalanobis = np.full(len(records), np.nan)
        if has_oos:
            with np.load(run["run_dir"] / "train_features.npz", allow_pickle=False) as train:
                if not np.array_equal(train["intent_id"], train_labels):
                    raise ValueError(f"Training features do not match the train split: {run['run_dir']}")
                detector = MahalanobisDetector(int(base["model"]["num_intents"])).fit(train["cls"], train["intent_id"], train["sample_id"])
            mahalanobis = detector.mahalanobis_scores(features["cls"])
        predictions = features["logits"].argmax(axis=1)
        rows = [{
            "sample_id": record["sample_id"],
            "split": record["split"],
            "gold_intent": "" if record["is_oos"] else int(record["intent_id"]),
            "pred_intent": int(predictions[index]),
            "msp_score": float(msp[index]),
            "mahalanobis_score": "" if np.isnan(mahalanobis[index]) else float(mahalanobis[index]),
        } for index, record in enumerate(records)]
        name = f"{arm_name}_seed{seed}_predictions.csv"
        write_csv(output / name, rows, fieldnames=FIELDS)
        files[name] = {
            "content_sha256": content_sha256(output / name),
            "run_dir": run["run_dir"].relative_to(PROJECT_ROOT).as_posix(),
            "best_checkpoint_sha256": sha256_file(run["run_dir"] / "best.pt"),
        }
        print(f"Predicted {arm_name} seed {seed}")
    write_json(output / "evaluation_manifest.json", {
        "created_at": utc_now(),
        "status": "completed",
        "study": study["study"],
        "role": study["role"],
        "study_config_content_sha256": content_sha256(study_path),
        "supervisor_approval": study["supervisor_approval"],
        "test_reuse_disclosure": study.get("test_reuse_disclosure"),
        "has_oos": has_oos,
        "sample_counts": {"test": len(datasets["test"]), "oos_test": len(datasets["oos_test"])},
        "mahalanobis": "refit from each run's train_features.npz (train in-scope only); threshold-free metrics only" if has_oos else None,
        "files": files,
    })
    print(f"Confirmatory test predictions: {output}")


if __name__ == "__main__":
    main()
