"""P4-S02: The single approved phase-4 evaluation (E19).

Gate: study.yaml frozen with complete supervisor approval and no earlier phase-4 evaluation. For every
run of the 2x2 design (n10 seeds 0-9, n25 seeds 0-4) this writes per-example predictions on the unseen
pool U (primary), the phase-3 IHT (replication, second use) and OOS-251, the validation-selected
thresholds, and centroid geometry used for the mechanism analysis.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT / p) for p in ("03_src", "phase3_scl/03_src", "phase4_dascl/03_src")]

from clinc_domain_aux.common import (
    content_sha256, environment_info, load_config, make_run_id, read_json, resolve_path, stable_hash, utc_now,
    write_csv, write_json,
)
from clinc_domain_aux.confirmatory import require_approval
from clinc_domain_aux.data import load_datasets
from clinc_domain_aux.detectors import MahalanobisDetector, select_threshold
from clinc_domain_aux.evaluation import msp_scores
from clinc_domain_aux.training import _data_hashes, predict_records
from phase3_scl.metrics import representation_geometry
from phase3_scl.study import intent_domain_array, load_features, load_split, results_dir, softmax, verified_run
from phase4_dascl.evaluation_sets import build_unseen
from phase4_dascl.geometry import domain_centroid_separation

STUDY = "phase4_dascl/01_configs/study.yaml"
FIELDS = ["sample_id", "eval_set", "gold_intent", "pred_intent", "confidence", "msp_score", "mahalanobis_score"]


def unseen_records(study: dict) -> list[dict]:
    """Rebuild U and check it against the manifest written before training."""
    phase3 = study["phase3_split"]
    records, _ = build_unseen(load_datasets(resolve_path(phase3["main"]), strict_counts=False),
                              load_datasets(resolve_path(phase3["n25"]), strict_counts=False),
                              per_intent=int(phase3["unseen_per_intent"]))
    manifest = read_json(resolve_path("phase4_dascl/07_metadata/unseen_manifest.json"))
    if stable_hash([[row["sample_id"], row["text"], row["intent_id"]] for row in records]) != manifest["content_hash"]:
        raise ValueError("Unseen pool U differs from the manifest fixed before training")
    return records


def evaluate_run(config: dict, seed: int, unseen: list[dict], tokenizer, output: Path, name: str) -> dict:
    run = verified_run(config, seed)
    split = load_split(config)
    if _data_hashes(split)["combined"] != run["data_hash"]:
        raise ValueError(f"Current split differs from the data used in training: {run['run_dir']}")
    sets = {"unseen": unseen, "iht": split["test"], "oos": split["oos_test"]}
    evaluated = [*unseen, *split["test"], *split["oos_test"]]
    features = predict_records(config, run["run_dir"], [*evaluated, *split["oos_val"]], tokenizer)
    count = len(evaluated)
    train = load_features(run["run_dir"], "train_features.npz")
    val = load_features(run["run_dir"], "val_features.npz")
    detector = MahalanobisDetector(int(config["model"]["num_intents"])).fit(train["cls"], train["intent_id"], train["sample_id"])
    msp = msp_scores(features["logits"], "logits")
    maha = detector.mahalanobis_scores(features["cls"])
    thresholds = {
        "msp": select_threshold(msp_scores(val["logits"], "logits"), msp[count:])["threshold"],
        "mahalanobis": select_threshold(detector.mahalanobis_scores(val["cls"]), maha[count:])["threshold"],
    }
    probabilities = softmax(features["logits"][:count])
    rows, position = [], 0
    for eval_set, records in sets.items():
        for record in records:
            rows.append({"sample_id": record["sample_id"], "eval_set": eval_set,
                         "gold_intent": "" if record["is_oos"] else int(record["intent_id"]),
                         "pred_intent": int(probabilities[position].argmax()), "confidence": float(probabilities[position].max()),
                         "msp_score": float(msp[position]), "mahalanobis_score": float(maha[position])})
            position += 1
    write_csv(output / "predictions" / name, rows, fieldnames=FIELDS)
    n_unseen = len(unseen)
    unseen_cls, unseen_gold = features["cls"][:n_unseen], features["intent_id"][:n_unseen]
    oos_cls = features["cls"][n_unseen + len(split["test"]):count]
    return {
        "file": name, "content_sha256": content_sha256(output / "predictions" / name),
        "run_dir": run["run_dir"].relative_to(PROJECT_ROOT).as_posix(), "best_epoch": int(run["best_epoch"]),
        "epochs_completed": int(run["epochs_completed"]), "thresholds": thresholds,
        "geometry": {**representation_geometry(train["cls"], train["intent_id"], unseen_cls, oos_cls),
                     **domain_centroid_separation(unseen_cls, unseen_gold, intent_domain_array(config))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default=STUDY)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    study = load_config(args.study)
    require_approval(study)
    for manifest in results_dir(study, "test").glob("*/evaluation_manifest.json"):
        raise ValueError(f"Phase-4 evaluation already exists: {manifest.parent}")
    unseen = unseen_records(study)
    output = results_dir(study, "test") / make_run_id("phase4_test")
    (output / "predictions").mkdir(parents=True)
    started = time.time()
    first = load_config(study["groups"]["n10"]["arms"]["ib_ce"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(first["model"]["name"], revision=first["model"]["revision"], local_files_only=True)
    runs = {}
    for group, spec in study["groups"].items():
        for arm, path in spec["arms"].items():
            config = load_config(path)
            for seed in spec["seeds"]:
                key = f"{group}__{arm}__seed{int(seed)}"
                runs[key] = evaluate_run(config, int(seed), unseen, tokenizer, output, f"{key}.csv")
                print(f"Evaluated {key}")
    write_json(output / "evaluation_manifest.json", {
        "created_at": utc_now(), "status": "completed", "elapsed_seconds": round(time.time() - started, 1),
        "study_config_content_sha256": content_sha256(resolve_path(args.study)),
        "unseen_manifest_sha256": content_sha256(resolve_path("phase4_dascl/07_metadata/unseen_manifest.json")),
        "supervisor_approval": study["supervisor_approval"], "environment": environment_info(), "runs": runs,
        "disclosure": "U is new; IHT and OOS-251 are used for the second time (phase 3 was the first).",
    })
    print(f"Phase-4 predictions: {output}")


if __name__ == "__main__":
    main()
