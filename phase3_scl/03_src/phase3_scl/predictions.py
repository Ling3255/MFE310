"""P3-M5: Loading the saved phase-3 test predictions (shared by the statistics and analysis steps)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from clinc_domain_aux.common import content_sha256, read_csv
from phase3_scl.study import results_dir

DETECTORS = ("msp", "mahalanobis", "knn")


def unique_test_run(study: dict) -> Path:
    runs = [path for path in results_dir(study, "test").glob("*") if (path / "evaluation_manifest.json").is_file()]
    if len(runs) != 1:
        raise ValueError("Expected exactly one completed phase-3 test evaluation")
    return runs[0]


def load_predictions(test_run: Path, manifest: dict) -> dict:
    """{(group, arm, seed): {variant: {'gold', 'pred', 'confidence', 'is': {det: arr}, 'oos': {det: arr}}}}"""
    data = {}
    for key, entry in manifest["runs"].items():
        path = test_run / "predictions" / entry["file"]
        if content_sha256(path) != entry["content_sha256"]:
            raise ValueError(f"Prediction file changed after evaluation: {entry['file']}")
        group, arm, seed = key.split("__")
        per_variant = defaultdict(lambda: {"ids": [], "gold": [], "pred": [], "confidence": [],
                                           "is": defaultdict(list), "oos": defaultdict(list)})
        for row in read_csv(path):
            bucket = per_variant[row["variant"]]
            side = "oos" if row["split"] == "oos_test" else "is"
            if side == "is":
                bucket["ids"].append(row["sample_id"])
                bucket["gold"].append(int(row["gold_intent"]))
                bucket["pred"].append(int(row["pred_intent"]))
                bucket["confidence"].append(float(row["confidence"]))
            for detector in DETECTORS:
                if row[f"{detector}_score"] != "":
                    bucket[side][detector].append(float(row[f"{detector}_score"]))
        data[(group, arm, int(seed.removeprefix("seed")))] = {
            variant: {"ids": values["ids"], "gold": np.asarray(values["gold"]), "pred": np.asarray(values["pred"]),
                      "confidence": np.asarray(values["confidence"]),
                      "is": {k: np.asarray(v) for k, v in values["is"].items()},
                      "oos": {k: np.asarray(v) for k, v in values["oos"].items()}}
            for variant, values in per_variant.items()
        }
    return data
