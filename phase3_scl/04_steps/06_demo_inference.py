"""P3-S06: Single-utterance demo for the presentation (CE vs SCL, seed 0, frozen thresholds).

Reuses M11 FrozenInference with the checkpoint, the Mahalanobis detector saved by 03_evaluate_test.py
and the validation-selected thresholds recorded in its manifest. Runs only after the test evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT / "03_src"), str(PROJECT_ROOT / "phase3_scl" / "03_src")]

from clinc_domain_aux.common import load_config, read_json
from clinc_domain_aux.data import load_label_maps
from clinc_domain_aux.inference import FrozenInference
from phase3_scl.predictions import unique_test_run
from phase3_scl.study import STUDY_CONFIG, load_study, verified_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default=STUDY_CONFIG)
    parser.add_argument("--text", action="append", required=True, help="Utterance to classify (repeatable)")
    parser.add_argument("--arm", choices=("ce_balanced", "scl", "both"), default="both")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    study = load_study(args.study)
    test_run = unique_test_run(study)
    manifest = read_json(test_run / "evaluation_manifest.json")
    arms = ("ce_balanced", "scl") if args.arm == "both" else (args.arm,)
    for arm in arms:
        config = load_config(study["main"]["arms"][arm])
        thresholds = manifest["runs"][f"n80__{arm}__seed0"]["thresholds"]
        labels, domains, intent_domain = load_label_maps(config["metadata_dir"])
        model = FrozenInference(config, verified_run(config, 0)["run_dir"], test_run / "detectors" / f"{arm}_seed0_mahalanobis.npz",
                                thresholds["msp"], thresholds["mahalanobis"], labels, domains, intent_domain)
        for text in args.text:
            result = model.predict(text)
            decision = "ACCEPT" if result["accepted_mahalanobis"] else "REJECT (out of scope)"
            print(json.dumps({"arm": arm, "text": text, "intent": result["intent"], "domain": result["domain"],
                              "mahalanobis_score": round(result["mahalanobis_score"], 2), "decision": decision}, ensure_ascii=False))


if __name__ == "__main__":
    main()
