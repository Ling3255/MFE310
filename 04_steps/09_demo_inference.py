"""S09: Run single-sentence inference with an approved frozen BERT model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "03_src"))

from clinc_domain_aux.common import load_config, resolve_path, utc_now, write_json
from clinc_domain_aux.data import load_label_maps
from clinc_domain_aux.inference import FrozenInference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="01_configs/04_frozen.yaml")
    parser.add_argument("--text", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", choices=("bert_ce", "bert_dom"), default="bert_dom")
    parser.add_argument("--output", default="10_exports/demo_inference.json")
    args = parser.parse_args()
    frozen = load_config(args.config)
    approval = frozen.get("supervisor_approval", {})
    if not frozen.get("frozen") or not approval.get("approved") or not all(approval.get(key) for key in ("confirmed_by", "confirmed_at", "evidence")):
        raise ValueError("Demo requires an explicitly approved frozen configuration")
    entries = [entry for entry in frozen["models"] if entry["model"] == args.model and int(entry["seed"]) == args.seed]
    if len(entries) != 1:
        raise ValueError("The requested frozen model/seed is not uniquely available")
    entry = entries[0]
    base = load_config(frozen["base_config"])
    labels, domains, intent_domain = load_label_maps(base["metadata_dir"])
    inference = FrozenInference(
        base, entry["run_dir"], entry["detector_path"],
        entry["thresholds"]["msp"]["threshold"], entry["thresholds"]["mahalanobis"]["threshold"],
        labels, domains, intent_domain,
    )
    result = inference.predict(
        args.text,
        include_domain_head_score=bool(frozen["optional_analyses"].get("domain_head_score", False)) and args.model == "bert_dom",
    )
    payload = {"created_at": utc_now(), "model": args.model, "seed": args.seed, **result}
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
