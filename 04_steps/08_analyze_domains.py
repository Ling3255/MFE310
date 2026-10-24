"""S08: Generate E7 domain analyses and frozen E8 OOS destinations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "03_src"))

from clinc_domain_aux.common import load_config, read_csv, read_json, resolve_path, utc_now, write_json
from clinc_domain_aux.data import load_datasets, load_label_maps
from clinc_domain_aux.domain_analysis import analyze_domain_predictions, analyze_oos_false_acceptance


def test_run(value: str | None) -> Path:
    root = resolve_path("09_results/02_test")
    if value:
        path = resolve_path(value)
        if not (path / "evaluation_manifest.json").is_file():
            raise ValueError(f"Not a completed frozen test run: {path}")
        return path
    runs = [path for path in root.iterdir() if path.is_dir() and (path / "evaluation_manifest.json").is_file()]
    if len(runs) != 1:
        raise ValueError("Specify --test-run; no unique completed frozen test run was found")
    return runs[0]


def numeric(row: dict) -> dict:
    result = dict(row)
    for key in ("gold_intent", "gold_domain", "pred_intent", "pred_domain_from_intent", "seed"):
        result[key] = None if row[key] == "" else int(row[key])
    for key in ("msp_score", "mahalanobis_score"):
        result[key] = None if row[key] == "" else float(row[key])
    for key in ("accepted_msp", "accepted_mahalanobis"):
        result[key] = None if row[key] == "" else row[key].lower() == "true"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="01_configs/04_frozen.yaml")
    parser.add_argument("--test-run")
    args = parser.parse_args()
    frozen = load_config(args.config)
    source = test_run(args.test_run)
    rows = [numeric(row) for row in read_csv(source / "predictions.csv")]
    base = load_config(frozen["base_config"])
    datasets = load_datasets(base["data"]["root"], strict_counts=True)
    label_to_id, domain_to_id, _ = load_label_maps(base["metadata_dir"])
    ce = [row for row in rows if row["model"] == "bert_ce"]
    dom = [row for row in rows if row["model"] == "bert_dom"]
    settings = frozen["evaluation"]
    domain = analyze_domain_predictions(
        ce, dom, label_to_id, domain_to_id, datasets["test"],
        resolve_path("09_results/05_tables"), resolve_path("09_results/04_figures"),
        resolve_path("09_results/06_cases"), analysis_seed=int(settings["case_seed"]),
        cases_per_category=int(settings["cases_per_category"]),
    )
    oos = None
    if frozen["optional_analyses"].get("oos_false_acceptance_destinations", False):
        oos = analyze_oos_false_acceptance(
            rows, label_to_id, domain_to_id, resolve_path("09_results/05_tables"), resolve_path("09_results/04_figures"),
        )
    summary = {
        "created_at": utc_now(), "source_test_run": str(source),
        "domain_files": domain["files"], "domain_summary": domain["summary"],
        "oos_files": None if oos is None else oos["files"],
    }
    write_json("09_results/analysis_manifest.json", summary)
    evidence_path = resolve_path("00_docs/evidence_map.json")
    evidence = read_json(evidence_path) if evidence_path.exists() else {"claims": []}
    evidence["claims"] = [claim for claim in evidence.get("claims", []) if claim.get("source_test_run") != str(source)]
    for claim_id, path in {**domain["files"], **({} if oos is None else oos["files"])}.items():
        evidence.setdefault("claims", []).append({
            "claim_id": claim_id, "source_prediction": str(source / "predictions.csv"),
            "source_test_run": str(source), "frozen_config_sha256": read_json(source / "evaluation_manifest.json")["frozen_config_sha256"],
            "script": "04_steps/08_analyze_domains.py", "output_path": path,
        })
    write_json(evidence_path, evidence)
    print(f"Domain analysis: {resolve_path('09_results/05_tables')}")
    print(f"Cases: {resolve_path('09_results/06_cases')}")


if __name__ == "__main__":
    main()
