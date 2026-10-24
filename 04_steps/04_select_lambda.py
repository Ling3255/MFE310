"""S04: Select one global positive lambda from validation runs only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "03_src"))

from clinc_domain_aux.common import load_config, read_json, resolve_path, run_name, sha256_file, utc_now, write_csv, write_json
from clinc_domain_aux.domain_analysis import lambda_sensitivity_curve


def verified_run(runs_dir: Path, lambda_domain: float, seed: int) -> dict:
    run_dir = runs_dir / run_name(lambda_domain, seed)
    completed = read_json(run_dir / "completed.json")
    if completed.get("status") != "completed" or completed.get("smoke", False):
        raise ValueError(f"Missing completed formal run: {run_dir}")
    if float(completed["lambda_domain"]) != lambda_domain or int(completed["seed"]) != seed:
        raise ValueError(f"Run identity mismatch: {run_dir}")
    for name, expected in completed["artifacts"].items():
        if sha256_file(run_dir / name) != expected:
            raise ValueError(f"Artifact hash mismatch: {run_dir / name}")
    completed["run_dir"] = str(run_dir)
    return completed


def select_lambda(validation_rows: list[dict], seeds: list[int], tolerance: float) -> dict:
    means = {}
    for lambda_domain in sorted({float(row["lambda_domain"]) for row in validation_rows}):
        rows = [row for row in validation_rows if float(row["lambda_domain"]) == lambda_domain]
        if {int(row["seed"]) for row in rows} != set(seeds) or len(rows) != len(seeds):
            raise ValueError(f"Incomplete seeds for lambda={lambda_domain}")
        means[lambda_domain] = float(np.mean([row["best_val_macro_f1"] for row in rows]))
    positive = {key: value for key, value in means.items() if key > 0}
    if not positive:
        raise ValueError("The lambda search contains no positive values")
    best = max(positive.values())
    eligible = sorted(key for key, value in positive.items() if value >= best - tolerance)
    return {
        "selected_lambda": eligible[0],
        "lambda_means": means,
        "best_positive_mean": best,
        "eligible_lambdas": eligible,
        "tolerance": tolerance,
        "selection_rule": "maximum three-seed positive-lambda mean, then smallest lambda within tolerance",
        "created_at": utc_now(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="01_configs/03_search.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    runs_dir = resolve_path(config["runs_dir"])
    rows = [verified_run(runs_dir, float(value), int(seed)) for value in config["lambda_grid"] for seed in config["seeds"]]
    selected = select_lambda(rows, [int(seed) for seed in config["seeds"]], float(config["selection"]["tolerance"]))
    output = resolve_path(config["validation_results_dir"])
    write_csv(output / "lambda_summary.csv", ({
        "lambda_domain": row["lambda_domain"],
        "seed": row["seed"],
        "best_epoch": row["best_epoch"],
        "val_macro_f1": row["best_val_macro_f1"],
        "config_hash": row["config_hash"],
        "best_checkpoint_hash": row["best_checkpoint_hash"],
    } for row in rows))
    write_json(output / "selected_lambda.json", selected)
    curve = lambda_sensitivity_curve(rows, selected["selected_lambda"], output, resolve_path("09_results/04_figures"))
    print(f"Selected lambda={selected['selected_lambda']:g} from validation only.")
    print(f"Selection: {output / 'selected_lambda.json'}")
    print(f"Curve: {curve['files']['lambda_sensitivity_figure']}")


if __name__ == "__main__":
    main()
