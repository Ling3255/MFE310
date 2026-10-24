"""S02: Train the E2 TF-IDF logistic-regression baseline."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "03_src"))

from clinc_domain_aux.baseline import train_baseline
from clinc_domain_aux.common import load_config, read_json, resolve_path
from clinc_domain_aux.data import load_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="01_configs/02_baseline.yaml")
    parser.add_argument("--data-root", help="Override the configured CLINC data directory")
    args = parser.parse_args()
    config = load_config(args.config)
    base = load_config(config.get("base_config", "01_configs/01_base.yaml"))
    datasets = load_datasets(args.data_root or base["data"]["root"], strict_counts=True)
    output = resolve_path(config["output_dir"])
    completed = train_baseline(datasets, config, output)
    validation_dir = resolve_path(config["validation_results_dir"])
    validation_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output / "c_selection.csv", validation_dir / "baseline_c_selection.csv")
    shutil.copy2(output / "metrics.json", validation_dir / "baseline_metrics.json")
    print(f"Selected C={completed['selected_c']:g}; val macro-F1={completed['best_val_macro_f1']:.6f}")
    if any(completed["convergence"].values()):
        raise SystemExit("A candidate did not converge; inspect c_selection.csv before formal use.")
    print(f"Baseline artifacts: {output}")


if __name__ == "__main__":
    main()
