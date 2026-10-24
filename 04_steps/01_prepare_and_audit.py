"""S01: Run E0 data preparation and E1 quality audits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "03_src"))

from clinc_domain_aux.audit import run_audit
from clinc_domain_aux.common import PROJECT_ROOT as COMMON_PROJECT_ROOT
from clinc_domain_aux.common import load_config, resolve_path, write_json
from clinc_domain_aux.data import load_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Directory containing data_full.json and domains.json")
    parser.add_argument("--output", default="07_metadata", help="Metadata output directory")
    parser.add_argument("--config", default="01_configs/01_base.yaml", help="Base YAML configuration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if COMMON_PROJECT_ROOT != PROJECT_ROOT:
        raise RuntimeError("The imported common module belongs to a different project root")
    config = load_config(args.config)
    metadata_dir = resolve_path(args.output)
    results_dir = resolve_path(config.get("results_dir", "09_results"))
    datasets = load_datasets(args.data_root, metadata_dir=metadata_dir, strict_counts=True)
    model = config["model"]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model["name"],
        revision=model["revision"],
        local_files_only=True,
    )
    summary = run_audit(
        datasets,
        output_dir=results_dir / "05_tables",
        figures_dir=results_dir / "04_figures",
        tokenizer=tokenizer,
        tokenizer_name=f"{model['name']}@{model['revision']}",
        max_length=int(model.get("max_length", 64)),
    )
    write_json(metadata_dir / "audit_summary.json", summary)
    print(f"Validated {sum(len(rows) for rows in datasets.values())} rows across six splits.")
    print(f"Metadata: {metadata_dir}")
    print(f"Audit tables: {results_dir / '05_tables'}")
    print(f"Audit figures: {results_dir / '04_figures'}")


if __name__ == "__main__":
    main()
