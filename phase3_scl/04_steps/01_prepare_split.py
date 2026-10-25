"""P3-S01: Build the fixed phase-3 split (and R2 low-resource subsets) from CLINC150 only.

Writes CLINC-format directories under ``split.output_root`` (outside the repository, like the raw
data) and a manifest with hashes, counts, exclusions and leakage checks under phase3_scl/07_metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT / "03_src"), str(PROJECT_ROOT / "phase3_scl" / "03_src")]

from clinc_domain_aux.common import content_sha256, resolve_path, utc_now, write_csv, write_json
from phase3_scl.splits import build_low_resource, build_main_split, check_no_test_leakage, describe, leakage_report
from phase3_scl.study import STUDY_CONFIG, load_study


def write_split(directory: Path, split: dict, domains: dict) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "data_full.json", split)
    write_json(directory / "domains.json", domains)
    return {name: content_sha256(directory / name) for name in ("data_full.json", "domains.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default=STUDY_CONFIG)
    parser.add_argument("--clinc-data", help="Override split.clinc_data (directory with data_full.json)")
    args = parser.parse_args()
    study = load_study(args.study)
    settings = study["split"]
    source = resolve_path(args.clinc_data or settings["clinc_data"])
    full = json.loads((source / "data_full.json").read_text(encoding="utf-8"))
    oos_plus = json.loads((source / "data_oos_plus.json").read_text(encoding="utf-8"))
    domains = json.loads((source / "domains.json").read_text(encoding="utf-8"))

    main_split, record = build_main_split(full, oos_plus, seed=int(settings["seed"]), test_per_intent=int(settings["test_per_intent"]))
    check_no_test_leakage(main_split)
    subsets = build_low_resource(main_split, seed=int(settings["low_resource_seed"]), sizes=settings["low_resource_sizes"])
    for subset in subsets.values():
        check_no_test_leakage(subset)

    root = resolve_path(settings["output_root"])
    hashes = {"main": write_split(root / "main", main_split, domains)}
    for size, subset in subsets.items():
        hashes[f"n{size}"] = write_split(root / f"n{size}", subset, domains)

    metadata = resolve_path("phase3_scl/07_metadata")
    write_json(metadata / "sample_origin.json", record["origin"])
    write_json(metadata / "split_manifest.json", {
        "created_at": utc_now(),
        "source": {name: content_sha256(source / name) for name in ("data_full.json", "data_oos_plus.json", "domains.json")},
        "settings": settings,
        "split_content_sha256": hashes,
        "counts": {"main": describe(main_split), **{f"n{size}": describe(subset)["train"] for size, subset in subsets.items()}},
        "excluded_from_internal_test": record["excluded_from_test"],
        "leakage_main": leakage_report(main_split),
        "note": "test = internal held-out IS (from CLINC train); oos_test = data_full oos_train + data_oos_plus-only OOS. "
                "Neither was used for evaluation in phases 1-2; the official CLINC test is not used.",
    })
    write_csv(metadata / "split_counts.csv", [
        {"split": name, **values} for name, values in describe(main_split).items()
    ])
    counts = {name: len(rows) for name, rows in main_split.items()}
    print(f"Main split: {counts}; low-resource train sizes: { {f'n{s}': len(v['train']) for s, v in subsets.items()} }")
    print(f"Data written under {root}; manifest in {metadata}")


if __name__ == "__main__":
    main()
