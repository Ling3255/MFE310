"""S10: Convert MASSIVE en-US into the project data layout for the external replication."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "03_src"))

from clinc_domain_aux.common import content_sha256, read_json, resolve_path, sha256_file, utc_now, write_json
from clinc_domain_aux.confirmatory import convert_massive
from clinc_domain_aux.data import load_datasets

# Pinned in 00_docs/06_confirmatory_plan.md; a different file must fail loudly.
EXPECTED = {
    "archive_url": "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz",
    "archive_sha256": "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577",
    "jsonl_sha256": "c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e",
    "counts": {"train": 11514, "val": 2033, "test": 2974},
    "num_intents": 60,
    "num_domains": 18,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--massive-jsonl", required=True, help="Path to 1.1/data/en-US.jsonl from the MASSIVE 1.1 archive")
    parser.add_argument("--archive", help="Optional path to the downloaded .tar.gz, to record and check its SHA-256")
    parser.add_argument("--output", default="../dataset/massive_en", help="Converted data directory")
    parser.add_argument("--metadata", default="07_metadata/massive_en", help="Label maps and manifest directory")
    args = parser.parse_args()

    source = resolve_path(args.massive_jsonl)
    if content_sha256(source) != EXPECTED["jsonl_sha256"]:
        raise ValueError("en-US.jsonl does not match the pinned MASSIVE 1.1 file")
    if args.archive and sha256_file(args.archive) != EXPECTED["archive_sha256"]:
        raise ValueError("Archive does not match the pinned MASSIVE 1.1 archive")
    with source.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    data, domains, id_map = convert_massive(rows)
    counts = {split: len(data[split]) for split in ("train", "val", "test")}
    if counts != EXPECTED["counts"]:
        raise ValueError(f"Unexpected split counts: {counts}")
    if sum(map(len, domains.values())) != EXPECTED["num_intents"] or len(domains) != EXPECTED["num_domains"]:
        raise ValueError("Unexpected number of intents or scenarios")

    output = resolve_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "data_full.json", data)
    write_json(output / "domains.json", domains)
    write_json(output / "massive_id_map.json", id_map)

    metadata = resolve_path(args.metadata)
    datasets = load_datasets(output, metadata_dir=metadata, strict_counts=False)
    manifest_path = metadata / "data_manifest.json"
    manifest = read_json(manifest_path)
    manifest.update({
        "dataset": "MASSIVE 1.1 en-US (scenarios used as domains; dev used as val; no OOS)",
        "source": EXPECTED["archive_url"],
        "git_commit": None,
        "generated_at": utc_now(),
        "license": {
            "name": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "file": "1.1/LICENSE",
            "sha256": None,
        },
        "upstream": {
            "archive_sha256": EXPECTED["archive_sha256"],
            "jsonl_file": "1.1/data/en-US.jsonl",
            "jsonl_content_sha256": EXPECTED["jsonl_sha256"],
        },
        "oos": None,
        "intents_missing_from_split": {
            split: sorted({intent for intents in domains.values() for intent in intents}
                          - {row["intent"] for row in datasets[split]})
            for split in ("train", "val", "test")
        },
        "per_intent_counts": {
            split: dict(sorted(Counter(row["intent"] for row in datasets[split]).items()))
            for split in ("train", "val", "test")
        },
        "macro_f1_rule": "average over intents present in the evaluated gold split",
    })
    write_json(manifest_path, manifest)
    print(f"Converted MASSIVE en-US: {counts}; {len(domains)} scenarios")
    print(f"Intents missing per split: {manifest['intents_missing_from_split']}")
    print(f"Data: {output}\nMetadata: {metadata}")


if __name__ == "__main__":
    main()
