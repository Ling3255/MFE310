"""S03: Train one paired BERT-CE or BERT-Dom run."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "03_src"))

from clinc_domain_aux.common import load_config, resolve_path, run_name
from clinc_domain_aux.data import load_datasets, load_label_maps
from clinc_domain_aux.training import train_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="01_configs/01_base.yaml")
    parser.add_argument("--lambda-domain", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--allow-download", action="store_true", help="Permit Hugging Face downloads for initial model acquisition")
    parser.add_argument("--data-root", help="Override the configured CLINC data directory")
    args = parser.parse_args()
    if args.lambda_domain < 0:
        parser.error("--lambda-domain must be non-negative")
    if not args.allow_download:
        # Also covers BertModel.from_pretrained; kept out of the config so config_hash is unchanged.
        os.environ["HF_HUB_OFFLINE"] = "1"
    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = args.data_root
    datasets = load_datasets(config["data"]["root"], strict_counts=not args.smoke)
    metadata = resolve_path(config["metadata_dir"])
    label_to_id, domain_to_id, _ = load_label_maps(metadata)
    if len(label_to_id) != int(config["model"]["num_intents"]) or len(domain_to_id) != int(config["model"]["num_domains"]):
        raise ValueError("Model head sizes disagree with saved label maps")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"], revision=config["model"]["revision"],
        local_files_only=not args.allow_download,
    )
    name = run_name(args.lambda_domain, args.seed)
    run_dir = resolve_path(config["runs_dir"]) / ("smoke" if args.smoke else "") / name
    result = train_run(
        config, args.lambda_domain, args.seed, datasets, tokenizer, run_dir,
        smoke=args.smoke, stop_after_epoch=args.stop_after_epoch,
    )
    print(f"Run status: {result['status']}; directory: {run_dir}")


if __name__ == "__main__":
    main()
