"""S07: Recompute paired statistics and overlap sensitivity from frozen predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "03_src"))

from clinc_domain_aux.common import load_config, read_csv, read_json, resolve_path, write_csv, write_json
from clinc_domain_aux.data import load_datasets, load_label_maps
from clinc_domain_aux.statistics import (
    intent_stratified_paired_bootstrap, paired_seed_differences,
    summarize_seeds, train_test_overlap_sensitivity,
)


def resolve_test_run(value: str | None) -> Path:
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


def typed_is_rows(rows: list[dict], model: str, seed: int | None = None) -> list[dict]:
    selected = [row for row in rows if row["split"] == "test" and row["model"] == model and (seed is None or int(row["seed"]) == seed)]
    return [{**row, "gold_intent": int(row["gold_intent"]), "pred_intent": int(row["pred_intent"]), "seed": int(row["seed"])} for row in selected]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="01_configs/04_frozen.yaml")
    parser.add_argument("--test-run")
    args = parser.parse_args()
    frozen = load_config(args.config)
    test_run = resolve_test_run(args.test_run)
    rows = read_csv(test_run / "predictions.csv")
    metrics = read_json(test_run / "metrics.json")
    base = load_config(frozen["base_config"])
    datasets = load_datasets(base["data"]["root"], strict_counts=True)
    label_to_id, _, names = load_label_maps(base["metadata_dir"])
    domain_ids = {domain: index for index, domain in enumerate(sorted(set(names.values())))}
    intent_domain = np.empty(len(label_to_id), dtype=np.int64)
    for intent, index in label_to_id.items():
        intent_domain[index] = domain_ids[names[intent]]
    ce = {seed: typed_is_rows(rows, "bert_ce", seed) for seed in (0, 1, 2)}
    dom = {seed: typed_is_rows(rows, "bert_dom", seed) for seed in (0, 1, 2)}
    ce_f1 = {seed: metrics[f"bert_ce_seed{seed}"]["msp"]["macro_f1"] for seed in ce}
    dom_f1 = {seed: metrics[f"bert_dom_seed{seed}"]["msp"]["macro_f1"] for seed in dom}
    differences = paired_seed_differences(ce_f1, dom_f1)
    settings = frozen["evaluation"]
    result = intent_stratified_paired_bootstrap(
        ce, dom, num_intents=len(label_to_id),
        n_bootstrap=int(settings["bootstrap_iterations"]),
        samples_per_intent=int(settings["bootstrap_samples_per_intent"]),
        seed=int(settings["bootstrap_seed"]),
    )
    output = resolve_path("09_results/03_statistics")
    write_json(output / "paired_bootstrap.json", {
        "source_test_run": str(test_run),
        "observed_seed_deltas": result["observed_seed_deltas"],
        "observed_mean_delta": result["observed_mean_delta"],
        "confidence_interval": result["confidence_interval"],
        "confidence_level": result["confidence_level"],
        "n_bootstrap": result["n_bootstrap"],
        "samples_per_intent": result["samples_per_intent"],
        "analysis_seed": result["seed"],
        "condition": result["condition"],
        "interpretation_rule": "An interval crossing zero is insufficient evidence of improvement, not evidence of equivalence.",
    })
    write_csv(output / "bootstrap_mean_deltas.csv", ({
        "iteration": index,
        "mean_dom_minus_ce_macro_f1": float(value),
        "difference_percentage_points": float(value * 100),
    } for index, value in enumerate(result["bootstrap_mean_deltas"])))
    np.savez_compressed(output / "bootstrap_distributions.npz", **{
        "mean_deltas": result["bootstrap_mean_deltas"],
        **{f"seed_{seed}_deltas": values for seed, values in result["bootstrap_seed_deltas"].items()},
    })
    summary_rows = []
    for model in ("bert_ce", "bert_dom"):
        for metric in ("macro_f1", "accuracy", "cross_domain_error_rate", "within_domain_error_rate"):
            summary = summarize_seeds({seed: metrics[f"{model}_seed{seed}"]["msp"][metric] for seed in (0, 1, 2)})
            summary_rows.append({"model": model, "metric": metric, **summary, "values": str(summary["values"])})
    write_csv(output / "seed_summary.csv", summary_rows)
    write_json(output / "paired_seed_differences.json", differences)
    sensitivity_rows = []
    groups = [("B0", 0, typed_is_rows(rows, "B0"))]
    groups += [(model, seed, values) for model, collection in (("bert_ce", ce), ("bert_dom", dom)) for seed, values in collection.items()]
    for model, seed, predictions in groups:
        sensitivity = train_test_overlap_sensitivity(
            predictions, datasets["train"], intent_domain,
            test_rows=datasets["test"], num_intents=len(label_to_id),
        )
        sensitivity_rows.append({
            "model": model, "seed": seed, **sensitivity["metrics"],
            "n_total": sensitivity["n_total"], "n_excluded": sensitivity["n_excluded"],
            "n_remaining": sensitivity["n_remaining"],
        })
    write_csv(output / "overlap_sensitivity.csv", sensitivity_rows)
    print(f"Mean paired macro-F1 delta: {differences['mean_delta']:.6f}")
    print(f"95% paired bootstrap interval: {result['confidence_interval']}")
    print(f"Statistics: {output}")


if __name__ == "__main__":
    main()
