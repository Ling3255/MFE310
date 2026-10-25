"""P3-S04: Pre-registered statistics on the saved phase-3 test predictions (no new inference).

Primary (Bonferroni, alpha 0.025, 97.5% intervals):
  H7  scl vs ce_balanced, FPR@95%IS-TPR with Mahalanobis on IHT + OOS-251 (lower is better, SESOI 1 pp)
  H8  scl vs ce_balanced, macro-F1 non-inferiority (97.5% lower bound > -0.3 pp)
Secondary (95%, descriptive): other OOS scores, AUROC, calibration, error types, sampler ablation (A'),
R2 low-resource gains and R3 robustness under perturbation. B0 is reported as a single model.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT / "03_src"), str(PROJECT_ROOT / "phase3_scl" / "03_src")]

from clinc_domain_aux.common import load_config, read_json, stable_hash, utc_now, write_csv, write_json
from clinc_domain_aux.confirmatory import (
    auroc_oos, cross_domain_error, decide, fpr95_is_positive, macro_f1_present, seed_level_tests,
    two_level_bootstrap, two_level_oos_bootstrap, within_domain_error,
)
from phase3_scl.metrics import expected_calibration_error
from phase3_scl.predictions import load_predictions, unique_test_run
from phase3_scl.study import STUDY_CONFIG, intent_domain_array, load_study, results_dir


def run_metrics(entry: dict, domains: np.ndarray, num_labels: int, thresholds: dict, bins: int) -> dict:
    gold, pred = entry["gold"], entry["pred"]
    row = {
        "macro_f1": macro_f1_present(gold, pred, num_labels), "accuracy": float(np.mean(gold == pred)),
        "cross_domain_error": cross_domain_error(gold, pred, domains),
        "within_domain_error": within_domain_error(gold, pred, domains),
        "ece": expected_calibration_error(entry["confidence"], gold == pred, bins)["ece"],
    }
    for detector, is_scores in entry["is"].items():
        oos_scores = entry["oos"][detector]
        row[f"auroc_{detector}"] = auroc_oos(is_scores, oos_scores)
        row[f"fpr95_{detector}"] = fpr95_is_positive(is_scores, oos_scores)
        if detector in thresholds:
            row[f"is_acceptance_{detector}"] = float(np.mean(is_scores < thresholds[detector]))
            row[f"oos_false_acceptance_{detector}"] = float(np.mean(oos_scores < thresholds[detector]))
    return row


def compare(data: dict, group: str, reference: str, treatment: str, seeds: list[int], variant: str, metric: str,
            *, num_labels: int, domains: np.ndarray, level: float, alpha: float, settings: dict, sesoi: float | None = None) -> dict:
    """Paired seed-level test plus two-level bootstrap for one metric; returns a JSON-ready record."""
    ref = [data[(group, reference, s)][variant] for s in seeds]
    trt = [data[(group, treatment, s)][variant] for s in seeds]
    if any(r["ids"] != ref[0]["ids"] or t["ids"] != ref[0]["ids"] for r, t in zip(ref, trt)):
        raise ValueError("Paired runs disagree on the test sample order")
    gold = ref[0]["gold"]
    # Deterministic per-comparison seed (Python's built-in hash() of strings is randomised per process).
    boot_seed = int(settings["bootstrap_seed"]) + int(stable_hash([group, reference, treatment, variant, metric])[:8], 16) % 10_000
    if metric in ("macro_f1", "cross_domain_error", "within_domain_error", "accuracy"):
        fn = {"macro_f1": lambda g, p: macro_f1_present(g, p, num_labels),
              "cross_domain_error": lambda g, p: cross_domain_error(g, p, domains),
              "within_domain_error": lambda g, p: within_domain_error(g, p, domains),
              "accuracy": lambda g, p: float(np.mean(g == p))}[metric]
        better = "higher" if metric in ("macro_f1", "accuracy") else "lower"
        deltas = [fn(gold, t["pred"]) - fn(gold, r["pred"]) for r, t in zip(ref, trt)]
        boot = two_level_bootstrap(gold, np.stack([r["pred"] for r in ref]), np.stack([t["pred"] for t in trt]), fn,
                                   n_boot=int(settings["bootstrap_iterations"]), seed=boot_seed, level=level)
    else:
        kind, detector = metric.split("_", 1)
        fn = {"fpr95": fpr95_is_positive, "auroc": auroc_oos}[kind]
        better = "lower" if kind == "fpr95" else "higher"
        deltas = [fn(t["is"][detector], t["oos"][detector]) - fn(r["is"][detector], r["oos"][detector]) for r, t in zip(ref, trt)]
        boot = two_level_oos_bootstrap(
            gold, (np.stack([r["is"][detector] for r in ref]), np.stack([r["oos"][detector] for r in ref])),
            (np.stack([t["is"][detector] for t in trt]), np.stack([t["oos"][detector] for t in trt])), fn,
            n_boot=int(settings["oos_bootstrap_iterations"]), seed=boot_seed, level=level)
    distribution = boot.pop("distribution")
    tests = seed_level_tests(deltas)
    return {"group": group, "reference": reference, "treatment": treatment, "variant": variant, "metric": metric,
            "level": level, "seed_deltas": dict(zip(map(str, seeds), map(float, deltas))), "seed_tests": tests,
            "bootstrap": boot, "decision": decide(boot["ci"], tests, better=better, alpha=alpha, sesoi=sesoi),
            "_distribution": distribution}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default=STUDY_CONFIG)
    args = parser.parse_args()
    study = load_study(args.study)
    settings = study["analysis"]
    test_run = unique_test_run(study)
    manifest = read_json(test_run / "evaluation_manifest.json")
    data = load_predictions(test_run, manifest)
    reference_config = load_config(study["main"]["arms"]["ce_balanced"])
    domains, num_labels = intent_domain_array(reference_config), int(reference_config["model"]["num_intents"])

    per_run = []
    for (group, arm, seed), variants in sorted(data.items()):
        thresholds = manifest["runs"][f"{group}__{arm}__seed{seed}"]["thresholds"]
        for variant, entry in variants.items():
            per_run.append({"group": group, "arm": arm, "seed": seed, "variant": variant,
                            **run_metrics(entry, domains, num_labels, thresholds, int(settings["ece_bins"]))})
    output = results_dir(study, "statistics")
    write_csv(output / "per_run_metrics.csv", per_run, fieldnames=sorted({k for row in per_run for k in row},
              key=lambda k: (k not in ("group", "arm", "seed", "variant"), k)))

    main_seeds = [int(s) for s in study["main"]["seeds"]]
    low_seeds = [int(s) for s in study["low_resource"]["seeds"]]
    common = {"num_labels": num_labels, "domains": domains, "settings": settings}
    primary = dict(level=float(settings["level_primary"]), alpha=float(settings["alpha_primary"]))
    secondary = dict(level=float(settings["level_secondary"]), alpha=1 - float(settings["level_secondary"]))
    results = {
        "H7": compare(data, "n80", "ce_balanced", "scl", main_seeds, "clean", "fpr95_mahalanobis",
                      sesoi=float(settings["sesoi_fpr95"]), **primary, **common),
        "H8": compare(data, "n80", "ce_balanced", "scl", main_seeds, "clean", "macro_f1", **primary, **common),
    }
    margin = float(settings["noninferiority_margin_f1"])
    results["H8"]["non_inferior"] = bool(results["H8"]["bootstrap"]["ci"][0] > -margin)
    results["H8"]["noninferiority_margin"] = margin
    secondary_results = []
    for metric in ("auroc_mahalanobis", "fpr95_msp", "fpr95_knn", "auroc_msp", "auroc_knn", "accuracy",
                   "cross_domain_error", "within_domain_error"):
        secondary_results.append(compare(data, "n80", "ce_balanced", "scl", main_seeds, "clean", metric, **secondary, **common))
    for metric in ("macro_f1", "fpr95_mahalanobis"):  # sampler ablation: does class-balanced batching alone matter?
        secondary_results.append(compare(data, "n80", "ce_balanced", "ce_standard", main_seeds, "clean", metric, **secondary, **common))
        for size in study["low_resource"]["sizes"]:  # R2
            secondary_results.append(compare(data, size, "ce_balanced", "scl", low_seeds, "clean", metric, **secondary, **common))
        for variant in sorted({v for v in data[("n80", "scl", main_seeds[0])] if v != "clean"}):  # R3
            secondary_results.append(compare(data, "n80", "ce_balanced", "scl", main_seeds, variant, metric, **secondary, **common))
    ece = {arm: [row["ece"] for row in per_run if row["group"] == "n80" and row["arm"] == arm and row["variant"] == "clean"]
           for arm in ("ce_balanced", "scl")}
    distributions = {}
    for name, record in [*results.items(), *((f"secondary_{i}", r) for i, r in enumerate(secondary_results))]:
        distributions[name] = record.pop("_distribution")
    np.savez_compressed(output / "bootstrap_distributions.npz", **distributions)
    write_json(output / "comparisons.json", {
        "created_at": utc_now(), "source_test_run": test_run.relative_to(PROJECT_ROOT).as_posix(),
        "primary": results, "secondary": secondary_results,
        "ece_seed_level": {"deltas": [b - a for a, b in zip(ece["ce_balanced"], ece["scl"])],
                           "tests": seed_level_tests([b - a for a, b in zip(ece["ce_balanced"], ece["scl"])])},
        "note": "Secondary comparisons are descriptive (no multiplicity correction).",
    })
    write_summary(output / "main_results.md", per_run, results)
    for name, record in results.items():
        ci = record["bootstrap"]["ci"]
        print(f"{name}: delta {100 * record['seed_tests']['mean']:+.3f} pp, {100 * record['level']:.1f}% CI "
              f"[{100 * ci[0]:+.3f}, {100 * ci[1]:+.3f}] pp, Wilcoxon p={record['seed_tests']['wilcoxon_pvalue_two_sided']:.4f}"
              f" -> {record['decision']['verdict']}{' (non-inferior)' if record.get('non_inferior') else ''}")
    print(f"Statistics: {output}")


def write_summary(path: Path, per_run: list[dict], primary: dict) -> None:
    """Report-ready Markdown: clean-test mean (sd) per arm and the primary decisions."""
    columns = ["macro_f1", "accuracy", "cross_domain_error", "auroc_mahalanobis", "fpr95_mahalanobis", "fpr95_msp",
               "fpr95_knn", "ece", "is_acceptance_mahalanobis", "oos_false_acceptance_mahalanobis"]
    lines = ["| group | arm | n | " + " | ".join(columns) + " |", "|" + "---|" * (len(columns) + 3)]
    groups = defaultdict(list)
    for row in per_run:
        if row["variant"] == "clean":
            groups[(row["group"], row["arm"])].append(row)
    for (group, arm), rows in sorted(groups.items()):
        cells = []
        for column in columns:
            values = [row[column] for row in rows if column in row]
            cells.append("-" if not values else f"{np.mean(values):.4f}" + (f" ({np.std(values, ddof=1):.4f})" if len(values) > 1 else ""))
        lines.append(f"| {group} | {arm} | {len(rows)} | " + " | ".join(cells) + " |")
    lines += ["", "| hypothesis | delta (pp) | interval (pp) | Wilcoxon p | verdict |", "|---|---|---|---|---|"]
    for name, record in primary.items():
        ci = record["bootstrap"]["ci"]
        verdict = record["decision"]["verdict"] + (f"; non-inferior={record['non_inferior']}" if "non_inferior" in record else "")
        lines.append(f"| {name} {record['metric']} | {100 * record['seed_tests']['mean']:+.3f} | "
                     f"[{100 * ci[0]:+.3f}, {100 * ci[1]:+.3f}] ({100 * record['level']:.1f}%) | "
                     f"{record['seed_tests']['wilcoxon_pvalue_two_sided']:.4f} | {verdict} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
