"""P4-S03: Pre-registered statistics on the saved phase-4 predictions (no new inference).

Primary (n10, unseen pool U, Bonferroni alpha 0.025, 97.5% intervals):
  H1  interaction (da_scl - da_ce) - (ib_scl - ib_ce) on macro-F1 > 0   (SESOI 0.5 pp)
  H2  the same interaction on within-domain error < 0                  (mechanism)
Secondary (95%, descriptive): simple effects, cross-domain error, accuracy, OOS (Mahalanobis / MSP),
IHT replication and the n25 replication (5 seeds: intervals and direction only).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT / p) for p in ("03_src", "phase3_scl/03_src", "phase4_dascl/03_src")]

from clinc_domain_aux.common import content_sha256, load_config, read_csv, read_json, stable_hash, utc_now, write_csv, write_json
from clinc_domain_aux.confirmatory import (
    auroc_oos, cross_domain_error, decide, fpr95_is_positive, macro_f1_present, seed_level_tests, within_domain_error,
)
from phase3_scl.predictions import unique_test_run
from phase3_scl.study import intent_domain_array, results_dir
from phase4_dascl.contrasts import contrast_bootstrap, contrast_bootstrap_oos, seed_contrasts

STUDY = "phase4_dascl/01_configs/study.yaml"
IS_SETS = ("unseen", "iht")


def load(test_run: Path, manifest: dict) -> dict:
    """{(group, arm, seed): {eval_set: {'ids','gold','pred','msp','maha'}}} with OOS rows under 'oos'."""
    data = {}
    for key, entry in manifest["runs"].items():
        path = test_run / "predictions" / entry["file"]
        if content_sha256(path) != entry["content_sha256"]:
            raise ValueError(f"Prediction file changed after evaluation: {entry['file']}")
        buckets = defaultdict(lambda: defaultdict(list))
        for row in read_csv(path):
            bucket = buckets[row["eval_set"]]
            bucket["ids"].append(row["sample_id"])
            bucket["gold"].append(int(row["gold_intent"]) if row["gold_intent"] != "" else -1)
            bucket["pred"].append(int(row["pred_intent"]))
            bucket["msp"].append(float(row["msp_score"]))
            bucket["maha"].append(float(row["mahalanobis_score"]))
        group, arm, seed = key.split("__")
        data[(group, arm, int(seed.removeprefix("seed")))] = {
            name: {field: (values if field == "ids" else np.asarray(values)) for field, values in bucket.items()}
            for name, bucket in buckets.items()
        }
    return data


def metric_functions(num_labels: int, domains: np.ndarray) -> dict:
    return {
        "macro_f1": (lambda g, p: macro_f1_present(g, p, num_labels), "higher"),
        "accuracy": (lambda g, p: float(np.mean(g == p)), "higher"),
        "within_domain_error": (lambda g, p: within_domain_error(g, p, domains), "lower"),
        "cross_domain_error": (lambda g, p: cross_domain_error(g, p, domains), "lower"),
    }


def evaluate_contrast(data, group, seeds, eval_set, weights, metric, functions, *, level, alpha, n_boot, base_seed, sesoi=None):
    arms = list(weights)
    entries = {arm: [data[(group, arm, s)] for s in seeds] for arm in arms}
    reference_ids = entries[arms[0]][0][eval_set]["ids"]
    if any(e[eval_set]["ids"] != reference_ids for arm in arms for e in entries[arm]):
        raise ValueError("Arms disagree on the evaluation order")
    gold = entries[arms[0]][0][eval_set]["gold"]
    boot_seed = base_seed + int(stable_hash([group, eval_set, sorted(weights.items()), metric])[:8], 16) % 10_000
    if metric in functions:
        fn, better = functions[metric]
        per_seed = {arm: [fn(gold, e[eval_set]["pred"]) for e in entries[arm]] for arm in arms}
        boot = contrast_bootstrap(gold, {arm: np.stack([e[eval_set]["pred"] for e in entries[arm]]) for arm in arms},
                                  weights, fn, n_boot=n_boot, seed=boot_seed, level=level)
    else:
        kind, score = metric.split("_", 1)  # e.g. fpr95_maha, auroc_msp
        fn, better = {"fpr95": (fpr95_is_positive, "lower"), "auroc": (auroc_oos, "higher")}[kind]
        per_seed = {arm: [fn(e[eval_set][score], e["oos"][score]) for e in entries[arm]] for arm in arms}
        boot = contrast_bootstrap_oos(gold, {arm: (np.stack([e[eval_set][score] for e in entries[arm]]),
                                                   np.stack([e["oos"][score] for e in entries[arm]])) for arm in arms},
                                      weights, fn, n_boot=n_boot, seed=boot_seed, level=level)
    contrast = seed_contrasts(per_seed, weights)
    tests = seed_level_tests(contrast)
    boot.pop("distribution")
    record = {"group": group, "eval_set": eval_set, "weights": dict(weights), "metric": metric,
              "arm_means": {arm: float(np.mean(v)) for arm, v in per_seed.items()},
              "seed_contrasts": dict(zip(map(str, seeds), map(float, contrast))), "seed_tests": tests, "bootstrap": boot,
              "decision": decide(boot["ci"], tests, better=better, alpha=alpha, sesoi=sesoi)}
    if len(seeds) < 6:
        record["note"] = "fewer than 6 seeds: exact two-sided Wilcoxon cannot reach p < 0.05; interval and direction only"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default=STUDY)
    args = parser.parse_args()
    study = load_config(args.study)
    settings = study["analysis"]
    test_run = unique_test_run(study)
    manifest = read_json(test_run / "evaluation_manifest.json")
    data = load(test_run, manifest)
    config = load_config(study["groups"]["n10"]["arms"]["ib_ce"])
    functions = metric_functions(int(config["model"]["num_intents"]), intent_domain_array(config))
    output = results_dir(study, "statistics")

    per_run = []
    for (group, arm, seed), sets in sorted(data.items()):
        for eval_set in IS_SETS:
            entry = sets[eval_set]
            per_run.append({"group": group, "arm": arm, "seed": seed, "eval_set": eval_set,
                            **{name: fn(entry["gold"], entry["pred"]) for name, (fn, _) in functions.items()},
                            **{f"{kind}_{score}": f(entry[score], sets["oos"][score]) for kind, f in
                               (("fpr95", fpr95_is_positive), ("auroc", auroc_oos)) for score in ("maha", "msp")}})
    write_csv(output / "per_run_metrics.csv", per_run)

    common = {"n_boot": int(settings["bootstrap_iterations"]), "base_seed": int(settings["bootstrap_seed"])}
    group, eval_set = settings["primary_group"], settings["primary_eval_set"]
    seeds = [int(s) for s in study["groups"][group]["seeds"]]
    interaction = study["contrasts"]["interaction"]
    primary = dict(level=float(settings["level_primary"]), alpha=float(settings["alpha_primary"]), **common)
    results = {
        "H1": evaluate_contrast(data, group, seeds, eval_set, interaction, "macro_f1", functions,
                                sesoi=float(settings["sesoi_interaction_f1"]), **primary),
        "H2": evaluate_contrast(data, group, seeds, eval_set, interaction, "within_domain_error", functions, **primary),
    }
    secondary_settings = dict(level=float(settings["level_secondary"]), alpha=1 - float(settings["level_secondary"]),
                              n_boot=int(settings["secondary_bootstrap_iterations"]), base_seed=int(settings["bootstrap_seed"]))
    secondary = []
    for group_name, spec in study["groups"].items():
        group_seeds = [int(s) for s in spec["seeds"]]
        for evaluation in IS_SETS:
            for name, weights in study["contrasts"].items():
                for metric in ("macro_f1", "within_domain_error", "cross_domain_error", "fpr95_maha"):
                    if (group_name, evaluation, name) == (group, eval_set, "interaction") and metric in ("macro_f1", "within_domain_error"):
                        continue  # already the primary tests
                    secondary.append(evaluate_contrast(data, group_name, group_seeds, evaluation, weights, metric, functions,
                                                       **secondary_settings) | {"contrast": name})
    write_json(output / "comparisons.json", {
        "created_at": utc_now(), "source_test_run": test_run.relative_to(PROJECT_ROOT).as_posix(),
        "primary": results, "secondary": secondary,
        "note": "Secondary comparisons are descriptive (no multiplicity correction).",
    })
    write_summary(output / "main_results.md", per_run, results)
    for name, record in results.items():
        ci = record["bootstrap"]["ci"]
        print(f"{name} ({record['metric']}): interaction {100 * record['seed_tests']['mean']:+.3f} pp, "
              f"{100 * record['bootstrap']['level']:.1f}% CI [{100 * ci[0]:+.3f}, {100 * ci[1]:+.3f}] pp, "
              f"Wilcoxon p={record['seed_tests']['wilcoxon_pvalue_two_sided']:.4f} -> {record['decision']['verdict']}")
    print(f"Statistics: {output}")


def write_summary(path: Path, per_run: list[dict], primary: dict) -> None:
    """2x2 tables (mean (sd) over seeds) per group and evaluation set, then the primary decisions."""
    columns = ["macro_f1", "within_domain_error", "cross_domain_error", "fpr95_maha", "auroc_maha"]
    cells = defaultdict(list)
    for row in per_run:
        cells[(row["group"], row["eval_set"], row["arm"])].append(row)
    lines = []
    for group in sorted({row["group"] for row in per_run}):
        for eval_set in IS_SETS:
            lines += [f"### {group} / {eval_set}", "", "| arm | n | " + " | ".join(columns) + " |", "|" + "---|" * (len(columns) + 2)]
            for arm in ("ib_ce", "ib_scl", "da_ce", "da_scl"):
                rows = cells[(group, eval_set, arm)]
                values = [f"{np.mean([r[c] for r in rows]):.4f} ({np.std([r[c] for r in rows], ddof=1):.4f})" for c in columns]
                lines.append(f"| {arm} | {len(rows)} | " + " | ".join(values) + " |")
            lines.append("")
    lines += ["| hypothesis | interaction (pp) | interval (pp) | Wilcoxon p | verdict |", "|---|---|---|---|---|"]
    for name, record in primary.items():
        ci = record["bootstrap"]["ci"]
        lines.append(f"| {name} {record['metric']} | {100 * record['seed_tests']['mean']:+.3f} | "
                     f"[{100 * ci[0]:+.3f}, {100 * ci[1]:+.3f}] ({100 * record['bootstrap']['level']:.1f}%) | "
                     f"{record['seed_tests']['wilcoxon_pvalue_two_sided']:.4f} | {record['decision']['verdict']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
