"""P4-S04: Mechanism and error analysis for phase 4 (no new inference).

Outputs under phase4_dascl/09_results/analysis (primary group n10, unseen pool U unless stated):
  per_domain.csv                 intent F1 and within-domain error per domain and arm
  candidate_confusions.csv       the 16 confusable pairs predeclared in phase 1
  geometry_summary.csv           same-/other-domain centroid separation and compactness per group and arm
  sampler_summary.csv            batch composition from the pre-training audit (P4-S01)
  cases.csv                      fixed-seed ib_scl vs da_scl repairs / regressions (fill 'interpretation' by hand)
  figures/interaction_*.png      2x2 interaction plots (macro-F1, within-domain error) for n10 and n25
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT / p) for p in ("03_src", "phase3_scl/03_src", "phase4_dascl/03_src")]

from clinc_domain_aux.common import load_config, read_csv, read_json, resolve_path, write_csv
from clinc_domain_aux.confirmatory import macro_f1_present, within_domain_error
from clinc_domain_aux.data import load_label_maps
from phase3_scl.predictions import unique_test_run
from phase3_scl.study import intent_domain_array, results_dir
from phase4_dascl.evaluation_sets import load_unseen

STUDY = "phase4_dascl/01_configs/study.yaml"
ARMS = ("ib_ce", "ib_scl", "da_ce", "da_scl")


def load_rows(test_run: Path, manifest: dict, key: str, eval_set: str) -> list[dict]:
    return [row for row in read_csv(test_run / "predictions" / manifest["runs"][key]["file"]) if row["eval_set"] == eval_set]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default=STUDY)
    args = parser.parse_args()
    study = load_config(args.study)
    test_run = unique_test_run(study)
    manifest = read_json(test_run / "evaluation_manifest.json")
    config = load_config(study["groups"]["n10"]["arms"]["ib_ce"])
    num_labels = int(config["model"]["num_intents"])
    domains = intent_domain_array(config)
    label_to_id, domain_to_id, _ = load_label_maps(config["metadata_dir"])
    id_to_intent = {v: k for k, v in label_to_id.items()}
    id_to_domain = {v: k for k, v in domain_to_id.items()}
    output = results_dir(study, "analysis")
    figures = results_dir(study, "analysis", "figures")
    seeds = [int(s) for s in study["groups"]["n10"]["seeds"]]

    preds, gold, ids = {}, None, None
    for arm in ARMS:
        for seed in seeds:
            rows = load_rows(test_run, manifest, f"n10__{arm}__seed{seed}", "unseen")
            preds[(arm, seed)] = np.asarray([int(r["pred_intent"]) for r in rows])
            gold = np.asarray([int(r["gold_intent"]) for r in rows]) if gold is None else gold
            ids = [r["sample_id"] for r in rows] if ids is None else ids

    # Per-domain intent F1 and within-domain error (mean over seeds).
    table = []
    for domain_id, domain in sorted(id_to_domain.items()):
        rows = domains[gold] == domain_id
        record = {"domain": domain, "n": int(rows.sum())}
        for arm in ARMS:
            f1 = [macro_f1_present(gold[rows], preds[(arm, s)][rows], num_labels) for s in seeds]
            within = [within_domain_error(gold[rows], preds[(arm, s)][rows], domains) for s in seeds]
            record[f"{arm}_f1"], record[f"{arm}_within_error"] = float(np.mean(f1)), float(np.mean(within))
        record["interaction_f1"] = (record["da_scl_f1"] - record["da_ce_f1"]) - (record["ib_scl_f1"] - record["ib_ce_f1"])
        table.append(record)
    write_csv(output / "per_domain.csv", table)

    # Predeclared confusable pairs (phase 1, chosen without any predictions).
    pairs = read_csv(resolve_path("09_results/05_tables/candidate_confusions.csv"))
    write_csv(output / "candidate_confusions.csv", [{
        "intent_a": p["intent_a"], "intent_b": p["intent_b"], "same_domain": p["domain_a"] == p["domain_b"],
        **{f"{arm}_mean_errors": float(np.mean([
            np.sum((gold == label_to_id[p["intent_a"]]) & (preds[(arm, s)] == label_to_id[p["intent_b"]]))
            + np.sum((gold == label_to_id[p["intent_b"]]) & (preds[(arm, s)] == label_to_id[p["intent_a"]])) for s in seeds]))
           for arm in ARMS},
    } for p in pairs])

    # Geometry per group and arm, from the evaluation manifest.
    geometry = defaultdict(list)
    for key, entry in manifest["runs"].items():
        group, arm, _ = key.split("__")
        geometry[(group, arm)].append(entry["geometry"])
    write_csv(output / "geometry_summary.csv", [{
        "group": group, "arm": arm, "n_runs": len(values),
        **{f"{metric}_mean": float(np.mean([v[metric] for v in values])) for metric in values[0]},
    } for (group, arm), values in sorted(geometry.items())])

    # Batch composition evidence from the pre-training audit.
    audit = read_csv(resolve_path("phase4_dascl/07_metadata/sampler_audit.csv"))
    summary = defaultdict(list)
    for row in audit:
        summary[(row["group"], row["sampler"])].append(row)
    write_csv(output / "sampler_summary.csv", [{
        "group": group, "sampler": sampler, "seeds": len(rows),
        **{key: float(np.mean([float(r[key]) for r in rows])) for key in
           ("mean_positives_per_anchor", "mean_same_domain_negatives_per_anchor", "mean_other_domain_negatives_per_anchor",
            "mean_domains_per_batch", "batches")},
    } for (group, sampler), rows in sorted(summary.items())])

    write_cases(output, study, gold, ids, preds, id_to_intent, id_to_domain, domains)
    plot_interactions(figures, study, test_run, manifest, num_labels, domains)
    print(f"Analysis written to {output}")


def write_cases(output, study, gold, ids, preds, id_to_intent, id_to_domain, domains) -> None:
    """Seed-0 repairs and regressions of da_scl relative to ib_scl, labelled by error type.

    Prediction files do not store text, so U is rebuilt from the phase-3 split for the case table.
    """
    unseen, _, _ = load_unseen(study)
    text_of = {row["sample_id"]: row["text"] for row in unseen}
    rng = np.random.default_rng(int(study["analysis"]["case_seed"]))
    k = int(study["analysis"]["cases_per_pool"])
    ib, da = preds[("ib_scl", 0)], preds[("da_scl", 0)]
    pools = {"ib_scl_wrong_da_scl_right": np.flatnonzero((ib != gold) & (da == gold)),
             "ib_scl_right_da_scl_wrong": np.flatnonzero((ib == gold) & (da != gold))}
    cases = []
    for pool, indices in pools.items():
        for index in rng.permutation(indices)[:k]:
            wrong = ib[index] if pool.startswith("ib_scl_wrong") else da[index]
            cases.append({"pool": pool, "sample_id": ids[index], "text": text_of[ids[index]],
                          "gold": id_to_intent[int(gold[index])], "gold_domain": id_to_domain[int(domains[gold[index]])],
                          "ib_scl_pred": id_to_intent[int(ib[index])], "da_scl_pred": id_to_intent[int(da[index])],
                          "error_type": "within_domain" if domains[wrong] == domains[gold[index]] else "cross_domain",
                          "interpretation": ""})
    write_csv(output / "cases.csv", cases)


def plot_interactions(figures, study, test_run, manifest, num_labels, domains) -> None:
    """Classic 2x2 interaction plots: lines = loss, x = sampler; error bars = sd over seeds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = {"macro-F1": lambda g, p: macro_f1_present(g, p, num_labels),
               "within-domain error": lambda g, p: within_domain_error(g, p, domains)}
    for group, spec in study["groups"].items():
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        for axis, (title, fn) in zip(axes, metrics.items()):
            for loss, marker in (("ce", "o"), ("scl", "s")):
                means, sds = [], []
                for sampler in ("ib", "da"):
                    values = []
                    for seed in spec["seeds"]:
                        rows = load_rows(test_run, manifest, f"{group}__{sampler}_{loss}__seed{int(seed)}", "unseen")
                        values.append(fn(np.asarray([int(r["gold_intent"]) for r in rows]),
                                         np.asarray([int(r["pred_intent"]) for r in rows])))
                    means.append(np.mean(values)), sds.append(np.std(values, ddof=1))
                axis.errorbar([0, 1], means, yerr=sds, marker=marker, capsize=3, label=loss.upper())
            axis.set_xticks([0, 1], ["intent-balanced", "domain-aware"]), axis.set_title(f"{group}: {title} (U)")
            axis.legend(fontsize=7)
        fig.tight_layout(), fig.savefig(figures / f"interaction_{group}.png", dpi=200), plt.close(fig)


if __name__ == "__main__":
    main()
