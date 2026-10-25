"""P3-S05: Explain when, why and how SCL changes behaviour (no new test inference).

Outputs under phase3_scl/09_results/analysis:
  per_domain_f1.csv, length_buckets.csv, candidate_confusions.csv   subgroup analysis
  error_taxonomy.csv, oos_false_accept_destinations.csv             failure modes (Mahalanobis val threshold)
  geometry_summary.csv                                              representation compactness / separation
  cases.csv                                                         fixed-seed examples for manual interpretation
  figures/*.png                                                     reliability, R2 size curve, R3 noise, t-SNE
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT / "03_src"), str(PROJECT_ROOT / "phase3_scl" / "03_src")]

from clinc_domain_aux.common import load_config, read_csv, read_json, resolve_path, write_csv
from clinc_domain_aux.confirmatory import fpr95_is_positive, macro_f1_present
from clinc_domain_aux.data import load_label_maps
from phase3_scl.metrics import expected_calibration_error
from phase3_scl.predictions import load_predictions, unique_test_run
from phase3_scl.study import STUDY_CONFIG, intent_domain_array, load_features, load_split, load_study, results_dir, verified_run

ARMS = ("ce_balanced", "scl")


def per_class_f1(gold: np.ndarray, pred: np.ndarray, num_labels: int) -> np.ndarray:
    tp = np.bincount(gold[gold == pred], minlength=num_labels).astype(float)
    denominator = np.bincount(gold, minlength=num_labels) + np.bincount(pred, minlength=num_labels)
    return np.divide(2 * tp, denominator, out=np.zeros(num_labels), where=denominator > 0)


def mean_over_seeds(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default=STUDY_CONFIG)
    args = parser.parse_args()
    study = load_study(args.study)
    settings = study["analysis"]
    test_run = unique_test_run(study)
    manifest = read_json(test_run / "evaluation_manifest.json")
    data = load_predictions(test_run, manifest)
    config = load_config(study["main"]["arms"]["ce_balanced"])
    num_labels = int(config["model"]["num_intents"])
    domains = intent_domain_array(config)
    label_to_id, domain_to_id, _ = load_label_maps(config["metadata_dir"])
    id_to_intent = {index: name for name, index in label_to_id.items()}
    id_to_domain = {index: name for name, index in domain_to_id.items()}
    split = load_split(config)
    text_of = {row["sample_id"]: row["text"] for row in [*split["test"], *split["oos_test"]]}
    seeds = [int(seed) for seed in study["main"]["seeds"]]
    output = results_dir(study, "analysis")
    figures = results_dir(study, "analysis", "figures")
    clean = {(arm, seed): data[("n80", arm, seed)]["clean"] for arm in ARMS for seed in seeds}
    gold = clean[(ARMS[0], seeds[0])]["gold"]
    ids = clean[(ARMS[0], seeds[0])]["ids"]

    # Subgroup 1: per-domain intent F1 (mean over seeds).
    rows = []
    for domain_id, domain in sorted(id_to_domain.items()):
        mask = domains == domain_id
        values = {arm: [per_class_f1(gold, clean[(arm, s)]["pred"], num_labels)[mask].mean() for s in seeds] for arm in ARMS}
        rows.append({"domain": domain, **{f"{arm}_f1_mean": mean_over_seeds(values[arm])[0] for arm in ARMS},
                     "delta_scl_minus_ce": float(np.mean(values["scl"]) - np.mean(values["ce_balanced"]))})
    write_csv(output / "per_domain_f1.csv", rows)

    # Subgroup 2: accuracy by utterance-length tertile.
    lengths = np.asarray([len(text_of[sample_id].split()) for sample_id in ids])
    edges = np.quantile(lengths, [1 / 3, 2 / 3])
    bucket = np.digitize(lengths, edges, right=True)
    write_csv(output / "length_buckets.csv", [{
        "bucket": ["short", "medium", "long"][b], "n": int((bucket == b).sum()),
        "words_range": f"{lengths[bucket == b].min()}-{lengths[bucket == b].max()}",
        **{f"{arm}_accuracy": float(np.mean([np.mean(clean[(arm, s)]["pred"][bucket == b] == gold[bucket == b]) for s in seeds])) for arm in ARMS},
    } for b in range(3) if (bucket == b).any()])  # tied quantiles can leave a bucket empty

    # Subgroup 3: the 16 confusable intent pairs predeclared in phase 1 (independent of any predictions).
    pairs = read_csv(resolve_path("09_results/05_tables/candidate_confusions.csv"))
    write_csv(output / "candidate_confusions.csv", [{
        "intent_a": pair["intent_a"], "intent_b": pair["intent_b"],
        **{f"{arm}_mean_errors": float(np.mean([
            np.sum((gold == label_to_id[pair["intent_a"]]) & (clean[(arm, s)]["pred"] == label_to_id[pair["intent_b"]]))
            + np.sum((gold == label_to_id[pair["intent_b"]]) & (clean[(arm, s)]["pred"] == label_to_id[pair["intent_a"]]))
            for s in seeds])) for arm in ARMS},
    } for pair in pairs])

    # Failure modes at the deployment operating point (Mahalanobis threshold chosen on validation).
    taxonomy, destinations = [], defaultdict(lambda: defaultdict(list))
    for arm in ARMS:
        counts = defaultdict(list)
        for seed in seeds:
            entry = clean[(arm, seed)]
            tau = manifest["runs"][f"n80__{arm}__seed{seed}"]["thresholds"]["mahalanobis"]
            pred, accepted = entry["pred"], entry["is"]["mahalanobis"] < tau
            wrong = pred != gold
            same_domain = domains[pred] == domains[gold]
            counts["within_domain_confusion"].append(int(np.sum(wrong & same_domain)))
            counts["cross_domain_confusion"].append(int(np.sum(wrong & ~same_domain)))
            counts["misclassified_but_rejected"].append(int(np.sum(wrong & ~accepted)))
            counts["is_false_reject"].append(int(np.sum(~accepted)))
            oos_accepted = entry["oos"]["mahalanobis"] < tau
            counts["oos_false_accept"].append(int(oos_accepted.sum()))
        taxonomy.append({"arm": arm, **{key: mean_over_seeds(values)[0] for key, values in counts.items()},
                         "n_is": len(gold), "n_oos": int(len(clean[(arm, seeds[0])]["oos"]["mahalanobis"]))})
    write_csv(output / "error_taxonomy.csv", taxonomy)

    oos_rows = {}
    for arm in ARMS:
        for seed in seeds:
            entry = manifest["runs"][f"n80__{arm}__seed{seed}"]
            tau = entry["thresholds"]["mahalanobis"]
            for row in read_csv(test_run / "predictions" / entry["file"]):
                if row["variant"] == "clean" and row["split"] == "oos_test" and float(row["mahalanobis_score"]) < tau:
                    destinations[arm][id_to_domain[int(domains[int(row["pred_intent"])])]].append(seed)
                    if seed == 0:
                        oos_rows.setdefault(arm, []).append(row)
    write_csv(output / "oos_false_accept_destinations.csv", [{
        "domain": domain, **{f"{arm}_mean_count": len(destinations[arm][domain]) / len(seeds) for arm in ARMS},
    } for domain in sorted(id_to_domain.values())])

    # Representation geometry (why): compactness / separation per arm and training size.
    geometry = defaultdict(list)
    for key, entry in manifest["runs"].items():
        if "geometry" in entry:
            group, arm, _ = key.split("__")
            geometry[(group, arm)].append(entry["geometry"])
    write_csv(output / "geometry_summary.csv", [{
        "group": group, "arm": arm, "n_runs": len(values),
        **{f"{metric}_mean": float(np.mean([v[metric] for v in values])) for metric in values[0]},
    } for (group, arm), values in sorted(geometry.items())])

    write_cases(output, study, clean, ids, gold, text_of, id_to_intent, id_to_domain, domains, manifest, oos_rows)
    plot_figures(figures, study, data, clean, seeds, gold, int(settings["ece_bins"]), config, domains)
    print(f"Analysis written to {output}")


def write_cases(output, study, clean, ids, gold, text_of, id_to_intent, id_to_domain, domains, manifest, oos_rows) -> None:
    """Fixed-seed samples (seed-0 models) from four pools; 'interpretation' is filled in manually."""
    rng = np.random.default_rng(int(study["analysis"]["case_seed"]))
    k = int(study["analysis"]["cases_per_pool"])
    ce, scl = clean[("ce_balanced", 0)], clean[("scl", 0)]
    tau = manifest["runs"]["n80__scl__seed0"]["thresholds"]["mahalanobis"]
    pools = {
        "ce_wrong_scl_right": np.flatnonzero((ce["pred"] != gold) & (scl["pred"] == gold)),
        "ce_right_scl_wrong": np.flatnonzero((ce["pred"] == gold) & (scl["pred"] != gold)),
        "scl_is_false_reject": np.flatnonzero(scl["is"]["mahalanobis"] >= tau),
    }
    cases = []
    for pool, indices in pools.items():
        for index in rng.permutation(indices)[:k]:
            cases.append({"pool": pool, "sample_id": ids[index], "text": text_of[ids[index]],
                          "gold": id_to_intent[int(gold[index])], "gold_domain": id_to_domain[int(domains[gold[index]])],
                          "ce_pred": id_to_intent[int(ce["pred"][index])], "scl_pred": id_to_intent[int(scl["pred"][index])],
                          "scl_mahalanobis": float(scl["is"]["mahalanobis"][index]), "interpretation": ""})
    accepted = oos_rows.get("scl", [])
    for position in rng.permutation(len(accepted))[:k]:
        row = accepted[position]
        cases.append({"pool": "scl_oos_false_accept", "sample_id": row["sample_id"], "text": text_of[row["sample_id"]],
                      "gold": "oos", "gold_domain": "", "ce_pred": "", "scl_pred": id_to_intent[int(row["pred_intent"])],
                      "scl_mahalanobis": float(row["mahalanobis_score"]), "interpretation": ""})
    write_csv(output / "cases.csv", cases)


def plot_figures(figures, study, data, clean, seeds, gold, bins, config, domains) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Reliability diagram (seed-pooled clean test).
    fig, axis = plt.subplots(figsize=(4, 4))
    for arm in ARMS:
        confidence = np.concatenate([clean[(arm, s)]["confidence"] for s in seeds])
        correct = np.concatenate([clean[(arm, s)]["pred"] == gold for s in seeds])
        table = expected_calibration_error(confidence, correct, bins)
        axis.plot([b["confidence"] for b in table["bins"]], [b["accuracy"] for b in table["bins"]], marker="o",
                  label=f"{arm} (ECE {table['ece']:.3f})")
    axis.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axis.set_xlabel("confidence"), axis.set_ylabel("accuracy"), axis.legend(fontsize=7)
    fig.tight_layout(), fig.savefig(figures / "reliability.png", dpi=200), plt.close(fig)

    # R2: SCL - CE gain against training size, and R3: metric under each perturbation.
    comparisons = read_json(results_dir(study, "statistics") / "comparisons.json")
    records = [comparisons["primary"]["H7"], comparisons["primary"]["H8"], *comparisons["secondary"]]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for axis, metric in zip(axes, ("macro_f1", "fpr95_mahalanobis")):
        points = sorted((int(r["group"].removeprefix("n")), r) for r in records
                        if r["metric"] == metric and r["treatment"] == "scl" and r["variant"] == "clean")
        sizes = [size for size, _ in points]
        axis.errorbar(sizes, [100 * r["seed_tests"]["mean"] for _, r in points],
                      yerr=np.abs(np.array([[100 * (r["seed_tests"]["mean"] - r["bootstrap"]["ci"][0]),
                                             100 * (r["bootstrap"]["ci"][1] - r["seed_tests"]["mean"])] for _, r in points]).T),
                      marker="o", capsize=3)
        axis.axhline(0, color="k", linewidth=0.8), axis.set_xscale("log"), axis.set_xticks(sizes, [str(s) for s in sizes])
        axis.set_xlabel("training utterances per intent"), axis.set_ylabel(f"SCL - CE, {metric} (pp)")
        axis.set_title("bars: 95% CI (n10, n25); 97.5% CI (n80, primary)", fontsize=7)
    fig.tight_layout(), fig.savefig(figures / "r2_size_effect.png", dpi=200), plt.close(fig)

    variants = sorted({variant for variant in data[("n80", "scl", seeds[0])]}, key=lambda v: (v != "clean", v))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    num_labels = int(config["model"]["num_intents"])
    for axis, (title, getter) in zip(axes, (
        ("macro-F1", lambda e: macro_f1_present(e["gold"], e["pred"], num_labels)),
        ("FPR95 (Mahalanobis)", lambda e: fpr95_is_positive(e["is"]["mahalanobis"], e["oos"]["mahalanobis"])),
    )):
        for arm in ARMS:
            axis.plot(variants, [np.mean([getter(data[("n80", arm, s)][v]) for s in seeds]) for v in variants], marker="o", label=arm)
        axis.set_title(title), axis.tick_params(axis="x", rotation=30), axis.legend(fontsize=7)
    fig.tight_layout(), fig.savefig(figures / "r3_perturbations.png", dpi=200), plt.close(fig)

    # t-SNE of validation CLS features (seed 0), coloured by domain: a qualitative view of compactness.
    from sklearn.manifold import TSNE

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for axis, arm in zip(axes, ARMS):
        run = verified_run(load_config(study["main"]["arms"][arm]), 0)
        val = load_features(run["run_dir"], "val_features.npz")
        embedded = TSNE(n_components=2, random_state=0, init="pca").fit_transform(val["cls"])
        axis.scatter(embedded[:, 0], embedded[:, 1], c=domains[val["intent_id"]], cmap="tab10", s=2)
        axis.set_title(f"{arm} (val, seed 0)"), axis.set_xticks([]), axis.set_yticks([])
    fig.tight_layout(), fig.savefig(figures / "tsne_val_seed0.png", dpi=200), plt.close(fig)


if __name__ == "__main__":
    main()
