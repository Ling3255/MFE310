"""S12: Seed-level tests and two-level bootstrap for a confirmatory or exploratory study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "03_src"))

from clinc_domain_aux.common import read_csv, read_json, resolve_path, sha256_file, utc_now, write_csv, write_json
from clinc_domain_aux.confirmatory import (
    auroc_oos, cross_domain_error, decide, fpr95_is_positive, load_study, macro_f1_present, output_root,
    seed_level_tests, study_arms, two_level_bootstrap, two_level_oos_bootstrap, verify_run, within_domain_error,
)
from clinc_domain_aux.data import load_label_maps


def intent_domain_array(base: dict) -> np.ndarray:
    label_to_id, domain_to_id, intent_to_domain = load_label_maps(base["metadata_dir"])
    mapping = np.empty(len(label_to_id), dtype=np.int64)
    for intent, index in label_to_id.items():
        mapping[index] = domain_to_id[intent_to_domain[intent]]
    return mapping


def load_val(base: dict, arms: list[dict], seeds: list[int]) -> dict:
    """Validation predictions come from each run's exported val_features.npz (no new inference)."""
    gold, sample_ids, preds = None, None, {}
    for arm in arms:
        for seed in seeds:
            run = verify_run(base, float(arm["lambda_domain"]), seed)
            with np.load(run["run_dir"] / "val_features.npz", allow_pickle=False) as values:
                ids, labels, logits = values["sample_id"], values["intent_id"], values["logits"]
            if gold is None:
                gold, sample_ids = labels.astype(np.int64), ids
            elif not np.array_equal(ids, sample_ids) or not np.array_equal(labels, gold):
                raise ValueError(f"Validation order differs: {run['run_dir']}")
            preds[(arm["name"], seed)] = logits.argmax(axis=1).astype(np.int64)
    return {"gold": gold, "preds": preds, "oos": None, "source": "08_runs/*/val_features.npz"}


def resolve_test_run(study: dict, value: str | None) -> Path:
    if value:
        return resolve_path(value)
    root = output_root(study) / "test"
    runs = [path for path in root.glob("*") if (path / "evaluation_manifest.json").is_file()] if root.exists() else []
    if len(runs) != 1:
        raise ValueError("Specify --test-run; no unique completed confirmatory test run was found")
    return runs[0]


def load_test(study: dict, arms: list[dict], seeds: list[int], test_run: Path) -> dict:
    manifest = read_json(test_run / "evaluation_manifest.json")
    gold, preds, scores = None, {}, {}
    for arm in arms:
        for seed in seeds:
            name = f"{arm['name']}_seed{seed}_predictions.csv"
            if sha256_file(test_run / name) != manifest["files"][name]["sha256"]:
                raise ValueError(f"Prediction file changed after evaluation: {name}")
            rows = read_csv(test_run / name)
            in_scope = [row for row in rows if row["split"] == "test"]
            oos = [row for row in rows if row["split"] == "oos_test"]
            labels = np.asarray([int(row["gold_intent"]) for row in in_scope], dtype=np.int64)
            if gold is None:
                gold = labels
            elif not np.array_equal(gold, labels):
                raise ValueError(f"Test gold labels differ: {name}")
            preds[(arm["name"], seed)] = np.asarray([int(row["pred_intent"]) for row in in_scope], dtype=np.int64)
            if manifest["has_oos"]:
                for detector in ("msp", "mahalanobis"):
                    field = f"{detector}_score"
                    scores[(arm["name"], seed, detector)] = (
                        np.asarray([float(row[field]) for row in in_scope]),
                        np.asarray([float(row[field]) for row in oos]),
                    )
    return {"gold": gold, "preds": preds, "oos": scores if manifest["has_oos"] else None,
            "source": test_run.relative_to(PROJECT_ROOT).as_posix()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--test-run", help="S11 output directory (defaults to the unique one)")
    args = parser.parse_args()
    study, base = load_study(args.config)
    if args.split == "test" and study["role"] == "exploratory":
        raise ValueError("Exploratory studies have no test analysis")
    arms = study_arms(study)
    seeds = [int(seed) for seed in study["seeds"]]
    analysis = study["analysis"]
    alpha, n_boot, boot_seed = float(analysis["alpha"]), int(analysis["bootstrap_iterations"]), int(analysis["bootstrap_seed"])
    data = load_val(base, arms, seeds) if args.split == "val" else load_test(study, arms, seeds, resolve_test_run(study, args.test_run))
    gold, preds = data["gold"], data["preds"]
    num_labels = int(base["model"]["num_intents"])
    domains = intent_domain_array(base)
    metrics = {
        "macro_f1": (lambda g, p: macro_f1_present(g, p, num_labels), "higher"),
        "cross_domain_error_rate": (lambda g, p: cross_domain_error(g, p, domains), "lower"),
        "within_domain_error_rate": (lambda g, p: within_domain_error(g, p, domains), "lower"),
    }

    per_seed = []
    for arm in arms:
        for seed in seeds:
            row = {"arm": arm["name"], "lambda_domain": float(arm["lambda_domain"]), "seed": seed}
            row.update({name: fn(gold, preds[(arm["name"], seed)]) for name, (fn, _) in metrics.items()})
            row["accuracy"] = float(np.mean(gold == preds[(arm["name"], seed)]))
            if data["oos"]:
                for detector in ("msp", "mahalanobis"):
                    is_scores, oos_scores = data["oos"][(arm["name"], seed, detector)]
                    row[f"auroc_oos_{detector}"] = auroc_oos(is_scores, oos_scores)
                    row[f"fpr95_is_positive_{detector}"] = fpr95_is_positive(is_scores, oos_scores)
            per_seed.append(row)

    reference = arms[0]
    comparisons, distributions = [], {}
    for index, treatment in enumerate(arms[1:]):
        comparison = {"treatment": treatment["name"], "lambda_domain": float(treatment["lambda_domain"]),
                      "reference": reference["name"], "metrics": {}}
        ref_stack = np.stack([preds[(reference["name"], seed)] for seed in seeds])
        trt_stack = np.stack([preds[(treatment["name"], seed)] for seed in seeds])
        for offset, (name, (fn, better)) in enumerate(metrics.items()):
            deltas = [fn(gold, trt_stack[i]) - fn(gold, ref_stack[i]) for i in range(len(seeds))]
            tests = seed_level_tests(deltas)
            boot = two_level_bootstrap(gold, ref_stack, trt_stack, fn, n_boot=n_boot, seed=boot_seed + 100 * index + offset)
            distributions[f"{treatment['name']}__{name}"] = boot.pop("distribution")
            sesoi = float(analysis["sesoi_macro_f1"]) if name == "macro_f1" else None
            comparison["metrics"][name] = {
                "seed_deltas": dict(zip(map(str, seeds), map(float, deltas))), "seed_tests": tests, "bootstrap": boot,
                "decision": decide(boot["ci95"], tests, better=better, alpha=alpha, sesoi=sesoi),
            }
        if data["oos"]:
            oos_metrics = {"auroc_oos": (auroc_oos, "higher"), "fpr95_is_positive": (fpr95_is_positive, "lower")}
            for d_index, detector in enumerate(("msp", "mahalanobis")):
                ref_is = np.stack([data["oos"][(reference["name"], s, detector)][0] for s in seeds])
                ref_oos = np.stack([data["oos"][(reference["name"], s, detector)][1] for s in seeds])
                trt_is = np.stack([data["oos"][(treatment["name"], s, detector)][0] for s in seeds])
                trt_oos = np.stack([data["oos"][(treatment["name"], s, detector)][1] for s in seeds])
                for m_index, (name, (fn, better)) in enumerate(oos_metrics.items()):
                    deltas = [fn(trt_is[i], trt_oos[i]) - fn(ref_is[i], ref_oos[i]) for i in range(len(seeds))]
                    tests = seed_level_tests(deltas)
                    boot = two_level_oos_bootstrap(
                        gold, (ref_is, ref_oos), (trt_is, trt_oos), fn,
                        n_boot=int(analysis["oos_bootstrap_iterations"]), seed=boot_seed + 100 * index + 10 + 2 * d_index + m_index,
                    )
                    distributions[f"{treatment['name']}__{name}_{detector}"] = boot.pop("distribution")
                    comparison["metrics"][f"{name}_{detector}"] = {
                        "seed_deltas": dict(zip(map(str, seeds), map(float, deltas))), "seed_tests": tests, "bootstrap": boot,
                        "decision": decide(boot["ci95"], tests, better=better, alpha=alpha), "role": "secondary, descriptive",
                    }
        comparisons.append(comparison)

    output = output_root(study) / args.split
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_seed_metrics.csv", per_seed)
    np.savez_compressed(output / "bootstrap_distributions.npz", **distributions)
    note = {
        "val": "Validation sanity check from saved val features. Not used to choose any setting in a confirmatory study.",
        "test": "Single approved test evaluation; see evaluation manifest for disclosure.",
    }[args.split]
    if study["role"] == "exploratory":
        note = "Exploratory: validation results may guide which variant to pre-register next; no confirmatory claim."
    write_json(output / "summary.json", {
        "created_at": utc_now(), "study": study["study"], "role": study["role"], "split": args.split,
        "source": data["source"], "seeds": seeds, "n_rows": int(len(gold)),
        "n_intents_present": int(len(np.unique(gold))), "macro_f1_rule": "average over intents present in gold",
        "analysis": analysis, "note": note, "comparisons": comparisons,
    })
    for comparison in comparisons:
        primary = comparison["metrics"]["macro_f1"]
        mechanism = comparison["metrics"]["cross_domain_error_rate"]
        print(f"[{args.split}] {comparison['treatment']} - {comparison['reference']}: "
              f"macro-F1 delta {100 * primary['seed_tests']['mean']:+.3f} pp, "
              f"CI95 [{100 * primary['bootstrap']['ci95'][0]:+.3f}, {100 * primary['bootstrap']['ci95'][1]:+.3f}] pp, "
              f"Wilcoxon p={primary['seed_tests']['wilcoxon_pvalue_two_sided']:.4f} -> {primary['decision']['verdict']}; "
              f"cross-domain delta {100 * mechanism['seed_tests']['mean']:+.3f} pp -> {mechanism['decision']['verdict']}")
    print(f"Statistics: {output}")


if __name__ == "__main__":
    main()
