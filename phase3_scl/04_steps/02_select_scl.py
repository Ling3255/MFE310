"""P3-S02: E12 development selection on validation data only.

For every dev run (reference CE and the six SupCon candidates, seeds 100-102) this computes val
macro-F1 and Mahalanobis AUROC (val IS vs oos_val, detector fitted on train features). The
pre-registered rule picks the candidate with the highest mean AUROC whose mean val macro-F1 is within
``f1_tolerance`` of the reference (ties: smaller weight, then smaller temperature). The KNN k is chosen
on the reference runs. Writes the selection record and materialises the selected SCL configs.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT / "03_src"), str(PROJECT_ROOT / "phase3_scl" / "03_src")]

from clinc_domain_aux.common import content_sha256, load_config, resolve_path, save_yaml, utc_now, write_csv, write_json
from clinc_domain_aux.confirmatory import auroc_oos, fpr95_is_positive, macro_f1_present
from clinc_domain_aux.detectors import KNNDetector, MahalanobisDetector
from clinc_domain_aux.training import predict_records
from phase3_scl.study import STUDY_CONFIG, load_features, load_split, load_study, msp, results_dir, verified_run


def evaluate_dev_run(config: dict, seed: int, tokenizer, k_grid: list[int] | None) -> dict:
    """Validation metrics of one dev run; KNN AUROCs only when ``k_grid`` is given."""
    run = verified_run(config, seed)
    train = load_features(run["run_dir"], "train_features.npz")
    val = load_features(run["run_dir"], "val_features.npz")
    oos = predict_records(config, run["run_dir"], load_split(config)["oos_val"], tokenizer)
    num_labels = int(config["model"]["num_intents"])
    detector = MahalanobisDetector(num_labels).fit(train["cls"], train["intent_id"])
    is_maha, oos_maha = detector.mahalanobis_scores(val["cls"]), detector.mahalanobis_scores(oos["cls"])
    row = {
        "seed": seed,
        "val_macro_f1": macro_f1_present(val["intent_id"], val["logits"].argmax(axis=1), num_labels),
        "val_auroc_mahalanobis": auroc_oos(is_maha, oos_maha),
        "val_fpr95_mahalanobis": fpr95_is_positive(is_maha, oos_maha),
        "val_auroc_msp": auroc_oos(msp(val["logits"]), msp(oos["logits"])),
        "best_epoch": int(run["best_epoch"]),
    }
    for k in k_grid or []:
        knn = KNNDetector(k).fit(train["cls"])
        row[f"val_auroc_knn_k{k}"] = auroc_oos(knn.score(val["cls"]), knn.score(oos["cls"]))
    return row


def materialise(source: dict, runs_dir: str, scl: dict, path: Path, header: str) -> str:
    """Write a generated config; an existing file must be identical (configs are frozen once used)."""
    config = copy.deepcopy(source)
    config["runs_dir"] = runs_dir
    config["training"]["supervised_contrastive"] = dict(scl)
    tmp = path.with_suffix(".tmp")
    save_yaml(tmp, config)
    text = f"# {header}\n" + tmp.read_text(encoding="utf-8")
    tmp.unlink()
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"{path} already exists with different content; generated configs must not change")
    path.write_text(text, encoding="utf-8")
    return content_sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default=STUDY_CONFIG)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    study = load_study(args.study)
    dev = study["dev"]
    seeds = [int(seed) for seed in dev["seeds"]]
    reference = load_config(dev["reference"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(reference["model"]["name"], revision=reference["model"]["revision"], local_files_only=True)

    rows = []
    k_grid = [int(k) for k in dev["knn_k_grid"]]
    for seed in seeds:
        rows.append({"config": dev["reference"], "weight": 0.0, "temperature": None,
                     **evaluate_dev_run(reference, seed, tokenizer, k_grid)})
    candidates = {}
    for path in dev["candidates"]:
        config = load_config(path)
        scl = config["training"]["supervised_contrastive"]
        candidates[path] = (config, scl)
        for seed in seeds:
            rows.append({"config": path, "weight": float(scl["weight"]), "temperature": float(scl["temperature"]),
                         **evaluate_dev_run(config, seed, tokenizer, None)})
        print(f"Evaluated {path}")

    def mean(path: str, key: str) -> float:
        return float(np.mean([row[key] for row in rows if row["config"] == path]))

    reference_f1 = mean(dev["reference"], "val_macro_f1")
    k_means = {k: mean(dev["reference"], f"val_auroc_knn_k{k}") for k in k_grid}
    knn_k = min(k for k, value in k_means.items() if value == max(k_means.values()))
    summary = []
    for path, (_, scl) in candidates.items():
        summary.append({"config": path, "weight": float(scl["weight"]), "temperature": float(scl["temperature"]),
                        "mean_val_macro_f1": mean(path, "val_macro_f1"),
                        "mean_val_auroc_mahalanobis": mean(path, "val_auroc_mahalanobis"),
                        "eligible": mean(path, "val_macro_f1") >= reference_f1 - float(dev["f1_tolerance"])})
    eligible = [item for item in summary if item["eligible"]]
    output = results_dir(study, "dev")
    write_csv(output / "dev_runs.csv", rows)
    write_csv(output / "dev_candidates.csv", summary)
    record = {
        "created_at": utc_now(), "split": "validation only",
        "rule": "max mean val Mahalanobis AUROC among candidates with mean val macro-F1 >= reference - tolerance; "
                "ties: smaller weight, then smaller temperature",
        "reference": {"config": dev["reference"], "mean_val_macro_f1": reference_f1,
                      "mean_val_auroc_mahalanobis": mean(dev["reference"], "val_auroc_mahalanobis")},
        "candidates": summary, "knn": {"k_grid": k_grid, "mean_val_auroc": k_means, "selected_k": knn_k},
        "selected": None,
    }
    if eligible:
        best = max(item["mean_val_auroc_mahalanobis"] for item in eligible)
        chosen = min((item for item in eligible if item["mean_val_auroc_mahalanobis"] == best),
                     key=lambda item: (item["weight"], item["temperature"]))
        scl = candidates[chosen["config"]][1]
        generated = resolve_path("phase3_scl/01_configs/generated")
        hashes = {"arm_scl_selected.yaml": materialise(
            candidates[chosen["config"]][0], "phase3_scl/08_runs/scl_selected", scl, generated / "arm_scl_selected.yaml",
            f"Arm B (controlled improvement), generated by 02_select_scl.py from {chosen['config']}. Do not edit.")}
        for size in study["split"]["low_resource_sizes"]:
            name = f"arm_scl_selected_n{size}.yaml"
            hashes[name] = materialise(
                load_config(f"phase3_scl/01_configs/arm_ce_balanced_n{size}.yaml"), f"phase3_scl/08_runs/scl_selected_n{size}",
                scl, generated / name, f"R2 low-resource ({size}/intent) arm B, generated by 02_select_scl.py. Do not edit.")
        record["selected"] = {**chosen, "generated_configs": hashes}
    write_json(output / "selection.json", record)
    plot_grid(summary, output / "dev_grid.png")
    print(f"Reference val macro-F1 {reference_f1:.4f}; KNN k={knn_k}")
    print(f"Selected: {record['selected']['config'] if record['selected'] else 'none (H7/H8 not run)'}")


def plot_grid(summary: list[dict], path: Path) -> None:
    """Heatmap of mean val Mahalanobis AUROC over the weight x temperature grid (ablation 2)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weights = sorted({item["weight"] for item in summary})
    temperatures = sorted({item["temperature"] for item in summary})
    grid = np.full((len(weights), len(temperatures)), np.nan)
    for item in summary:
        grid[weights.index(item["weight"]), temperatures.index(item["temperature"])] = item["mean_val_auroc_mahalanobis"]
    fig, axis = plt.subplots(figsize=(4, 3.2))
    image = axis.imshow(grid, cmap="viridis")
    axis.set_xticks(range(len(temperatures)), [f"tau={t:g}" for t in temperatures])
    axis.set_yticks(range(len(weights)), [f"w={w:g}" for w in weights])
    for (i, j), value in np.ndenumerate(grid):
        axis.text(j, i, f"{value:.4f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, label="mean val AUROC (Mahalanobis)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
