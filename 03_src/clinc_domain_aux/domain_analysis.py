"""M10: Lambda sensitivity, domain error analyses, cases, and OOS destinations."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from .common import write_csv, write_json
from .evaluation import classification_metrics, per_domain_intent_f1


Rows = Sequence[Mapping[str, Any]]


def _result(files: Mapping[str, str | os.PathLike[str]], summary: Mapping[str, Any]) -> dict[str, Any]:
    return {"files": {name: str(path) for name, path in files.items()}, "summary": dict(summary)}


def _figure(path: str | os.PathLike[str]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(target, dpi=180, bbox_inches="tight")
    plt.close()
    return str(target)


def _seed(value: Any) -> int:
    return int(value)


def _metric_value(row: Mapping[str, Any], metric: str) -> float:
    if metric in row:
        value = row[metric]
    elif metric == "val_macro_f1" and "best_val_macro_f1" in row:
        value = row["best_val_macro_f1"]
    elif metric == "val_macro_f1" and "macro_f1" in row:
        value = row["macro_f1"]
    else:
        raise ValueError(f"validation row lacks {metric!r}")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{metric} must be finite")
    return number


def lambda_sensitivity_curve(
    validation_rows: Rows,
    selected_lambda: float,
    tables_dir: str | os.PathLike[str],
    figures_dir: str | os.PathLike[str],
    *,
    metric: str = "val_macro_f1",
    seeds: Sequence[int] = (0, 1, 2),
) -> dict[str, Any]:
    """Write per-seed validation values and plot means, CE, and selected lambda."""
    expected_seeds = tuple(int(seed) for seed in seeds)
    if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("seeds must be non-empty and unique")
    values: dict[float, dict[int, float]] = {}
    for row in validation_rows:
        lambda_value = float(row["lambda_domain"] if "lambda_domain" in row else row["lambda"])
        seed = _seed(row["seed"])
        if seed not in expected_seeds:
            continue
        per_seed = values.setdefault(lambda_value, {})
        if seed in per_seed:
            raise ValueError(f"duplicate validation row for lambda={lambda_value}, seed={seed}")
        per_seed[seed] = _metric_value(row, metric)
    if not values:
        raise ValueError("no validation rows match the requested seeds")
    for lambda_value, per_seed in values.items():
        if set(per_seed) != set(expected_seeds):
            raise ValueError(f"lambda={lambda_value} does not contain exactly the requested seeds")
    if 0.0 not in values:
        raise ValueError("lambda sensitivity requires the lambda=0 CE reference")
    selected = float(selected_lambda)
    if selected not in values:
        raise ValueError("selected_lambda has no validation values")

    ordered_lambdas = sorted(values)
    table_rows = []
    for lambda_value in ordered_lambdas:
        for seed in expected_seeds:
            table_rows.append({
                "lambda_domain": lambda_value,
                "seed": seed,
                "metric": metric,
                "value": values[lambda_value][seed],
                "is_ce_reference": lambda_value == 0.0,
                "is_selected": lambda_value == selected,
            })
        seed_values = np.asarray([values[lambda_value][seed] for seed in expected_seeds], dtype=np.float64)
        table_rows.append({
            "lambda_domain": lambda_value,
            "seed": "mean",
            "metric": metric,
            "value": float(seed_values.mean()),
            "is_ce_reference": lambda_value == 0.0,
            "is_selected": lambda_value == selected,
        })
    table_path = Path(tables_dir) / "lambda_sensitivity.csv"
    write_csv(table_path, table_rows)

    figure_path = Path(figures_dir) / "lambda_sensitivity.png"
    fig, axis = plt.subplots(figsize=(8, 5))
    offsets = np.linspace(-0.012, 0.012, len(expected_seeds))
    for index, seed in enumerate(expected_seeds):
        x = [lambda_value + offsets[index] for lambda_value in ordered_lambdas]
        y = [values[lambda_value][seed] for lambda_value in ordered_lambdas]
        axis.scatter(x, y, s=34, alpha=0.8, label=f"seed {seed}", zorder=3)
    means = [float(np.mean([values[value][seed] for seed in expected_seeds])) for value in ordered_lambdas]
    axis.plot(ordered_lambdas, means, color="black", marker="o", linewidth=2, label="three-seed mean")
    ce_mean = float(np.mean([values[0.0][seed] for seed in expected_seeds]))
    axis.axhline(ce_mean, color="tab:blue", linestyle="--", linewidth=1.4, label="CE mean")
    axis.axvline(selected, color="tab:red", linestyle=":", linewidth=1.8, label=f"selected λ={selected:g}")
    axis.set_xlabel("Domain loss weight λ")
    axis.set_ylabel(metric)
    axis.set_xticks(ordered_lambdas)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    saved_figure = _figure(figure_path)
    return _result(
        {"lambda_sensitivity_table": table_path, "lambda_sensitivity_figure": saved_figure},
        {
            "metric": metric,
            "seeds": list(expected_seeds),
            "selected_lambda": selected,
            "ce_mean": ce_mean,
            "lambda_means": dict(zip(ordered_lambdas, means)),
        },
    )


def _reverse_mapping(mapping: Mapping[str, Any], name: str) -> dict[int, str]:
    if not mapping:
        raise ValueError(f"{name} must not be empty")
    result: dict[int, str] = {}
    for label, value in mapping.items():
        identifier = int(value)
        if identifier < 0 or identifier in result:
            raise ValueError(f"{name} IDs must be non-negative and unique")
        result[identifier] = str(label)
    if set(result) != set(range(len(result))):
        raise ValueError(f"{name} IDs must be contiguous from zero")
    return result


def _analysis_mappings(
    label_to_id: Mapping[str, Any],
    domain_to_id: Mapping[str, Any],
    test_rows: Rows,
) -> tuple[dict[int, str], dict[int, str], np.ndarray, dict[str, str]]:
    id_to_intent = _reverse_mapping(label_to_id, "label_to_id")
    id_to_domain = _reverse_mapping(domain_to_id, "domain_to_id")
    intent_to_domain_name: dict[str, str] = {}
    texts: dict[str, str] = {}
    for row in test_rows:
        sample_id = str(row["sample_id"])
        if sample_id in texts:
            raise ValueError(f"duplicate test sample_id: {sample_id}")
        texts[sample_id] = str(row["text"])
        intent = row.get("intent")
        domain = row.get("domain")
        if intent is None or domain is None:
            continue
        if str(intent) not in label_to_id or str(domain) not in domain_to_id:
            raise ValueError("test rows contain an intent or domain absent from the mappings")
        previous = intent_to_domain_name.get(str(intent))
        if previous is not None and previous != str(domain):
            raise ValueError(f"inconsistent domain mapping for intent {intent!r}")
        intent_to_domain_name[str(intent)] = str(domain)
    if set(intent_to_domain_name) != set(label_to_id):
        raise ValueError("test_rows must define exactly one domain for every mapped intent")
    intent_to_domain = np.empty(len(id_to_intent), dtype=np.int64)
    for intent, intent_id in label_to_id.items():
        intent_to_domain[int(intent_id)] = int(domain_to_id[intent_to_domain_name[str(intent)]])
    return id_to_intent, id_to_domain, intent_to_domain, texts


def _is_rows(rows: Rows, split: str) -> list[Mapping[str, Any]]:
    selected = [row for row in rows if "split" not in row or str(row["split"]) == split]
    if not selected:
        raise ValueError(f"no IS prediction rows for split {split!r}")
    if any(row.get("gold_intent") is None for row in selected):
        raise ValueError("domain analysis requires in-scope gold labels")
    return selected


def _grouped(rows: Rows, split: str) -> dict[int, list[Mapping[str, Any]]]:
    groups: dict[int, list[Mapping[str, Any]]] = {}
    for row in _is_rows(rows, split):
        groups.setdefault(_seed(row.get("seed", 0)), []).append(row)
    for seed, group in groups.items():
        identifiers = [str(row["sample_id"]) for row in group]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"duplicate sample_id within seed {seed}")
    return groups


def _aligned_predictions(
    ce_rows: Rows,
    dom_rows: Rows,
    num_intents: int,
    intent_to_domain: np.ndarray,
    split: str,
) -> dict[int, list[tuple[Mapping[str, Any], Mapping[str, Any]]]]:
    ce_groups = _grouped(ce_rows, split)
    dom_groups = _grouped(dom_rows, split)
    if set(ce_groups) != set(dom_groups):
        raise ValueError("CE and Dom seed sets differ")
    aligned: dict[int, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for seed in sorted(ce_groups):
        dom_by_id = {str(row["sample_id"]): row for row in dom_groups[seed]}
        ce_ids = {str(row["sample_id"]) for row in ce_groups[seed]}
        if ce_ids != set(dom_by_id):
            raise ValueError(f"CE and Dom test sample sets differ for seed {seed}")
        pairs = [(ce_row, dom_by_id[str(ce_row["sample_id"])]) for ce_row in ce_groups[seed]]
        for ce_row, dom_row in pairs:
            gold = int(ce_row["gold_intent"])
            pred_ce = int(ce_row["pred_intent"])
            pred_dom = int(dom_row["pred_intent"])
            if int(dom_row["gold_intent"]) != gold:
                raise ValueError("CE and Dom gold intent labels differ")
            if not all(0 <= value < num_intents for value in (gold, pred_ce, pred_dom)):
                raise ValueError("intent IDs are outside the label mapping")
            expected_gold_domain = int(intent_to_domain[gold])
            if int(ce_row.get("gold_domain", expected_gold_domain)) != expected_gold_domain:
                raise ValueError("gold_domain disagrees with the intent-domain mapping")
            if int(dom_row.get("gold_domain", expected_gold_domain)) != expected_gold_domain:
                raise ValueError("CE and Dom gold domain labels differ")
            for row, pred in ((ce_row, pred_ce), (dom_row, pred_dom)):
                expected = int(intent_to_domain[pred])
                if int(row.get("pred_domain_from_intent", expected)) != expected:
                    raise ValueError("pred_domain_from_intent disagrees with the mapping")
        aligned[seed] = pairs
    return aligned


def _arrays(pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]], model: str) -> tuple[np.ndarray, np.ndarray]:
    index = 0 if model == "ce" else 1
    return (
        np.asarray([int(pair[0]["gold_intent"]) for pair in pairs], dtype=np.int64),
        np.asarray([int(pair[index]["pred_intent"]) for pair in pairs], dtype=np.int64),
    )


def _confusion(gold: np.ndarray, pred: np.ndarray, num_domains: int) -> np.ndarray:
    return np.bincount(gold * num_domains + pred, minlength=num_domains**2).reshape(num_domains, num_domains).astype(np.float64)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=np.float64), where=totals > 0)


def _mean_seed_matrices(matrices: Mapping[int, np.ndarray]) -> np.ndarray:
    return np.mean(np.stack([matrices[seed] for seed in sorted(matrices)]), axis=0)


def _heatmap(
    matrix: np.ndarray,
    labels: Sequence[str],
    path: Path,
    title: str,
    *,
    difference: bool = False,
    normalized: bool = False,
) -> str:
    size = max(7.0, len(labels) * 0.72)
    fig, axis = plt.subplots(figsize=(size + 1.5, size))
    if difference:
        limit = float(np.max(np.abs(matrix)))
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-limit if limit else -1.0, vmax=limit if limit else 1.0)
    else:
        image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0 if normalized else None)
    axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels=labels)
    axis.set_xlabel("Predicted domain")
    axis.set_ylabel("Gold domain")
    axis.set_title(title)
    threshold = float(matrix.min() + (matrix.max() - matrix.min()) / 2) if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text = f"{value:.3f}" if normalized or difference else f"{value:.1f}"
            axis.text(column, row, text, ha="center", va="center", fontsize=7, color="white" if value > threshold else "black")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _figure(path)


def _domain_f1(confusion: np.ndarray) -> np.ndarray:
    true_positive = np.diag(confusion)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    return np.divide(2 * true_positive, denominator, out=np.zeros_like(true_positive), where=denominator > 0)


def _mean_rows(rows: list[dict[str, Any]], keys: Sequence[str], value_fields: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    result = list(rows)
    for group_key, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        mean_row = dict(zip(keys, group_key))
        mean_row["seed"] = "mean"
        for field in value_fields:
            mean_row[field] = float(np.mean([float(row[field]) for row in group]))
        for field in group[0]:
            if field not in mean_row and field != "seed":
                mean_row[field] = group[0][field]
        result.append(mean_row)
    return result


def _plot_domain_metric(rows: list[dict[str, Any]], value: str, path: Path, ylabel: str) -> str:
    domains = list(dict.fromkeys(row["domain"] for row in rows if row["seed"] == "mean" and row["model"] == "ce"))
    fig, axis = plt.subplots(figsize=(max(9, len(domains)), 5))
    x = np.arange(len(domains))
    for offset, model, color in ((-0.18, "ce", "tab:blue"), (0.18, "dom", "tab:orange")):
        lookup = {row["domain"]: float(row[value]) for row in rows if row["seed"] == "mean" and row["model"] == model}
        axis.bar(x + offset, [lookup[domain] for domain in domains], width=0.36, label=model.upper(), color=color)
    axis.set_xticks(x, domains, rotation=35, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel(ylabel)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _figure(path)


def _top_confusions(
    aligned: Mapping[int, Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]],
    id_to_intent: Mapping[int, str],
    id_to_domain: Mapping[int, str],
    intent_to_domain: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    seeds = sorted(aligned)
    counts: dict[str, dict[int, dict[tuple[int, int], int]]] = {"ce": {}, "dom": {}}
    for model in counts:
        for seed in seeds:
            gold, pred = _arrays(aligned[seed], model)
            counts[model][seed] = {}
            for gold_id, pred_id in zip(gold, pred):
                if gold_id != pred_id:
                    pair = (int(gold_id), int(pred_id))
                    counts[model][seed][pair] = counts[model][seed].get(pair, 0) + 1
    pairs = set().union(*(set(counts["ce"][seed]) for seed in seeds))
    ordered = sorted(
        pairs,
        key=lambda pair: (
            -float(np.mean([counts["ce"][seed].get(pair, 0) for seed in seeds])),
            id_to_intent[pair[0]],
            id_to_intent[pair[1]],
        ),
    )[:top_k]
    result = []
    for rank, (gold, pred) in enumerate(ordered, start=1):
        row: dict[str, Any] = {
            "rank": rank,
            "gold_intent_id": gold,
            "gold_intent": id_to_intent[gold],
            "pred_intent_id": pred,
            "pred_intent": id_to_intent[pred],
            "gold_domain": id_to_domain[int(intent_to_domain[gold])],
            "pred_domain": id_to_domain[int(intent_to_domain[pred])],
            "same_domain": bool(intent_to_domain[gold] == intent_to_domain[pred]),
        }
        for model in ("ce", "dom"):
            for seed in seeds:
                row[f"{model}_seed{seed}_errors"] = counts[model][seed].get((gold, pred), 0)
            row[f"{model}_mean_errors"] = float(np.mean([counts[model][seed].get((gold, pred), 0) for seed in seeds]))
        result.append(row)
    return result


def _plot_top_confusions(rows: list[dict[str, Any]], path: Path) -> str:
    fig, axis = plt.subplots(figsize=(10, max(4, 0.42 * max(len(rows), 1))))
    if rows:
        labels = [f"{row['gold_intent']} → {row['pred_intent']}" for row in rows]
        y = np.arange(len(rows))
        axis.barh(y - 0.18, [row["ce_mean_errors"] for row in rows], height=0.36, label="CE")
        axis.barh(y + 0.18, [row["dom_mean_errors"] for row in rows], height=0.36, label="Dom")
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
    else:
        axis.text(0.5, 0.5, "No CE intent confusions", ha="center", va="center")
        axis.set_axis_off()
    axis.set_xlabel("Three-seed mean error count")
    axis.legend()
    fig.tight_layout()
    return _figure(path)


def sample_error_cases(
    ce_rows: Rows,
    dom_rows: Rows,
    label_to_id: Mapping[str, Any],
    domain_to_id: Mapping[str, Any],
    test_rows: Rows,
    cases_dir: str | os.PathLike[str],
    *,
    analysis_seed: int = 20261024,
    cases_per_category: int = 5,
    split: str = "test",
) -> dict[str, Any]:
    """Sample seed-0 repairs and regressions without generating interpretations."""
    if cases_per_category < 0:
        raise ValueError("cases_per_category must be non-negative")
    id_to_intent, id_to_domain, intent_to_domain, texts = _analysis_mappings(label_to_id, domain_to_id, test_rows)
    aligned = _aligned_predictions(ce_rows, dom_rows, len(id_to_intent), intent_to_domain, split)
    if 0 not in aligned:
        raise ValueError("error-case analysis requires model seed 0")
    pools: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {"ce_error_dom_correct": [], "ce_correct_dom_error": []}
    for pair in aligned[0]:
        ce, dom = pair
        gold = int(ce["gold_intent"])
        ce_correct = int(ce["pred_intent"]) == gold
        dom_correct = int(dom["pred_intent"]) == gold
        if not ce_correct and dom_correct:
            pools["ce_error_dom_correct"].append(pair)
        elif ce_correct and not dom_correct:
            pools["ce_correct_dom_error"].append(pair)
    rng = np.random.default_rng(analysis_seed)
    cases: list[dict[str, Any]] = []
    pool_sizes = {}
    for category, pool in pools.items():
        pool_sizes[category] = len(pool)
        count = min(cases_per_category, len(pool))
        selected = rng.permutation(len(pool))[:count] if count < len(pool) else np.arange(len(pool))
        for index in selected:
            ce, dom = pool[int(index)]
            sample_id = str(ce["sample_id"])
            if sample_id not in texts:
                raise ValueError(f"test_rows lacks text for {sample_id}")
            gold = int(ce["gold_intent"])
            ce_pred = int(ce["pred_intent"])
            dom_pred = int(dom["pred_intent"])
            cases.append({
                "category": category,
                "model_seed": 0,
                "analysis_seed": int(analysis_seed),
                "sample_id": sample_id,
                "text": texts[sample_id],
                "gold_intent_id": gold,
                "gold_intent": id_to_intent[gold],
                "gold_domain_id": int(intent_to_domain[gold]),
                "gold_domain": id_to_domain[int(intent_to_domain[gold])],
                "ce_pred_intent_id": ce_pred,
                "ce_pred_intent": id_to_intent[ce_pred],
                "ce_pred_domain_id": int(intent_to_domain[ce_pred]),
                "ce_pred_domain": id_to_domain[int(intent_to_domain[ce_pred])],
                "ce_msp_score": float(ce["msp_score"]),
                "dom_pred_intent_id": dom_pred,
                "dom_pred_intent": id_to_intent[dom_pred],
                "dom_pred_domain_id": int(intent_to_domain[dom_pred]),
                "dom_pred_domain": id_to_domain[int(intent_to_domain[dom_pred])],
                "dom_msp_score": float(dom["msp_score"]),
                "interpretation": "",
            })
    payload = {
        "model_seed": 0,
        "analysis_seed": int(analysis_seed),
        "cases_per_category": int(cases_per_category),
        "pool_sizes": pool_sizes,
        "sampling": "uniform without replacement within each seed-0 pool; all rows when pool is smaller",
        "interpretation_status": "blank for manual analysis",
        "cases": cases,
    }
    json_path = Path(cases_dir) / "error_cases.json"
    csv_path = Path(cases_dir) / "error_cases.csv"
    write_json(json_path, payload)
    fieldnames = list(cases[0]) if cases else [
        "category", "model_seed", "analysis_seed", "sample_id", "text", "gold_intent", "gold_domain",
        "ce_pred_intent", "ce_pred_domain", "ce_msp_score", "dom_pred_intent", "dom_pred_domain",
        "dom_msp_score", "interpretation",
    ]
    write_csv(csv_path, cases, fieldnames=fieldnames)
    return _result(
        {"error_cases_json": json_path, "error_cases_table": csv_path},
        {"model_seed": 0, "analysis_seed": int(analysis_seed), "pool_sizes": pool_sizes, "n_cases": len(cases)},
    )


def analyze_domain_predictions(
    ce_rows: Rows,
    dom_rows: Rows,
    label_to_id: Mapping[str, Any],
    domain_to_id: Mapping[str, Any],
    test_rows: Rows,
    tables_dir: str | os.PathLike[str],
    figures_dir: str | os.PathLike[str],
    cases_dir: str | os.PathLike[str],
    *,
    analysis_seed: int = 20261024,
    cases_per_category: int = 5,
    top_k: int = 15,
    split: str = "test",
) -> dict[str, Any]:
    """Generate all IS E7 tables, figures, and seed-0 comparison cases."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    id_to_intent, id_to_domain, intent_to_domain, _ = _analysis_mappings(label_to_id, domain_to_id, test_rows)
    aligned = _aligned_predictions(ce_rows, dom_rows, len(id_to_intent), intent_to_domain, split)
    seeds = sorted(aligned)
    num_domains = len(id_to_domain)
    domain_names = [id_to_domain[index] for index in range(num_domains)]
    matrices: dict[str, dict[int, np.ndarray]] = {"ce": {}, "dom": {}}
    outcome_rows: list[dict[str, Any]] = []
    intent_f1_rows: list[dict[str, Any]] = []
    domain_f1_rows: list[dict[str, Any]] = []
    for model in ("ce", "dom"):
        for seed in seeds:
            gold, pred = _arrays(aligned[seed], model)
            gold_domain = intent_to_domain[gold]
            pred_domain = intent_to_domain[pred]
            matrix = _confusion(gold_domain, pred_domain, num_domains)
            matrices[model][seed] = matrix
            metrics = classification_metrics(gold, pred, intent_to_domain, num_intents=len(id_to_intent))
            for outcome, field in (
                ("correct", "accuracy"),
                ("within_domain_error", "within_domain_error_rate"),
                ("cross_domain_error", "cross_domain_error_rate"),
            ):
                outcome_rows.append({
                    "model": model,
                    "seed": seed,
                    "outcome": outcome,
                    "rate": metrics[field],
                    "count": int(round(metrics[field] * len(gold))),
                    "n_samples": len(gold),
                })
            per_domain = per_domain_intent_f1(gold, pred, intent_to_domain, num_intents=len(id_to_intent))
            domain_scores = _domain_f1(matrix)
            for domain in range(num_domains):
                intent_f1_rows.append({
                    "model": model,
                    "seed": seed,
                    "domain_id": domain,
                    "domain": id_to_domain[domain],
                    "n_intents": int(np.sum(intent_to_domain == domain)),
                    "intent_macro_f1": per_domain[domain],
                })
                domain_f1_rows.append({
                    "model": model,
                    "seed": seed,
                    "domain_id": domain,
                    "domain": id_to_domain[domain],
                    "domain_f1": float(domain_scores[domain]),
                })

    files: dict[str, str] = {}
    confusion_counts: list[dict[str, Any]] = []
    confusion_normalized: list[dict[str, Any]] = []
    confusion_differences: list[dict[str, Any]] = []
    means = {model: _mean_seed_matrices(matrices[model]) for model in ("ce", "dom")}
    for model in ("ce", "dom"):
        for seed in [*seeds, "mean"]:
            matrix = means[model] if seed == "mean" else matrices[model][seed]
            normalized = _normalize_rows(matrix)
            for gold_domain in range(num_domains):
                for pred_domain in range(num_domains):
                    base = {
                        "model": model,
                        "seed": seed,
                        "gold_domain_id": gold_domain,
                        "gold_domain": id_to_domain[gold_domain],
                        "pred_domain_id": pred_domain,
                        "pred_domain": id_to_domain[pred_domain],
                    }
                    confusion_counts.append({**base, "count": float(matrix[gold_domain, pred_domain])})
                    confusion_normalized.append({**base, "proportion": float(normalized[gold_domain, pred_domain])})
    for seed in [*seeds, "mean"]:
        ce_matrix = means["ce"] if seed == "mean" else matrices["ce"][seed]
        dom_matrix = means["dom"] if seed == "mean" else matrices["dom"][seed]
        difference = dom_matrix - ce_matrix
        normalized_difference = _normalize_rows(dom_matrix) - _normalize_rows(ce_matrix)
        for gold_domain in range(num_domains):
            for pred_domain in range(num_domains):
                confusion_differences.append({
                    "seed": seed,
                    "gold_domain_id": gold_domain,
                    "gold_domain": id_to_domain[gold_domain],
                    "pred_domain_id": pred_domain,
                    "pred_domain": id_to_domain[pred_domain],
                    "count_dom_minus_ce": float(difference[gold_domain, pred_domain]),
                    "row_normalized_dom_minus_ce": float(normalized_difference[gold_domain, pred_domain]),
                })
    for name, rows in (
        ("domain_confusion_counts", confusion_counts),
        ("domain_confusion_row_normalized", confusion_normalized),
        ("domain_confusion_dom_minus_ce", confusion_differences),
    ):
        path = Path(tables_dir) / f"{name}.csv"
        write_csv(path, rows)
        files[name] = str(path)
    for model in ("ce", "dom"):
        for normalized in (False, True):
            matrix = _normalize_rows(means[model]) if normalized else means[model]
            name = f"domain_confusion_{model}_{'row_normalized' if normalized else 'counts'}"
            files[name] = _heatmap(
                matrix,
                domain_names,
                Path(figures_dir) / f"{name}.png",
                f"{model.upper()} three-seed mean ({'row-normalized' if normalized else 'counts'})",
                normalized=normalized,
            )
    for normalized in (False, True):
        ce = _normalize_rows(means["ce"]) if normalized else means["ce"]
        dom = _normalize_rows(means["dom"]) if normalized else means["dom"]
        name = f"domain_confusion_dom_minus_ce{'_row_normalized' if normalized else '_counts'}"
        files[name] = _heatmap(
            dom - ce,
            domain_names,
            Path(figures_dir) / f"{name}.png",
            f"Dom − CE three-seed mean ({'row-normalized' if normalized else 'counts'})",
            difference=True,
            normalized=normalized,
        )

    outcome_summary = _mean_rows(outcome_rows, ["model", "outcome"], ["rate", "count", "n_samples"])
    intent_f1_summary = _mean_rows(intent_f1_rows, ["model", "domain_id", "domain"], ["intent_macro_f1", "n_intents"])
    domain_f1_summary = _mean_rows(domain_f1_rows, ["model", "domain_id", "domain"], ["domain_f1"])
    for name, rows in (
        ("mutually_exclusive_outcomes", outcome_summary),
        ("per_domain_intent_f1", intent_f1_summary),
        ("domain_f1", domain_f1_summary),
    ):
        path = Path(tables_dir) / f"{name}.csv"
        write_csv(path, rows)
        files[name] = str(path)

    fig, axis = plt.subplots(figsize=(7, 5))
    x = np.arange(2)
    bottoms = np.zeros(2)
    for outcome, color in (("correct", "tab:green"), ("within_domain_error", "tab:orange"), ("cross_domain_error", "tab:red")):
        rates = []
        for model in ("ce", "dom"):
            values = [float(row["rate"]) for row in outcome_summary if row["seed"] == "mean" and row["model"] == model and row["outcome"] == outcome]
            rates.append(values[0])
        axis.bar(x, np.asarray(rates) * 100, bottom=bottoms * 100, label=outcome, color=color)
        bottoms += np.asarray(rates)
    axis.set_xticks(x, ["CE", "Dom"])
    axis.set_ylim(0, 100)
    axis.set_ylabel("Percentage of all IS test samples")
    axis.legend()
    fig.tight_layout()
    files["mutually_exclusive_outcomes_figure"] = _figure(Path(figures_dir) / "mutually_exclusive_outcomes.png")
    files["per_domain_intent_f1_figure"] = _plot_domain_metric(
        intent_f1_summary, "intent_macro_f1", Path(figures_dir) / "per_domain_intent_f1.png", "Within-domain 15-intent macro-F1"
    )
    files["domain_f1_figure"] = _plot_domain_metric(
        domain_f1_summary, "domain_f1", Path(figures_dir) / "domain_f1.png", "Domain-class F1"
    )

    top_rows = _top_confusions(aligned, id_to_intent, id_to_domain, intent_to_domain, top_k)
    top_path = Path(tables_dir) / "top_intent_confusions.csv"
    top_fields = list(top_rows[0]) if top_rows else [
        "rank", "gold_intent_id", "gold_intent", "pred_intent_id", "pred_intent", "gold_domain", "pred_domain",
        "same_domain", *[f"ce_seed{seed}_errors" for seed in seeds], "ce_mean_errors",
        *[f"dom_seed{seed}_errors" for seed in seeds], "dom_mean_errors",
    ]
    write_csv(top_path, top_rows, fieldnames=top_fields)
    files["top_intent_confusions"] = str(top_path)
    files["top_intent_confusions_figure"] = _plot_top_confusions(top_rows, Path(figures_dir) / "top_intent_confusions.png")

    cases = sample_error_cases(
        ce_rows, dom_rows, label_to_id, domain_to_id, test_rows, cases_dir,
        analysis_seed=analysis_seed, cases_per_category=cases_per_category, split=split,
    )
    files.update(cases["files"])
    return _result(files, {
        "seeds": seeds,
        "n_samples_per_seed": {seed: len(aligned[seed]) for seed in seeds},
        "n_domains": num_domains,
        "n_intents": len(id_to_intent),
        "top_confusions": len(top_rows),
        "top_order": "CE three-seed mean errors, descending; Dom reuses this order",
        "outcome_denominator": "all IS test samples",
        "cases": cases["summary"],
    })


def analyze_oos_false_acceptance(
    prediction_rows: Rows,
    label_to_id: Mapping[str, Any],
    domain_to_id: Mapping[str, Any],
    tables_dir: str | os.PathLike[str],
    figures_dir: str | os.PathLike[str] | None = None,
    *,
    split: str = "oos_test",
) -> dict[str, Any]:
    """Count accepted OOS predictions by domain for both frozen detectors."""
    id_to_intent = _reverse_mapping(label_to_id, "label_to_id")
    id_to_domain = _reverse_mapping(domain_to_id, "domain_to_id")
    rows = [row for row in prediction_rows if "split" not in row or str(row["split"]) == split]
    if not rows:
        raise ValueError(f"no OOS prediction rows for split {split!r}")
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        pred = int(row["pred_intent"])
        if pred not in id_to_intent:
            raise ValueError("OOS pred_intent is outside the label mapping")
        groups.setdefault((str(row["model"]), _seed(row.get("seed", 0))), []).append(row)
    output_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for (model, seed), group in sorted(groups.items()):
        total = len(group)
        for detector in ("msp", "mahalanobis"):
            field = f"accepted_{detector}"
            accepted = [row for row in group if bool(row[field])]
            n_accepted = len(accepted)
            summary_rows.append({
                "model": model,
                "seed": seed,
                "detector": detector,
                "n_all_oos": total,
                "n_false_accepted_oos": n_accepted,
                "oos_false_acceptance_rate": n_accepted / total,
            })
            for domain in range(len(id_to_domain)):
                count = sum(int(row.get("pred_domain_from_intent", -1)) == domain for row in accepted)
                output_rows.append({
                    "model": model,
                    "seed": seed,
                    "detector": detector,
                    "domain_id": domain,
                    "domain": id_to_domain[domain],
                    "count": count,
                    "proportion_of_all_oos": count / total,
                    "proportion_of_false_accepted_oos": count / n_accepted if n_accepted else 0.0,
                    "n_all_oos": total,
                    "n_false_accepted_oos": n_accepted,
                })
    files: dict[str, str] = {}
    for name, output in (("oos_false_acceptance_destinations", output_rows), ("oos_false_acceptance_summary", summary_rows)):
        path = Path(tables_dir) / f"{name}.csv"
        write_csv(path, output)
        files[name] = str(path)
    if figures_dir is not None:
        destinations_mean = _mean_rows(
            output_rows,
            ["model", "detector", "domain_id", "domain"],
            ["count", "proportion_of_all_oos", "proportion_of_false_accepted_oos", "n_all_oos", "n_false_accepted_oos"],
        )
        combos = list(dict.fromkeys((row["model"], row["detector"]) for row in summary_rows))
        fig, axes = plt.subplots(len(combos), 1, figsize=(11, max(3, 3 * len(combos))), squeeze=False)
        for axis, (model, detector) in zip(axes[:, 0], combos):
            selected = [row for row in destinations_mean if row["seed"] == "mean" and row["model"] == model and row["detector"] == detector]
            axis.bar([row["domain"] for row in selected], [100 * row["proportion_of_false_accepted_oos"] for row in selected])
            axis.set_title(f"{model} / {detector}: accepted OOS destinations (seed mean)")
            axis.set_ylabel("% of accepted OOS")
            axis.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        files["oos_false_acceptance_destinations_figure"] = _figure(Path(figures_dir) / "oos_false_acceptance_destinations.png")
    return _result(files, {
        "n_groups": len(groups),
        "n_all_oos_by_group": {f"{model}:seed{seed}": len(group) for (model, seed), group in groups.items()},
        "detectors": ["msp", "mahalanobis"],
        "proportion_denominators": ["all OOS", "false-accepted OOS"],
    })
