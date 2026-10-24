"""M12: Confirmatory study helpers: run verification, MASSIVE conversion, two-level statistics."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from .common import load_config, read_json, resolve_path, run_name, sha256_file, stable_hash

MASSIVE_SPLITS = {"train": "train", "dev": "val", "test": "test"}


# ---------------------------------------------------------------- study configuration

def require_approval(study: Mapping[str, Any]) -> None:
    """Test-time steps run only on a frozen study with complete supervisor approval."""
    if study.get("role") == "exploratory":
        raise ValueError("Exploratory studies are validation-only and never evaluated on test")
    if not study.get("frozen"):
        raise ValueError("Study configuration is not frozen")
    approval = study.get("supervisor_approval") or {}
    missing = [key for key in ("confirmed_by", "confirmed_at", "evidence") if not approval.get(key)]
    if not approval.get("approved") or missing:
        raise ValueError(f"Test evaluation blocked: supervisor approval incomplete ({', '.join(missing) or 'approved=false'})")
    try:
        datetime.fromisoformat(str(approval["confirmed_at"]).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("supervisor_approval.confirmed_at must be an ISO timestamp") from error


def study_arms(study: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Reference arm first, then treatments; each arm has name and lambda_domain."""
    reference = dict(study["reference"])
    treatments = [dict(item) for item in study["treatments"]]
    if float(reference["lambda_domain"]) != 0.0:
        raise ValueError("The reference arm must be lambda_domain=0 (BERT-CE)")
    if not treatments or any(float(item["lambda_domain"]) <= 0 for item in treatments):
        raise ValueError("Treatments must have positive lambda_domain")
    names = [reference["name"], *[item["name"] for item in treatments]]
    if len(set(names)) != len(names):
        raise ValueError("Arm names must be unique")
    return [reference, *treatments]


def verify_run(base: Mapping[str, Any], lambda_domain: float, seed: int) -> dict[str, Any]:
    """Check a finished formal run and that it used ``base`` unchanged (no --data-root override)."""
    run_dir = resolve_path(base["runs_dir"]) / run_name(lambda_domain, seed)
    completed = read_json(run_dir / "completed.json")
    if completed.get("status") != "completed" or completed.get("smoke", False):
        raise ValueError(f"Missing completed formal run: {run_dir}")
    if float(completed["lambda_domain"]) != float(lambda_domain) or int(completed["seed"]) != int(seed):
        raise ValueError(f"Run identity mismatch: {run_dir}")
    expected = stable_hash({
        "config": dict(base), "lambda_domain": float(lambda_domain), "seed": int(seed), "smoke": False,
    })
    if completed["config_hash"] != expected:
        raise ValueError(f"Run was not trained with the unchanged base config: {run_dir}")
    for name, digest in completed["artifacts"].items():
        if sha256_file(run_dir / name) != digest:
            raise ValueError(f"Artifact changed after training: {run_dir / name}")
    return {**completed, "run_dir": run_dir}


def load_study(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    study = load_config(path)
    base = load_config(study["base_config"])
    if resolve_path(base["runs_dir"]) != resolve_path(study.get("runs_dir", base["runs_dir"])):
        raise ValueError("Study runs_dir must match the base config runs_dir")
    return study, base


# ---------------------------------------------------------------- MASSIVE conversion

def convert_massive(rows: Sequence[Mapping[str, Any]], locale: str = "en-US") -> tuple[dict[str, list], dict[str, list[str]], dict[str, str]]:
    """Map MASSIVE rows to the CLINC data_full/domains layout.

    Scenarios become domains; dev becomes val; OOS splits are empty because
    MASSIVE has no out-of-scope class. Returns (data_full, domains, id_map).
    """
    data: dict[str, list] = {"train": [], "val": [], "test": [], "oos_train": [], "oos_val": [], "oos_test": []}
    id_map: dict[str, str] = {}
    scenario_of: dict[str, str] = {}
    for row in rows:
        if row["locale"] != locale:
            raise ValueError(f"Unexpected locale {row['locale']!r}")
        split = MASSIVE_SPLITS.get(row["partition"])
        if split is None:
            raise ValueError(f"Unknown partition {row['partition']!r}")
        intent, scenario = str(row["intent"]), str(row["scenario"])
        if scenario_of.setdefault(intent, scenario) != scenario:
            raise ValueError(f"Intent {intent!r} appears in more than one scenario")
        id_map[f"{split}:{len(data[split]):05d}"] = str(row["id"])
        data[split].append([str(row["utt"]), intent])
    domains: dict[str, list[str]] = {}
    for intent, scenario in sorted(scenario_of.items()):
        domains.setdefault(scenario, []).append(intent)
    return data, dict(sorted(domains.items())), id_map


# ---------------------------------------------------------------- metrics

def macro_f1_present(gold: np.ndarray, pred: np.ndarray, num_labels: int) -> float:
    """Macro-F1 over labels that occur in ``gold`` (MASSIVE test lacks one intent)."""
    true_positive = np.bincount(gold[gold == pred], minlength=num_labels).astype(np.float64)
    gold_count = np.bincount(gold, minlength=num_labels)
    pred_count = np.bincount(pred, minlength=num_labels)
    present = gold_count > 0
    return float(np.mean(2.0 * true_positive[present] / (gold_count[present] + pred_count[present])))


def cross_domain_error(gold: np.ndarray, pred: np.ndarray, intent_domain: np.ndarray) -> float:
    return float(np.mean(intent_domain[gold] != intent_domain[pred]))


def within_domain_error(gold: np.ndarray, pred: np.ndarray, intent_domain: np.ndarray) -> float:
    return float(np.mean((gold != pred) & (intent_domain[gold] == intent_domain[pred])))


def auroc_oos(is_scores: np.ndarray, oos_scores: np.ndarray) -> float:
    """OOS-positive AUROC via average ranks (ties count one half)."""
    ranks = stats.rankdata(np.concatenate([is_scores, oos_scores]))
    n_is, n_oos = len(is_scores), len(oos_scores)
    return float((ranks[n_is:].sum() - n_oos * (n_oos + 1) / 2.0) / (n_is * n_oos))


def fpr95_is_positive(is_scores: np.ndarray, oos_scores: np.ndarray, target: float = 0.95) -> float:
    """OOS false acceptance at the first operating point accepting >= target of IS (ties grouped).

    Matches ``evaluation.oos_metrics``: accept iff s <= t, t = the ceil(target*n)-th smallest IS score.
    """
    k = math.ceil(target * len(is_scores) - 1e-12)
    threshold = np.partition(is_scores, k - 1)[k - 1]
    return float(np.mean(oos_scores <= threshold))


# ---------------------------------------------------------------- statistics

def stratified_indices(groups: Sequence[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Resample each intent group with replacement, keeping its original size."""
    return np.concatenate([group[rng.integers(0, len(group), size=len(group))] for group in groups])


def intent_groups(gold: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(gold == label) for label in np.unique(gold)]


def wilcoxon_exact(values: Sequence[float]) -> float:
    """Two-sided exact Wilcoxon signed-rank p-value by enumerating all sign flips.

    Zeros are dropped and tied |d| get average ranks; independent of SciPy's version-specific
    choice between exact and normal-approximation methods.
    """
    nonzero = np.asarray(values, dtype=np.float64)
    nonzero = nonzero[nonzero != 0]
    m = len(nonzero)
    if m == 0:
        return 1.0
    if m > 20:
        raise ValueError("Exact enumeration supports at most 20 non-zero paired differences")
    ranks = stats.rankdata(np.abs(nonzero))
    observed = ranks[nonzero > 0].sum()
    signs = (np.arange(2 ** m)[:, None] >> np.arange(m)[None, :]) & 1
    null = signs @ ranks
    center = ranks.sum() / 2.0
    return float(np.mean(np.abs(null - center) >= abs(observed - center) - 1e-9))


def seed_level_tests(deltas: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(deltas, dtype=np.float64)
    if len(values) < 3:
        raise ValueError("At least three paired seeds are required")
    if values.std(ddof=1) == 0:
        t_statistic, t_pvalue = None, None  # undefined for identical deltas; Wilcoxon still decides
    else:
        t_result = stats.ttest_1samp(values, 0.0)
        t_statistic, t_pvalue = float(t_result.statistic), float(t_result.pvalue)
    wilcoxon_p = wilcoxon_exact(values)
    return {
        "n_seeds": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "n_positive": int((values > 0).sum()),
        "t_statistic": t_statistic,
        "t_pvalue_two_sided": t_pvalue,
        "wilcoxon_pvalue_two_sided": wilcoxon_p,
        "wilcoxon_method": "exact sign-flip enumeration; zeros dropped, ties average-ranked",
    }


def two_level_bootstrap(
    gold: np.ndarray,
    reference: np.ndarray,
    treatment: np.ndarray,
    metric,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Resample paired seeds and intent-stratified test rows; return mean(treatment - reference).

    ``reference``/``treatment`` have shape (n_seeds, n_rows); ``metric(gold, pred)`` returns a float.
    Seeds are resampled as (reference, treatment) pairs because they share an initialization.
    """
    if reference.shape != treatment.shape or reference.shape[1] != len(gold):
        raise ValueError("Prediction arrays must be (n_seeds, n_rows) and match gold")
    rng = np.random.default_rng(seed)
    groups = intent_groups(gold)
    n_seeds = reference.shape[0]
    distribution = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        seeds = rng.integers(0, n_seeds, size=n_seeds)
        rows = stratified_indices(groups, rng)
        sampled_gold = gold[rows]
        distribution[b] = np.mean([
            metric(sampled_gold, treatment[s, rows]) - metric(sampled_gold, reference[s, rows]) for s in seeds
        ])
    observed = float(np.mean([metric(gold, treatment[s]) - metric(gold, reference[s]) for s in range(n_seeds)]))
    low, high = np.quantile(distribution, [0.025, 0.975])
    return {"observed": observed, "ci95": [float(low), float(high)], "n_boot": int(n_boot), "seed": int(seed),
            "distribution": distribution}


def two_level_oos_bootstrap(
    gold: np.ndarray,
    reference: tuple[np.ndarray, np.ndarray],
    treatment: tuple[np.ndarray, np.ndarray],
    metric,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Like ``two_level_bootstrap`` for (IS scores, OOS scores) arrays of shape (n_seeds, n)."""
    ref_is, ref_oos = reference
    trt_is, trt_oos = treatment
    rng = np.random.default_rng(seed)
    groups = intent_groups(gold)
    n_seeds, n_oos = ref_is.shape[0], ref_oos.shape[1]
    distribution = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        seeds = rng.integers(0, n_seeds, size=n_seeds)
        rows = stratified_indices(groups, rng)
        oos_rows = rng.integers(0, n_oos, size=n_oos)
        distribution[b] = np.mean([
            metric(trt_is[s, rows], trt_oos[s, oos_rows]) - metric(ref_is[s, rows], ref_oos[s, oos_rows]) for s in seeds
        ])
    observed = float(np.mean([metric(trt_is[s], trt_oos[s]) - metric(ref_is[s], ref_oos[s]) for s in range(n_seeds)]))
    low, high = np.quantile(distribution, [0.025, 0.975])
    return {"observed": observed, "ci95": [float(low), float(high)], "n_boot": int(n_boot), "seed": int(seed),
            "distribution": distribution}


def decide(ci95: Sequence[float], seed_tests: Mapping[str, Any], *, better: str, alpha: float, sesoi: float | None = None) -> dict[str, Any]:
    """Pre-registered decision: both the two-level CI and the seed-level Wilcoxon test must agree.

    ``better`` is "higher" (macro-F1) or "lower" (error rates).
    """
    low, high = ci95
    significant = seed_tests["wilcoxon_pvalue_two_sided"] < alpha
    if better == "higher":
        supported, contradicted = low > 0 and significant, high < 0 and significant
        effect = seed_tests["mean"]
    elif better == "lower":
        supported, contradicted = high < 0 and significant, low > 0 and significant
        effect = -seed_tests["mean"]
    else:
        raise ValueError("better must be 'higher' or 'lower'")
    verdict = "supported" if supported else "contradicted" if contradicted else "inconclusive"
    result = {"verdict": verdict, "alpha": alpha, "criterion": "two-level 95% CI excludes 0 AND seed-level Wilcoxon p < alpha"}
    if sesoi is not None:
        result["sesoi"] = sesoi
        result["meets_sesoi"] = bool(effect >= sesoi)
    return result


def output_root(study: Mapping[str, Any]) -> Path:
    return resolve_path(study["output_dir"])
