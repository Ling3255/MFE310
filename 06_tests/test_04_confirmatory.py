"""T04: Confirmatory-study helpers: MASSIVE conversion, metrics, two-level bootstrap, gates."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.metrics import f1_score

from clinc_domain_aux.common import sha256_file, stable_hash, write_json
from clinc_domain_aux.confirmatory import (
    auroc_oos, convert_massive, decide, fpr95_is_positive, macro_f1_present, require_approval,
    seed_level_tests, stratified_indices, intent_groups, two_level_bootstrap, verify_run,
)
from clinc_domain_aux.data import load_datasets
from clinc_domain_aux.evaluation import oos_metrics


def massive_rows():
    rows = []
    for index, (partition, scenario, intent) in enumerate([
        ("train", "alarm", "alarm_set"), ("train", "alarm", "alarm_query"), ("train", "music", "music_likeness"),
        ("dev", "alarm", "alarm_set"), ("test", "music", "music_likeness"), ("test", "alarm", "alarm_query"),
    ]):
        rows.append({"id": str(index), "locale": "en-US", "partition": partition, "scenario": scenario,
                     "intent": intent, "utt": f"utterance {index}"})
    return rows


def test_convert_massive_maps_dev_to_val_scenarios_to_domains_and_loads(tmp_path):
    data, domains, id_map = convert_massive(massive_rows())
    assert {split: len(rows) for split, rows in data.items()} == {
        "train": 3, "val": 1, "test": 2, "oos_train": 0, "oos_val": 0, "oos_test": 0}
    assert domains == {"alarm": ["alarm_query", "alarm_set"], "music": ["music_likeness"]}
    assert id_map["test:00001"] == "5"
    write_json(tmp_path / "data_full.json", data)
    write_json(tmp_path / "domains.json", domains)
    loaded = load_datasets(tmp_path, strict_counts=False)
    assert [row["domain"] for row in loaded["test"]] == ["music", "alarm"]
    conflicting = massive_rows() + [{**massive_rows()[0], "id": "9", "scenario": "music"}]
    with pytest.raises(ValueError):
        convert_massive(conflicting)


def test_macro_f1_present_ignores_intents_absent_from_gold():
    gold = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 3])  # label 3 never occurs in gold
    expected = f1_score(gold, pred, labels=[0, 1, 2], average="macro")
    assert macro_f1_present(gold, pred, num_labels=5) == pytest.approx(expected)


def test_fast_oos_metrics_match_evaluation_module_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(20):
        is_scores = np.round(rng.normal(0, 1, 137), 1)
        oos_scores = np.round(rng.normal(1, 1, 41), 1)
        reference = oos_metrics(is_scores, oos_scores)
        assert auroc_oos(is_scores, oos_scores) == pytest.approx(reference["auroc_oos"])
        assert fpr95_is_positive(is_scores, oos_scores) == pytest.approx(reference["fpr95_is_positive"])


def test_stratified_resampling_keeps_intent_sizes_and_is_reproducible():
    gold = np.repeat([0, 1, 2], [5, 1, 9])
    groups = intent_groups(gold)
    first = stratified_indices(groups, np.random.default_rng(3))
    second = stratified_indices(groups, np.random.default_rng(3))
    assert np.array_equal(first, second)
    assert np.array_equal(np.bincount(gold[first]), np.bincount(gold))


def test_two_level_bootstrap_zero_for_identical_arms_and_positive_for_better_arm():
    rng = np.random.default_rng(1)
    gold = np.repeat(np.arange(10), 20)
    reference = np.stack([np.where(rng.random(200) < 0.8, gold, (gold + 1) % 10) for _ in range(5)])
    metric = lambda g, p: macro_f1_present(g, p, 10)
    same = two_level_bootstrap(gold, reference, reference.copy(), metric, n_boot=50, seed=7)
    assert same["observed"] == 0 and same["ci95"] == [0.0, 0.0]
    better = reference.copy()
    better[:, ::4] = gold[::4]  # treatment fixes a quarter of rows in every seed
    result = two_level_bootstrap(gold, reference, better, metric, n_boot=200, seed=7)
    assert result["ci95"][0] > 0
    again = two_level_bootstrap(gold, reference, better, metric, n_boot=200, seed=7)
    assert np.array_equal(result["distribution"], again["distribution"])


def test_seed_tests_and_preregistered_decision_rule():
    tests = seed_level_tests([0.003, 0.002, 0.004, 0.001, 0.002, 0.003, 0.002, 0.001, 0.003, 0.002])
    assert tests["n_positive"] == 10 and tests["wilcoxon_pvalue_two_sided"] < 0.01
    assert decide([0.001, 0.003], tests, better="higher", alpha=0.05, sesoi=0.002)["verdict"] == "supported"
    assert decide([-0.001, 0.003], tests, better="higher", alpha=0.05)["verdict"] == "inconclusive"
    assert decide([0.001, 0.003], tests, better="lower", alpha=0.05)["verdict"] == "contradicted"
    flat = seed_level_tests([0.0, 0.0, 0.0])
    assert flat["t_statistic"] is None and flat["wilcoxon_pvalue_two_sided"] == 1.0


def test_require_approval_blocks_exploratory_and_incomplete_studies():
    with pytest.raises(ValueError):
        require_approval({"role": "exploratory", "frozen": True})
    with pytest.raises(ValueError):
        require_approval({"role": "primary_confirmation", "frozen": True,
                          "supervisor_approval": {"approved": True, "confirmed_by": "x", "confirmed_at": "2026-10-25T09:00:00Z"}})
    require_approval({"role": "primary_confirmation", "frozen": True, "supervisor_approval": {
        "approved": True, "confirmed_by": "x", "confirmed_at": "2026-10-25T09:00:00Z", "evidence": "link"}})


def test_verify_run_rejects_runs_trained_with_a_different_config(tmp_path):
    base = {"runs_dir": str(tmp_path), "data": {"root": "../dataset/data"}, "training": {"epochs": 5}}
    run_dir = tmp_path / "dom_lam0p1_seed3"
    run_dir.mkdir()
    (run_dir / "best.pt").write_bytes(b"weights")
    completed = {
        "status": "completed", "smoke": False, "lambda_domain": 0.1, "seed": 3,
        "config_hash": stable_hash({"config": base, "lambda_domain": 0.1, "seed": 3, "smoke": False}),
        "artifacts": {"best.pt": sha256_file(run_dir / "best.pt")},
    }
    (run_dir / "completed.json").write_text(json.dumps(completed), encoding="utf-8")
    assert verify_run(base, 0.1, 3)["seed"] == 3
    overridden = {**base, "data": {"root": "/abs/path/data"}}  # e.g. trained with --data-root
    with pytest.raises(ValueError):
        verify_run(overridden, 0.1, 3)
    (run_dir / "best.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        verify_run(base, 0.1, 3)


def test_exact_wilcoxon_matches_scipy_without_ties_and_handles_ties_and_zeros():
    from scipy import stats
    from clinc_domain_aux.confirmatory import wilcoxon_exact

    values = np.array([0.3, -0.1, 0.5, 0.2, 0.9, -0.4, 0.7, 0.6, 0.8, 1.1])
    assert wilcoxon_exact(values) == pytest.approx(stats.wilcoxon(values, method="exact").pvalue)
    assert wilcoxon_exact(np.ones(10)) == pytest.approx(2 / 1024)
    tied = np.array([-2, -1, -1, 0, -3, -1, -2, 1, -1, -2]) / 4500  # cross-domain deltas are multiples of 1/N
    assert 0 < wilcoxon_exact(tied) < 0.05
    assert wilcoxon_exact(np.zeros(5)) == 1.0


def test_fpr95_matches_evaluation_module_at_real_test_sizes():
    rng = np.random.default_rng(5)
    for n_is in (4500, 2974):
        is_scores = np.round(rng.normal(0, 1, n_is), 2)
        oos_scores = np.round(rng.normal(1, 1, 1000), 2)
        assert fpr95_is_positive(is_scores, oos_scores) == oos_metrics(is_scores, oos_scores)["fpr95_is_positive"]
