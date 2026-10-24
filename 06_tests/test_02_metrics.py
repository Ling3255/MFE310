"""T02: Small, deterministic checks for M07, M08, and M09."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from clinc_domain_aux.detectors import MahalanobisDetector, select_threshold
from clinc_domain_aux.evaluation import (
    classification_metrics,
    frozen_threshold_metrics,
    msp_scores,
    oos_metrics,
    per_domain_intent_f1,
)
from clinc_domain_aux.statistics import (
    exact_train_test_overlap,
    intent_stratified_paired_bootstrap,
    paired_seed_differences,
    summarize_seeds,
    train_test_overlap_sensitivity,
)


def test_classification_hand_example_and_fixed_label_set() -> None:
    domains = [0, 0, 0, 1]
    gold = [0, 0, 1, 2, 3]
    pred = [0, 1, 1, 3, 2]
    result = classification_metrics(gold, pred, domains, num_intents=4)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 0 + 0) / 4)
    assert result["macro_f1"] == pytest.approx(f1_score(gold, pred, labels=range(4), average="macro", zero_division=0))
    assert result["accuracy"] == pytest.approx(0.4)
    assert result["cross_domain_error_rate"] == pytest.approx(0.4)
    assert result["within_domain_error_rate"] == pytest.approx(0.2)
    assert result["accuracy"] + result["cross_domain_error_rate"] + result["within_domain_error_rate"] == pytest.approx(1.0)
    assert result["domain_macro_f1"] == pytest.approx((0.75 + 0) / 2)
    by_domain = per_domain_intent_f1(gold, pred, domains, num_intents=4)
    assert by_domain[0] == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)
    assert by_domain[1] == 0.0


def test_msp_logits_and_probabilities_have_explicit_direction() -> None:
    probabilities = np.asarray([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]])
    logits = np.log(probabilities)
    np.testing.assert_allclose(msp_scores(probabilities, "probabilities"), [0.3, 0.5])
    np.testing.assert_allclose(msp_scores(logits, "logits"), [0.3, 0.5])
    with pytest.raises(ValueError):
        msp_scores(probabilities, "scores")


def manual_oos_fixture() -> tuple[np.ndarray, np.ndarray]:
    return np.arange(1, 21, dtype=float) / 100, np.asarray([0.05, 0.18, 0.30, 0.40, 0.50])


def test_metric_document_hand_example_and_permutation() -> None:
    in_scope, out_of_scope = manual_oos_fixture()
    result = oos_metrics(in_scope, out_of_scope)
    assert result["fpr95_is_positive"] == pytest.approx(0.4)
    assert result["fpr95_actual_is_acceptance"] == pytest.approx(0.95)
    frozen = frozen_threshold_metrics(in_scope, out_of_scope, tau=0.175)
    assert frozen["is_acceptance_at_val95"] == pytest.approx(0.85)
    assert frozen["oos_fa_at_val95"] == pytest.approx(0.20)
    assert frozen["oos_recall_at_val95"] == pytest.approx(0.80)
    rng = np.random.default_rng(19)
    permutation = rng.permutation(len(in_scope))
    reordered = oos_metrics(in_scope[permutation], out_of_scope[rng.permutation(len(out_of_scope))])
    assert reordered["auroc_oos"] == pytest.approx(result["auroc_oos"])
    assert reordered["fpr95_is_positive"] == pytest.approx(result["fpr95_is_positive"])


def test_auroc_direction_ties_identical_scores_and_missing_classes() -> None:
    assert oos_metrics([0.1, 0.2], [0.3, 0.4])["auroc_oos"] == 1.0
    assert oos_metrics([0.3, 0.4], [0.1, 0.2])["auroc_oos"] == 0.0
    tied = oos_metrics([0.1] * 19 + [0.2], [0.1, 0.3])
    assert tied["fpr95_actual_is_acceptance"] == pytest.approx(0.95)
    assert tied["fpr95_is_positive"] == pytest.approx(0.5)
    identical = oos_metrics([0.2, 0.2], [0.2, 0.2])
    assert identical["auroc_oos"] == pytest.approx(0.5)
    assert identical["fpr95_is_positive"] == 1.0
    assert identical["fpr95_actual_is_acceptance"] == 1.0
    with pytest.raises(ValueError):
        oos_metrics([], [0.1])
    with pytest.raises(ValueError):
        oos_metrics([0.1], [])
    with pytest.raises(ValueError):
        frozen_threshold_metrics([0.1], [], 0.2)


def test_threshold_selection_nextafter_ties_and_all_identical() -> None:
    result = select_threshold(np.arange(20, dtype=float), [18.0, 19.0, 20.0])
    expected = np.nextafter(18.0, np.inf)
    assert result["threshold"] == expected
    assert result["is_acceptance"] == pytest.approx(0.95)
    assert result["oos_recall"] == pytest.approx(2 / 3)
    assert np.sum(np.arange(20) < result["threshold"]) == 19

    identical = select_threshold([0.2] * 20, [0.1, 0.3])
    assert identical["threshold"] == np.nextafter(0.2, np.inf)
    assert identical["is_acceptance"] == 1.0
    assert identical["oos_recall"] == pytest.approx(0.5)


def test_mahalanobis_within_class_fit_and_npz_roundtrip(tmp_path) -> None:
    features = np.asarray([
        [0.0, 0.0], [2.0, 0.0], [0.0, 2.0],
        [10.0, 10.0], [12.0, 10.0], [10.0, 12.0],
    ], dtype=np.float32)
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    detector = MahalanobisDetector(num_intents=2).fit(features, labels, sample_ids=[f"train:{i}" for i in range(6)])
    np.testing.assert_allclose(detector.class_means_, [[2 / 3, 2 / 3], [32 / 3, 32 / 3]])
    assert detector.precision_.dtype == np.float64
    scores = detector.mahalanobis_scores(detector.class_means_)
    np.testing.assert_allclose(scores, 0.0, atol=1e-10)
    assert np.isfinite(detector.mahalanobis_scores(features)).all()

    path = tmp_path / "detector.npz"
    detector.save(path)
    loaded = MahalanobisDetector.load(path)
    assert loaded.sample_hash_ == detector.sample_hash_
    assert loaded.metadata["dtype"] == "float64"
    np.testing.assert_allclose(loaded.mahalanobis_scores(features), detector.mahalanobis_scores(features))
    with pytest.raises(ValueError):
        detector.mahalanobis_scores([[np.nan, 0.0]])


def prediction_rows(seed: int, offset: int) -> list[dict]:
    rows = []
    for intent in range(2):
        for item in range(3):
            sample_id = f"test:{intent * 3 + item}"
            ce_pred = intent if item < 2 else 1 - intent
            dom_pred = intent if item != 0 or intent != offset % 2 else 1 - intent
            rows.append({"sample_id": sample_id, "seed": seed, "gold_intent": intent, "ce_pred": ce_pred, "dom_pred": dom_pred})
    return rows


def bootstrap_inputs() -> tuple[dict, dict]:
    ce, dom = {}, {}
    for seed in [0, 1]:
        rows = prediction_rows(seed, seed)
        ce[seed] = [{**row, "pred_intent": row.pop("ce_pred")} for row in rows]
        dom_rows = prediction_rows(seed, seed)
        dom[seed] = [{**row, "pred_intent": row.pop("dom_pred")} for row in dom_rows]
    return ce, dom


def test_bootstrap_shared_indices_reproducible_and_independently_recomputed() -> None:
    ce, dom = bootstrap_inputs()
    first = intent_stratified_paired_bootstrap(
        ce, dom, num_intents=2, n_bootstrap=25, samples_per_intent=3, return_indices=True
    )
    second = intent_stratified_paired_bootstrap(
        ce, dom, num_intents=2, n_bootstrap=25, samples_per_intent=3, return_indices=True
    )
    np.testing.assert_array_equal(first["bootstrap_indices"], second["bootstrap_indices"])
    np.testing.assert_allclose(first["bootstrap_mean_deltas"], second["bootstrap_mean_deltas"])
    assert first["n_bootstrap"] == 25
    assert "not retraining" in first["condition"]

    indices = first["bootstrap_indices"]
    for seed in [0, 1]:
        gold = np.asarray([row["gold_intent"] for row in ce[seed]])
        ce_pred = np.asarray([row["pred_intent"] for row in ce[seed]])
        dom_pred = np.asarray([row["pred_intent"] for row in dom[seed]])
        recomputed = []
        for draw in indices:
            sampled_gold = gold[draw.ravel()]
            ce_f1 = f1_score(sampled_gold, ce_pred[draw.ravel()], labels=[0, 1], average="macro", zero_division=0)
            dom_f1 = f1_score(sampled_gold, dom_pred[draw.ravel()], labels=[0, 1], average="macro", zero_division=0)
            recomputed.append(dom_f1 - ce_f1)
        np.testing.assert_allclose(first["bootstrap_seed_deltas"][seed], recomputed)
    np.testing.assert_allclose(
        first["bootstrap_mean_deltas"],
        (first["bootstrap_seed_deltas"][0] + first["bootstrap_seed_deltas"][1]) / 2,
    )


def test_seed_statistics_and_exact_overlap_sensitivity() -> None:
    summary = summarize_seeds({0: 0.2, 1: 0.4, 2: 0.6})
    assert summary["mean"] == pytest.approx(0.4)
    assert summary["std"] == pytest.approx(0.2)
    paired = paired_seed_differences({0: 0.5, 1: 0.6, 2: 0.4}, {0: 0.6, 1: 0.7, 2: 0.3})
    assert paired["seed_deltas"] == {0: pytest.approx(0.1), 1: pytest.approx(0.1), 2: pytest.approx(-0.1)}
    assert paired["mean_delta"] == pytest.approx(1 / 30)

    rows = [
        {"sample_id": "test:0", "text": "same", "gold_intent": 0, "pred_intent": 1},
        {"sample_id": "test:1", "text": "unique", "gold_intent": 1, "pred_intent": 1},
    ]
    overlap = exact_train_test_overlap([{"text": "same"}], rows)
    assert overlap["test_sample_ids"] == ["test:0"]
    sensitivity = train_test_overlap_sensitivity(rows, ["same"], [0, 0], num_intents=2)
    assert sensitivity["n_remaining"] == 1
    assert sensitivity["metrics"]["macro_f1"] == pytest.approx(0.5)
