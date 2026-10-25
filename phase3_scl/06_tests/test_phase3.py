"""P3-T: Phase-3 unit tests. Run: PYTHONPATH=03_src:phase3_scl/03_src pytest phase3_scl/06_tests -q"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset
from transformers import BertConfig

from clinc_domain_aux.confirmatory import macro_f1_present, two_level_bootstrap
from clinc_domain_aux.contrastive import ClassBalancedBatchSampler, supervised_contrastive_loss
from clinc_domain_aux.detectors import KNNDetector
from clinc_domain_aux.domain_aux import build_model, load_model
from clinc_domain_aux.training import train_run
from phase3_scl.metrics import expected_calibration_error, representation_geometry
from phase3_scl.perturb import char_noise, make_variants, strip_punctuation, variant_of
from phase3_scl.splits import build_low_resource, build_main_split, check_no_test_leakage, normalize


# ----------------------------------------------------------------- SupCon loss

def naive_supcon(z: torch.Tensor, labels: torch.Tensor, tau: float) -> torch.Tensor:
    losses = []
    for i in range(len(z)):
        positives = [p for p in range(len(z)) if p != i and labels[p] == labels[i]]
        if not positives:
            continue
        denominator = torch.logsumexp(torch.stack([z[i] @ z[a] / tau for a in range(len(z)) if a != i]), 0)
        losses.append(-torch.stack([z[i] @ z[p] / tau - denominator for p in positives]).mean())
    return torch.stack(losses).mean()


def test_supcon_matches_definition_and_is_permutation_invariant():
    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(12, 5), dim=1)
    labels = torch.tensor([0, 0, 1, 1, 1, 2, 3, 3, 4, 0, 2, 5])  # label 4 and 5 have no positive
    loss = supervised_contrastive_loss(z, labels, 0.2)
    torch.testing.assert_close(loss, naive_supcon(z, labels, 0.2))
    order = torch.randperm(12)
    torch.testing.assert_close(supervised_contrastive_loss(z[order], labels[order], 0.2), loss)


def test_supcon_without_positives_is_zero_and_keeps_graph():
    z = torch.nn.functional.normalize(torch.randn(4, 3), dim=1).requires_grad_()
    loss = supervised_contrastive_loss(z, torch.arange(4), 0.1)
    assert loss.item() == 0.0
    loss.backward()
    assert z.grad is not None


def test_supcon_prefers_compact_classes():
    labels = torch.tensor([0, 0, 1, 1])
    tight = torch.tensor([[1.0, 0.0], [0.99, 0.14], [-1.0, 0.0], [-0.99, 0.14]])
    mixed = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.99, 0.14], [-0.99, 0.14]])
    norm = torch.nn.functional.normalize
    assert supervised_contrastive_loss(norm(tight, dim=1), labels, 0.1) < supervised_contrastive_loss(norm(mixed, dim=1), labels, 0.1)


# ----------------------------------------------------------------- class-balanced sampler

def test_class_balanced_sampler_covers_each_example_once_with_same_label_groups():
    labels = [label for label in range(6) for _ in range(10)]  # 10 per class -> groups of 4, 4, 2
    generator = torch.Generator().manual_seed(3)
    sampler = ClassBalancedBatchSampler(labels, classes_per_batch=2, samples_per_class=4, generator=generator)
    batches = list(sampler)
    assert len(batches) == len(sampler)
    flat = [index for batch in batches for index in batch]
    assert sorted(flat) == list(range(60))
    for batch in batches:
        assert len(batch) <= 8
        counts = np.bincount([labels[i] for i in batch], minlength=6)
        assert all(count in (0, 2, 4, 6, 8) or count <= 4 for count in counts)
    again = list(ClassBalancedBatchSampler(labels, 2, 4, torch.Generator().manual_seed(3)))
    other = list(ClassBalancedBatchSampler(labels, 2, 4, torch.Generator().manual_seed(4)))
    assert again == batches and other != batches


# ----------------------------------------------------------------- split construction

def clinc_fixture():
    intents = ["a_x", "b_y", "c_z"]
    train = [[f"{intent} utterance {i}", intent] for intent in intents for i in range(10)]
    train[0] = ["shared text", "a_x"]          # duplicates a val text -> excluded from internal test
    train[11] = ["conflict text", "b_y"]       # same text labelled differently in val -> excluded
    full = {
        "train": train,
        "val": [["shared text", "a_x"], ["conflict text", "c_z"], ["val only", "b_y"]],
        "test": [["official test", "a_x"]],
        "oos_train": [["oos one", "oos"], ["oos two", "oos"]],
        "oos_val": [["oos val", "oos"]],
        "oos_test": [["oos official", "oos"]],
    }
    plus = {"oos_train": [["OOS one", "oos"], ["oos three", "oos"]]}  # "OOS one" duplicates after normalisation
    return full, plus


def test_main_split_counts_exclusions_and_no_leakage():
    full, plus = clinc_fixture()
    split, record = build_main_split(full, plus, seed=1, test_per_intent=3)
    assert len(split["test"]) == 9 and len(split["train"]) == 21
    test_texts = {normalize(text) for text, _ in split["test"]}
    assert not test_texts & {"shared text", "conflict text"}
    assert [text for text, _ in split["oos_test"]] == ["oos one", "oos two", "oos three"]
    assert split["oos_train"] == [] and split["val"] == full["val"]
    check_no_test_leakage(split)
    assert record["origin"]["oos_test:00002"] == "data_oos_plus.oos_train:00001"
    again, _ = build_main_split(full, plus, seed=1, test_per_intent=3)
    assert again == split


def test_low_resource_subsets_are_nested_and_per_intent():
    full, plus = clinc_fixture()
    split, _ = build_main_split(full, plus, seed=1, test_per_intent=3)
    subsets = build_low_resource(split, seed=5, sizes=[2, 4])
    small, large = ({tuple(row) for row in subsets[size]["train"]} for size in (2, 4))
    assert small < large
    assert all(sum(1 for _, intent in subsets[2]["train"] if intent == name) == 2 for name in ("a_x", "b_y", "c_z"))
    assert subsets[2]["test"] == split["test"]


def test_leakage_check_raises():
    full, plus = clinc_fixture()
    split, _ = build_main_split(full, plus, seed=1, test_per_intent=3)
    split["val"] = split["val"] + [split["test"][0]]
    with pytest.raises(ValueError):
        check_no_test_leakage(split)


# ----------------------------------------------------------------- perturbations and metrics

def test_perturbations_are_deterministic_and_non_empty():
    records = [{"sample_id": "test:00000", "text": "Can you book a table, please?"}]
    first = make_variants(records, seed=7, noise_rates=[0.1], punctuation=True)
    assert first == make_variants(records, seed=7, noise_rates=[0.1], punctuation=True)
    assert [variant_of(r["sample_id"]) for r in first] == ["noise10", "nopunct"]
    assert first[1]["text"] == "Can you book a table please"
    assert char_noise("abc", 0.0, np.random.default_rng(0)) == "abc"
    assert char_noise("a", 1.0, np.random.default_rng(1)) != ""
    assert strip_punctuation("?!") == "?!"


def test_ece_hand_example():
    confidence = np.array([0.9, 0.9, 0.6, 0.6])
    correct = np.array([1, 0, 1, 1])
    result = expected_calibration_error(confidence, correct, bins=10)
    assert result["ece"] == pytest.approx(0.5 * abs(0.5 - 0.9) + 0.5 * abs(1.0 - 0.6))


def test_geometry_detects_tighter_clusters_and_far_oos():
    rng = np.random.default_rng(0)
    centres = np.eye(4)
    labels = np.repeat(np.arange(4), 30)
    loose = centres[labels] + rng.normal(0, 0.3, (120, 4))
    tight = centres[labels] + rng.normal(0, 0.05, (120, 4))
    oos = -np.ones((10, 4)) + rng.normal(0, 0.05, (10, 4))
    loose_geo = representation_geometry(loose, labels, loose[:8], oos)
    tight_geo = representation_geometry(tight, labels, tight[:8], oos)
    assert tight_geo["within_between_ratio"] < loose_geo["within_between_ratio"]
    assert tight_geo["oos_minus_test_distance"] > 0


def test_knn_detector_scores_outliers_higher():
    rng = np.random.default_rng(1)
    bank = np.vstack([rng.normal(0, 0.1, (50, 3)) + [1, 0, 0], rng.normal(0, 0.1, (50, 3)) + [0, 1, 0]])
    detector = KNNDetector(5).fit(bank)
    scores = detector.score(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    assert scores[0] < scores[1]
    with pytest.raises(ValueError):
        KNNDetector(0)


def test_bootstrap_interval_widens_with_level():
    rng = np.random.default_rng(2)
    gold = np.repeat(np.arange(5), 10)
    ref = np.stack([np.where(rng.random(50) < 0.8, gold, (gold + 1) % 5) for _ in range(4)])
    trt = np.stack([np.where(rng.random(50) < 0.85, gold, (gold + 1) % 5) for _ in range(4)])
    fn = lambda g, p: macro_f1_present(g, p, 5)  # noqa: E731
    narrow = two_level_bootstrap(gold, ref, trt, fn, n_boot=300, seed=1)
    wide = two_level_bootstrap(gold, ref, trt, fn, n_boot=300, seed=1, level=0.975)
    assert narrow["ci"] == narrow["ci95"]
    assert wide["ci"][0] <= narrow["ci"][0] and wide["ci"][1] >= narrow["ci"][1]


# ----------------------------------------------------------------- training integration (tiny BERT, CPU)

def tiny_bert() -> dict:
    return BertConfig(vocab_size=48, hidden_size=24, num_hidden_layers=1, num_attention_heads=2,
                      intermediate_size=48, max_position_embeddings=32).to_dict()


class SyntheticTokens(Dataset):
    def __init__(self, split: str, count: int) -> None:
        self.rows = [{"sample_id": f"{split}:{i:05d}", "input_ids": [2, 4 + (i * 3) % 35, 5 + i % 4, 3],
                      "intent_id": i % 4, "domain_id": (i % 4) // 2} for i in range(count)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def phase3_config(scl: tuple[float, float] | None) -> dict:
    training = {"learning_rate": 0.001, "weight_decay": 0.01, "epochs": 2, "patience": 2, "effective_batch_size": 4,
                "micro_batch_size": 4, "gradient_accumulation": 1, "warmup_ratio": 0.1, "max_grad_norm": 1.0,
                "num_workers": 0, "deterministic": True, "amp": False, "device": "cpu", "sampler": "class_balanced",
                "classes_per_batch": 2, "samples_per_class": 2, "initialization_dir": "../init",
                "best_checkpoint_weights_only": True, "keep_last_checkpoint": False}
    if scl:
        training["supervised_contrastive"] = {"weight": scl[0], "temperature": scl[1]}
    return {"model": {"encoder_config": tiny_bert(), "num_intents": 4, "num_domains": 2, "dropout": 0.1, "max_length": 8},
            "training": training}


def test_scl_training_run_pairs_with_ce_and_saves_weights_only(tmp_path):
    datasets = {"train": SyntheticTokens("train", 16), "val": SyntheticTokens("val", 8)}
    ce = train_run(phase3_config(None), 0.0, 3, datasets, None, tmp_path / "ce" / "ce_seed3", smoke=True)
    scl = train_run(phase3_config((0.5, 0.1)), 0.0, 3, datasets, None, tmp_path / "scl" / "ce_seed3", smoke=True)
    assert ce["initialization_hash"] == scl["initialization_hash"]  # shared initialization file
    ce_order = json.loads((tmp_path / "ce" / "ce_seed3" / "sample_order.json").read_text())
    scl_order = json.loads((tmp_path / "scl" / "ce_seed3" / "sample_order.json").read_text())
    assert ce_order["epochs"] == scl_order["epochs"]                # identical batches: only the loss differs
    log = [json.loads(line) for line in (tmp_path / "scl" / "ce_seed3" / "train_log.jsonl").read_text().splitlines()]
    assert all("train_loss_scl" in row for row in log)
    assert not (tmp_path / "scl" / "ce_seed3" / "last.pt").exists() and "last.pt" not in scl["artifacts"]
    checkpoint = torch.load(tmp_path / "scl" / "ce_seed3" / "best.pt", map_location="cpu", weights_only=False)
    assert "optimizer_state" not in checkpoint
    model = build_model({"encoder_config": tiny_bert()}, 4, 2, 0.0, "cpu")
    load_model(model, tmp_path / "scl" / "ce_seed3" / "best.pt")
    # Re-invoking a completed run is a no-op that re-validates its artefacts.
    assert train_run(phase3_config((0.5, 0.1)), 0.0, 3, datasets, None, tmp_path / "scl" / "ce_seed3", smoke=True)["status"] == "completed"


def test_scl_requires_matching_batch_geometry(tmp_path):
    config = phase3_config((0.5, 0.1))
    config["training"]["classes_per_batch"] = 3
    with pytest.raises(ValueError):
        train_run(config, 0.0, 1, {"train": SyntheticTokens("train", 8), "val": SyntheticTokens("val", 4)}, None, tmp_path / "unused", smoke=True)
