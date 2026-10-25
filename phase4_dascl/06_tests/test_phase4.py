"""P4-T: Phase-4 unit tests.

Run: PYTHONPATH=03_src:phase3_scl/03_src:phase4_dascl/03_src pytest phase4_dascl/06_tests -q
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset
from transformers import BertConfig

from clinc_domain_aux.confirmatory import macro_f1_present
from clinc_domain_aux.contrastive import ClassBalancedBatchSampler, DomainAwareBatchSampler
from clinc_domain_aux.training import train_run
from phase4_dascl.audit import batch_composition
from phase4_dascl.contrasts import contrast_bootstrap, seed_contrasts
from phase4_dascl.evaluation_sets import build_unseen
from phase4_dascl.geometry import domain_centroid_separation

INTERACTION = {"da_scl": 1, "da_ce": -1, "ib_scl": -1, "ib_ce": 1}


def synthetic_labels(intents_per_domain=(3, 3, 3, 3, 3, 3), per_intent=10):
    labels, domains, intent = [], [], 0
    for domain, count in enumerate(intents_per_domain):
        for _ in range(count):
            labels += [intent] * per_intent
            domains += [domain] * per_intent
            intent += 1
    return labels, domains


# ----------------------------------------------------------------- domain-aware sampler

def test_domain_aware_sampler_covers_every_example_once_with_two_domains_per_batch():
    labels, domains = synthetic_labels()
    sampler = DomainAwareBatchSampler(labels, domains, 2, 2, 4, torch.Generator().manual_seed(0))
    batches = list(sampler)
    assert len(batches) == len(sampler) == 15  # 6 domains x 5 blocks, two blocks per batch
    assert sorted(i for batch in batches for i in batch) == list(range(len(labels)))
    assert all(len({domains[i] for i in batch}) == 2 for batch in batches)
    assert all(len(batch) <= 16 for batch in batches)
    again = list(DomainAwareBatchSampler(labels, domains, 2, 2, 4, torch.Generator().manual_seed(0)))
    assert again == batches


def test_domain_aware_length_is_exact_for_unequal_domains():
    labels, domains = synthetic_labels(intents_per_domain=(1, 3, 2, 5), per_intent=7)
    sampler = DomainAwareBatchSampler(labels, domains, 2, 2, 4, torch.Generator().manual_seed(1))
    batches = list(sampler)
    assert len(batches) == len(sampler)
    assert sorted(i for batch in batches for i in batch) == list(range(len(labels)))


def test_domain_aware_sampler_rejects_intent_in_two_domains():
    with pytest.raises(ValueError):
        DomainAwareBatchSampler([0, 0, 1], [0, 1, 1], 2, 1, 2, torch.Generator())


def test_domain_aware_batches_contain_more_same_domain_negatives():
    labels, domains = synthetic_labels()
    domain_aware = list(DomainAwareBatchSampler(labels, domains, 2, 2, 4, torch.Generator().manual_seed(3)))
    intent_balanced = list(ClassBalancedBatchSampler(labels, 4, 4, torch.Generator().manual_seed(3)))
    da = batch_composition(domain_aware, labels, domains)
    ib = batch_composition(intent_balanced, labels, domains)
    assert da["every_example_once"] and ib["every_example_once"]
    assert da["mean_same_domain_negatives_per_anchor"] > ib["mean_same_domain_negatives_per_anchor"]


def test_batch_composition_hand_example():
    result = batch_composition([[0, 1, 2, 3]], labels=[0, 0, 1, 2], domains=[0, 0, 0, 1])
    assert result["mean_positives_per_anchor"] == 0.5
    assert result["mean_same_domain_negatives_per_anchor"] == 1.0
    assert result["mean_other_domain_negatives_per_anchor"] == 1.5
    assert result["every_example_once"]


# ----------------------------------------------------------------- contrasts

def test_interaction_contrast_per_seed_and_zero_for_identical_arms():
    values = {"da_scl": [0.9, 0.8], "da_ce": [0.7, 0.7], "ib_scl": [0.75, 0.6], "ib_ce": [0.7, 0.6]}
    np.testing.assert_allclose(seed_contrasts(values, INTERACTION), [0.15, 0.1])
    rng = np.random.default_rng(0)
    gold = np.repeat(np.arange(5), 20)
    base = np.stack([np.where(rng.random(100) < 0.8, gold, (gold + 1) % 5) for _ in range(4)])
    fn = lambda g, p: macro_f1_present(g, p, 5)  # noqa: E731
    same = contrast_bootstrap(gold, {arm: base.copy() for arm in INTERACTION}, INTERACTION, fn, n_boot=50, seed=1, level=0.95)
    assert same["ci"] == [0.0, 0.0]


def test_interaction_bootstrap_detects_a_gain_only_in_da_scl():
    rng = np.random.default_rng(1)
    gold = np.repeat(np.arange(5), 20)
    base = np.stack([np.where(rng.random(100) < 0.8, gold, (gold + 1) % 5) for _ in range(4)])
    improved = base.copy()
    improved[:, ::4] = gold[::4]  # da_scl fixes a quarter of the rows in every seed
    predictions = {"ib_ce": base, "ib_scl": base, "da_ce": base, "da_scl": improved}
    fn = lambda g, p: macro_f1_present(g, p, 5)  # noqa: E731
    result = contrast_bootstrap(gold, predictions, INTERACTION, fn, n_boot=200, seed=2, level=0.975)
    assert result["ci"][0] > 0


# ----------------------------------------------------------------- unseen pool and geometry

def record(sample_id, text, intent):
    return {"sample_id": sample_id, "text": text, "intent": intent, "intent_id": 0 if intent == "a" else 1, "domain_id": 0}


def test_build_unseen_excludes_low_resource_training_and_listed_texts():
    train = [record(f"train:{i:05d}", f"{intent} text {i}", intent) for i, intent in enumerate(["a"] * 5 + ["b"] * 5)]
    main = {"train": train, "val": [], "test": [], "oos_val": [], "oos_test": []}
    n25 = {"train": [train[0], train[1], train[5], train[6]]}
    unseen, origin, counts = build_unseen(main, n25, excluded_texts=["A TEXT 4"])  # matched after normalisation
    assert [row["sample_id"] for row in unseen] == [f"unseen:{i:05d}" for i in range(5)]
    assert origin["unseen:00000"] == "train:00002" and counts == {"a": 2, "b": 3}
    assert all(row["split"] == "unseen" for row in unseen)


def test_build_unseen_train_val_duplicate_must_be_excluded():
    """Regression for the 2026-10-25 audit stop: CLINC train shares three texts with val."""
    train = [record(f"train:{i:05d}", text, intent) for i, (text, intent) in
             enumerate([("hey what's up", "a"), ("a other", "a"), ("b one", "b"), ("b two", "b")])]
    main = {"train": train, "val": [record("val:00000", "hey what's up", "a")], "test": [], "oos_val": [], "oos_test": []}
    with pytest.raises(ValueError):
        build_unseen(main, {"train": []}, excluded_texts=[])
    unseen, _, counts = build_unseen(main, {"train": []}, excluded_texts=["hey what's up"])
    assert counts == {"a": 1, "b": 2} and len(unseen) == 3
    with pytest.raises(ValueError):  # an intent may not disappear from U
        build_unseen(main, {"train": []}, excluded_texts=["hey what's up", "a other"])


def test_domain_centroid_separation_reflects_close_same_domain_intents():
    labels = np.repeat(np.arange(4), 5)
    intent_domain = np.array([0, 0, 1, 1])
    separated = np.eye(4)[labels]
    crowded_centres = np.array([[1.0, 0, 0, 0], [1.0, 0.1, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
    crowded = crowded_centres[labels]
    far = domain_centroid_separation(separated, labels, intent_domain)
    near = domain_centroid_separation(crowded, labels, intent_domain)
    assert far["same_domain_nearest_centroid_distance"] > near["same_domain_nearest_centroid_distance"]
    assert far["other_domain_nearest_centroid_distance"] == pytest.approx(1.0)


# ----------------------------------------------------------------- training integration (tiny BERT, CPU)

class SyntheticTokens(Dataset):
    def __init__(self, split: str, count: int) -> None:
        self.rows = [{"sample_id": f"{split}:{i:05d}", "input_ids": [2, 4 + (i * 3) % 35, 5 + i % 4, 3],
                      "intent_id": i % 4, "domain_id": (i % 4) // 2} for i in range(count)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def domain_aware_config(samples_per_intent: int = 2) -> dict:
    tiny = BertConfig(vocab_size=48, hidden_size=24, num_hidden_layers=1, num_attention_heads=2,
                      intermediate_size=48, max_position_embeddings=32).to_dict()
    training = {"learning_rate": 0.001, "weight_decay": 0.01, "epochs": 2, "patience": 2, "effective_batch_size": 4,
                "micro_batch_size": 4, "gradient_accumulation": 1, "warmup_ratio": 0.1, "max_grad_norm": 1.0,
                "num_workers": 0, "deterministic": True, "amp": False, "device": "cpu", "sampler": "domain_aware",
                "domains_per_batch": 2, "intents_per_domain": 1, "samples_per_intent": samples_per_intent,
                "initialization_dir": "../init", "best_checkpoint_weights_only": True, "keep_last_checkpoint": False,
                "supervised_contrastive": {"weight": 0.5, "temperature": 0.1}}
    return {"model": {"encoder_config": tiny, "num_intents": 4, "num_domains": 2, "dropout": 0.1, "max_length": 8},
            "training": training}


def test_domain_aware_training_run_completes(tmp_path):
    datasets = {"train": SyntheticTokens("train", 16), "val": SyntheticTokens("val", 8)}
    result = train_run(domain_aware_config(), 0.0, 5, datasets, None, tmp_path / "da" / "ce_seed5", smoke=True)
    assert result["status"] == "completed"
    assert (tmp_path / "da" / "ce_seed5" / "sample_order.json").exists()


def test_domain_aware_batch_geometry_must_match_micro_batch(tmp_path):
    with pytest.raises(ValueError):
        train_run(domain_aware_config(samples_per_intent=3), 0.0, 1,
                  {"train": SyntheticTokens("train", 8), "val": SyntheticTokens("val", 4)}, None, tmp_path / "x", smoke=True)
