"""T01: Small-fixture tests for M02 IDs, maps, assertions, and OOS exclusion."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "03_src"))

from clinc_domain_aux.data import ClincDataset, load_datasets, load_label_maps, make_dataloader


def _fixture(root: Path) -> Path:
    data_root = root / "data"
    data_root.mkdir()
    domains = {"z_domain": ["zeta", "alpha"], "a_domain": ["beta"]}
    data = {
        "train": [["Alpha raw  text", "alpha"], ["Beta text", "beta"]],
        "val": [["Alpha val", "alpha"]],
        "test": [["Zeta test", "zeta"]],
        "oos_train": [["Unknown request", "oos"]],
        "oos_val": [["Another unknown", "oos"]],
        "oos_test": [["Third unknown", "oos"]],
    }
    (data_root / "domains.json").write_text(json.dumps(domains), encoding="utf-8")
    (data_root / "data_full.json").write_text(json.dumps(data), encoding="utf-8")
    return data_root


class _Tokenizer:
    def __call__(self, text, max_length, return_tensors, **kwargs):
        import torch

        ids = [ord(character) % 100 + 1 for character in text][: max_length - 2]
        ids = [101, *ids, 102]
        ids += [0] * (max_length - len(ids))
        mask = [int(value != 0) for value in ids]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])}


def test_stable_ids_sorted_maps_and_original_text(tmp_path):
    data_root = _fixture(tmp_path)
    first = load_datasets(data_root, tmp_path / "metadata_a", strict_counts=False)
    second = load_datasets(data_root, tmp_path / "metadata_b", strict_counts=False)

    assert [row["sample_id"] for row in first["train"]] == ["train:00000", "train:00001"]
    assert first["train"][0]["text"] == "Alpha raw  text"
    assert [row["sample_id"] for row in second["train"]] == [row["sample_id"] for row in first["train"]]
    label_to_id, domain_to_id, intent_to_domain = load_label_maps(tmp_path / "metadata_a")
    assert label_to_id == {"alpha": 0, "beta": 1, "zeta": 2}
    assert domain_to_id == {"a_domain": 0, "z_domain": 1}
    assert intent_to_domain == {"alpha": "z_domain", "beta": "a_domain", "zeta": "z_domain"}
    assert first["oos_train"][0]["intent_id"] == -1
    assert first["oos_train"][0]["domain_id"] == -1
    assert first["oos_train"][0]["domain"] is None
    manifest = json.loads((tmp_path / "metadata_a" / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["train"] == 2
    assert manifest["strict_counts"] is False
    assert len(manifest["files"]["data_full.json"]["sha256"]) == 64


def test_oos_is_excluded_from_training_dataset_and_loader(tmp_path):
    datasets = load_datasets(_fixture(tmp_path), strict_counts=False)
    records = datasets["train"] + datasets["oos_train"]
    dataset = ClincDataset(records, _Tokenizer(), max_length=12, in_scope_only=True)
    loader = make_dataloader(records, _Tokenizer(), batch_size=2, max_length=12, in_scope_only=True)

    assert len(dataset) == 2
    item = dataset[0]
    assert set(item) == {"input_ids", "attention_mask", "intent_id", "domain_id", "sample_id"}
    batch = next(iter(loader))
    assert batch["sample_id"] == ["train:00000", "train:00001"]
    assert batch["input_ids"].shape == (2, 12)
    assert batch["attention_mask"].shape == (2, 12)
    assert all(int(value) >= 0 for value in batch["intent_id"])


def test_official_count_assertion_fails_on_small_fixture(tmp_path):
    with pytest.raises(ValueError, match="Official split counts"):
        load_datasets(_fixture(tmp_path), strict_counts=True)


def test_duplicate_intent_across_domains_fails_even_without_strict_counts(tmp_path):
    data_root = _fixture(tmp_path)
    domains = json.loads((data_root / "domains.json").read_text(encoding="utf-8"))
    domains["a_domain"].append("alpha")
    (data_root / "domains.json").write_text(json.dumps(domains), encoding="utf-8")
    with pytest.raises(ValueError, match="more than one domain"):
        load_datasets(data_root, strict_counts=False)
