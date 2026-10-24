"""T03: Paired initialization, lambda-zero gradients, ordering, and resume."""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
from transformers import BertConfig

from clinc_domain_aux.common import hash_state_dict, read_json, read_jsonl, set_seed
from clinc_domain_aux.domain_aux import (
    build_model,
    create_common_initialization,
    load_common_initialization,
)
from clinc_domain_aux.training import _flush_window, _optimizer, train_run


def tiny_config(dropout: float = 0.1) -> BertConfig:
    return BertConfig(
        vocab_size=48,
        hidden_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=48,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        max_position_embeddings=32,
        type_vocab_size=2,
    )


def token_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[2, 7, 9, 3], [2, 11, 13, 3], [2, 5, 6, 3]]),
        "attention_mask": torch.ones(3, 4, dtype=torch.long),
        "token_type_ids": torch.zeros(3, 4, dtype=torch.long),
        "intent_id": torch.tensor([0, 1, 2]),
        "domain_id": torch.tensor([0, 1, 0]),
    }


def test_lambda_zero_gradient_and_shared_dropout() -> None:
    set_seed(7)
    model = build_model(tiny_config(), 3, 2, 0.0, "cpu")
    reference = build_model(tiny_config(), 3, 2, 0.0, "cpu")
    reference.load_state_dict(model.state_dict())
    batch = token_batch()
    shared_inputs = []
    model.intent_head.register_forward_hook(lambda _module, args, _out: shared_inputs.append(args[0]))
    model.domain_head.register_forward_hook(lambda _module, args, _out: shared_inputs.append(args[0]))
    model.eval()
    reference.eval()

    output = model(**batch)
    output["loss"].backward()
    reference_output = reference(**batch)
    pure_intent_loss = F.cross_entropy(
        reference_output["intent_logits"], batch["intent_id"]
    )
    pure_intent_loss.backward()

    assert set(output) == {"intent_logits", "domain_logits", "cls", "loss_intent", "loss_domain", "loss"}
    assert shared_inputs[0] is shared_inputs[1]
    assert not output["loss_domain"].requires_grad
    torch.testing.assert_close(output["loss"], pure_intent_loss, rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        other = dict(reference.named_parameters())[name]
        if name.startswith("domain_head."):
            assert parameter.grad is None
        elif parameter.grad is None or other.grad is None:
            assert parameter.grad is other.grad
        else:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)


def test_common_initialization_and_optimizer_groups(tmp_path) -> None:
    path = tmp_path / "common_initialization_seed4.pt"
    created = create_common_initialization(tiny_config(), 3, 2, 4, path)
    ce = build_model(tiny_config(), 3, 2, 0.0, "cpu")
    domain = build_model(tiny_config(), 3, 2, 0.5, "cpu")
    ce_loaded = load_common_initialization(ce, path)
    domain_loaded = load_common_initialization(domain, path)

    assert created["initialization_hash"] == ce_loaded["initialization_hash"]
    assert ce_loaded["initialization_hash"] == domain_loaded["initialization_hash"]
    assert hash_state_dict(ce.state_dict()) == hash_state_dict(domain.state_dict())
    optimizer = _optimizer(ce, {"learning_rate": 2e-5, "weight_decay": 0.01}, 0.0)
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert all(id(parameter) not in optimized for parameter in ce.domain_head.parameters())
    assert optimizer.param_groups[0]["weight_decay"] == 0.01
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


class SyntheticTokens(Dataset):
    def __init__(self, split: str, count: int) -> None:
        self.rows = []
        for index in range(count):
            intent = index % 3
            self.rows.append({
                "sample_id": f"{split}:{index:05d}",
                "input_ids": [2, 4 + ((index * 3) % 35), 5 + intent, 3],
                "intent_id": intent,
                "domain_id": intent % 2,
            })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def training_config() -> dict:
    return {
        "model": {
            "encoder_config": tiny_config().to_dict(),
            "num_intents": 3,
            "num_domains": 2,
            "dropout": 0.1,
            "max_length": 8,
        },
        "training": {
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "epochs": 2,
            "patience": 2,
            "effective_batch_size": 4,
            "micro_batch_size": 2,
            "gradient_accumulation": 2,
            "warmup_ratio": 0.1,
            "max_grad_norm": 1.0,
            "num_workers": 0,
            "deterministic": True,
            "amp": False,
            "device": "cpu",
            "resume": True,
        },
    }


def test_cross_lambda_order_and_checkpoint_resume(tmp_path) -> None:
    datasets = {"train": SyntheticTokens("train", 6), "val": SyntheticTokens("val", 4)}
    config = training_config()
    ce_dir = tmp_path / "ce_seed9"
    resumed_dir = tmp_path / "dom_lam0p5_seed9"
    reference_dir = tmp_path / "dom_reference_seed9"

    ce = train_run(config, 0.0, 9, datasets, None, ce_dir, smoke=True)
    partial = train_run(
        config, 0.5, 9, datasets, None, resumed_dir, smoke=True, stop_after_epoch=1
    )
    assert partial["status"] == "partial"
    resumed = train_run(config, 0.5, 9, datasets, None, resumed_dir, smoke=True)
    reference = train_run(config, 0.5, 9, datasets, None, reference_dir, smoke=True)

    ce_order = read_json(ce_dir / "sample_order.json")
    resumed_order = read_json(resumed_dir / "sample_order.json")
    reference_order = read_json(reference_dir / "sample_order.json")
    assert ce_order["epochs"] == resumed_order["epochs"] == reference_order["epochs"]
    assert ce["initialization_hash"] == resumed["initialization_hash"] == reference["initialization_hash"]

    resumed_last = torch.load(resumed_dir / "last.pt", map_location="cpu", weights_only=False)
    reference_last = torch.load(reference_dir / "last.pt", map_location="cpu", weights_only=False)
    assert resumed_last["epoch"] == reference_last["epoch"] == 2
    assert resumed_last["optimizer_state"]["state"]
    assert resumed_last["scheduler_state"]["last_epoch"] == reference_last["scheduler_state"]["last_epoch"]
    torch.testing.assert_close(
        resumed_last["rng_state"]["torch"], reference_last["rng_state"]["torch"], rtol=0, atol=0
    )
    for name, tensor in resumed_last["model_state"].items():
        torch.testing.assert_close(tensor, reference_last["model_state"][name], rtol=0, atol=0)

    resumed_log = [{k: v for k, v in row.items() if k != "timestamp"}
                   for row in read_jsonl(resumed_dir / "train_log.jsonl")]
    reference_log = [{k: v for k, v in row.items() if k != "timestamp"}
                     for row in read_jsonl(reference_dir / "train_log.jsonl")]
    assert resumed_log == reference_log
    for name in (
        "config.yaml", "environment.json", "train_log.jsonl", "checkpoint_selection.json",
        "best.pt", "last.pt", "completed.json", "train_features.npz", "val_features.npz",
    ):
        assert (resumed_dir / name).is_file()
    with np.load(resumed_dir / "train_features.npz") as features:
        assert {"sample_id", "intent_id", "domain_id", "cls", "logits"}.issubset(features.files)
        assert features["cls"].shape == (6, 24)
        assert features["logits"].shape == (6, 3)
    assert json.loads((resumed_dir / "completed.json").read_text())["status"] == "completed"


def test_last_accumulation_window_uses_actual_sample_count() -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = AdamW([parameter], lr=0.0)
    scheduler = LambdaLR(optimizer, lambda _step: 1.0)
    window = [
        ({"intent_id": torch.zeros(2)}, {"loss": parameter * 2.0}),
        ({"intent_id": torch.zeros(1)}, {"loss": parameter * 3.0}),
    ]
    captured = []
    parameter.register_hook(lambda gradient: captured.append(gradient.clone()))
    _flush_window(window, None, optimizer, scheduler, max_grad_norm=100.0)
    torch.testing.assert_close(sum(captured), torch.tensor(7.0 / 3.0))
    assert parameter.grad is None
