"""M06: Paired initialization, training, checkpoint selection, and feature export."""

from __future__ import annotations

import copy
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import Dataset

from .common import (
    append_jsonl,
    environment_info,
    hash_array,
    hash_state_dict,
    load_config,
    read_json,
    resolve_path,
    save_yaml,
    set_seed,
    sha256_file,
    stable_hash,
    utc_now,
    write_json,
)
from .contrastive import ClassBalancedBatchSampler, DomainAwareBatchSampler, supervised_contrastive_loss
from .domain_aux import (
    DualHeadBert,
    build_model,
    create_common_initialization,
    load_common_initialization,
    resolve_device,
)


def _value(sample: Any, key: str, default: Any = None) -> Any:
    if isinstance(sample, Mapping):
        return sample.get(key, default)
    return getattr(sample, key, default)


def _plain_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    return value


def _dataset_hash(dataset: Dataset) -> str:
    rows = []
    for index in range(len(dataset)):
        sample = dataset[index]
        row = {
            key: _plain_scalar(_value(sample, key))
            for key in ("sample_id", "text", "intent_id", "domain_id")
        }
        if row["text"] is None:
            row["input_ids"] = _plain_scalar(_value(sample, "input_ids"))
        rows.append(row)
    return stable_hash(rows)


def _data_hashes(datasets: Mapping[str, Dataset]) -> dict[str, Any]:
    splits = {split: _dataset_hash(dataset) for split, dataset in sorted(datasets.items())}
    return {"splits": splits, "combined": stable_hash(splits)}


def _collate(samples: Sequence[Any], tokenizer: Any, max_length: int) -> dict[str, Any]:
    texts = [_value(sample, "text") for sample in samples]
    if tokenizer is not None and all(text is not None for text in texts):
        encoded = tokenizer(
            list(texts),
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch = {key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)}
    else:
        input_values = [_value(sample, "input_ids") for sample in samples]
        if any(value is None for value in input_values):
            raise ValueError("Samples need text with a tokenizer, or pre-tokenized input_ids")
        input_values = [torch.as_tensor(value, dtype=torch.long) for value in input_values]
        mask_values = [_value(sample, "attention_mask") for sample in samples]
        type_values = [_value(sample, "token_type_ids") for sample in samples]
        width = max(int(value.numel()) for value in input_values)
        input_ids = torch.zeros(len(samples), width, dtype=torch.long)
        attention_mask = torch.zeros(len(samples), width, dtype=torch.long)
        token_type_ids = torch.zeros(len(samples), width, dtype=torch.long)
        for index, value in enumerate(input_values):
            input_ids[index, : value.numel()] = value
            if mask_values[index] is None:
                attention_mask[index, : value.numel()] = 1
            else:
                mask = torch.as_tensor(mask_values[index], dtype=torch.long)
                attention_mask[index, : mask.numel()] = mask
            if type_values[index] is not None:
                types = torch.as_tensor(type_values[index], dtype=torch.long)
                token_type_ids[index, : types.numel()] = types
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    batch["intent_id"] = torch.as_tensor(
        [int(_plain_scalar(_value(sample, "intent_id"))) for sample in samples], dtype=torch.long
    )
    batch["domain_id"] = torch.as_tensor(
        [int(_plain_scalar(_value(sample, "domain_id"))) for sample in samples], dtype=torch.long
    )
    sample_ids = [_value(sample, "sample_id") for sample in samples]
    if any(sample_id is None for sample_id in sample_ids):
        raise ValueError("Every sample needs a stable sample_id")
    batch["sample_id"] = [str(sample_id) for sample_id in sample_ids]
    return batch


def _make_loader(
    dataset: Dataset,
    tokenizer: Any,
    training: Mapping[str, Any],
    device: torch.device,
    shuffle: bool,
    seed: int,
    sampler_generator: torch.Generator | None = None,
) -> tuple[DataLoader, torch.Generator]:
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(seed) + 1_000_003)
    common = {
        "num_workers": int(training.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
        "generator": loader_generator,
        "collate_fn": lambda samples: _collate(samples, tokenizer, int(training.get("max_length", 64))),
    }
    if shuffle and training.get("sampler") == "class_balanced":
        # Phase 3: groups of same-intent examples per batch so SupCon always has positives.
        labels = [int(_plain_scalar(_value(dataset[index], "intent_id"))) for index in range(len(dataset))]
        batch_sampler = ClassBalancedBatchSampler(
            labels, int(training["classes_per_batch"]), int(training["samples_per_class"]), sampler_generator,
        )
        return DataLoader(dataset, batch_sampler=batch_sampler, **common), loader_generator
    if shuffle and training.get("sampler") == "domain_aware":
        # Phase 4: batches of domain blocks, raising same-domain hard negatives for SupCon.
        labels = [int(_plain_scalar(_value(dataset[index], "intent_id"))) for index in range(len(dataset))]
        domains = [int(_plain_scalar(_value(dataset[index], "domain_id"))) for index in range(len(dataset))]
        batch_sampler = DomainAwareBatchSampler(
            labels, domains, int(training["domains_per_batch"]), int(training["intents_per_domain"]),
            int(training["samples_per_intent"]), sampler_generator,
        )
        return DataLoader(dataset, batch_sampler=batch_sampler, **common), loader_generator
    sampler = RandomSampler(dataset, generator=sampler_generator) if shuffle else None
    loader = DataLoader(
        dataset,
        batch_size=int(training.get("micro_batch_size", 16)),
        shuffle=False,
        sampler=sampler,
        **common,
    )
    return loader, loader_generator


def _model_inputs(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device)
        for key in ("input_ids", "attention_mask", "token_type_ids")
        if key in batch
    }


def _optimizer(model: DualHeadBert, training: Mapping[str, Any], lambda_domain: float) -> AdamW:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if lambda_domain == 0 and name.startswith("domain_head."):
            continue
        exempt = parameter.ndim == 1 or name.endswith(".bias") or "LayerNorm" in name
        (no_decay if exempt else decay).append(parameter)
    if not decay and not no_decay:
        raise ValueError("The model has no trainable parameters")
    weight_decay = float(training.get("weight_decay", 0.01))
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(training.get("learning_rate", 2e-5)),
    )


def _scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))

    return LambdaLR(optimizer, multiplier)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _flush_window(
    window: list[tuple[Mapping[str, Any], Mapping[str, torch.Tensor | None]]],
    model: DualHeadBert,
    optimizer: AdamW,
    scheduler: LambdaLR,
    max_grad_norm: float,
) -> None:
    sample_count = sum(int(batch["intent_id"].numel()) for batch, _ in window)
    if sample_count <= 0:
        raise ValueError("Cannot optimize an empty accumulation window")
    for batch, output in window:
        if output["loss"] is None:
            raise ValueError("Training requires intent labels")
        weight = float(batch["intent_id"].numel()) / float(sample_count)
        (output["loss"] * weight).backward()
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    clip_grad_norm_(parameters, max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    window.clear()


def _train_epoch(
    model: DualHeadBert,
    loader: DataLoader,
    device: torch.device,
    optimizer: AdamW,
    scheduler: LambdaLR,
    accumulation: int,
    max_grad_norm: float,
    scl: tuple[float, float] | None = None,
) -> tuple[dict[str, float | int], list[str]]:
    """Train one epoch; ``scl=(weight, temperature)`` uses (1-w)*CE + w*SupCon on L2-normalized CLS."""
    model.train()
    window: list[tuple[Mapping[str, Any], Mapping[str, torch.Tensor | None]]] = []
    order: list[str] = []
    intent_total = 0.0
    domain_total = 0.0
    combined_total = 0.0
    domain_count = 0
    scl_total = 0.0
    sample_total = 0
    updates = 0

    for batch_index, batch in enumerate(loader):
        inputs = _model_inputs(batch, device)
        output = model(
            **inputs,
            intent_id=batch["intent_id"].to(device),
            domain_id=batch["domain_id"].to(device),
        )
        if scl is not None:
            weight, temperature = scl
            loss_scl = supervised_contrastive_loss(
                F.normalize(output["cls"], dim=-1), batch["intent_id"].to(device), temperature,
            )
            output = {**output, "loss": (1.0 - weight) * output["loss_intent"] + weight * loss_scl}
            scl_total += float(loss_scl.detach().cpu()) * int(batch["intent_id"].numel())
        size = int(batch["intent_id"].numel())
        intent_loss = float(output["loss_intent"].detach().cpu())
        combined_loss = float(output["loss"].detach().cpu())
        intent_total += intent_loss * size
        combined_total += combined_loss * size
        if output["loss_domain"] is not None:
            domain_total += float(output["loss_domain"].detach().cpu()) * size
            domain_count += size
        sample_total += size
        order.extend(str(value) for value in batch["sample_id"])
        window.append((batch, output))
        if len(window) == accumulation or batch_index == len(loader) - 1:
            _flush_window(window, model, optimizer, scheduler, max_grad_norm)
            updates += 1

    metrics = {
        "train_loss_intent": intent_total / sample_total,
        "train_loss_domain": domain_total / domain_count if domain_count else None,
        "train_loss": combined_total / sample_total,
        "train_samples": sample_total,
        "optimizer_steps": updates,
    }
    if scl is not None:
        metrics["train_loss_scl"] = scl_total / sample_total
    return metrics, order


@torch.no_grad()
def extract_features(
    model: DualHeadBert,
    dataloader: DataLoader,
    device: str | torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Export raw CLS and logits in dataloader order without applying dropout."""
    target = next(model.parameters()).device if device is None else resolve_device(device)
    was_training = model.training
    model.eval()
    result: dict[str, list[Any]] = {
        "sample_id": [],
        "intent_id": [],
        "domain_id": [],
        "cls": [],
        "logits": [],
        "domain_logits": [],
    }
    for batch in dataloader:
        output = model(**_model_inputs(batch, target))
        result["sample_id"].extend(str(value) for value in batch["sample_id"])
        result["intent_id"].append(batch["intent_id"].numpy())
        result["domain_id"].append(batch["domain_id"].numpy())
        result["cls"].append(output["cls"].cpu().numpy())
        result["logits"].append(output["intent_logits"].cpu().numpy())
        result["domain_logits"].append(output["domain_logits"].cpu().numpy())
    if was_training:
        model.train()
    if not result["sample_id"]:
        return {
            "sample_id": np.asarray([], dtype=np.str_),
            "intent_id": np.asarray([], dtype=np.int64),
            "domain_id": np.asarray([], dtype=np.int64),
            "cls": np.empty((0, model.encoder.config.hidden_size), dtype=np.float32),
            "logits": np.empty((0, model.num_intents), dtype=np.float32),
            "domain_logits": np.empty((0, model.num_domains), dtype=np.float32),
        }
    return {
        "sample_id": np.asarray(result["sample_id"], dtype=np.str_),
        "intent_id": np.concatenate(result["intent_id"]).astype(np.int64),
        "domain_id": np.concatenate(result["domain_id"]).astype(np.int64),
        "cls": np.concatenate(result["cls"]),
        "logits": np.concatenate(result["logits"]),
        "domain_logits": np.concatenate(result["domain_logits"]),
    }


def _evaluate(
    model: DualHeadBert,
    loader: DataLoader,
    device: torch.device,
    num_intents: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    features = extract_features(model, loader, device)
    predictions = features["logits"].argmax(axis=1)
    metrics = {
        "val_macro_f1": float(f1_score(
            features["intent_id"], predictions, labels=np.arange(num_intents),
            average="macro", zero_division=0,
        )),
        "val_accuracy": float(accuracy_score(features["intent_id"], predictions)),
    }
    return metrics, features


def _checkpoint(
    model: DualHeadBert,
    optimizer: AdamW,
    scheduler: LambdaLR,
    epoch: int,
    best_epoch: int,
    best_macro_f1: float,
    epochs_without_improvement: int,
    rng: Mapping[str, Any],
    sampler_generator: torch.Generator,
    loader_generator: torch.Generator,
    config_hash: str,
    data_hash: str,
    initialization_hash: str,
    order_hash: str,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "state_hash": hash_state_dict(model.state_dict()),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "rng_state": copy.deepcopy(dict(rng)),
        "sampler_generator_state": sampler_generator.get_state(),
        "loader_generator_state": loader_generator.get_state(),
        "best_epoch": int(best_epoch),
        "best_macro_f1": float(best_macro_f1),
        "epochs_without_improvement": int(epochs_without_improvement),
        "config_hash": config_hash,
        "data_hash": data_hash,
        "initialization_hash": initialization_hash,
        "train_sample_order_hash": order_hash,
    }


def _save_checkpoint(path: Path, checkpoint: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(checkpoint), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _save_features(path: Path, features: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **features)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _initialization_path(config: Mapping[str, Any], run_dir: Path, seed: int) -> Path:
    training = config.get("training", {})
    configured = training.get("initialization_path")
    if configured is not None:
        return resolve_path(str(configured).format(seed=seed), run_dir)
    directory = resolve_path(training.get("initialization_dir", run_dir.parent), run_dir)
    return directory / f"common_initialization_seed{seed}.pt"


def train_run(
    config: Mapping[str, Any],
    lambda_domain: float,
    seed: int,
    datasets: Mapping[str, Dataset],
    tokenizer: Any,
    run_dir: str | os.PathLike[str],
    smoke: bool = False,
    stop_after_epoch: int | None = None,
) -> dict[str, Any]:
    """Train one lambda/seed pair, or resume its last checkpoint.

    ``stop_after_epoch`` is a smoke/debug interruption hook. Omit it on resume.
    """
    if lambda_domain < 0:
        raise ValueError("lambda_domain must be non-negative")
    if "train" not in datasets or "val" not in datasets:
        raise ValueError("datasets must contain train and val")
    training = config.get("training", {})
    model_config = config.get("model", {})
    effective_batch = int(training.get("effective_batch_size", 32))
    micro_batch = int(training.get("micro_batch_size", 16))
    accumulation = int(training.get("gradient_accumulation", 2))
    if effective_batch != micro_batch * accumulation:
        raise ValueError("effective_batch_size must equal micro_batch_size * gradient_accumulation")
    if training.get("amp", False):
        raise ValueError("This training module currently requires training.amp: false")
    epochs = int(training.get("epochs", 5))
    patience = int(training.get("patience", 2))
    if training.get("sampler") == "class_balanced":
        if int(training["classes_per_batch"]) * int(training["samples_per_class"]) != micro_batch:
            raise ValueError("classes_per_batch * samples_per_class must equal micro_batch_size")
    if training.get("sampler") == "domain_aware":
        size = int(training["domains_per_batch"]) * int(training["intents_per_domain"]) * int(training["samples_per_intent"])
        if size != micro_batch:
            raise ValueError("domains_per_batch * intents_per_domain * samples_per_intent must equal micro_batch_size")
    scl = None
    if training.get("supervised_contrastive") is not None:
        scl_values = training["supervised_contrastive"]
        scl = (float(scl_values["weight"]), float(scl_values["temperature"]))
        if not 0.0 <= scl[0] <= 1.0 or scl[1] <= 0 or float(lambda_domain) != 0.0:
            raise ValueError("SupCon needs weight in [0, 1], temperature > 0 and lambda_domain = 0")
    device = resolve_device(training.get("device", "auto"))
    run_path = resolve_path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    data_hashes = _data_hashes(datasets)
    config_hash = stable_hash({
        "config": config, "lambda_domain": float(lambda_domain),
        "seed": int(seed), "smoke": bool(smoke),
    })
    initialization_path = _initialization_path(config, run_path, int(seed))
    completed_path = run_path / "completed.json"
    if completed_path.exists():
        completed = read_json(completed_path)
        if completed.get("config_hash") != config_hash or completed.get("data_hash") != data_hashes["combined"]:
            raise ValueError("Completed run configuration or data hash does not match")
        if completed.get("initialization_file_hash") != sha256_file(initialization_path):
            raise ValueError("Completed run initialization file does not match")
        if completed.get("status") != "completed":
            raise ValueError("Completed marker has an invalid status")
        for name, expected_hash in completed.get("artifacts", {}).items():
            if sha256_file(run_path / name) != expected_hash:
                raise ValueError(f"Completed run artifact hash does not match: {run_path / name}")
        return completed

    run_config = copy.deepcopy(dict(config))
    run_config.update({
        "lambda_domain": float(lambda_domain),
        "seed": int(seed),
        "smoke": bool(smoke),
        "config_hash": config_hash,
        "data_hash": data_hashes["combined"],
        "data_hashes": data_hashes,
        "initialization_path": str(initialization_path),
    })
    environment_path = run_path / "environment.json"
    if not environment_path.exists():
        write_json(environment_path, {
            **environment_info(), "device": str(device), "smoke": bool(smoke),
        })

    if not initialization_path.exists():
        create_common_initialization(
            model_config, int(model_config["num_intents"]), int(model_config["num_domains"]),
            int(seed), initialization_path, device="cpu",
        )

    set_seed(int(seed), bool(training.get("deterministic", False)))
    model = build_model(
        model_config, int(model_config["num_intents"]), int(model_config["num_domains"]),
        float(lambda_domain), device,
    )
    initialization = load_common_initialization(model, initialization_path, device)
    initialization_hash = initialization["initialization_hash"]
    run_config["initialization_hash"] = initialization_hash
    run_config["initialization_file_hash"] = initialization["file_hash"]
    config_path = run_path / "config.yaml"
    if config_path.exists():
        saved_config = load_config(config_path)
        if saved_config.get("config_hash") != config_hash:
            raise ValueError("Resume configuration hash does not match config.yaml")
    else:
        save_yaml(config_path, run_config)

    set_seed(int(seed), bool(training.get("deterministic", False)))
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(int(seed))
    train_loader, train_loader_generator = _make_loader(
        datasets["train"], tokenizer, {**training, "max_length": model_config.get("max_length", 64)},
        device, True, int(seed), sampler_generator,
    )
    val_loader, _ = _make_loader(
        datasets["val"], tokenizer, {**training, "max_length": model_config.get("max_length", 64)},
        device, False, int(seed) + 17,
    )
    optimizer = _optimizer(model, training, float(lambda_domain))
    windows_per_epoch = math.ceil(len(train_loader) / accumulation)
    scheduler = _scheduler(
        optimizer,
        int(math.ceil(windows_per_epoch * epochs * float(training.get("warmup_ratio", 0.1)))),
        windows_per_epoch * epochs,
    )

    start_epoch = 0
    best_epoch = 0
    best_macro_f1 = -math.inf
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    order_records: list[dict[str, Any]] = []
    selection_path = run_path / "checkpoint_selection.json"
    order_path = run_path / "sample_order.json"
    last_path = run_path / "last.pt"
    resume = bool(training.get("resume", True))
    if resume and last_path.exists():
        checkpoint = _load_checkpoint(last_path, device)
        for key, expected in (
            ("config_hash", config_hash), ("data_hash", data_hashes["combined"]),
            ("initialization_hash", initialization_hash),
        ):
            if checkpoint.get(key) != expected:
                raise ValueError(f"Resume checkpoint {key} mismatch")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        _restore_rng(checkpoint["rng_state"])
        sampler_generator.set_state(checkpoint["sampler_generator_state"])
        train_loader_generator.set_state(checkpoint["loader_generator_state"])
        start_epoch = int(checkpoint["epoch"])
        best_epoch = int(checkpoint["best_epoch"])
        best_macro_f1 = float(checkpoint["best_macro_f1"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        if selection_path.exists():
            history = read_json(selection_path).get("history", [])
        if order_path.exists():
            order_records = read_json(order_path).get("epochs", [])
        if len(history) != start_epoch or len(order_records) != start_epoch:
            raise ValueError("Resume logs and last.pt describe different epochs")

    for epoch_index in range(start_epoch, epochs):
        train_metrics, sample_order = _train_epoch(
            model, train_loader, device, optimizer, scheduler, accumulation,
            float(training.get("max_grad_norm", 1.0)), scl,
        )
        val_metrics, _ = _evaluate(model, val_loader, device, int(model_config["num_intents"]))
        epoch = epoch_index + 1
        order_hash = hash_array(np.asarray(sample_order, dtype=np.str_))
        improved = val_metrics["val_macro_f1"] > best_macro_f1
        if improved:
            best_macro_f1 = val_metrics["val_macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        row = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "improved": improved,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_macro_f1,
            "train_sample_order_hash": order_hash,
            "timestamp": utc_now(),
        }
        append_jsonl(run_path / "train_log.jsonl", row)
        history.append({key: value for key, value in row.items() if key != "timestamp"})
        order_records.append({"epoch": epoch, "sample_ids": sample_order, "sha256": order_hash})
        write_json(order_path, {
            "seed": int(seed), "sample_order_hash": stable_hash(order_records), "epochs": order_records,
        })

        checkpoint = _checkpoint(
            model, optimizer, scheduler, epoch, best_epoch, best_macro_f1,
            epochs_without_improvement, _rng_state(), sampler_generator,
            train_loader_generator, config_hash, data_hashes["combined"],
            initialization_hash, order_hash,
        )
        _save_checkpoint(last_path, checkpoint)
        if improved:
            if training.get("best_checkpoint_weights_only", False):
                keep = ("format_version", "epoch", "model_state", "state_hash", "config_hash", "data_hash", "initialization_hash")
                _save_checkpoint(run_path / "best.pt", {key: checkpoint[key] for key in keep})
            else:
                _save_checkpoint(run_path / "best.pt", checkpoint)
        write_json(selection_path, {
            "metric": "val_macro_f1",
            "selection_rule": "strictly greater; ties keep the earlier epoch",
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_macro_f1,
            "patience": patience,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "completed": False,
        })
        if stop_after_epoch is not None and epoch >= stop_after_epoch and epoch < epochs:
            return {
                "status": "partial", "epoch": epoch, "best_epoch": best_epoch,
                "best_val_macro_f1": best_macro_f1, "run_dir": str(run_path),
            }
        if epochs_without_improvement >= patience:
            break

    best_checkpoint = _load_checkpoint(run_path / "best.pt", device)
    model.load_state_dict(best_checkpoint["model_state"])
    export_train_loader, _ = _make_loader(
        datasets["train"], tokenizer, {**training, "max_length": model_config.get("max_length", 64)},
        device, False, int(seed) + 31,
    )
    export_val_loader, _ = _make_loader(
        datasets["val"], tokenizer, {**training, "max_length": model_config.get("max_length", 64)},
        device, False, int(seed) + 47,
    )
    train_features = extract_features(model, export_train_loader, device)
    val_features = extract_features(model, export_val_loader, device)
    _save_features(run_path / "train_features.npz", train_features)
    _save_features(run_path / "val_features.npz", val_features)

    selection = read_json(selection_path)
    selection["completed"] = True
    write_json(selection_path, selection)
    epochs_completed = int(_load_checkpoint(last_path, device)["epoch"])
    keep_last = bool(training.get("keep_last_checkpoint", True))
    if not keep_last:
        last_path.unlink()  # resume state is no longer needed once the run is complete
    artifacts = {}
    for name in (
        "config.yaml", "environment.json", "train_log.jsonl", "checkpoint_selection.json",
        "sample_order.json", "best.pt", "last.pt", "train_features.npz", "val_features.npz",
    ):
        if name == "last.pt" and not keep_last:
            continue
        artifacts[name] = sha256_file(run_path / name)
    completed = {
        "status": "completed",
        "created_at": utc_now(),
        "lambda_domain": float(lambda_domain),
        "seed": int(seed),
        "smoke": bool(smoke),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "epochs_completed": epochs_completed,
        "config_hash": config_hash,
        "data_hash": data_hashes["combined"],
        "data_hashes": data_hashes,
        "initialization_hash": initialization_hash,
        "initialization_file_hash": sha256_file(initialization_path),
        "best_checkpoint_hash": hash_state_dict(best_checkpoint["model_state"]),
        "artifacts": artifacts,
        "run_dir": str(run_path),
    }
    write_json(completed_path, completed)
    return completed


def predict_records(
    config: Mapping[str, Any],
    run_dir: str | os.PathLike[str],
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    device: str | torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Load the selected checkpoint and export logits/CLS for any frozen split."""
    from .domain_aux import load_model

    run_path = resolve_path(run_dir)
    completed = read_json(run_path / "completed.json")
    if completed.get("status") != "completed" or completed.get("smoke", False):
        raise ValueError(f"Only completed formal runs can be evaluated: {run_path}")
    training = config.get("training", {})
    model_config = config.get("model", {})
    target = resolve_device(device or training.get("device", "auto"))
    model = build_model(
        model_config,
        int(model_config["num_intents"]),
        int(model_config["num_domains"]),
        float(completed["lambda_domain"]),
        target,
    )
    model.float()
    load_model(model, run_path / "best.pt", device=target, strict=True)
    loader, _ = _make_loader(
        records,
        tokenizer,
        {**training, "max_length": model_config.get("max_length", 64)},
        target,
        False,
        int(completed["seed"]),
    )
    return extract_features(model, loader, target)
