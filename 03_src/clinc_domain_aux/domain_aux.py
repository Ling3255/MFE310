"""M05: Shared BERT representations with intent and domain auxiliary heads."""

from __future__ import annotations

import os
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertConfig, BertModel

from .common import hash_state_dict, resolve_path, set_seed, sha256_file


class DualHeadBert(nn.Module):
    """Use one raw CLS representation and one dropout output for both heads."""

    def __init__(
        self,
        encoder: BertModel,
        num_intents: int,
        num_domains: int,
        lambda_domain: float = 0.0,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        if num_intents < 2 or num_domains < 2:
            raise ValueError("num_intents and num_domains must both be at least 2")
        if lambda_domain < 0:
            raise ValueError("lambda_domain must be non-negative")
        self.encoder = encoder
        self.num_intents = int(num_intents)
        self.num_domains = int(num_domains)
        self.lambda_domain = float(lambda_domain)
        probability = encoder.config.hidden_dropout_prob if dropout is None else float(dropout)
        self.shared_dropout = nn.Dropout(probability)
        hidden_size = int(encoder.config.hidden_size)
        self.intent_head = nn.Linear(hidden_size, self.num_intents)
        self.domain_head = nn.Linear(hidden_size, self.num_domains)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
        intent_labels: torch.Tensor | None = None,
        domain_labels: torch.Tensor | None = None,
        intent_id: torch.Tensor | None = None,
        domain_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if labels is not None:
            if isinstance(labels, Mapping):
                intent_labels = labels.get("intent_id", labels.get("intent_labels", intent_labels))
                domain_labels = labels.get("domain_id", labels.get("domain_labels", domain_labels))
            else:
                intent_labels = labels
        if intent_id is not None:
            intent_labels = intent_id
        if domain_id is not None:
            domain_labels = domain_id

        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = encoded.last_hidden_state[:, 0, :]
        shared = self.shared_dropout(cls)
        intent_logits = self.intent_head(shared)
        domain_logits = self.domain_head(shared)

        loss_intent = None
        loss_domain = None
        loss = None
        if intent_labels is not None:
            loss_intent = F.cross_entropy(intent_logits, intent_labels.long())
            if domain_labels is not None:
                domain_source = domain_logits.detach() if self.lambda_domain == 0 else domain_logits
                loss_domain = F.cross_entropy(domain_source, domain_labels.long(), ignore_index=-1)
            elif self.lambda_domain > 0:
                raise ValueError("domain labels are required when lambda_domain is positive")
            loss = loss_intent
            if self.lambda_domain > 0:
                if loss_domain is None:
                    raise ValueError("domain labels are required when lambda_domain is positive")
                loss = loss + self.lambda_domain * loss_domain

        return {
            "intent_logits": intent_logits,
            "domain_logits": domain_logits,
            "cls": cls,
            "loss_intent": loss_intent,
            "loss_domain": loss_domain,
            "loss": loss,
        }


def _model_config(config: Any) -> Any:
    if isinstance(config, Mapping) and "model" in config:
        return config["model"]
    return config


def resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_model(
    config: Any,
    num_intents: int,
    num_domains: int,
    lambda_domain: float,
    device: str | torch.device | None = "cpu",
) -> DualHeadBert:
    """Build from a BertConfig locally or from a configured pretrained checkpoint."""
    model_config = _model_config(config)
    if isinstance(model_config, BertConfig):
        encoder = BertModel(model_config)
        dropout = None
    elif isinstance(model_config, str):
        encoder = BertModel.from_pretrained(model_config)
        dropout = None
    elif isinstance(model_config, Mapping):
        encoder_values = model_config.get("encoder_config", model_config.get("bert_config"))
        if encoder_values is None and "name" not in model_config and "hidden_size" in model_config:
            encoder_values = model_config
        if encoder_values is not None:
            if isinstance(encoder_values, BertConfig):
                bert_config = encoder_values
            else:
                bert_config = BertConfig(**dict(encoder_values))
            encoder = BertModel(bert_config)
        elif "name" in model_config:
            arguments: dict[str, Any] = {}
            for key in ("revision", "cache_dir", "local_files_only", "trust_remote_code"):
                if key in model_config:
                    arguments[key] = model_config[key]
            encoder = BertModel.from_pretrained(model_config["name"], **arguments)
        else:
            raise ValueError("model config requires name or encoder_config")
        dropout = model_config.get("dropout")
    else:
        raise TypeError("config must be a model mapping, full configuration, BertConfig, or model name")

    model = DualHeadBert(
        encoder=encoder,
        num_intents=num_intents,
        num_domains=num_domains,
        lambda_domain=lambda_domain,
        dropout=dropout,
    )
    return model.to(resolve_device(device))


def _torch_save(value: Any, path: str | os.PathLike[str]) -> Path:
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _torch_load(path: str | os.PathLike[str], device: str | torch.device | None = "cpu") -> Any:
    try:
        return torch.load(resolve_path(path), map_location=resolve_device(device), weights_only=False)
    except TypeError:
        return torch.load(resolve_path(path), map_location=resolve_device(device))


def save_model(
    model: DualHeadBert,
    path: str | os.PathLike[str],
    metadata: Mapping[str, Any] | None = None,
) -> str:
    state = model.state_dict()
    state_hash = hash_state_dict(state)
    _torch_save(
        {
            "format_version": 1,
            "state_dict": state,
            "state_hash": state_hash,
            "metadata": dict(metadata or {}),
        },
        path,
    )
    return state_hash


def load_model(
    model: DualHeadBert,
    path: str | os.PathLike[str],
    device: str | torch.device | None = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint = _torch_load(path, device)
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    expected_hash = checkpoint.get("state_hash") if isinstance(checkpoint, Mapping) else None
    actual_hash = hash_state_dict(state)
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError(f"Model state hash mismatch: {path}")
    incompatible = model.load_state_dict(state, strict=strict)
    model.to(resolve_device(device))
    return {
        "state_hash": actual_hash,
        "metadata": checkpoint.get("metadata", {}) if isinstance(checkpoint, Mapping) else {},
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def save_common_initialization(
    model: DualHeadBert,
    path: str | os.PathLike[str],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state_hash = save_model(model, path, metadata=metadata)
    return {
        "path": str(resolve_path(path)),
        "initialization_hash": state_hash,
        "file_hash": sha256_file(path),
    }


def create_common_initialization(
    config: Any,
    num_intents: int,
    num_domains: int,
    seed: int,
    path: str | os.PathLike[str],
    device: str | torch.device | None = "cpu",
) -> dict[str, Any]:
    """Create the sole initialization for one seed; callers reuse the saved file."""
    set_seed(seed)
    model = build_model(config, num_intents, num_domains, 0.0, device)
    return save_common_initialization(model, path, metadata={"seed": int(seed)})


def load_common_initialization(
    model: DualHeadBert,
    path: str | os.PathLike[str],
    device: str | torch.device | None = "cpu",
) -> dict[str, Any]:
    loaded = load_model(model, path, device=device, strict=True)
    return {
        "path": str(resolve_path(path)),
        "initialization_hash": loaded["state_hash"],
        "file_hash": sha256_file(path),
        "metadata": loaded["metadata"],
    }
