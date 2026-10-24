"""M11: Frozen single-sentence intent, domain, and OOS inference."""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from .common import load_config, resolve_path
from .detectors import MahalanobisDetector
from .domain_aux import build_model, load_model, resolve_device
from .evaluation import msp_scores


class FrozenInference:
    """Load one frozen checkpoint and detector for FP32 single-sentence inference.

    Model and tokenizer use the exact revision in ``config['model']``. Local
    files are required by default so missing artifacts fail without networking.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | str | os.PathLike[str],
        run_dir: str | os.PathLike[str],
        detector_path: str | os.PathLike[str],
        tau_msp: float,
        tau_mahalanobis: float,
        label_to_id: Mapping[str, Any],
        domain_to_id: Mapping[str, Any],
        intent_to_domain: Mapping[str, str],
        *,
        device: str | torch.device | None = "cpu",
        local_files_only: bool = True,
    ) -> None:
        self.config = load_config(config) if isinstance(config, (str, os.PathLike)) else copy.deepcopy(dict(config))
        model_config = self.config.get("model")
        if not isinstance(model_config, Mapping):
            raise ValueError("config must contain a model mapping")
        if not model_config.get("name") or not model_config.get("revision"):
            raise ValueError("model name and a fixed revision are required for frozen inference")
        self.label_to_id = self._validate_id_mapping(label_to_id, "label_to_id")
        self.domain_to_id = self._validate_id_mapping(domain_to_id, "domain_to_id")
        if set(intent_to_domain) != set(self.label_to_id):
            raise ValueError("intent_to_domain must cover exactly the mapped intents")
        if any(domain not in self.domain_to_id for domain in intent_to_domain.values()):
            raise ValueError("intent_to_domain contains an unmapped domain")
        self.intent_to_domain = {str(intent): str(domain) for intent, domain in intent_to_domain.items()}
        self.id_to_intent = {identifier: intent for intent, identifier in self.label_to_id.items()}
        self.id_to_domain = {identifier: domain for domain, identifier in self.domain_to_id.items()}
        self.tau_msp = self._threshold(tau_msp, "tau_msp")
        self.tau_mahalanobis = self._threshold(tau_mahalanobis, "tau_mahalanobis")
        self.device = resolve_device(device)
        self.max_length = int(model_config.get("max_length", self.config.get("training", {}).get("max_length", 64)))
        if self.max_length < 2:
            raise ValueError("model max_length must be at least 2")

        tokenizer_arguments: dict[str, Any] = {
            "revision": model_config["revision"],
            "local_files_only": bool(local_files_only),
        }
        for key in ("cache_dir", "trust_remote_code"):
            if key in model_config:
                tokenizer_arguments[key] = model_config[key]
        self.tokenizer = AutoTokenizer.from_pretrained(model_config["name"], **tokenizer_arguments)

        frozen_model_config = copy.deepcopy(dict(model_config))
        frozen_model_config["local_files_only"] = bool(local_files_only)
        num_intents = int(frozen_model_config.get("num_intents", len(self.label_to_id)))
        num_domains = int(frozen_model_config.get("num_domains", len(self.domain_to_id)))
        if num_intents != len(self.label_to_id) or num_domains != len(self.domain_to_id):
            raise ValueError("model head sizes disagree with the label mappings")
        lambda_domain = float(self.config.get("lambda_domain", 0.0))
        self.model = build_model(
            frozen_model_config,
            num_intents,
            num_domains,
            lambda_domain,
            self.device,
        )
        checkpoint_path = resolve_path(run_dir) / "best.pt"
        self.checkpoint = load_model(self.model, checkpoint_path, device=self.device, strict=True)
        self.model.float()
        self.model.eval()
        self.model.requires_grad_(False)
        self.detector = MahalanobisDetector.load(resolve_path(detector_path))
        if self.detector.num_intents != num_intents:
            raise ValueError("detector num_intents disagrees with the model")
        if self.detector.n_features_in_ != int(self.model.encoder.config.hidden_size):
            raise ValueError("detector feature dimension disagrees with the model encoder")
        self.checkpoint_path = checkpoint_path
        self.detector_path = resolve_path(detector_path)
        self.dtype = torch.float32

    @staticmethod
    def _validate_id_mapping(mapping: Mapping[str, Any], name: str) -> dict[str, int]:
        if not isinstance(mapping, Mapping) or not mapping:
            raise ValueError(f"{name} must be a non-empty mapping")
        converted = {str(label): int(identifier) for label, identifier in mapping.items()}
        identifiers = list(converted.values())
        if len(set(converted)) != len(mapping):
            raise ValueError(f"{name} labels must remain unique after string conversion")
        if len(set(identifiers)) != len(identifiers) or set(identifiers) != set(range(len(identifiers))):
            raise ValueError(f"{name} IDs must be unique and contiguous from zero")
        return converted

    @staticmethod
    def _threshold(value: Any, name: str) -> float:
        threshold = float(value)
        if not np.isfinite(threshold):
            raise ValueError(f"{name} must be finite")
        return threshold

    @torch.inference_mode()
    def predict(self, text: str, *, include_domain_head_score: bool = False) -> dict[str, Any]:
        """Run one forward pass; both acceptance decisions use the intent head."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        output = self.model(**inputs)
        intent_logits = output["intent_logits"]
        cls = output["cls"]
        if intent_logits is None or cls is None:
            raise ValueError("model did not return intent logits and CLS features")
        intent_id = int(intent_logits.argmax(dim=-1).item())
        intent = self.id_to_intent[intent_id]
        domain = self.intent_to_domain[intent]
        intent_score = float(msp_scores(intent_logits.detach().cpu().numpy(), "logits")[0])
        mahalanobis_score = float(self.detector.mahalanobis_scores(cls.detach().cpu().numpy())[0])
        result: dict[str, Any] = {
            "text": text,
            "intent_id": intent_id,
            "intent": intent,
            "domain_id": self.domain_to_id[domain],
            "domain": domain,
            "msp_score": intent_score,
            "mahalanobis_score": mahalanobis_score,
            "accepted_msp": bool(intent_score < self.tau_msp),
            "accepted_mahalanobis": bool(mahalanobis_score < self.tau_mahalanobis),
            "threshold_rule": "reject iff OOS score >= tau; accept iff score < tau",
        }
        if include_domain_head_score:
            domain_logits = output["domain_logits"]
            if domain_logits is None:
                raise ValueError("model did not return domain logits")
            result["domain_head_score"] = float(msp_scores(domain_logits.detach().cpu().numpy(), "logits")[0])
        return result
