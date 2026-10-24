"""M01: Configuration, paths, deterministic seeds, hashing, and run records."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(path: str | os.PathLike[str], base_dir: str | os.PathLike[str] | None = None) -> Path:
    value = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if value.is_absolute():
        return value.resolve()
    return (Path(base_dir) if base_dir is not None else PROJECT_ROOT).joinpath(value).resolve()


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    return config


def save_yaml(path: str | os.PathLike[str], value: Mapping[str, Any]) -> None:
    _atomic_write(path, yaml.safe_dump(dict(value), sort_keys=False, allow_unicode=True))


def read_json(path: str | os.PathLike[str]) -> Any:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | os.PathLike[str], value: Any) -> None:
    _atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    _atomic_write(path, text)


def append_jsonl(path: str | os.PathLike[str], row: Mapping[str, Any]) -> None:
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")


def read_csv(path: str | os.PathLike[str]) -> list[dict[str, str]]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: str | os.PathLike[str],
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    materialized = list(rows)
    if fieldnames is None:
        if not materialized:
            raise ValueError("fieldnames are required when writing an empty CSV")
        fieldnames = list(materialized[0])
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        writer.writerows(materialized)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with resolve_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256_bytes(encoded)


def hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return stable_hash({"dtype": str(array.dtype), "shape": list(array.shape), "data": sha256_bytes(array.tobytes())})


def hash_state_dict(state_dict: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def lambda_token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def run_name(lambda_domain: float, seed: int) -> str:
    return f"ce_seed{seed}" if float(lambda_domain) == 0 else f"dom_lam{lambda_token(lambda_domain)}_seed{seed}"


def parse_run_name(name: str) -> tuple[float, int]:
    match = re.fullmatch(r"(?:ce|dom_lam([0-9p]+))_seed([0-9]+)", name)
    if not match:
        raise ValueError(f"Invalid run name: {name}")
    token, seed = match.groups()
    return (0.0 if token is None else float(token.replace("p", "."))), int(seed)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def environment_info() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "scikit-learn", "numpy", "pandas", "scipy", "matplotlib", "seaborn", "PyYAML", "pytest", "joblib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    info: dict[str, Any] = {
        "created_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }
    try:
        import torch

        info["torch"] = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        }
    except ImportError:
        info["torch"] = None
    return info


def git_commit(path: str | os.PathLike[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(resolve_path(path)), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def make_run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}_{stamp}_{os.getpid()}"


def _atomic_write(path: str | os.PathLike[str], text: str) -> None:
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
