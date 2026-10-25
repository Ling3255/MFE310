"""P3-M2: Deterministic input perturbations for the robustness test R3.

Character noise imitates typing and ASR-like errors: each alphabetic character is, with
probability ``rate``, swapped with its right neighbour, deleted, or followed by a random
lowercase letter. The generator is seeded per sample, so every model sees identical inputs.
"""

from __future__ import annotations

import re
import string
from typing import Any, Mapping, Sequence

import numpy as np

_PUNCTUATION = re.compile(f"[{re.escape(string.punctuation)}]")


def char_noise(text: str, rate: float, rng: np.random.Generator) -> str:
    """Apply swap/delete/insert edits to alphabetic characters with probability ``rate``."""
    if not 0 <= rate <= 1:
        raise ValueError("rate must be in [0, 1]")
    chars = list(text)
    output: list[str] = []
    index = 0
    while index < len(chars):
        char = chars[index]
        if char.isalpha() and rng.random() < rate:
            operation = rng.integers(3)
            if operation == 0 and index + 1 < len(chars):  # swap with the next character
                output.extend([chars[index + 1], char])
                index += 2
                continue
            if operation == 1:  # delete
                index += 1
                continue
            output.extend([char, string.ascii_lowercase[rng.integers(26)]])  # insert after
            index += 1
            continue
        output.append(char)
        index += 1
    noisy = "".join(output).strip()
    return noisy if noisy else text  # never produce an empty utterance


def strip_punctuation(text: str) -> str:
    """Remove punctuation and collapse whitespace (ASR transcripts carry no punctuation)."""
    stripped = " ".join(_PUNCTUATION.sub(" ", text).split())
    return stripped if stripped else text


def make_variants(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    noise_rates: Sequence[float],
    punctuation: bool,
) -> list[dict[str, Any]]:
    """Return perturbed copies of ``records``; each copy's sample_id gets a ``#variant`` suffix."""
    variants = []
    for position, record in enumerate(records):
        for rate in noise_rates:
            rng = np.random.default_rng([int(seed), int(round(rate * 1000)), position])
            variants.append({**record, "sample_id": f"{record['sample_id']}#noise{int(round(rate * 100)):02d}",
                             "text": char_noise(record["text"], rate, rng)})
        if punctuation:
            variants.append({**record, "sample_id": f"{record['sample_id']}#nopunct",
                             "text": strip_punctuation(record["text"])})
    return variants


def variant_of(sample_id: str) -> str:
    """'clean' for original rows, otherwise the suffix after '#'."""
    return sample_id.split("#", 1)[1] if "#" in sample_id else "clean"
