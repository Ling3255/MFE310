"""M14: Supervised contrastive loss and class-balanced batch sampling (phase 3).

The loss follows Khosla et al. (2020) and its fine-tuning use in Gunel et al. (2021):
for each anchor with at least one same-label partner in the batch,

    L_i = -1/|P(i)| * sum_{p in P(i)} log( exp(z_i.z_p / tau) / sum_{a != i} exp(z_i.z_a / tau) )

averaged over anchors that have positives. Features must already be L2-normalized.
"""

from __future__ import annotations

import math
from typing import Iterator, Sequence

import torch


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return the mean SupCon loss over anchors that have at least one positive.

    ``features`` is (batch, dim) and L2-normalized; ``labels`` is (batch,). A batch without
    any positive pair returns a zero that keeps the autograd graph connected.
    """
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("features must be (batch, dim) and labels (batch,)")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch = len(labels)
    logits = features @ features.T / temperature
    self_mask = torch.eye(batch, dtype=torch.bool, device=features.device)
    logits = logits.masked_fill(self_mask, float("-inf"))
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()  # numerical stability only
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positives = (labels[:, None] == labels[None, :]) & ~self_mask
    positive_count = positives.sum(dim=1)
    anchors = positive_count > 0
    if not anchors.any():
        return features.sum() * 0.0
    mean_log_prob = log_prob.masked_fill(~positives, 0.0).sum(dim=1)[anchors] / positive_count[anchors]
    return -mean_log_prob.mean()


class ClassBalancedBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Yield batches of ``classes_per_batch`` groups, each holding up to ``samples_per_class``
    examples of one label, so that SupCon sees same-label positives in every batch.

    Each epoch visits every example exactly once: every label's shuffled examples are cut into
    groups, groups are dealt round by round (every label once per round, label order shuffled),
    and the resulting sequence is chunked into batches. All randomness comes from ``generator``,
    so the order is reproducible and its state can be checkpointed like a RandomSampler.
    """

    def __init__(
        self,
        labels: Sequence[int],
        classes_per_batch: int,
        samples_per_class: int,
        generator: torch.Generator,
    ) -> None:
        if classes_per_batch < 1 or samples_per_class < 2:
            raise ValueError("need classes_per_batch >= 1 and samples_per_class >= 2")
        self.labels = [int(label) for label in labels]
        self.classes_per_batch = int(classes_per_batch)
        self.samples_per_class = int(samples_per_class)
        self.generator = generator
        self.by_label: dict[int, list[int]] = {}
        for index, label in enumerate(self.labels):
            self.by_label.setdefault(label, []).append(index)
        groups = sum(math.ceil(len(indices) / self.samples_per_class) for indices in self.by_label.values())
        self._length = math.ceil(groups / self.classes_per_batch)

    def __len__(self) -> int:
        return self._length

    def _permute(self, items: list[int]) -> list[int]:
        order = torch.randperm(len(items), generator=self.generator).tolist()
        return [items[position] for position in order]

    def __iter__(self) -> Iterator[list[int]]:
        label_groups = {}
        for label in sorted(self.by_label):
            shuffled = self._permute(self.by_label[label])
            label_groups[label] = [
                shuffled[start:start + self.samples_per_class]
                for start in range(0, len(shuffled), self.samples_per_class)
            ]
        sequence: list[list[int]] = []
        for round_index in range(max(len(groups) for groups in label_groups.values())):
            active = [label for label in sorted(label_groups) if round_index < len(label_groups[label])]
            for label in self._permute(active):
                sequence.append(label_groups[label][round_index])
        for start in range(0, len(sequence), self.classes_per_batch):
            yield [index for group in sequence[start:start + self.classes_per_batch] for index in group]
