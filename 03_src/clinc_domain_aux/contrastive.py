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


class DomainAwareBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Phase 4: batches of ``domains_per_batch`` domain blocks, each block holding
    ``intents_per_domain`` groups of up to ``samples_per_intent`` same-intent examples.

    Compared with :class:`ClassBalancedBatchSampler` this raises the share of same-domain,
    different-intent negatives for SupCon. Per epoch (every example exactly once):

    1. each intent's shuffled examples are cut into groups of ``samples_per_intent``;
    2. within each domain, groups are dealt round by round (every intent of the domain once per round,
       order reshuffled each round) and the sequence is cut into blocks of ``intents_per_domain`` groups;
    3. each batch takes one block from each of the ``domains_per_batch`` domains with the most blocks
       left (ties broken at random), so the number of batches depends only on the block counts.
    """

    def __init__(
        self,
        labels: Sequence[int],
        domains: Sequence[int],
        domains_per_batch: int,
        intents_per_domain: int,
        samples_per_intent: int,
        generator: torch.Generator,
    ) -> None:
        if len(labels) != len(domains):
            raise ValueError("labels and domains must have the same length")
        if domains_per_batch < 1 or intents_per_domain < 1 or samples_per_intent < 2:
            raise ValueError("need domains_per_batch >= 1, intents_per_domain >= 1, samples_per_intent >= 2")
        self.domains_per_batch = int(domains_per_batch)
        self.intents_per_domain = int(intents_per_domain)
        self.samples_per_intent = int(samples_per_intent)
        self.generator = generator
        self.by_label: dict[int, list[int]] = {}
        self.domain_of: dict[int, int] = {}
        for index, (label, domain) in enumerate(zip(labels, domains)):
            label, domain = int(label), int(domain)
            if self.domain_of.setdefault(label, domain) != domain:
                raise ValueError(f"intent {label} appears in more than one domain")
            self.by_label.setdefault(label, []).append(index)
        self.labels_of: dict[int, list[int]] = {}
        for label, domain in sorted(self.domain_of.items()):
            self.labels_of.setdefault(domain, []).append(label)
        blocks = []
        for labels_in_domain in self.labels_of.values():
            groups = sum(math.ceil(len(self.by_label[label]) / self.samples_per_intent) for label in labels_in_domain)
            blocks.append(math.ceil(groups / self.intents_per_domain))
        self._length = self._count_batches(blocks)

    def _count_batches(self, blocks: list[int]) -> int:
        """Batches produced by the largest-remaining rule; tie-breaking never changes the count."""
        remaining, count = sorted(blocks, reverse=True), 0
        while remaining and remaining[0] > 0:
            for position in range(min(self.domains_per_batch, len(remaining))):
                if remaining[position] > 0:
                    remaining[position] -= 1
            remaining = sorted((value for value in remaining if value > 0), reverse=True)
            count += 1
        return count

    def __len__(self) -> int:
        return self._length

    def _permute(self, items: list) -> list:
        order = torch.randperm(len(items), generator=self.generator).tolist()
        return [items[position] for position in order]

    def __iter__(self) -> Iterator[list[int]]:
        groups = {}
        for label in sorted(self.by_label):
            shuffled = self._permute(self.by_label[label])
            groups[label] = [shuffled[start:start + self.samples_per_intent]
                             for start in range(0, len(shuffled), self.samples_per_intent)]
        remaining: dict[int, list[list[int]]] = {}
        for domain, labels_in_domain in sorted(self.labels_of.items()):
            sequence = []
            for round_index in range(max(len(groups[label]) for label in labels_in_domain)):
                active = [label for label in labels_in_domain if round_index < len(groups[label])]
                sequence.extend(groups[label][round_index] for label in self._permute(active))
            remaining[domain] = [
                [index for group in sequence[start:start + self.intents_per_domain] for index in group]
                for start in range(0, len(sequence), self.intents_per_domain)
            ]
        while any(remaining.values()):
            candidates = [domain for domain in sorted(remaining) if remaining[domain]]
            priority = torch.randperm(len(candidates), generator=self.generator).tolist()
            ordered = sorted(range(len(candidates)), key=lambda i: (-len(remaining[candidates[i]]), priority[i]))
            yield [index for i in ordered[:self.domains_per_batch] for index in remaining[candidates[i]].pop(0)]
