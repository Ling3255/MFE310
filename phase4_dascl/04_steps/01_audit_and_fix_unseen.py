"""P4-S01: Before any training — audit both samplers and fix the unseen evaluation pool U.

1. Sampler audit: for every phase-4 group and seed, replay epoch 1 of the intent-balanced (IB) and
   domain-aware (DA) samplers exactly as training will (same generator seed) and record per-anchor
   positives, same-domain negatives and other-domain negatives, plus coverage and batch counts.
2. Unseen pool: build U (phase-3 T3 minus the n25 subset) and record its size and content hash, so the
   evaluation step can verify it is unchanged.
No model is trained and no evaluation data is scored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT / p) for p in ("03_src", "phase3_scl/03_src", "phase4_dascl/03_src")]

from clinc_domain_aux.common import load_config, resolve_path, stable_hash, utc_now, write_csv, write_json
from clinc_domain_aux.contrastive import ClassBalancedBatchSampler, DomainAwareBatchSampler
from phase3_scl.study import load_split
from phase4_dascl.audit import batch_composition
from phase4_dascl.evaluation_sets import load_unseen

STUDY = "phase4_dascl/01_configs/study.yaml"


def make_sampler(training: dict, labels: list[int], domains: list[int], seed: int):
    generator = torch.Generator()
    generator.manual_seed(int(seed))  # identical to train_run's sampler_generator
    if training["sampler"] == "domain_aware":
        return DomainAwareBatchSampler(labels, domains, int(training["domains_per_batch"]),
                                       int(training["intents_per_domain"]), int(training["samples_per_intent"]), generator)
    return ClassBalancedBatchSampler(labels, int(training["classes_per_batch"]), int(training["samples_per_class"]), generator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default=STUDY)
    args = parser.parse_args()
    study = load_config(args.study)
    metadata = resolve_path("phase4_dascl/07_metadata")

    rows = []
    for group, spec in study["groups"].items():
        for arm in ("ib_ce", "da_ce"):  # the loss does not affect batching; *_scl arms share these orders
            config = load_config(spec["arms"][arm])
            train = load_split(config)["train"]
            labels = [row["intent_id"] for row in train]
            domains = [row["domain_id"] for row in train]
            for seed in spec["seeds"]:
                sampler = make_sampler(config["training"], labels, domains, int(seed))
                batches = list(sampler)
                if len(batches) != len(sampler):
                    raise ValueError(f"Sampler length mismatch for {group} {arm} seed {seed}")
                rows.append({"group": group, "sampler": arm.split("_")[0], "seed": int(seed),
                             **batch_composition(batches, labels, domains)})
    write_csv(metadata / "sampler_audit.csv", rows)

    unseen, origin, per_intent = load_unseen(study)
    write_json(metadata / "unseen_manifest.json", {
        "created_at": utc_now(), "count": len(unseen),
        "per_intent_min": min(per_intent.values()), "per_intent_max": max(per_intent.values()),
        "intents_below_max": {intent: n for intent, n in per_intent.items() if n < max(per_intent.values())},
        "content_hash": stable_hash([[row["sample_id"], row["text"], row["intent_id"]] for row in unseen]),
        "definition": "phase-3 T3 minus the n25 subset minus the phase-3 internal-test exclusions "
                      "(train texts duplicated in other splits or label-conflicting); never trained on by phase-4 models; "
                      "never evaluated",
        "origin": origin,
    })
    summary = {}
    for row in rows:
        summary.setdefault((row["group"], row["sampler"]), []).append(row["mean_same_domain_negatives_per_anchor"])
    for (group, sampler), values in sorted(summary.items()):
        print(f"{group} {sampler}: mean same-domain negatives per anchor {sum(values) / len(values):.2f}")
    print(f"Unseen pool U: {len(unseen)} utterances; audit and manifest in {metadata}")


if __name__ == "__main__":
    main()
