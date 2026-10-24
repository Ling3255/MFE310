"""M03: CLINC150 counts, duplicate, length, overlap, and confusion audits."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from .common import resolve_path, write_csv
from .data import SPLITS, Record, iter_records

WORD_RE = re.compile(r"[a-z]+")

CANDIDATE_CONFUSIONS = (
    ("bill_balance", "bill_due", "Both concern credit-card bills, but one asks for the amount owed and the other for due status or date."),
    ("credit_limit", "credit_limit_change", "Both concern card limits, distinguished by querying the current limit versus changing it."),
    ("calories", "nutrition_info", "Calorie requests are a specific subset of broader nutrition-information requests."),
    ("restaurant_reservation", "cancel_reservation", "Both concern restaurant reservations, with opposite creation and cancellation goals."),
    ("reminder", "reminder_update", "Both target reminders, distinguished by creating a reminder versus modifying an existing one."),
    ("todo_list", "todo_list_update", "Both target to-do lists, distinguished by querying or creating a list versus updating it."),
    ("pto_request", "pto_request_status", "Both concern PTO requests, distinguished by submitting a request versus checking its status."),
    ("timer", "alarm", "Both are time-based alerts, but a timer uses elapsed duration while an alarm uses a scheduled time."),
    ("balance", "rewards_balance", "The shared balance wording refers to different domain-specific balances: bank funds versus card rewards."),
    ("order", "order_checks", "The shared order wording distinguishes a purchase or delivery request from ordering bank checks."),
    ("shopping_list", "ingredients_list", "Both can request item lists for food preparation, but one is a shopping artifact and the other recipe ingredients."),
    ("date", "calendar", "Date queries and calendar requests both concern temporal information but have different assistant actions."),
    ("time", "timezone", "Time and timezone requests share temporal wording while asking for local time versus a zone conversion or difference."),
    ("change_language", "translate", "Both involve language, but one changes assistant settings and the other translates content."),
    ("book_hotel", "restaurant_reservation", "Both are reservation requests with similar date and party-size slots but different service domains."),
    ("meeting_schedule", "calendar_update", "Both can describe adding or checking meetings, distinguished by work scheduling versus a general calendar update."),
)


def run_audit(
    datasets: Mapping[str, Sequence[Record]],
    output_dir: str | Path,
    figures_dir: str | Path | None = None,
    tokenizer: Any = None,
    tokenizer_name: str | None = None,
    max_length: int = 64,
) -> dict[str, Any]:
    """Write all E1 tables and optionally the domain-overlap figure.

    Duplicate normalization only case-folds and collapses whitespace. Model
    input text is never modified. Candidate confusions use label semantics
    and train/val definitions only; no predictions or test examples are read.
    """
    tables = resolve_path(output_dir)
    figures = resolve_path(figures_dir) if figures_dir is not None else None
    counts_rows = _count_rows(datasets)
    duplicate_rows = _duplicate_rows(datasets)
    length_rows = _length_rows(datasets, tokenizer, tokenizer_name, max_length)
    jaccard_rows, jaccard_lookup = _jaccard_rows(datasets)
    confusion_rows = candidate_confusion_rows(datasets)
    write_csv(tables / "audit_counts.csv", counts_rows, fieldnames=("split", "domain", "intent", "count"))
    write_csv(
        tables / "duplicates.csv",
        duplicate_rows,
        fieldnames=(
            "normalization",
            "category",
            "split_a",
            "split_b",
            "unique_text_count",
            "sample_pair_count",
            "definition",
        ),
    )
    write_csv(tables / "length_stats.csv", length_rows, fieldnames=tuple(length_rows[0]))
    write_csv(tables / "domain_jaccard.csv", jaccard_rows, fieldnames=tuple(jaccard_rows[0]))
    write_csv(
        tables / "candidate_confusions.csv",
        confusion_rows,
        fieldnames=(
            "intent_a",
            "intent_b",
            "domain_a",
            "domain_b",
            "relation",
            "reason",
            "selection_basis",
        ),
    )
    if figures is not None:
        _plot_jaccard(jaccard_rows, figures / "fig_01_domain_overlap.png")
    return {
        "duplicate_policy": "Official splits and text remain unchanged; overlap removal is sensitivity analysis only.",
        "duplicates": duplicate_rows,
        "length_stats": length_rows,
        "candidate_confusion_count": len(confusion_rows),
        "candidate_confusion_basis": "Predeclared label semantics with train/val presence checks; no test predictions or examples used.",
        "word_rule": "Lowercase contiguous English letter sequences: [a-z]+ after case folding.",
        "jaccard_definition": "Train-only domain word sets; J(A,B)=|A intersection B|/|A union B|.",
        "domain_jaccard": jaccard_lookup,
        "limitations": [
            "English, crowdsourced, single-intent queries do not represent all languages, dialects, or multi-intent requests.",
            "Domain vocabulary overlap is descriptive lexical similarity, not a direct measure of semantic similarity.",
            "Examples were not exported in audit tables; review any future qualitative examples before public release.",
        ],
    }


def candidate_confusion_rows(datasets: Mapping[str, Sequence[Record]]) -> list[dict[str, Any]]:
    """Materialize the predeclared 16-pair list using only train and val."""
    available = {
        split: {row["intent"]: row["domain"] for row in datasets[split]}
        for split in ("train", "val")
    }
    rows = []
    for intent_a, intent_b, reason in CANDIDATE_CONFUSIONS:
        if any(intent not in available[split] for split in ("train", "val") for intent in (intent_a, intent_b)):
            continue
        domain_a = available["train"][intent_a]
        domain_b = available["train"][intent_b]
        rows.append(
            {
                "intent_a": intent_a,
                "intent_b": intent_b,
                "domain_a": domain_a,
                "domain_b": domain_b,
                "relation": "same_domain" if domain_a == domain_b else "cross_domain",
                "reason": reason,
                "selection_basis": "Predeclared from intent semantics; train/val definitions checked; no predictions used.",
            }
        )
    if len(available["train"]) == 150 and len(rows) != 16:
        raise AssertionError("The full CLINC150 audit must retain all 16 predeclared candidate pairs")
    return rows


def _count_rows(datasets: Mapping[str, Sequence[Record]]) -> list[dict[str, Any]]:
    rows = []
    for split in SPLITS:
        counts = Counter((row["domain"] or "", row["intent"]) for row in datasets[split])
        for (domain, intent), count in sorted(counts.items()):
            rows.append({"split": split, "domain": domain, "intent": intent, "count": count})
    return rows


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _duplicate_rows(datasets: Mapping[str, Sequence[Record]]) -> list[dict[str, Any]]:
    records = list(iter_records(datasets))
    output = []
    for normalization, key_function in (("exact", lambda row: row["text"]), ("normalized", lambda row: _normalized(row["text"]))):
        groups: dict[str, list[Record]] = defaultdict(list)
        for row in records:
            groups[key_function(row)].append(row)
        grouped = [group for group in groups.values() if len(group) > 1]
        output.append(_duplicate_summary(normalization, "all_duplicates", grouped))
        cross_groups = [group for group in grouped if len({row["split"] for row in group}) > 1]
        output.append(_duplicate_summary(normalization, "cross_split", cross_groups, cross_only=True))
        conflict_groups = [group for group in grouped if len({row["intent"] for row in group}) > 1]
        output.append(_duplicate_summary(normalization, "label_conflicts", conflict_groups, conflicts_only=True))
        for split_a, split_b in combinations(SPLITS, 2):
            pair_groups = [
                group
                for group in grouped
                if any(row["split"] == split_a for row in group)
                and any(row["split"] == split_b for row in group)
            ]
            pairs = sum(
                sum(1 for row in group if row["split"] == split_a)
                * sum(1 for row in group if row["split"] == split_b)
                for group in pair_groups
            )
            output.append(
                {
                    "normalization": normalization,
                    "category": "cross_split_pair",
                    "split_a": split_a,
                    "split_b": split_b,
                    "unique_text_count": len(pair_groups) if pairs else 0,
                    "sample_pair_count": pairs,
                    "definition": "Unique normalized keys and cross-split sample pairs for this split pair.",
                }
            )
    return output


def _duplicate_summary(
    normalization: str,
    category: str,
    groups: Sequence[Sequence[Record]],
    cross_only: bool = False,
    conflicts_only: bool = False,
) -> dict[str, Any]:
    pair_count = 0
    for group in groups:
        for row_a, row_b in combinations(group, 2):
            if cross_only and row_a["split"] == row_b["split"]:
                continue
            if conflicts_only and row_a["intent"] == row_b["intent"]:
                continue
            pair_count += 1
    return {
        "normalization": normalization,
        "category": category,
        "split_a": "",
        "split_b": "",
        "unique_text_count": len(groups),
        "sample_pair_count": pair_count,
        "definition": "unique_text_count counts duplicated text keys; sample_pair_count counts unordered sample pairs.",
    }


def _length_rows(
    datasets: Mapping[str, Sequence[Record]],
    tokenizer: Any,
    tokenizer_name: str | None,
    max_length: int,
) -> list[dict[str, Any]]:
    if max_length < 1:
        raise ValueError("max_length must be positive")
    rows = []
    for split in SPLITS:
        texts = [row["text"] for row in datasets[split]]
        words = [len(WORD_RE.findall(text.casefold())) for text in texts]
        row: dict[str, Any] = {
            "split": split,
            "count": len(texts),
            "max_length": max_length,
            "tokenizer": tokenizer_name or "",
            **_stats("word", words),
        }
        if tokenizer is not None and texts:
            encoded = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
            token_lengths = [len(ids) for ids in encoded["input_ids"]]
            truncated = sum(length > max_length for length in token_lengths)
            row.update(_stats("bert_token", token_lengths))
            row["truncated_count"] = truncated
            row["truncated_rate"] = truncated / len(texts)
        else:
            row.update(_stats("bert_token", []))
            row["truncated_count"] = ""
            row["truncated_rate"] = ""
        rows.append(row)
    return rows


def _stats(prefix: str, values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {f"{prefix}_{name}": "" for name in ("mean", "median", "min", "p95", "max")}
    return {
        f"{prefix}_mean": mean(values),
        f"{prefix}_median": median(values),
        f"{prefix}_min": min(values),
        f"{prefix}_p95": _percentile(values, 0.95),
        f"{prefix}_max": max(values),
    }


def _percentile(values: Sequence[int], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _train_domain_words(datasets: Mapping[str, Sequence[Record]]) -> dict[str, set[str]]:
    words: dict[str, set[str]] = defaultdict(set)
    for row in datasets["train"]:
        if row["domain"] is not None:
            words[row["domain"]].update(WORD_RE.findall(row["text"].casefold()))
    return dict(words)


def _jaccard_rows(datasets: Mapping[str, Sequence[Record]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    words = _train_domain_words(datasets)
    domains = sorted(words)
    lookup = {domain_a: {} for domain_a in domains}
    rows = []
    for domain_a in domains:
        row: dict[str, Any] = {"domain": domain_a}
        for domain_b in domains:
            union = words[domain_a] | words[domain_b]
            value = len(words[domain_a] & words[domain_b]) / len(union) if union else 0.0
            row[domain_b] = value
            lookup[domain_a][domain_b] = value
        rows.append(row)
    if not rows:
        rows = [{"domain": ""}]
    return rows, lookup


def _plot_jaccard(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    domains = [str(row["domain"]) for row in rows]
    matrix = [[float(row[domain]) for domain in domains] for row in rows]
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    axis.set_xticks(range(len(domains)), labels=domains, rotation=45, ha="right")
    axis.set_yticks(range(len(domains)), labels=domains)
    axis.set_title("CLINC150 train domain vocabulary Jaccard overlap")
    for row_index in range(len(domains)):
        for column_index in range(len(domains)):
            value = matrix[row_index][column_index]
            axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if value < 0.55 else "black")
    figure.colorbar(image, ax=axis, label="Jaccard index")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
