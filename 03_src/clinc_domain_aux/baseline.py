"""M04: Train-only TF-IDF and logistic-regression lexical baseline."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .common import environment_info, sha256_file, stable_hash, utc_now, write_csv, write_json
from .evaluation import classification_metrics, msp_scores


def _pipeline(config: Mapping[str, Any], c_value: float) -> Pipeline:
    vectorizer = dict(config["vectorizer"])
    vectorizer["ngram_range"] = tuple(vectorizer["ngram_range"])
    logistic = dict(config["logistic_regression"])
    return Pipeline([
        ("tfidf", TfidfVectorizer(**vectorizer)),
        ("classifier", LogisticRegression(C=float(c_value), **logistic)),
    ])


def predict_scores(pipeline: Pipeline, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    probabilities = pipeline.predict_proba(list(texts))
    decision = pipeline.decision_function(list(texts))
    return np.asarray(decision, dtype=np.float64), np.asarray(probabilities, dtype=np.float64)


def save_score_archive(
    path: str | Path,
    pipeline: Pipeline,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    logits, probabilities = predict_scores(pipeline, [row["text"] for row in records])
    values = {
        "sample_id": np.asarray([row["sample_id"] for row in records], dtype=np.str_),
        "intent_id": np.asarray([row["intent_id"] for row in records], dtype=np.int64),
        "logits": logits,
        "probabilities": probabilities,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **values)
    return values


def train_baseline(
    datasets: Mapping[str, Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Select C by validation macro-F1, retaining the smaller C on exact ties."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = list(datasets["train"])
    validation = list(datasets["val"])
    oos_validation = list(datasets["oos_val"])
    train_texts = [row["text"] for row in train]
    train_labels = np.asarray([row["intent_id"] for row in train], dtype=np.int64)
    val_labels = np.asarray([row["intent_id"] for row in validation], dtype=np.int64)
    num_intents = len(set(int(row["intent_id"]) for row in train))
    intent_domain = np.full(num_intents, -1, dtype=np.int64)
    domain_ids = sorted({int(row["domain_id"]) for row in train})
    for row in train:
        intent_domain[int(row["intent_id"])] = int(row["domain_id"])
    data_hash = stable_hash({
        split: [[row["sample_id"], row["text"], row["intent_id"], row["domain_id"]] for row in datasets[split]]
        for split in ("train", "val", "oos_val")
    })
    config_hash = stable_hash(config)
    selection_rows = []
    fitted: dict[float, Pipeline] = {}
    convergence: dict[float, list[str]] = {}
    for c_value in sorted(float(value) for value in config["c_grid"]):
        pipeline = _pipeline(config, c_value)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            pipeline.fit(train_texts, train_labels)
        messages = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
        convergence[c_value] = messages
        probabilities = pipeline.predict_proba([row["text"] for row in validation])
        predictions = probabilities.argmax(axis=1)
        metrics = classification_metrics(val_labels, predictions, intent_domain, num_intents=num_intents)
        selection_rows.append({
            "C": c_value,
            **metrics,
            "converged": not messages,
            "convergence_messages": json.dumps(messages, ensure_ascii=False),
            "n_iter": json.dumps(pipeline.named_steps["classifier"].n_iter_.tolist()),
        })
        fitted[c_value] = pipeline
    best_f1 = max(row["macro_f1"] for row in selection_rows)
    best_c = min(row["C"] for row in selection_rows if row["macro_f1"] == best_f1)
    best = fitted[best_c]
    joblib.dump(best, output / "pipeline.joblib")
    val_scores = save_score_archive(output / "val_scores.npz", best, validation)
    oos_val_scores = save_score_archive(output / "oos_val_scores.npz", best, oos_validation)
    write_csv(output / "c_selection.csv", selection_rows)
    predictions = val_scores["probabilities"].argmax(axis=1)
    metrics = classification_metrics(val_labels, predictions, intent_domain, num_intents=num_intents)
    write_json(output / "metrics.json", {
        "selected_c": best_c,
        "selection_metric": "macro_f1",
        "tie_rule": "smallest C on exact macro-F1 ties",
        **metrics,
    })
    checkpoint_hash = sha256_file(output / "pipeline.joblib")
    val_msp = msp_scores(val_scores["probabilities"], "probabilities")
    write_csv(
        output / "val_predictions.csv",
        ({
            "sample_id": str(row["sample_id"]),
            "split": "val",
            "gold_intent": int(row["intent_id"]),
            "gold_domain": int(row["domain_id"]),
            "pred_intent": int(predictions[index]),
            "pred_domain_from_intent": int(intent_domain[predictions[index]]),
            "msp_score": float(val_msp[index]),
            "seed": 0,
            "model": "B0",
            "checkpoint_hash": checkpoint_hash,
        } for index, row in enumerate(validation)),
    )
    write_json(output / "environment.json", environment_info())
    completed = {
        "status": "completed",
        "created_at": utc_now(),
        "model": "B0",
        "selected_c": best_c,
        "best_val_macro_f1": best_f1,
        "config_hash": config_hash,
        "data_hash": data_hash,
        "checkpoint_hash": checkpoint_hash,
        "convergence": {str(key): value for key, value in convergence.items()},
        "run_dir": str(output.resolve()),
        "domain_ids": domain_ids,
    }
    write_json(output / "completed.json", completed)
    return completed


def load_baseline(run_dir: str | Path) -> Pipeline:
    return joblib.load(Path(run_dir) / "pipeline.joblib")
