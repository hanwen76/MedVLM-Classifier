from __future__ import annotations

import re
from collections import Counter
from typing import Any


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return " ".join(text.split())


def extract_label(prediction: str, labels: list[str]) -> str | None:
    pred_n = normalize_text(prediction)
    label_pairs = [(label, normalize_text(label)) for label in labels]

    exact_matches = [label for label, label_n in label_pairs if pred_n == label_n]
    if len(exact_matches) == 1:
        return exact_matches[0]

    contained = [label for label, label_n in label_pairs if label_n and label_n in pred_n]
    if len(contained) == 1:
        return contained[0]
    if len(contained) > 1:
        return None

    return None


def count_mentioned_labels(prediction: str, labels: list[str]) -> int:
    pred_n = normalize_text(prediction)
    return sum(1 for label in labels if normalize_text(label) in pred_n)


def macro_f1(pred_labels: list[str | None], references: list[str], labels: list[str]) -> float:
    scores: list[float] = []
    for label in labels:
        tp = sum(pred == label and ref == label for pred, ref in zip(pred_labels, references))
        fp = sum(pred == label and ref != label for pred, ref in zip(pred_labels, references))
        fn = sum(pred != label and ref == label for pred, ref in zip(pred_labels, references))
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / max(len(scores), 1)


def compute_classification_metrics(
    predictions: list[str],
    references: list[str],
    labels: list[str] | None = None,
) -> dict[str, Any]:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")

    labels = labels or sorted(set(references))
    pred_labels = [extract_label(pred, labels) for pred in predictions]
    exact_correct = [pred == ref for pred, ref in zip(pred_labels, references)]
    inclusion_correct = [
        normalize_text(ref) in normalize_text(pred)
        for pred, ref in zip(predictions, references)
    ]
    mentioned_counts = [count_mentioned_labels(pred, labels) for pred in predictions]
    ambiguous = [count > 1 for count in mentioned_counts]

    total = len(references)
    pred_counter = Counter(pred for pred in pred_labels if pred is not None)
    return {
        "exact_match_accuracy": sum(exact_correct) / max(total, 1),
        "label_inclusion_accuracy": sum(inclusion_correct) / max(total, 1),
        "macro_f1": macro_f1(pred_labels, references, labels),
        "ambiguity_rate": sum(ambiguous) / max(total, 1),
        "unparsed_rate": sum(pred is None for pred in pred_labels) / max(total, 1),
        "exact_correct": sum(exact_correct),
        "inclusion_correct": sum(inclusion_correct),
        "total": total,
        "labels": labels,
        "pred_label_counts": dict(pred_counter),
    }
