from __future__ import annotations

from .classification_metrics import compute_classification_metrics, normalize_text


def normalize(text: str) -> str:
    return normalize_text(text)


def compute_closed_world_accuracy(predictions: list[str], references: list[str]) -> dict[str, float]:
    metrics = compute_classification_metrics(predictions, references)
    return {
        "accuracy": metrics["label_inclusion_accuracy"],
        "correct": metrics["inclusion_correct"],
        "total": metrics["total"],
        **metrics,
    }
