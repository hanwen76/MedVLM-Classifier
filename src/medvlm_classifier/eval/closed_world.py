from __future__ import annotations


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def compute_closed_world_accuracy(predictions: list[str], references: list[str]) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")

    correct = 0
    for pred, ref in zip(predictions, references):
        pred_n = normalize(pred)
        ref_n = normalize(ref)
        if ref_n in pred_n:
            correct += 1

    total = len(references)
    acc = correct / max(total, 1)
    return {"accuracy": acc, "correct": correct, "total": total}
