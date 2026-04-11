#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Check MedVLM training data quality")
    parser.add_argument("--classification_ann", type=str, required=True)
    parser.add_argument("--classification_image_root", type=str, required=True)
    parser.add_argument("--instruction_jsonl", type=str, required=True)
    parser.add_argument("--instruction_image_root", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all")
    parser.add_argument("--long_text_threshold", type=int, default=4000, help="char length threshold")
    parser.add_argument("--output_json", type=str, default="outputs/data_check_report.json")
    return parser.parse_args()


def load_classification(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    raise ValueError(f"Unsupported classification annotation format: {path.suffix}")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def analyze_classification(rows: list[dict], image_root: Path, max_samples: int) -> dict:
    if max_samples > 0:
        rows = rows[:max_samples]

    missing_image = 0
    empty_label = 0
    bad_row = 0
    labels = []

    examples_missing = []
    examples_bad = []

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            bad_row += 1
            if len(examples_bad) < 5:
                examples_bad.append({"idx": i, "reason": "not_dict", "row": str(row)[:300]})
            continue

        image_rel = str(row.get("image", "")).strip()
        label = str(row.get("label", "")).strip()

        if not label:
            empty_label += 1
        else:
            labels.append(label)

        if not image_rel or not (image_root / image_rel).exists():
            missing_image += 1
            if len(examples_missing) < 5:
                examples_missing.append({"idx": i, "image": image_rel})

    unique_labels = sorted(set(labels))
    return {
        "num_samples": len(rows),
        "missing_image": missing_image,
        "missing_image_ratio": (missing_image / len(rows)) if rows else 0.0,
        "empty_label": empty_label,
        "empty_label_ratio": (empty_label / len(rows)) if rows else 0.0,
        "bad_row": bad_row,
        "num_unique_labels": len(unique_labels),
        "top10_labels": unique_labels[:10],
        "examples_missing": examples_missing,
        "examples_bad": examples_bad,
    }


def analyze_instruction(rows: list[dict], image_root: Path, max_samples: int, long_text_threshold: int) -> dict:
    if max_samples > 0:
        rows = rows[:max_samples]

    missing_image = 0
    empty_text = 0
    missing_assistant = 0
    missing_user_image = 0
    long_text = 0
    bad_row = 0

    text_lengths = []
    examples_missing = []
    examples_text = []

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            bad_row += 1
            continue

        image_rel = str(row.get("image", "")).strip()
        text = str(row.get("text", ""))

        if not image_rel or not (image_root / image_rel).exists():
            missing_image += 1
            if len(examples_missing) < 5:
                examples_missing.append({"idx": i, "image": image_rel})

        if not text.strip():
            empty_text += 1
            continue

        length = len(text)
        text_lengths.append(length)
        if length >= long_text_threshold:
            long_text += 1

        if "ASSISTANT:" not in text:
            missing_assistant += 1
            if len(examples_text) < 5:
                examples_text.append({"idx": i, "reason": "missing_ASSISTANT", "preview": text[:200]})

        if "USER:" not in text or "<image>" not in text:
            missing_user_image += 1
            if len(examples_text) < 5:
                examples_text.append({"idx": i, "reason": "missing_USER_or_<image>", "preview": text[:200]})

    n = len(rows)
    return {
        "num_samples": n,
        "missing_image": missing_image,
        "missing_image_ratio": (missing_image / n) if n else 0.0,
        "empty_text": empty_text,
        "empty_text_ratio": (empty_text / n) if n else 0.0,
        "missing_assistant": missing_assistant,
        "missing_assistant_ratio": (missing_assistant / n) if n else 0.0,
        "missing_user_or_image_token": missing_user_image,
        "missing_user_or_image_token_ratio": (missing_user_image / n) if n else 0.0,
        "long_text_count": long_text,
        "long_text_ratio": (long_text / n) if n else 0.0,
        "text_len_avg": mean(text_lengths) if text_lengths else 0.0,
        "text_len_max": max(text_lengths) if text_lengths else 0,
        "text_len_min": min(text_lengths) if text_lengths else 0,
        "bad_row": bad_row,
        "examples_missing": examples_missing,
        "examples_text_issues": examples_text,
    }


def main() -> None:
    args = parse_args()

    cls_rows = load_classification(Path(args.classification_ann))
    inst_rows = load_jsonl(Path(args.instruction_jsonl))

    cls_report = analyze_classification(
        rows=cls_rows,
        image_root=Path(args.classification_image_root),
        max_samples=args.max_samples,
    )
    inst_report = analyze_instruction(
        rows=inst_rows,
        image_root=Path(args.instruction_image_root),
        max_samples=args.max_samples,
        long_text_threshold=args.long_text_threshold,
    )

    report = {
        "classification": cls_report,
        "instruction": inst_report,
        "risk_hint": {
            "likely_data_issue": (
                cls_report["missing_image_ratio"] > 0.01
                or cls_report["empty_label_ratio"] > 0.01
                or inst_report["missing_image_ratio"] > 0.01
                or inst_report["empty_text_ratio"] > 0.01
                or inst_report["missing_assistant_ratio"] > 0.01
            ),
            "likely_truncation_risk": inst_report["long_text_ratio"] > 0.2,
        },
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[done] report saved to {out}")


if __name__ == "__main__":
    main()
