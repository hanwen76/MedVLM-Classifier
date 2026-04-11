from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate reference-style prediction JSONL")
    parser.add_argument("--pred_jsonl", type=str, required=True, help="JSONL with fields pred,label")
    parser.add_argument("--output_json", type=str, default="outputs/reference_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.pred_jsonl)

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    correct = 0
    details = []
    for row in rows:
        label = str(row.get("label", ""))
        pred = str(row.get("pred", ""))
        ok = normalize(label) in normalize(pred)
        if ok:
            correct += 1
        details.append({"label": label, "pred": pred, "is_correct": ok})

    total = len(rows)
    accuracy = correct / max(total, 1)
    out = {"accuracy": accuracy, "correct": correct, "total": total, "details": details}

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(out)


if __name__ == "__main__":
    main()
