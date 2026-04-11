from __future__ import annotations

import argparse
import json
from pathlib import Path

from medvlm_classifier.data.formatters import PromptTemplateConfig, build_classification_dialog


def convert_classification_pairs() -> None:
    parser = argparse.ArgumentParser("Convert image-label pairs to instruction-style JSONL")
    parser.add_argument("--input_json", type=str, required=True, help="[{\"image\":...,\"label\":...}, ...]")
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--question", type=str, default="这张医学影像显示了什么？")
    args = parser.parse_args()

    with Path(args.input_json).open("r", encoding="utf-8") as f:
        data = json.load(f)

    cfg = PromptTemplateConfig(question=args.question)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for item in data:
            text = build_classification_dialog(item["label"], cfg)
            line = {"image": item["image"], "label": item["label"], "text": text, "source": "classification"}
            f.write(json.dumps(line, ensure_ascii=False) + "\\n")

    print(f"[convert] wrote {len(data)} samples to {out_path}")


if __name__ == "__main__":
    convert_classification_pairs()
