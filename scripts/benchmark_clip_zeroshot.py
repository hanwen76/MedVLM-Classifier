#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medvlm_classifier.eval.classification_metrics import compute_classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Zero-shot CLIP classification baseline")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--data_json", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--labels_json", type=str, default=None, help="JSON list or id->label dict. Defaults to labels in data_json.")
    parser.add_argument("--prompt_template", type=str, default="a medical image of {label}")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs/clip_zeroshot")
    return parser.parse_args()


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_labels(rows: list[dict], labels_json: str | None) -> list[str]:
    if labels_json is None:
        return sorted({str(row["label"]) for row in rows})
    raw = load_json(labels_json)
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, dict):
        try:
            return [str(v) for _, v in sorted(raw.items(), key=lambda kv: int(kv[0]))]
        except ValueError:
            return [str(v) for _, v in sorted(raw.items())]
    raise ValueError(f"Unsupported labels_json format: {type(raw).__name__}")


def encode_texts(
    model: CLIPModel,
    processor: CLIPProcessor,
    labels: list[str],
    prompt_template: str,
    device: torch.device,
) -> torch.Tensor:
    prompts = [prompt_template.format(label=label) for label in labels]
    inputs = processor(text=prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    return torch.nn.functional.normalize(features, dim=-1)


def main() -> None:
    args = parse_args()
    rows = load_json(args.data_json)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    labels = load_labels(rows, args.labels_json)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    model.eval()

    text_features = encode_texts(
        model=model,
        processor=processor,
        labels=labels,
        prompt_template=args.prompt_template,
        device=device,
    )

    image_root = Path(args.image_root)
    predictions: list[str] = []
    references: list[str] = []
    pred_rows: list[dict] = []

    for i in tqdm(range(0, len(rows), args.batch_size), desc="clip-zeroshot"):
        batch = rows[i : i + args.batch_size]
        images = [Image.open(image_root / item["image"]).convert("RGB") for item in batch]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            logits = image_features @ text_features.T
            probs = logits.softmax(dim=-1)
            pred_indices = probs.argmax(dim=-1).tolist()
            top_probs = probs.max(dim=-1).values.tolist()

        for item, pred_idx, score in zip(batch, pred_indices, top_probs):
            pred = labels[int(pred_idx)]
            ref = str(item["label"])
            predictions.append(pred)
            references.append(ref)
            pred_rows.append(
                {
                    "image": item["image"],
                    "label": ref,
                    "pred_label": pred,
                    "confidence": float(score),
                    "is_correct": pred == ref,
                }
            )

    metrics = compute_classification_metrics(predictions, references, labels=labels)
    out = {
        "clip_model": args.clip_model,
        "prompt_template": args.prompt_template,
        "metrics": metrics,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in pred_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(out["metrics"], ensure_ascii=False, indent=2))
    print("[done] outputs written to", output_dir)


if __name__ == "__main__":
    main()
