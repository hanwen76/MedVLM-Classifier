#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert LLaVA-Med style data to MedVLM-Classifier instruction JSONL. "
            "Supports both (image+conversations) and (messages+images) schemas."
        )
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help=(
            "Path to .json OR .parquet OR directory containing parquet shards "
            "(e.g., train-00000-of-00014.parquet)."
        ),
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="",
        help="Root directory for path-based images (legacy image+conversations schema).",
    )
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all")
    parser.add_argument(
        "--require_image_exists",
        action="store_true",
        help="For path-based images: skip sample if image file does not exist under image_root.",
    )
    parser.add_argument(
        "--first_round_only",
        action="store_true",
        help="Use first user+assistant round only; otherwise concatenate all rounds.",
    )
    parser.add_argument(
        "--embedded_image_dir",
        type=str,
        default="",
        help=(
            "For messages+images schema: where to save extracted images. "
            "Default: <output_jsonl_dir>/extracted_images"
        ),
    )
    parser.add_argument(
        "--image_format",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Output image format when extracting embedded images.",
    )
    return parser.parse_args()


def load_records(input_path: Path) -> list[dict[str, Any]]:
    if input_path.is_dir():
        parquet_files = sorted(input_path.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found under: {input_path}")
        ds = load_dataset("parquet", data_files=[str(x) for x in parquet_files], split="train")
        return [dict(x) for x in ds]

    if input_path.suffix.lower() == ".parquet":
        ds = load_dataset("parquet", data_files=str(input_path), split="train")
        return [dict(x) for x in ds]

    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_user_text(text: str) -> str:
    return text.replace("<image>", "").strip()


def build_dialog_from_conversations(conversations: list[dict], first_round_only: bool) -> str | None:
    users: list[str] = []
    assistants: list[str] = []

    for turn in conversations:
        role = str(turn.get("from", "")).strip().lower()
        value = str(turn.get("value", "")).strip()
        if role in {"human", "user"}:
            users.append(normalize_user_text(value))
        elif role in {"gpt", "assistant"}:
            assistants.append(value)

    if not users or not assistants:
        return None

    if first_round_only:
        user_text = users[0]
        assistant_text = assistants[0]
    else:
        user_text = "\n".join(x for x in users if x)
        assistant_text = "\n".join(x for x in assistants if x)

    if not user_text:
        user_text = "Please analyze this medical image."

    return f"USER: <image>\\n{user_text}\\nASSISTANT: {assistant_text}"


def build_dialog_from_messages(messages: list[dict], first_round_only: bool) -> str | None:
    users: list[str] = []
    assistants: list[str] = []

    for turn in messages:
        role = str(turn.get("role", "")).strip().lower()
        value = str(turn.get("content", "")).strip()
        if role in {"human", "user"}:
            users.append(normalize_user_text(value))
        elif role in {"gpt", "assistant"}:
            assistants.append(value)

    if not users or not assistants:
        return None

    if first_round_only:
        user_text = users[0]
        assistant_text = assistants[0]
    else:
        user_text = "\n".join(x for x in users if x)
        assistant_text = "\n".join(x for x in assistants if x)

    if not user_text:
        user_text = "Please analyze this medical image."

    return f"USER: <image>\\n{user_text}\\nASSISTANT: {assistant_text}"


def to_pil_image(obj: Any) -> Image.Image | None:
    if obj is None:
        return None

    if isinstance(obj, Image.Image):
        return obj.convert("RGB")

    if isinstance(obj, np.ndarray):
        return Image.fromarray(obj).convert("RGB")

    if isinstance(obj, dict):
        if obj.get("bytes"):
            return Image.open(io.BytesIO(obj["bytes"])).convert("RGB")
        if obj.get("path"):
            p = Path(obj["path"])
            if p.exists():
                return Image.open(p).convert("RGB")

    return None


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json)
    image_root = Path(args.image_root) if args.image_root else None
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    if args.max_samples > 0:
        records = records[: args.max_samples]

    extracted_dir = Path(args.embedded_image_dir) if args.embedded_image_dir else output_path.parent / "extracted_images"
    used_embedded_images = False

    kept = 0
    skipped_no_dialog = 0
    skipped_missing_image = 0
    skipped_bad_image = 0

    with output_path.open("w", encoding="utf-8") as fw:
        for idx, item in enumerate(tqdm(records, desc="convert-llava-med")):
            row_out: dict[str, Any] | None = None

            # Schema A: legacy LLaVA style with image path + conversations
            if "conversations" in item and "image" in item:
                image_rel = str(item.get("image", "")).strip()
                if not image_rel:
                    skipped_missing_image += 1
                    continue

                if args.require_image_exists:
                    if image_root is None:
                        raise ValueError("--image_root is required when --require_image_exists is set for path-based schema")
                    if not (image_root / image_rel).exists():
                        skipped_missing_image += 1
                        continue

                dialog = build_dialog_from_conversations(item.get("conversations", []), args.first_round_only)
                if dialog is None:
                    skipped_no_dialog += 1
                    continue

                row_out = {
                    "image": image_rel,
                    "text": dialog,
                    "source": "llava-med",
                    "sample_id": item.get("id", idx),
                }

            # Schema B: HF style with messages + embedded images
            elif "messages" in item and "images" in item:
                messages = item.get("messages", [])
                dialog = build_dialog_from_messages(messages, args.first_round_only)
                if dialog is None:
                    skipped_no_dialog += 1
                    continue

                images = item.get("images", [])
                if not images:
                    skipped_missing_image += 1
                    continue

                pil = to_pil_image(images[0])
                if pil is None:
                    skipped_bad_image += 1
                    continue

                used_embedded_images = True
                extracted_dir.mkdir(parents=True, exist_ok=True)
                filename = f"llava_med_{idx:07d}.{args.image_format}"
                out_img = extracted_dir / filename
                if args.image_format == "jpg":
                    pil.save(out_img, quality=95)
                else:
                    pil.save(out_img)

                row_out = {
                    "image": filename,
                    "text": dialog,
                    "source": "llava-med-hf",
                    "sample_id": item.get("id", idx),
                }

            else:
                skipped_no_dialog += 1
                continue

            fw.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            kept += 1

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "total_input": len(records),
        "kept": kept,
        "skipped_no_dialog": skipped_no_dialog,
        "skipped_missing_image": skipped_missing_image,
        "skipped_bad_image": skipped_bad_image,
        "schema_detected_embedded_images": used_embedded_images,
        "instruction_image_root": str(extracted_dir) if used_embedded_images else (str(image_root) if image_root else ""),
    }

    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
