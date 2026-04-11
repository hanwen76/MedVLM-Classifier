#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import medmnist
import numpy as np
from medmnist import INFO
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and export MedMNIST to MedVLM-Classifier classification format"
    )
    parser.add_argument("--dataset", type=str, default="pathmnist", help="e.g. pathmnist, dermamnist, chestmnist")
    parser.add_argument("--size", type=int, default=224, help="image size to export")
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/medmnist",
        help="root dir to save exported images and annotations",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="comma-separated splits",
    )
    parser.add_argument(
        "--max_samples_per_split",
        type=int,
        default=0,
        help="0 means export all",
    )
    parser.add_argument(
        "--save_rgb",
        action="store_true",
        help="convert all images to RGB before saving (recommended for VLM)",
    )
    return parser.parse_args()


def to_label_name(raw_label: np.ndarray, label_map: dict[int, str], is_multi_label: bool) -> tuple[str, list[int]]:
    arr = np.array(raw_label).reshape(-1)
    if is_multi_label:
        indices = [int(i) for i, v in enumerate(arr.tolist()) if int(v) == 1]
        if not indices:
            return "none", []
        names = [label_map[i] for i in indices]
        return "|".join(names), indices

    idx = int(arr[0])
    return label_map[idx], [idx]


def export_split(
    data_class,
    split: str,
    size: int,
    image_dir: Path,
    ann_path: Path,
    label_map: dict[int, str],
    is_multi_label: bool,
    max_samples: int,
    save_rgb: bool,
) -> int:
    dataset = data_class(split=split, download=True, size=size)
    n = len(dataset)
    if max_samples > 0:
        n = min(n, max_samples)

    rows: list[dict] = []
    for i in tqdm(range(n), desc=f"export-{split}"):
        image, label = dataset[i]

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if save_rgb:
            image = image.convert("RGB")

        filename = f"{split}_{i:07d}.png"
        image.save(image_dir / filename)

        label_name, label_ids = to_label_name(label, label_map, is_multi_label=is_multi_label)
        rows.append(
            {
                "image": filename,
                "label": label_name,
                "label_ids": label_ids,
                "split": split,
            }
        )

    with ann_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return n


def main() -> None:
    args = parse_args()
    dataset_flag = args.dataset.lower()

    if dataset_flag not in INFO:
        raise ValueError(f"Unknown MedMNIST dataset '{dataset_flag}'. Use medmnist.INFO keys.")

    info = INFO[dataset_flag]
    data_class = getattr(medmnist, info["python_class"])

    label_map = {int(k): v for k, v in info["label"].items()}
    task = info.get("task", "")
    is_multi_label = "multi-label" in task

    splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    out_root = Path(args.output_root) / dataset_flag
    image_dir = out_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    (out_root / "labels.json").write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")

    summary: dict[str, int] = {}
    for split in splits:
        ann_path = out_root / f"{split}.json"
        exported = export_split(
            data_class=data_class,
            split=split,
            size=args.size,
            image_dir=image_dir,
            ann_path=ann_path,
            label_map=label_map,
            is_multi_label=is_multi_label,
            max_samples=args.max_samples_per_split,
            save_rgb=args.save_rgb,
        )
        summary[split] = exported

    meta = {
        "dataset": dataset_flag,
        "task": task,
        "is_multi_label": is_multi_label,
        "size": args.size,
        "splits": summary,
        "output_root": str(out_root),
    }
    (out_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
