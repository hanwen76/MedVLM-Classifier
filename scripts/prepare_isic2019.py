#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path


LABEL_MAP_FULL = {
    "MEL": "melanoma",
    "NV": "melanocytic nevus",
    "BCC": "basal cell carcinoma",
    "AK": "actinic keratosis",
    "BKL": "benign keratosis",
    "DF": "dermatofibroma",
    "VASC": "vascular lesion",
    "SCC": "squamous cell carcinoma",
    "UNK": "none of the others",
}

CLASS_COLUMNS = tuple(LABEL_MAP_FULL.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ISIC 2019 into MedVLM-Classifier format"
    )
    parser.add_argument("--train_input", type=str, required=True, help="ISIC_2019_Training_Input.zip or extracted image dir")
    parser.add_argument("--train_gt", type=str, required=True, help="ISIC_2019_Training_GroundTruth.csv")
    parser.add_argument("--test_input", type=str, default=None, help="ISIC_2019_Test_Input.zip or extracted image dir")
    parser.add_argument("--test_gt", type=str, default=None, help="ISIC_2019_Test_GroundTruth.csv")
    parser.add_argument("--output_root", type=str, default="data/isic2019")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Fraction split from official training set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_style", choices=["code", "full"], default="full")
    parser.add_argument(
        "--keep_unknown_test",
        action="store_true",
        help="Keep UNK samples in test.json. By default they are dropped for closed-set evaluation.",
    )
    parser.add_argument(
        "--force_reextract",
        action="store_true",
        help="If input is zip, delete existing extracted dir and extract again.",
    )
    return parser.parse_args()


def ensure_dir_from_input(input_path: str, target_dir: Path, force_reextract: bool) -> Path:
    src = Path(input_path)
    if src.is_dir():
        if target_dir.exists() and force_reextract:
            if target_dir.is_symlink() or target_dir.is_file():
                target_dir.unlink()
            else:
                shutil.rmtree(target_dir)
        if not target_dir.exists():
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            target_dir.symlink_to(src.resolve(), target_is_directory=True)
        return target_dir

    if src.suffix.lower() != ".zip":
        raise ValueError(f"Expected a .zip file or directory, got: {src}")

    if target_dir.exists() and force_reextract:
        shutil.rmtree(target_dir)

    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(target_dir)

    return target_dir


def find_image_root(path: Path) -> Path:
    jpgs = list(path.rglob("*.jpg"))
    if not jpgs:
        raise FileNotFoundError(f"No .jpg files found under {path}")
    names = {p.parent.name for p in jpgs[: min(len(jpgs), 50)]}
    if len(names) == 1:
        return jpgs[0].parent
    return path


def parse_label(row: dict[str, str], label_style: str) -> str:
    active = [col for col in CLASS_COLUMNS if row.get(col, "0").strip() == "1.0" or row.get(col, "0").strip() == "1"]
    if len(active) != 1:
        raise ValueError(f"Expected exactly one active class for image {row.get('image')}, got {active}")
    label_code = active[0]
    if label_style == "code":
        return label_code
    return LABEL_MAP_FULL[label_code]


def load_ground_truth(csv_path: Path, label_style: str) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = str(row["image"]).strip()
            label_code = next(
                (col for col in CLASS_COLUMNS if row.get(col, "0").strip() in {"1", "1.0"}),
                None,
            )
            if label_code is None:
                raise ValueError(f"No positive class found for image {image_id}")
            label = label_code if label_style == "code" else LABEL_MAP_FULL[label_code]
            rows.append({"image_id": image_id, "label_code": label_code, "label": label})
    return rows


def build_records(rows: list[dict], image_root: Path, raw_root: Path) -> list[dict]:
    records: list[dict] = []
    for row in rows:
        image_name = f"{row['image_id']}.jpg"
        path = next(image_root.rglob(image_name), None)
        if path is None:
            raise FileNotFoundError(f"Image not found for {image_name} under {image_root}")
        records.append(
            {
                "image": str(path.relative_to(raw_root)).replace("\\", "/"),
                "label": row["label"],
                "label_code": row["label_code"],
            }
        )
    return records


def stratified_split(records: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        grouped[row["label"]].append(row)

    rng = random.Random(seed)
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for _, items in grouped.items():
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_ratio)))
        n_val = min(n_val, max(1, len(items) - 1))
        val_rows.extend(items[:n_val])
        train_rows.extend(items[n_val:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def save_json(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    raw_root = out_root / "raw"
    raw_train = raw_root / "train"
    raw_test = raw_root / "test"

    train_src = ensure_dir_from_input(args.train_input, raw_train, force_reextract=args.force_reextract)
    train_image_root = find_image_root(train_src)
    train_rows = load_ground_truth(Path(args.train_gt), label_style=args.label_style)
    train_records_all = build_records(train_rows, image_root=train_image_root, raw_root=raw_root)
    train_records, val_records = stratified_split(train_records_all, val_ratio=args.val_ratio, seed=args.seed)

    test_records: list[dict] = []
    if args.test_input and args.test_gt:
        test_src = ensure_dir_from_input(args.test_input, raw_test, force_reextract=args.force_reextract)
        test_image_root = find_image_root(test_src)
        test_rows = load_ground_truth(Path(args.test_gt), label_style=args.label_style)
        if not args.keep_unknown_test:
            test_rows = [row for row in test_rows if row["label_code"] != "UNK"]
        test_records = build_records(test_rows, image_root=test_image_root, raw_root=raw_root)

    labels = sorted({row["label"] for row in train_records_all})
    if args.keep_unknown_test and any(row["label_code"] == "UNK" for row in test_records):
        unknown_label = "UNK" if args.label_style == "code" else LABEL_MAP_FULL["UNK"]
        if unknown_label not in labels:
            labels.append(unknown_label)

    out_root.mkdir(parents=True, exist_ok=True)
    save_json(out_root / "train.json", train_records)
    save_json(out_root / "val.json", val_records)
    save_json(out_root / "test.json", test_records)
    (out_root / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "dataset": "isic2019",
        "label_style": args.label_style,
        "keep_unknown_test": args.keep_unknown_test,
        "image_root": str(raw_root),
        "num_train": len(train_records),
        "num_val": len(val_records),
        "num_test": len(test_records),
        "labels": labels,
        "source_train_input": str(Path(args.train_input).resolve()),
        "source_train_gt": str(Path(args.train_gt).resolve()),
        "source_test_input": str(Path(args.test_input).resolve()) if args.test_input else None,
        "source_test_gt": str(Path(args.test_gt).resolve()) if args.test_gt else None,
    }
    (out_root / "meta.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
