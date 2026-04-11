from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from medvlm_classifier.data import MedicalClassificationEvalDataset, PromptTemplateConfig
from medvlm_classifier.eval import compute_closed_world_accuracy
from medvlm_classifier.model import load_vlm_and_processor


def _configure_llava_processor(processor, model) -> None:
    # Forward-compatible fix for LLaVA image token expansion in recent transformers.
    cfg = getattr(model, "config", None)
    vision_cfg = getattr(cfg, "vision_config", None)
    patch_size = getattr(vision_cfg, "patch_size", None)
    select_strategy = getattr(cfg, "vision_feature_select_strategy", None)
    if patch_size is not None and not hasattr(processor, "patch_size"):
        processor.patch_size = patch_size
    if select_strategy is not None and not hasattr(processor, "vision_feature_select_strategy"):
        processor.vision_feature_select_strategy = select_strategy


def _pick_reference_dtype(model) -> torch.dtype | None:
    # Prefer vision tower dtype for LLaVA-like models.
    for name, p in model.named_parameters():
        if not p.is_floating_point():
            continue
        if "vision_tower" in name:
            return p.dtype

    # Fallback: most common floating dtype.
    counts: dict[torch.dtype, int] = {}
    for _, p in model.named_parameters():
        if not p.is_floating_point():
            continue
        counts[p.dtype] = counts.get(p.dtype, 0) + p.numel()
    if not counts:
        return None
    return max(counts.items(), key=lambda x: x[1])[0]


def _align_projector_dtype(model) -> None:
    target_dtype = _pick_reference_dtype(model)
    if target_dtype is None:
        return

    for attr in ("multi_modal_projector", "mm_projector", "projector"):
        module = getattr(model, attr, None)
        if module is not None:
            module.to(dtype=target_dtype)
            print(f"[eval] aligned {attr} dtype to {target_dtype}")
            return

    for name, module in model.named_modules():
        if name.endswith(("multi_modal_projector", "mm_projector", "projector")):
            module.to(dtype=target_dtype)
            print(f"[eval] aligned {name} dtype to {target_dtype}")
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Closed-world evaluation for MedVLM-Classifier")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--annotation", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--output_path", type=str, default="outputs/eval_closed_world.json")
    return parser.parse_args()


def collate_eval(batch):
    images = [x["image"] for x in batch]
    texts = [x["text"] for x in batch]
    labels = [x["label_name"] for x in batch]
    return {"images": images, "texts": texts, "labels": labels}


def run() -> None:
    args = parse_args()
    model, processor = load_vlm_and_processor(args.model_name_or_path)
    _configure_llava_processor(processor, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    _align_projector_dtype(model)
    model.eval()
    print(f"[eval] device={device}")

    ds = MedicalClassificationEvalDataset(
        annotation_path=args.annotation,
        image_root=args.image_root,
        prompt_cfg=PromptTemplateConfig(question="这张医学影像显示了什么？"),
    )
    if args.max_samples > 0:
        ds.samples = ds.samples[: args.max_samples]
    print(f"[eval] num_samples={len(ds)} batch_size={args.batch_size} max_new_tokens={args.max_new_tokens}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_eval)

    predictions: list[str] = []
    references: list[str] = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="eval"), start=1):
            model_inputs = processor(
                text=batch["texts"],
                images=batch["images"],
                return_tensors="pt",
                padding=True,
            )
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            generated = model.generate(**model_inputs, max_new_tokens=args.max_new_tokens)
            prompt_lengths = model_inputs["attention_mask"].sum(dim=1).tolist()
            texts = []
            for row, prompt_len in zip(generated, prompt_lengths):
                text = processor.decode(row[int(prompt_len) :], skip_special_tokens=True)
                texts.append(text)
            predictions.extend(texts)
            references.extend(batch["labels"])
            if i % args.log_every == 0:
                print(f"[eval] processed_batches={i} processed_samples={len(references)}")

    metrics = compute_closed_world_accuracy(predictions, references)
    out = {
        "metrics": metrics,
        "samples": [
            {"prediction": p, "reference": r, "is_correct": normalize_ref_in_pred(p, r)}
            for p, r in zip(predictions, references)
        ],
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[eval]", out["metrics"])
    print("[eval] details written to", output_path)


def normalize_ref_in_pred(pred: str, ref: str) -> bool:
    return " ".join(ref.lower().split()) in " ".join(pred.lower().split())


if __name__ == "__main__":
    run()
