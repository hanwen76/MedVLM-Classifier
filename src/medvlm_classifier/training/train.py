from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from medvlm_classifier.data import (
    InstructionTuneDataset,
    MedicalClassificationDataset,
    MixedBatchDataset,
    PromptTemplateConfig,
    VLMDataCollator,
)
from medvlm_classifier.model import freeze_all_except_projector, load_vlm_and_processor, summarize_trainable_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train MedVLM-Classifier with projector-only finetuning")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--classification_ann", type=str, required=True)
    parser.add_argument("--classification_image_root", type=str, default=None)
    parser.add_argument("--instruction_jsonl", type=str, required=True)
    parser.add_argument("--instruction_image_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/medvlm-classifier")
    parser.add_argument("--ratio", type=str, default="2:1", help="classification:instruction")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--virtual_size", type=int, default=12000)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--min_valid_label_tokens", type=int, default=1)
    parser.add_argument(
        "--force_trainable_fp32",
        action="store_true",
        help="Force trainable params to fp32. May cause dtype mismatch on some VLMs.",
    )
    parser.add_argument("--auto_lr_backoff", action="store_true", help="Automatically halve LR when a bad update is detected.")
    parser.add_argument("--min_lr", type=float, default=1e-7, help="Lower bound for auto LR backoff.")
    parser.add_argument(
        "--debug_no_skip",
        action="store_true",
        help="Do not skip problematic batches. Raise immediately for debugging.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--projector_keywords", nargs="*", default=[])
    return parser.parse_args()


def parse_ratio(ratio: str) -> tuple[int, int]:
    left, right = ratio.split(":")
    return int(left), int(right)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def patch_trainable_linear_autocast(model) -> int:
    """For trainable nn.Linear layers, cast input to weight dtype and cast output back.

    This enables fp32 projector weights to work with fp16 activations safely.
    """
    patched = 0
    for _, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not module.weight.requires_grad:
            continue

        def _forward(self, x):
            in_dtype = x.dtype
            out = F.linear(x.to(self.weight.dtype), self.weight, self.bias)
            return out.to(in_dtype)

        module.forward = MethodType(_forward, module)
        patched += 1
    return patched


def train() -> None:
    args = parse_args()
    set_seed(args.seed)

    model, processor = load_vlm_and_processor(args.model_name_or_path)
    freeze_all_except_projector(model, extra_keywords=args.projector_keywords)
    if args.force_trainable_fp32:
        # Keep trainable params in fp32 and patch trainable Linear forward for dtype alignment.
        for _, p in model.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()
        patched = patch_trainable_linear_autocast(model)
        print(f"[train] force_trainable_fp32 enabled, patched_trainable_linear_layers={patched}")
    stats = summarize_trainable_params(model)
    if stats["trainable"] == 0:
        raise RuntimeError(
            "No trainable parameters detected. Please check projector naming or pass --projector_keywords."
        )

    prompt_cfg = PromptTemplateConfig(question="这张医学影像显示了什么？")
    cls_ds = MedicalClassificationDataset(
        annotation_path=args.classification_ann,
        image_root=args.classification_image_root,
        prompt_cfg=prompt_cfg,
    )
    inst_ds = InstructionTuneDataset(
        instruction_jsonl=args.instruction_jsonl,
        image_root=args.instruction_image_root,
    )
    mixed = MixedBatchDataset(
        classification_dataset=cls_ds,
        instruction_dataset=inst_ds,
        cls_to_inst_ratio=parse_ratio(args.ratio),
        virtual_size=args.virtual_size,
        seed=args.seed,
    )

    collator = VLMDataCollator(processor=processor, max_length=args.max_length)
    loader = DataLoader(
        mixed,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    trainable_named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in trainable_named_params]
    trainable_dtypes = sorted({str(p.dtype) for p in params})
    print(f"[train] trainable_param_dtypes={trainable_dtypes}")

    # Pre-check finiteness before training starts.
    non_finite_before = [n for n, p in trainable_named_params if not torch.isfinite(p).all().item()]
    if non_finite_before:
        raise RuntimeError(
            "[sanity_check] non-finite trainable params before training: "
            + ", ".join(non_finite_before[:10])
        )
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    global_step = 0
    seen_steps = 0
    skipped_empty_label = 0
    skipped_nan_loss = 0
    skipped_bad_update = 0
    losses: list[float] = []

    for epoch in range(args.epochs):
        for batch in loader:
            seen_steps += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            valid_label_tokens = int((batch["labels"] != -100).sum().item())
            if valid_label_tokens < args.min_valid_label_tokens:
                if args.debug_no_skip:
                    raise RuntimeError(
                        f"[debug_no_skip] empty supervision at seen_step={seen_steps}, "
                        f"valid_label_tokens={valid_label_tokens}, min_required={args.min_valid_label_tokens}"
                    )
                skipped_empty_label += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            out = model(**batch)
            loss = out.loss
            if torch.isnan(loss) or torch.isinf(loss):
                if args.debug_no_skip:
                    raise RuntimeError(
                        f"[debug_no_skip] loss is invalid at seen_step={seen_steps}, "
                        f"loss={loss.item()}, valid_label_tokens={valid_label_tokens}"
                    )
                skipped_nan_loss += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            # Keep a light-weight backup of trainable params to recover from bad updates.
            param_backup = [p.data.detach().clone() for p in params]
            loss.backward()

            bad_grad_names = []
            for n, p in trainable_named_params:
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all().item():
                    bad_grad_names.append(n)
            if bad_grad_names:
                raise RuntimeError(
                    f"[debug] non-finite gradients at seen_step={seen_steps}. "
                    f"first_bad={bad_grad_names[0]} total_bad={len(bad_grad_names)}"
                )
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            bad_update_names = [n for n, p in trainable_named_params if not torch.isfinite(p).all().item()]
            bad_update = len(bad_update_names) > 0
            if bad_update:
                if args.debug_no_skip:
                    raise RuntimeError(
                        f"[debug_no_skip] bad parameter update at seen_step={seen_steps}, "
                        f"lr={optimizer.param_groups[0]['lr']:.2e}, first_bad={bad_update_names[0]}, total_bad={len(bad_update_names)}"
                    )
                skipped_bad_update += 1
                for p, old in zip(params, param_backup):
                    p.data.copy_(old)
                optimizer.zero_grad(set_to_none=True)

                if args.auto_lr_backoff:
                    for group in optimizer.param_groups:
                        old_lr = group["lr"]
                        group["lr"] = max(old_lr * 0.5, args.min_lr)
                continue

            scheduler.step()

            losses.append(float(loss.item()))
            global_step += 1
            if global_step % args.log_every == 0:
                print(
                    f"[train] epoch={epoch} step={global_step} loss={loss.item():.4f} "
                    f"valid_label_tokens={valid_label_tokens} skipped_empty={skipped_empty_label} "
                    f"skipped_nan={skipped_nan_loss} skipped_bad_update={skipped_bad_update} lr={optimizer.param_groups[0]['lr']:.2e}"
                )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    metadata = {
        "trainable_stats": stats,
        "avg_loss": sum(losses) / max(len(losses), 1),
        "steps": global_step,
        "seen_steps": seen_steps,
        "skipped_empty_label_batches": skipped_empty_label,
        "skipped_nan_loss_batches": skipped_nan_loss,
        "skipped_bad_update_batches": skipped_bad_update,
        "ratio": args.ratio,
        "freeze_policy": "freeze_vision_encoder_and_llm_train_projector_only",
        "trainable_param_dtypes": trainable_dtypes,
        "force_trainable_fp32": args.force_trainable_fp32,
        "grad_clip_norm": args.grad_clip_norm,
        "min_valid_label_tokens": args.min_valid_label_tokens,
        "auto_lr_backoff": args.auto_lr_backoff,
        "min_lr": args.min_lr,
        "final_lr": optimizer.param_groups[0]["lr"],
    }
    with (output_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("[done] Model and processor saved to", output_dir)
    print("[done] Training summary:", metadata)


if __name__ == "__main__":
    train()
