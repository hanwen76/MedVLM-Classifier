#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medvlm_classifier.eval.classification_metrics import compute_classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("CoOp-style CLIP prompt tuning baseline")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--labels_json", type=str, default=None)
    parser.add_argument("--label_prompt_template", type=str, default="{label}")
    parser.add_argument("--n_ctx", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs/clip_prompt_tuning")
    return parser.parse_args()


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_labels(rows: list[dict[str, Any]], labels_json: str | None) -> list[str]:
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


class JsonImageDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], image_root: Path, label_to_idx: dict[str, int]):
        self.rows = rows
        self.image_root = image_root
        self.label_to_idx = label_to_idx

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.rows[idx]
        return {
            "image": Image.open(self.image_root / item["image"]).convert("RGB"),
            "label": str(item["label"]),
            "target": self.label_to_idx[str(item["label"])],
            "image_path": item["image"],
        }


def collate_images(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [item["image"] for item in batch],
        "targets": torch.tensor([item["target"] for item in batch], dtype=torch.long),
        "labels": [item["label"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
    }


class ClipPromptLearner(nn.Module):
    def __init__(
        self,
        model: CLIPModel,
        tokenizer,
        labels: list[str],
        n_ctx: int,
        label_prompt_template: str,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.labels = labels
        self.n_ctx = n_ctx
        self.label_texts = [label_prompt_template.format(label=label) for label in labels]

        hidden_size = model.config.text_config.hidden_size
        self.context = nn.Parameter(torch.empty(n_ctx, hidden_size))
        nn.init.normal_(self.context, std=0.02)

        self.label_token_ids = [
            tokenizer(text, add_special_tokens=False)["input_ids"]
            for text in self.label_texts
        ]
        self.max_length = model.config.text_config.max_position_embeddings
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def _build_inputs(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_embedding = self.model.text_model.embeddings.token_embedding
        embeds: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        eos_positions: list[int] = []

        for label_ids in self.label_token_ids:
            max_label_len = self.max_length - self.n_ctx - 2
            label_ids = label_ids[:max_label_len]

            fixed_ids = torch.tensor(
                [self.bos_token_id] + label_ids + [self.eos_token_id],
                dtype=torch.long,
                device=device,
            )
            fixed_emb = token_embedding(fixed_ids)
            prompt = torch.cat(
                [
                    fixed_emb[:1],
                    self.context.to(device),
                    fixed_emb[1:],
                ],
                dim=0,
            )
            length = prompt.size(0)
            eos_positions.append(length - 1)

            if length < self.max_length:
                pad = torch.zeros(
                    self.max_length - length,
                    prompt.size(-1),
                    dtype=prompt.dtype,
                    device=device,
                )
                prompt = torch.cat([prompt, pad], dim=0)
            embeds.append(prompt)

            mask = torch.zeros(self.max_length, dtype=torch.long, device=device)
            mask[:length] = 1
            masks.append(mask)

        return torch.stack(embeds), torch.stack(masks), torch.tensor(eos_positions, device=device)

    def forward(self) -> torch.Tensor:
        device = self.context.device
        inputs_embeds, attention_mask, eos_positions = self._build_inputs(device)
        text_model = self.model.text_model
        batch_size, seq_len, _ = inputs_embeds.shape
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        hidden = inputs_embeds + text_model.embeddings.position_embedding(position_ids)

        dtype = hidden.dtype
        causal_mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)

        padding_mask = (1 - attention_mask[:, None, None, :]).to(dtype=dtype) * torch.finfo(dtype).min
        encoder_outputs = text_model.encoder(
            inputs_embeds=hidden,
            attention_mask=padding_mask,
            causal_attention_mask=causal_mask,
            return_dict=True,
        )
        hidden = text_model.final_layer_norm(encoder_outputs.last_hidden_state)
        pooled = hidden[torch.arange(hidden.size(0), device=device), eos_positions]
        text_features = self.model.text_projection(pooled)
        return torch.nn.functional.normalize(text_features, dim=-1)


def encode_images(model: CLIPModel, processor: CLIPProcessor, images: list[Image.Image], device: torch.device) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return torch.nn.functional.normalize(features, dim=-1)


def evaluate(
    model: CLIPModel,
    processor: CLIPProcessor,
    prompt_learner: ClipPromptLearner,
    loader: DataLoader,
    labels: list[str],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    prompt_learner.eval()
    predictions: list[str] = []
    references: list[str] = []
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        text_features = prompt_learner()
        scale = model.logit_scale.exp()
        for batch in tqdm(loader, desc="eval"):
            image_features = encode_images(model, processor, batch["images"], device)
            logits = scale * image_features @ text_features.T
            probs = logits.softmax(dim=-1)
            pred_indices = probs.argmax(dim=-1).tolist()
            confidences = probs.max(dim=-1).values.tolist()

            for path, ref, pred_idx, confidence in zip(
                batch["image_paths"], batch["labels"], pred_indices, confidences
            ):
                pred = labels[int(pred_idx)]
                predictions.append(pred)
                references.append(ref)
                rows.append(
                    {
                        "image": path,
                        "label": ref,
                        "pred_label": pred,
                        "confidence": float(confidence),
                        "is_correct": pred == ref,
                    }
                )

    return compute_classification_metrics(predictions, references, labels=labels), rows


def main() -> None:
    args = parse_args()
    train_rows = load_json(args.train_json)
    test_rows = load_json(args.test_json)
    if args.max_train_samples > 0:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_test_samples > 0:
        test_rows = test_rows[: args.max_test_samples]

    labels = load_labels(train_rows + test_rows, args.labels_json)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    prompt_learner = ClipPromptLearner(
        model=model,
        tokenizer=processor.tokenizer,
        labels=labels,
        n_ctx=args.n_ctx,
        label_prompt_template=args.label_prompt_template,
    ).to(device)

    train_ds = JsonImageDataset(train_rows, Path(args.image_root), label_to_idx)
    test_ds = JsonImageDataset(test_rows, Path(args.image_root), label_to_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_images)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_images)

    optimizer = torch.optim.AdamW([prompt_learner.context], lr=args.lr, weight_decay=args.weight_decay)
    loss_history: list[dict[str, Any]] = []

    for epoch in range(args.epochs):
        prompt_learner.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for batch in tqdm(train_loader, desc=f"train-epoch-{epoch + 1}"):
            targets = batch["targets"].to(device)
            image_features = encode_images(model, processor, batch["images"], device)
            text_features = prompt_learner()
            logits = model.logit_scale.exp() * image_features @ text_features.T
            loss = nn.functional.cross_entropy(logits, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * targets.numel()
            total += targets.numel()
            correct += int((logits.argmax(dim=-1) == targets).sum().item())

        record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(total, 1),
            "train_accuracy": correct / max(total, 1),
        }
        loss_history.append(record)
        print(json.dumps(record, ensure_ascii=False))

    metrics, pred_rows = evaluate(
        model=model,
        processor=processor,
        prompt_learner=prompt_learner,
        loader=test_loader,
        labels=labels,
        device=device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"context": prompt_learner.context.detach().cpu(), "labels": labels}, output_dir / "prompt.pt")
    summary = {
        "clip_model": args.clip_model,
        "label_prompt_template": args.label_prompt_template,
        "n_ctx": args.n_ctx,
        "epochs": args.epochs,
        "lr": args.lr,
        "metrics": metrics,
        "history": loss_history,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in pred_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("[done] outputs written to", output_dir)


if __name__ == "__main__":
    main()
