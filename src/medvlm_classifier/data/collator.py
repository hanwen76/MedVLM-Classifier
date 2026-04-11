from __future__ import annotations

from typing import Any

import torch


class VLMDataCollator:
    """Collate image-text pairs into model-ready tensors with label masking."""

    def __init__(self, processor, max_length: int = 1024, assistant_prefix: str = "ASSISTANT:"):
        self.processor = processor
        self.max_length = max_length
        self.assistant_prefix = assistant_prefix

    def _mask_prompt_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return labels

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [item["text"] for item in batch]
        images = [item["image"] for item in batch]

        if any(img is None for img in images):
            first_valid = next((img for img in images if img is not None), None)
            if first_valid is None:
                raise ValueError("Batch has no valid images. This VLM setup expects image-conditioned batches.")
            images = [img if img is not None else first_valid.copy() for img in images]

        model_inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = model_inputs["input_ids"]
        labels = self._mask_prompt_tokens(input_ids)

        if self.assistant_prefix:
            prefix_ids = self.processor.tokenizer(self.assistant_prefix, add_special_tokens=False)["input_ids"]
            prefix_len = len(prefix_ids)
            for i in range(input_ids.size(0)):
                seq = input_ids[i].tolist()
                start = -1
                for j in range(max(1, len(seq) - prefix_len + 1)):
                    if seq[j : j + prefix_len] == prefix_ids:
                        start = j + prefix_len
                        break
                if start > 0:
                    labels[i, :start] = -100

        model_inputs["labels"] = labels
        return model_inputs
