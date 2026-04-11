from __future__ import annotations

from typing import Iterable

from transformers import AutoModelForVision2Seq, AutoProcessor


PROJECTOR_KEYWORDS = (
    "projector",
    "multi_modal_projector",
    "mm_projector",
    "visual_projection",
)


def load_vlm_and_processor(model_name_or_path: str, torch_dtype="auto"):
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    return model, processor


def _is_projector_param(name: str, extra_keywords: Iterable[str] | None = None) -> bool:
    keywords = list(PROJECTOR_KEYWORDS)
    if extra_keywords:
        keywords.extend(extra_keywords)
    name_l = name.lower()
    return any(keyword in name_l for keyword in keywords)


def freeze_all_except_projector(model, extra_keywords: Iterable[str] | None = None) -> None:
    """Freeze Vision Encoder + LLM; only projector-related params remain trainable."""
    for name, param in model.named_parameters():
        param.requires_grad = _is_projector_param(name, extra_keywords=extra_keywords)


def summarize_trainable_params(model) -> dict[str, int]:
    total = 0
    trainable = 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
