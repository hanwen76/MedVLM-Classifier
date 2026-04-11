from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from .formatters import PromptTemplateConfig, build_classification_dialog


@dataclass
class ClassificationSample:
    image_path: str
    label: str


def build_eval_prompt(config: PromptTemplateConfig | None = None) -> str:
    config = config or PromptTemplateConfig()
    return (
        f"{config.user_prefix} <image>\n"
        f"{config.question}\n"
        f"{config.assistant_prefix}"
    )


class MedicalClassificationDataset(Dataset):
    """Supports CSV/JSON/JSONL medical image classification annotations."""

    def __init__(self, annotation_path: str, image_root: str | None = None, prompt_cfg: PromptTemplateConfig | None = None):
        self.annotation_path = Path(annotation_path)
        self.image_root = Path(image_root) if image_root else self.annotation_path.parent
        self.prompt_cfg = prompt_cfg or PromptTemplateConfig()
        self.samples = self._load_samples(self.annotation_path)

    def _load_samples(self, path: Path) -> list[ClassificationSample]:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return self._load_csv(path)
        if suffix == ".json":
            return self._load_json(path)
        if suffix == ".jsonl":
            return self._load_jsonl(path)
        raise ValueError(f"Unsupported annotation format: {suffix}")

    def _load_csv(self, path: Path) -> list[ClassificationSample]:
        rows: list[ClassificationSample] = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(ClassificationSample(image_path=row["image"], label=row["label"]))
        return rows

    def _load_json(self, path: Path) -> list[ClassificationSample]:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return [ClassificationSample(image_path=item["image"], label=item["label"]) for item in data]

    def _load_jsonl(self, path: Path) -> list[ClassificationSample]:
        rows: list[ClassificationSample] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                rows.append(ClassificationSample(image_path=item["image"], label=item["label"]))
        return rows

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image_path = self.image_root / sample.image_path
        image = Image.open(image_path).convert("RGB")
        dialog = build_classification_dialog(label=sample.label, config=self.prompt_cfg)
        return {
            "image": image,
            "text": dialog,
            "label_name": sample.label,
            "source": "classification",
        }


class MedicalClassificationEvalDataset(Dataset):
    """Classification dataset for inference-time prompts without answer leakage."""

    def __init__(self, annotation_path: str, image_root: str | None = None, prompt_cfg: PromptTemplateConfig | None = None):
        self.annotation_path = Path(annotation_path)
        self.image_root = Path(image_root) if image_root else self.annotation_path.parent
        self.prompt_cfg = prompt_cfg or PromptTemplateConfig()
        self.samples = MedicalClassificationDataset(
            annotation_path=annotation_path,
            image_root=image_root,
            prompt_cfg=prompt_cfg,
        ).samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image_path = self.image_root / sample.image_path
        image = Image.open(image_path).convert("RGB")
        prompt = build_eval_prompt(config=self.prompt_cfg)
        return {
            "image": image,
            "text": prompt,
            "label_name": sample.label,
            "source": "classification_eval",
        }


class InstructionTuneDataset(Dataset):
    """Generic instruction tuning data reader from JSONL.

    JSONL format:
    {"text": "USER: ... ASSISTANT: ...", "image": "optional/path.png"}
    """

    def __init__(self, instruction_jsonl: str, image_root: str | None = None):
        self.path = Path(instruction_jsonl)
        self.image_root = Path(image_root) if image_root else self.path.parent
        self.samples = self._load_jsonl(self.path)

    def _load_jsonl(self, path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return rows

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        image = None
        if item.get("image"):
            image = Image.open(self.image_root / item["image"]).convert("RGB")
        return {
            "image": image,
            "text": item["text"],
            "label_name": item.get("label"),
            "source": "instruction",
        }
