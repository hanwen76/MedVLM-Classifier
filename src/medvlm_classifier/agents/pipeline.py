from __future__ import annotations

import torch
from PIL.Image import Image

from medvlm_classifier.agents import ImageWikiQARetriever


class ClassifyAndRetrievePipeline:
    """Extension point: classify image then trigger ImageWikiQA retrieval."""

    def __init__(self, model, processor, retriever: ImageWikiQARetriever, prompt: str = "这张医学影像显示了什么？"):
        self.model = model
        self.processor = processor
        self.retriever = retriever
        self.prompt = prompt

    def __call__(self, image: Image, max_new_tokens: int = 32) -> dict:
        device = next(self.model.parameters()).device
        model_inputs = self.processor(
            text=[f"USER: <image>\\n{self.prompt}\\nASSISTANT:"],
            images=[image],
            return_tensors="pt",
        )
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        with torch.no_grad():
            generated = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]

        disease = text.split("ASSISTANT:")[-1].strip()
        retrieved = self.retriever.retrieve(disease)
        return {
            "prediction_text": text,
            "predicted_disease": disease,
            "retrieval": retrieved,
        }
