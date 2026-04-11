from __future__ import annotations

from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)


def load_model_and_processor_by_id(model_id: str, torch_dtype="auto"):
    model_id_l = model_id.lower()

    if "llava-v1.6" in model_id_l:
        processor = LlavaNextProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        return model, processor

    if "blip2" in model_id_l:
        processor = Blip2Processor.from_pretrained(model_id, trust_remote_code=True)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        return model, processor

    if "instructblip" in model_id_l:
        processor = InstructBlipProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        return model, processor

    if "llava" in model_id_l:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        return model, processor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    return model, processor
