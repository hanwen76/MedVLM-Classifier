from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image
from tqdm import trange

from medvlm_classifier.model.factory import load_model_and_processor_by_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Reference-style inference aligned with VLMClassifier-main/main_results/main.py")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True, help="JSONL with fields image,label,split")
    parser.add_argument("--class_path", type=str, required=True, help="JSON list of class names")
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--including_label", action="store_true")
    parser.add_argument("--n_labels", type=int, default=1000)
    parser.add_argument("--chain_of_thought", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--fixed_order", action="store_true")
    parser.add_argument("--init_prompt", type=str, default="What type of object is in this photo?")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(model_id: str, question: str, chain_of_thought: bool) -> str:
    model_id_l = model_id.lower()

    if "llava-v1.6-mistral" in model_id_l:
        if chain_of_thought:
            return f"[INST] <image>\\n{question}[/INST] Let's think step by step."
        return f"[INST] <image>\\n{question}[/INST]"

    if "llava-v1.6-vicuna" in model_id_l:
        prefix = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
            "USER: <image>\\n"
        )
        if chain_of_thought:
            return f"{prefix}{question} ASSISTANT: Let's think step by step."
        return f"{prefix}{question} ASSISTANT:"

    if "blip" in model_id_l:
        if chain_of_thought:
            return f"Question: {question} Let's think step by step. Answer:"
        return f"Question: {question} Answer:"

    if chain_of_thought:
        return f"USER: <image>\\n{question}\\nASSISTANT: Let's think step by step."
    return f"USER: <image>\\n{question}\\nASSISTANT:"


def extract_pred(model_id: str, text: str) -> str:
    model_id_l = model_id.lower()
    if "mistral" in model_id_l:
        return text.split("[/INST]")[-1].strip()
    if "blip" in model_id_l:
        return text.split("Answer:")[-1].strip()
    return text.split("ASSISTANT:")[-1].strip()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    model, processor = load_model_and_processor_by_id(args.model_id, torch_dtype=torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    data = [item for item in load_jsonl(args.data_path) if item.get("split", args.split) == args.split]
    classes = json.load(open(args.class_path, "r", encoding="utf-8"))
    random.shuffle(data)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and output_path.exists():
        done = load_jsonl(str(output_path))
        data = data[len(done) :]

    with output_path.open("a", encoding="utf-8") as fw:
        for i in trange(0, len(data), args.batch_size):
            batch = data[i : i + args.batch_size]
            images = [Image.open(item["image"]).convert("RGB") for item in batch]

            if args.including_label:
                choices = []
                for item in batch:
                    if args.fixed_order:
                        if args.n_labels != len(classes):
                            raise ValueError("When --fixed_order is enabled, --n_labels must equal len(classes).")
                        choices.append(classes)
                    else:
                        label = item["label"]
                        negatives = random.sample(sorted(list(set(classes) - {label})), args.n_labels - 1)
                        cur_choices = [label] + negatives
                        random.shuffle(cur_choices)
                        choices.append(cur_choices)
                questions = [f"{args.init_prompt} Choose one from \"{', '.join(c)}\"." for c in choices]
            else:
                questions = [args.init_prompt for _ in batch]

            prompts = [build_prompt(args.model_id, q, args.chain_of_thought) for q in questions]
            inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=512 if args.chain_of_thought else args.max_new_tokens,
                )
            generated_text = processor.batch_decode(output, skip_special_tokens=True)

            for item, text in zip(batch, generated_text):
                row = dict(item)
                row["output"] = text
                row["pred"] = extract_pred(args.model_id, text)
                fw.write(json.dumps(row, ensure_ascii=False) + "\\n")
                fw.flush()


if __name__ == "__main__":
    main()
