#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rawtext_dpo.config import load_config
from rawtext_dpo.metrics import masked_mean_nll
from rawtext_dpo.serialization import serialize_branch, serialize_pair
from rawtext_dpo.tokenization import tokenize_serialized


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def evaluate_item(model, tokenizer, row, *, arrangement, order, scoring_mode, max_length):
    if scoring_mode == "full_sequence":
        serialized = serialize_pair(
            row["prompt"], row["chosen"], row["rejected"],
            arrangement=arrangement, order=order,
        )
        item = tokenize_serialized(tokenizer, serialized, max_length=max_length)
        if item["over_limit"]:
            return None
        chosen_loss, chosen_tokens = masked_mean_nll(model, item, "chosen")
        rejected_loss, rejected_tokens = masked_mean_nll(model, item, "rejected")
        return {
            "chosen_tokens": chosen_tokens,
            "rejected_tokens": rejected_tokens,
            "chosen_loss": chosen_loss,
            "rejected_loss": rejected_loss,
        }

    if scoring_mode != "isolated_branch":
        raise ValueError(f"Unknown scoring mode: {scoring_mode}")
    semantic_values = {"chosen": row["chosen"], "rejected": row["rejected"]}
    outputs = {}
    for semantic, answer in semantic_values.items():
        label = semantic if arrangement == "aligned" else ("rejected" if semantic == "chosen" else "chosen")
        serialized = serialize_branch(row["prompt"], answer, semantic=semantic, label=label)
        item = tokenize_serialized(tokenizer, serialized, max_length=max_length)
        if item["over_limit"]:
            return None
        loss, tokens = masked_mean_nll(model, item, semantic)
        outputs[f"{semantic}_loss"] = loss
        outputs[f"{semantic}_tokens"] = tokens
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/sanity.yaml"))
    parser.add_argument("--subset", required=True, help="Subset directory containing eval.jsonl")
    parser.add_argument("--model", default=None, help="Base model name/path, or LoRA adapter directory")
    parser.add_argument("--base-model", default=None, help="Base model for loading a LoRA adapter")
    parser.add_argument("--model-variant", default="base")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    model_name = args.model or config["model"]["name"]
    max_length = int(config["training"]["max_seq_length"])
    subset = Path(args.subset)
    rows = read_jsonl(subset / "eval.jsonl", args.limit)
    is_lora = args.model_variant.lower() in {"lora", "adapter"}
    base_model_name = args.base_model or config["model"]["name"]
    tokenizer_name = model_name if is_lora else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if is_lora:
        if not args.model:
            raise ValueError("--model must point to a LoRA adapter directory when --model-variant=lora")
        model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype="auto")
        model = PeftModel.from_pretrained(model, args.model)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    output_path = Path(args.output or config["output"]["root_dir"]) / "results" / f"{args.model_variant}_evaluation.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id", "prompt_id", "model_variant", "arrangement", "order", "scoring_mode",
        "chosen_tokens", "rejected_tokens", "chosen_loss", "rejected_loss", "loss_difference",
        "truncated",
    ]
    written = 0
    skipped = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for arrangement in config["evaluation"]["arrangements"]:
                for order in config["evaluation"]["orders"]:
                    for scoring_mode in config["evaluation"]["scoring_modes"]:
                        result = evaluate_item(
                            model, tokenizer, row,
                            arrangement=arrangement,
                            order=order,
                            scoring_mode=scoring_mode,
                            max_length=max_length,
                        )
                        if result is None:
                            skipped += 1
                            continue
                        writer.writerow({
                            "sample_id": row["sample_id"],
                            "prompt_id": row.get("prompt_id"),
                            "model_variant": args.model_variant,
                            "arrangement": arrangement,
                            "order": order,
                            "scoring_mode": scoring_mode,
                            "chosen_tokens": result["chosen_tokens"],
                            "rejected_tokens": result["rejected_tokens"],
                            "chosen_loss": result["chosen_loss"],
                            "rejected_loss": result["rejected_loss"],
                            "loss_difference": result["chosen_loss"] - result["rejected_loss"],
                            "truncated": False,
                        })
                        written += 1
    print(json.dumps({"output": str(output_path), "written": written, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
