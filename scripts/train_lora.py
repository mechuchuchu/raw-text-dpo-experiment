#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rawtext_dpo.config import load_config
from rawtext_dpo.serialization import serialize_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/sanity.yaml"))
    parser.add_argument("--subset", required=True, help="Subset directory containing train.jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    model_name = config["model"]["name"]
    max_length = int(config["training"]["max_seq_length"])
    subset = Path(args.subset)
    output_dir = Path(args.output or config["output"]["root_dir"]) / "checkpoints" / config["experiment_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    lora = config["model"]
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM",
        r=int(lora["lora_r"]),
        lora_alpha=int(lora["lora_alpha"]),
        lora_dropout=float(lora["lora_dropout"]),
        target_modules=list(lora["target_modules"]),
        bias="none",
    ))
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=str(subset / "train.jsonl"), split="train")

    def tokenize(row):
        serialized = serialize_pair(
            row["prompt"], row["chosen"], row["rejected"],
            arrangement="aligned",
            order=config["training"].get("serialization_order", "forward"),
        )
        return tokenizer(
            serialized.text,
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    training = config["training"]
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        num_train_epochs=float(training["num_train_epochs"]),
        logging_steps=int(training["logging_steps"]),
        save_steps=int(training["save_steps"]),
        save_strategy="steps",
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"saved adapter and tokenizer to {output_dir / 'final'}")


if __name__ == "__main__":
    main()
