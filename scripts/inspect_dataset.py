#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_dataset

from rawtext_dpo.config import load_config
from rawtext_dpo.data import normalize_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/sanity.yaml"))
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    config = load_config(args.config)
    dataset_config = config["dataset"]
    dataset = load_dataset(
        dataset_config["name"],
        revision=dataset_config.get("revision"),
        split=dataset_config.get("split", "train"),
        streaming=True,
    )
    filters: Counter = Counter()
    valid = 0
    preference_types: Counter = Counter()
    message_counts: Counter = Counter()
    first_example = None
    for raw_row in dataset.take(args.limit):
        normalized, reason = normalize_row(
            raw_row,
            single_turn_only=False,
            drop_empty_answers=False,
        )
        if normalized is None:
            filters[reason or "unknown"] += 1
            continue
        valid += 1
        preference_types[normalized["preference_type"]] += 1
        message_counts[len(raw_row["chosen"])] += 1
        if first_example is None:
            first_example = {
                "keys": list(raw_row),
                "chosen_roles": [m.get("role") for m in raw_row["chosen"]],
                "rejected_roles": [m.get("role") for m in raw_row["rejected"]],
                "metadata": {
                    "chosen_model": raw_row.get("chosen_model"),
                    "rejected_model": raw_row.get("rejected_model"),
                    "prompt_id": raw_row.get("prompt_id"),
                    "preference_type": raw_row.get("preference_type"),
                },
            }
    print(json.dumps({
        "dataset": dataset_config["name"],
        "split": dataset_config.get("split", "train"),
        "inspected_rows": args.limit,
        "valid_rows_under_general_normalization": valid,
        "filter_counts": dict(filters),
        "preference_types": dict(preference_types),
        "chosen_message_count": dict(message_counts),
        "first_example": first_example,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
