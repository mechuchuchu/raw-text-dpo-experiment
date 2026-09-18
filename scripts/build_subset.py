#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rawtext_dpo.config import load_config, write_json
from rawtext_dpo.data import load_normalized_rows, sample_and_split, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/sanity.yaml"))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    dataset_config = config["dataset"]
    output_root = Path(args.output_dir or config["output"]["root_dir"])
    run_id = f"{config['experiment_name']}_seed{seed}_n{dataset_config['sample_size']}"
    output_dir = output_root / "subsets" / run_id

    rows, filters = load_normalized_rows(config)
    train_rows, eval_rows = sample_and_split(
        rows,
        sample_size=int(dataset_config["sample_size"]),
        eval_fraction=float(dataset_config["eval_fraction"]),
        seed=seed,
    )
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "eval.jsonl", eval_rows)
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "dataset": dataset_config,
        "total_valid_rows": len(rows),
        "sample_rows": len(train_rows) + len(eval_rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "filter_counts": dict(filters),
        "train_ids": [row["sample_id"] for row in train_rows],
        "eval_ids": [row["sample_id"] for row in eval_rows],
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), **{k: manifest[k] for k in ("total_valid_rows", "train_rows", "eval_rows", "filter_counts")}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
