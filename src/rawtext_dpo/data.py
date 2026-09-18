from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset


def _content(message: dict[str, Any]) -> Any:
    return message.get("content") if isinstance(message, dict) else None


def normalize_row(
    row: dict[str, Any],
    *,
    single_turn_only: bool = True,
    drop_empty_answers: bool = True,
) -> tuple[dict[str, Any] | None, str | None]:
    """Convert a Dolci row into the compact experiment representation."""
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not isinstance(chosen, list) or not isinstance(rejected, list):
        return None, "invalid_message_list"

    chosen_roles = [m.get("role") for m in chosen if isinstance(m, dict)]
    rejected_roles = [m.get("role") for m in rejected if isinstance(m, dict)]
    if single_turn_only and (
        chosen_roles != ["user", "assistant"]
        or rejected_roles != ["user", "assistant"]
    ):
        return None, "not_single_turn"
    if len(chosen) < 2 or len(rejected) < 2:
        return None, "too_few_messages"

    prompt = _content(chosen[0])
    chosen_answer = _content(chosen[-1])
    rejected_answer = _content(rejected[-1])
    if not all(isinstance(value, str) for value in (prompt, chosen_answer, rejected_answer)):
        return None, "non_string_content"
    if drop_empty_answers and not all(value.strip() for value in (prompt, chosen_answer, rejected_answer)):
        return None, "empty_content"

    prompt_id = row.get("prompt_id")
    group_id = str(prompt_id) if prompt_id else hashlib.sha256(prompt.encode()).hexdigest()
    return {
        "sample_id": group_id,
        "prompt_id": prompt_id,
        "prompt": prompt,
        "chosen": chosen_answer,
        "rejected": rejected_answer,
        "chosen_model": row.get("chosen_model"),
        "rejected_model": row.get("rejected_model"),
        "preference_type": row.get("preference_type"),
        "group_id": group_id,
    }, None


def load_normalized_rows(config: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter]:
    dataset_config = config["dataset"]
    dataset = load_dataset(
        dataset_config["name"],
        revision=dataset_config.get("revision"),
        split=dataset_config.get("split", "train"),
    )
    rows: list[dict[str, Any]] = []
    filters: Counter = Counter()
    for raw_row in dataset:
        normalized, reason = normalize_row(
            raw_row,
            single_turn_only=dataset_config.get("single_turn_only", True),
            drop_empty_answers=dataset_config.get("drop_empty_answers", True),
        )
        if normalized is None:
            filters[reason or "unknown"] += 1
            continue
        rows.append(normalized)
    return rows, filters


def sample_and_split(
    rows: list[dict[str, Any]],
    *,
    sample_size: int,
    eval_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if sample_size > len(rows):
        raise ValueError(f"Requested {sample_size} rows, but only {len(rows)} are valid")
    rng = random.Random(seed)
    selected = list(rows)
    rng.shuffle(selected)
    selected = selected[:sample_size]

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[row["group_id"]].append(row)
    group_ids = list(groups)
    rng.shuffle(group_ids)
    target_eval = max(1, round(sample_size * eval_fraction))

    eval_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    eval_count = 0
    for group_id in group_ids:
        group_rows = groups[group_id]
        if eval_count < target_eval:
            eval_rows.extend(group_rows)
            eval_count += len(group_rows)
        else:
            train_rows.extend(group_rows)
    return train_rows, eval_rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
