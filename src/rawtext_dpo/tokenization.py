from __future__ import annotations

from typing import Any

import torch

from .serialization import SerializedText


def tokenize_serialized(
    tokenizer: Any,
    serialized: SerializedText,
    *,
    max_length: int | None = None,
) -> dict[str, Any]:
    encoded = tokenizer(
        serialized.text,
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]
    offsets = encoded["offset_mapping"][0].tolist()
    over_limit = max_length is not None and input_ids.numel() > max_length

    masks: dict[str, torch.Tensor] = {}
    for span in serialized.spans:
        mask = torch.zeros(input_ids.shape[0], dtype=torch.bool)
        for index, (start, end) in enumerate(offsets):
            if end > span.start and start < span.end:
                mask[index] = True
        if span.semantic in masks:
            masks[span.semantic] |= mask
        else:
            masks[span.semantic] = mask

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_masks": masks,
        "num_tokens": int(input_ids.numel()),
        "over_limit": bool(over_limit),
    }


def truncate_tokenized(item: dict[str, Any], max_length: int) -> dict[str, Any]:
    """Apply left-preserving truncation while keeping target masks aligned."""
    if item["num_tokens"] <= max_length:
        return item
    result = dict(item)
    result["input_ids"] = item["input_ids"][:max_length]
    result["attention_mask"] = item["attention_mask"][:max_length]
    result["target_masks"] = {
        key: value[:max_length] for key, value in item["target_masks"].items()
    }
    result["num_tokens"] = max_length
    result["over_limit"] = True
    return result
