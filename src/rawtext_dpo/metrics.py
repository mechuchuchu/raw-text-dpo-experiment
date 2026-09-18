from __future__ import annotations

from typing import Any

import torch


@torch.no_grad()
def masked_mean_nll(model: Any, item: dict[str, Any], target: str) -> tuple[float, int]:
    input_ids = item["input_ids"].unsqueeze(0).to(model.device)
    attention_mask = item["attention_mask"].unsqueeze(0).to(model.device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]
    token_nll = -log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)[0]
    target_mask = item["target_masks"].get(target)
    if target_mask is None:
        raise KeyError(f"Missing target mask: {target}")
    target_mask = target_mask[1:].to(model.device)
    count = int(target_mask.sum().item())
    if count == 0:
        raise ValueError(f"No target tokens available for {target}")
    return float(token_nll[target_mask].mean().item()), count
