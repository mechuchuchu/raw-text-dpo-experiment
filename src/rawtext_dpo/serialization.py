from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    semantic: str
    label: str
    answer: str
    start: int
    end: int


@dataclass(frozen=True)
class SerializedText:
    text: str
    spans: tuple[Span, ...]


def _segment(label: str, answer: str) -> str:
    return f"###{label}\n{answer}"


def serialize_pair(
    prompt: str,
    chosen: str,
    rejected: str,
    *,
    arrangement: str = "aligned",
    order: str = "forward",
) -> SerializedText:
    if arrangement not in {"aligned", "swapped"}:
        raise ValueError(f"Unknown arrangement: {arrangement}")
    if order not in {"forward", "reverse"}:
        raise ValueError(f"Unknown order: {order}")

    label_answers = {
        "chosen": ("chosen", chosen),
        "rejected": ("rejected", rejected),
    }
    if arrangement == "swapped":
        label_answers = {
            "chosen": ("rejected", rejected),
            "rejected": ("chosen", chosen),
        }
    labels = ["chosen", "rejected"] if order == "forward" else ["rejected", "chosen"]

    prefix = f"###prompt\n{prompt}\n\n"
    pieces = [prefix]
    spans: list[Span] = []
    cursor = len(prefix)
    for index, label in enumerate(labels):
        semantic, answer = label_answers[label]
        segment = _segment(label, answer)
        answer_start = cursor + len(f"###{label}\n")
        pieces.append(segment)
        spans.append(Span(semantic, label, answer, answer_start, answer_start + len(answer)))
        cursor += len(segment)
        if index < len(labels) - 1:
            pieces.append("\n\n")
            cursor += 2
    return SerializedText("".join(pieces), tuple(spans))


def serialize_branch(
    prompt: str,
    answer: str,
    *,
    semantic: str,
    label: str,
) -> SerializedText:
    prefix = f"###prompt\n{prompt}\n\n###{label}\n"
    return SerializedText(
        prefix + answer,
        (Span(semantic, label, answer, len(prefix), len(prefix) + len(answer)),),
    )
