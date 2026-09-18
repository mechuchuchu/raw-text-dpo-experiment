import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rawtext_dpo.serialization import serialize_branch, serialize_pair


def test_aligned_forward_spans():
    result = serialize_pair("P", "C", "R")
    assert result.text == "###prompt\nP\n\n###chosen\nC\n\n###rejected\nR"
    assert [(span.semantic, span.label, span.answer) for span in result.spans] == [
        ("chosen", "chosen", "C"),
        ("rejected", "rejected", "R"),
    ]
    assert result.text[result.spans[0].start:result.spans[0].end] == "C"
    assert result.text[result.spans[1].start:result.spans[1].end] == "R"


def test_swapped_reverse_keeps_semantic_spans():
    result = serialize_pair("P", "C", "R", arrangement="swapped", order="reverse")
    assert result.text == "###prompt\nP\n\n###rejected\nC\n\n###chosen\nR"
    assert [(span.semantic, span.label) for span in result.spans] == [
        ("chosen", "rejected"),
        ("rejected", "chosen"),
    ]


def test_branch_serialization():
    result = serialize_branch("P", "C", semantic="chosen", label="rejected")
    assert result.text == "###prompt\nP\n\n###rejected\nC"
    assert result.text[result.spans[0].start:result.spans[0].end] == "C"
