import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rawtext_dpo.data import normalize_row, sample_and_split


def row(prompt="P", chosen="C", rejected="R"):
    return {
        "chosen": [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}],
        "rejected": [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected}],
        "prompt_id": prompt,
        "chosen_model": "a",
        "rejected_model": "b",
        "preference_type": "test",
    }


def test_normalize_single_turn():
    normalized, reason = normalize_row(row())
    assert reason is None
    assert normalized["prompt"] == "P"
    assert normalized["chosen"] == "C"
    assert normalized["rejected"] == "R"


def test_normalize_rejects_empty_answer():
    normalized, reason = normalize_row(row(rejected=""))
    assert normalized is None
    assert reason == "empty_content"


def test_split_has_no_group_overlap():
    rows = [normalize_row(row(f"p{i}", f"c{i}", f"r{i}"))[0] for i in range(20)]
    train, evaluation = sample_and_split(rows, sample_size=20, eval_fraction=0.2, seed=42)
    assert not {x["group_id"] for x in train} & {x["group_id"] for x in evaluation}
    assert len(train) + len(evaluation) == 20
