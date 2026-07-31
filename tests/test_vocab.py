"""Vocabulary integrity: the binding V2 map and the recorded V1/V2 index
divergence that forbids mixing label spaces."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run() -> None:
    vocab = json.loads(
        (PROJECT_ROOT / "data/v2/answer_vocab_v2.json").read_text()
    )
    assert vocab["top_k"] == 100
    assert len(vocab["answers"]) == 100
    mapping = vocab["answer_to_index"]
    assert len(mapping) == 100
    assert sorted(mapping.values()) == list(range(100)), "indices not 0..99"
    assert set(mapping) == set(vocab["answers"])

    comparison = json.loads(
        (
            PROJECT_ROOT
            / "artifacts/v2_00_protocol/v1_v2_vocab_index_comparison.json"
        ).read_text()
    )
    assert comparison["answer_sets_identical"] is True
    assert comparison["n_answers_with_changed_index"] == 11
    assert comparison["v1_vocabulary_size"] == 100
    assert comparison["v2_vocabulary_size"] == 100
