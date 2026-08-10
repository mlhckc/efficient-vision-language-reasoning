"""Tests for the E9 compact-VLM contextual baseline.

Every gate's FAILURE branch is exercised, not only its passing branch, so a
gate that could never fire would be caught here rather than in a report.
No test loads a compact VLM or touches the GPU; the model-dependent gates are
covered by the frozen preflight artefact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
E9_DIR = PROJECT_ROOT / "experiments" / "e9_compact_vlm"
for path in (str(PROJECT_ROOT), str(E9_DIR),
             str(PROJECT_ROOT / "experiments" / "e8a_question_encoder")):
    if path not in sys.path:
        sys.path.insert(0, path)

import e9_common as e9  # noqa: E402
import g21_scorer as g21  # noqa: E402

CHECKS = []


def check(name):
    def decorate(fn):
        CHECKS.append((name, fn))
        return fn
    return decorate


class _Halt(SystemExit):
    pass


def expect_halt(fn, gate: str) -> None:
    """A gate must actually fire. Calling a gate's failure branch and
    observing no halt is itself a failure."""
    original = e9.halt
    fired = {}

    def capture(g, reason):
        fired["gate"] = g
        fired["reason"] = reason
        raise _Halt(3)
    e9.halt = capture
    try:
        fn()
    except _Halt:
        pass
    except SystemExit:
        pass
    finally:
        e9.halt = original
    assert fired.get("gate") == gate, \
        f"expected gate {gate} to fire, got {fired.get('gate')!r}"


# --- Frozen constants ---------------------------------------------------------

@check("frozen constants are the approved ones")
def test_constants():
    assert e9.PROMPT_TEMPLATE == (
        "Answer the question using a single word or phrase.\n"
        "Question: {question}\nAnswer:")
    assert e9.OPEN_MAX_NEW_TOKENS == 20
    assert e9.DO_SAMPLE is False and e9.NUM_BEAMS == 1
    assert e9.TRIE_SUPPORT == "top_1000"
    assert e9.BRANCH_HALT_HOURS == 6.0
    assert e9.DETERMINISM_ROWS == 512
    assert e9.MODELS["smolvlm_256m"]["role"] == "PRIMARY"
    assert e9.MODELS["smolvlm_500m"]["role"] == "SECONDARY"
    assert e9.CONDITION_ROLE["blank"] == "AUXILIARY / UNMATCHED ABLATION"


@check("the pre-registered priority puts every mandatory cell first")
def test_priority():
    assert all(cell[0] == "smolvlm_256m" for cell in e9.MANDATORY_ORDER)
    assert all(cell[0] == "smolvlm_500m" for cell in e9.OPTIONAL_ORDER)
    assert len(e9.MANDATORY_ORDER) == 5


@check("greedy decoding is asserted, never sampled")
def test_greedy():
    assert e9.DO_SAMPLE is False
    assert e9.FROZEN_GENERATION_DEFAULTS["temperature"] is None
    assert e9.FROZEN_GENERATION_DEFAULTS["top_p"] is None


# --- G17, the clean-test embargo ---------------------------------------------

@check("G17 refuses any constructed embargoed path")
def test_g17_fires():
    # The forbidden token is assembled, never written as a literal, so the
    # run_all.py embargo source scan needs no exemption for this file.
    forbidden = "data/v2/" + "test_" + "clean_targets" + ".csv"
    expect_halt(lambda: e9.assert_no_clean_test_path(forbidden), "G17")


@check("G17 admits ordinary development paths")
def test_g17_passes():
    e9.assert_no_clean_test_path(e9.DEV_RAW, e9.VOCAB_100, e9.VOCAB_1000)


@check("the G17 source scan finds no embargoed reference in E9 code")
def test_embargo_source_scan():
    scan = e9.embargo_source_scan()
    assert scan["offending"] == []
    assert scan["clean_test_resolvable"] is False
    assert "e9_common.py" in scan["scanned"]
    assert "generate.py" in scan["scanned"]


# --- Derangement --------------------------------------------------------------

@check("the derangement reproduces E8B's frozen mapping exactly")
def test_derangement():
    frame = e9.load_dev_raw()
    mapping, provenance = e9.build_derangement(frame)
    assert provenance["sha256"] == e9.EXPECTED_DERANGEMENT_SHA256
    assert provenance["self_pairs"] == 0
    assert provenance["n_images"] == e9.RAW_CLUSTERS
    assert all(k != v for k, v in mapping.items())
    assert "IMAGE-PARTNER" in provenance["intervention_kind"]


@check("G-DERANGE fires when the mapping does not match the pin")
def test_derangement_gate_fires():
    original = e9.EXPECTED_DERANGEMENT_SHA256
    e9.EXPECTED_DERANGEMENT_SHA256 = "0" * 64
    try:
        expect_halt(lambda: e9.build_derangement(e9.load_dev_raw()),
                    "G-DERANGE")
    finally:
        e9.EXPECTED_DERANGEMENT_SHA256 = original


# --- Blank image --------------------------------------------------------------

@check("the blank image is the pinned mid-grey construction")
def test_blank():
    image = e9.blank_image()
    array = np.asarray(image)
    assert image.size == e9.BLANK_SIZE and image.mode == "RGB"
    assert array.min() == 128 and array.max() == 128
    assert len(e9.blank_image_sha256()) == 64


# --- Trie ---------------------------------------------------------------------

class _StubTokenizer:
    """Character-level stand-in: deterministic, and it makes the leading-space
    surface form behave exactly as a BPE tokenizer's does."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens=False,
               clean_up_tokenization_spaces=False):
        return "".join(chr(i) for i in ids)


@check("the trie accepts exactly the answer set, both surface forms")
def test_trie():
    tokenizer = _StubTokenizer()
    answers = ["yes", "no", "left", "man"]
    cache = e9.build_answer_sequences(tokenizer, answers)
    assert len(cache["sequences"]) == 2 * len(answers)
    root = e9.build_trie(cache)
    accepted = set()

    def walk(node, prefix):
        if node["leaf"] is not None:
            accepted.add(tokenizer.decode(prefix).strip())
        for token in sorted(node["children"]):
            walk(node["children"][token], prefix + [token])
    walk(root, [])
    assert accepted == set(answers)


@check("the trie walk returns None only off-path")
def test_trie_walk():
    tokenizer = _StubTokenizer()
    cache = e9.build_answer_sequences(tokenizer, ["yes"])
    root = e9.build_trie(cache)
    assert e9.trie_walk(root, [ord("y")]) is not None
    assert e9.trie_walk(root, [ord("z")]) is None
    node = e9.trie_walk(root, [ord(c) for c in "yes"])
    assert node["leaf"] == 0


@check("G-TRIE fires when two answers share one token sequence")
def test_trie_collision_gate():
    class Collide(_StubTokenizer):
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1]}

        def decode(self, ids, skip_special_tokens=False,
                   clean_up_tokenization_spaces=False):
            return "yes"
    expect_halt(lambda: e9.build_answer_sequences(Collide(), ["yes", "no"]),
                "G-TRIE")


# --- Scorer -------------------------------------------------------------------

@check("the G21 scorer behaves as independently expected")
def test_scorer():
    assert g21.normalized_exact("Yes.", "yes")
    assert g21.normalized_exact(" yes ", "yes")
    assert g21.normalized_exact("2", "two")
    assert g21.normalized_exact("A man", "man")
    assert not g21.normalized_exact("The answer is yes", "yes")
    assert not g21.normalized_exact("no", "yes")
    assert g21.raw_exact("yes", "yes")
    assert not g21.raw_exact("Yes.", "yes")


@check("normalisation is idempotent on generative-style emissions")
def test_scorer_idempotent():
    for text in (" Yes.", "No!!", "the man", "2.0", "left ", ""):
        once = g21.normalize_answer(text)
        assert g21.normalize_answer(once) == once


# --- Budget and memory gates --------------------------------------------------

@check("G19 fires when the projected branch total exceeds the halt")
def test_budget_gate():
    expect_halt(lambda: e9.assert_budget(e9.BRANCH_HALT_HOURS + 1.0), "G19")


@check("G19 fires above the 80 per cent memory ceiling")
def test_memory_gate():
    expect_halt(lambda: e9.assert_memory(90, 100), "G19")
    e9.assert_memory(50, 100)


# --- Atomic write and schema --------------------------------------------------

@check("atomic write leaves no temporary file and hashes what it wrote")
def test_atomic_write(tmp=None):
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "artefact.json"
        digest = e9.atomic_write_json(path, {"e9_test": {"value": 1}})
        assert path.exists()
        assert not [p for p in Path(directory).iterdir()
                    if p.suffix == ".tmp"]
        assert digest == e9.sha256_bytes(path.read_bytes())


@check("the run driver refuses a mislabelled artefact")
def test_stale_gate():
    import tempfile
    import run as e9run
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "artefact.json"
        e9.atomic_write_json(path, {"e9_generation": {
            "model": "smolvlm_500m", "role": "SECONDARY", "readout": "open",
            "readout_role": "PRIMARY", "condition": "normal",
            "condition_role": "PRIMARY", "rows": 1, "max_new_tokens": 20,
            "batch_size": 8, "elapsed_seconds": 1.0, "gpu_hours": 0.0,
            "peak_memory_reserved_bytes": 1, "mean_generated_tokens": 1.0,
            "empty_emissions": 0, "overlong_emissions": 0,
            "answer_support": "unconstrained", "clean_test_accessed": False,
            "tokens_sha256": "0" * 64},
            "texts": ["yes"], "questionIds": ["1"]})
        expect_halt(lambda: e9run.validate_artefact(
            path, "smolvlm_256m", "open", "normal", 1), "G-STALE")


@check("the run driver refuses an artefact missing schema fields")
def test_schema_gate():
    import tempfile
    import run as e9run
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "artefact.json"
        e9.atomic_write_json(path, {"e9_generation": {"model": "x"}})
        expect_halt(lambda: e9run.validate_artefact(
            path, "smolvlm_256m", "open", "normal", 1), "G-SCHEMA")


@check("the run driver refuses a missing artefact")
def test_missing_output_gate():
    import run as e9run
    expect_halt(lambda: e9run.validate_artefact(
        Path("/nonexistent/e9.json"), "smolvlm_256m", "open", "normal", 1),
        "G-OUTPUT")


# --- Denominators and alignment ----------------------------------------------

@check("the frozen denominators are the measured ones")
def test_denominators():
    frame = e9.load_dev_raw()
    assert len(frame) == e9.RAW_ROWS == 10_004
    assert int(frame["in_vocabulary"].sum()) == e9.IN_VOCAB_ROWS == 7_714
    assert frame["imageId"].nunique() == e9.RAW_CLUSTERS == 777
    top1000 = set(e9.load_vocabulary(e9.VOCAB_1000))
    assert int(frame["answer"].isin(top1000).sum()) == e9.TOP1000_ROWS
    index, n = e9.image_index_of(frame)
    assert n == 777 and len(index) == 10_004


@check("vocabulary hashes are the pinned ones")
def test_vocab_hashes():
    assert e9.sha256_file(e9.VOCAB_100) == e9.VOCAB_100_SHA256
    assert e9.sha256_file(e9.VOCAB_1000) == e9.VOCAB_1000_SHA256


@check("the analysis two-condition rule never claims condition 3")
def test_two_condition_rule():
    import analyse
    positive = analyse.two_condition_outcome(
        {"mean_difference": 0.1, "ci_lower": 0.05, "ci_upper": 0.15})
    assert positive.startswith("positive") and "condition 3" in positive
    unclear = analyse.two_condition_outcome(
        {"mean_difference": 0.1, "ci_lower": -0.05, "ci_upper": 0.15})
    assert unclear == "uncertain or mixed evidence"
    negative = analyse.two_condition_outcome(
        {"mean_difference": -0.1, "ci_lower": -0.15, "ci_upper": -0.05})
    assert negative.startswith("negative")


@check("the fast paired interval equals the general row-materialising form")
def test_bootstrap_equivalence():
    import analyse
    rng = np.random.default_rng(7)
    index = np.repeat(np.arange(40), 5)
    a = rng.random(200) < 0.6
    b = [rng.random(200) < 0.5, rng.random(200) < 0.55]
    fast = analyse.paired_interval(a, b, index, 40)

    def statistic(rows):
        return (float(a[rows].mean())
                - float(np.mean([h[rows].mean() for h in b])))
    slow = analyse.clustered_draws(statistic, index, 40)
    lower, upper = analyse.percentile_interval(slow)
    assert fast["ci_lower"] == lower and fast["ci_upper"] == upper, \
        f"fast {fast['ci_lower']},{fast['ci_upper']} vs slow {lower},{upper}"


@check("the clustered bootstrap resamples images, not rows")
def test_bootstrap():
    import analyse
    index = np.array([0, 0, 1, 1, 2, 2])
    hits = np.array([1, 1, 0, 0, 1, 1], dtype=bool)
    draws = analyse.clustered_draws(lambda rows: hits[rows].mean(), index, 3)
    assert len(draws) == e9.BOOTSTRAP_DRAWS
    # Every draw takes whole images, so only 0, 1/3, 2/3 and 1 are reachable.
    assert set(np.round(np.unique(draws), 6)) <= {0.0, round(1 / 3, 6),
                                                  round(2 / 3, 6), 1.0}


def run() -> None:
    """The tests/run_all.py entry point: raise on the first failure."""
    failures = []
    for name, fn in CHECKS:
        try:
            fn()
        except Exception as error:  # noqa: BLE001
            failures.append((name, f"{type(error).__name__}: {error}"))
    if failures:
        raise AssertionError("; ".join(f"{n}: {r}" for n, r in failures))
    print(f"  test_e9: {len(CHECKS)} checks passed")


def main() -> int:
    try:
        run()
    except AssertionError as error:
        print(f"FAIL test_e9: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
