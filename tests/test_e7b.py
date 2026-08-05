"""E7b serial-benchmark tests: gates, registry, timing math, immutability.

CPU-only: no GPU work, no encoder forward pass, no store opened. Every
load-bearing gate is exercised as a known-positive and a known-negative,
following the project convention that a check that cannot fail is not
evidence. The embargoed clean-test target is never read or named.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.e7b_serial_efficiency import run as e7b  # noqa: E402

VERBOSE = False

_CHECKS: list[tuple[str, bool]] = []
_INSIDE_KNOWN_NEGATIVE = False


def check(name: str, condition: bool, detail: str = "") -> None:
    if not _INSIDE_KNOWN_NEGATIVE:
        _CHECKS.append((name, bool(condition)))
        if VERBOSE:
            print(f"  [{'PASS' if condition else 'FAIL'}] {name}"
                  + (f" — {detail}" if detail else ""))
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def must_fail(name: str, thunk) -> None:
    global _INSIDE_KNOWN_NEGATIVE
    _INSIDE_KNOWN_NEGATIVE = True
    try:
        thunk()
        raised = False
    except (AssertionError, KeyError, RuntimeError, ValueError, IndexError,
            TypeError, SystemExit, FileNotFoundError):
        raised = True
    finally:
        _INSIDE_KNOWN_NEGATIVE = False
    check(f"{name} (known-negative raises)", raised,
          "" if raised else "the unsafe path was NOT refused")


# --- Registry integrity -------------------------------------------------------

def test_registry() -> None:
    check("eight systems registered", len(e7b.SYSTEMS) == 8)
    check("question_only is the only context-only system",
          [n for n, s in e7b.SYSTEMS.items() if s["context_only"]]
          == ["question_only"])
    check("the fp16-cached class is exactly the three token systems",
          sorted(n for n, s in e7b.SYSTEMS.items() if s["fp16_cached"])
          == ["e8a_a0p", "e8a_a1", "reasoner"])
    for name, spec in e7b.SYSTEMS.items():
        checkpoint = PROJECT_ROOT / spec["checkpoint"]
        check(f"checkpoint exists for {name}", checkpoint.exists(),
              spec["checkpoint"])
        check(f"seed-0 policy in {name}'s checkpoint name",
              "seed0" in checkpoint.name)
        value = e7b.stored_accuracy(spec["seed0_accuracy"])
        check(f"stored seed-0 accuracy resolves for {name}",
              0.30 < value < 0.70, str(value))
        context = e7b.multiseed_context(spec["multiseed"])
        check(f"multi-seed context resolves for {name}",
              0.30 < context["mean"] < 0.70 and context["std"] >= 0)
    must_fail("an unknown accuracy route fails closed",
              lambda: e7b.stored_accuracy(("results/experiments/"
                                           "v2_07_scaling/results.json",
                                           "no_such_key")))


# --- Stage-name contracts -----------------------------------------------------

def test_stage_names() -> None:
    multimodal = e7b.build_pipeline("fusion")
    check("global multimodal pipeline has nine stages (no projection)",
          len(multimodal.stage_names) == 9
          and "S8_projection" not in multimodal.stage_names)
    floor = e7b.build_pipeline("question_only")
    check("question_only pipeline has five stages (no image branch)",
          len(floor.stage_names) == 5
          and floor.stage_names[0] == "S5_tokenise")
    reasoner = e7b.build_pipeline("reasoner")
    check("reasoner pipeline has nine stages (no projection)",
          len(reasoner.stage_names) == 9
          and "S8_projection" not in reasoner.stage_names)
    a1 = e7b.build_pipeline("e8a_a1")
    check("A1 pipeline has ten stages including the projection",
          len(a1.stage_names) == 10 and "S8_projection" in a1.stage_names)
    check("every pipeline ends at answer decode",
          all(p.stage_names[-1] == "S10_decode"
              for p in (multimodal, floor, reasoner, a1)))


# --- Reproduction gate --------------------------------------------------------

def _rows(cached_index, serial_index, cached_logits, serial_logits):
    serial = {"q1": {"index": serial_index, "answer": "yes",
                     "imageId": "img1", "logits": serial_logits}}
    cached = {"q1": {"index": cached_index, "logits": cached_logits}}
    return serial, cached


def test_reproduction_gate() -> None:
    agree_serial, agree_cached = _rows(2, 2, [0.0, 0.1, 0.9],
                                       [0.0, 0.1, 0.9])
    record = e7b.reproduction_gate("fusion", agree_serial, agree_cached)
    check("exact agreement passes the fp32 gate",
          record["agreements"] == 1 and not record["disagreements"])

    flip_serial, flip_cached = _rows(2, 1, [0.0, 0.1, 0.9],
                                     [0.0, 0.9, 0.1])
    must_fail("any fp32 disagreement halts",
              lambda: e7b.reproduction_gate("fusion", flip_serial,
                                            flip_cached))

    # fp16 class: a near tie within twice the logit perturbation is
    # explained; a large-margin flip is not.
    near_serial, near_cached = _rows(2, 1, [0.0, 0.5000, 0.5002],
                                     [0.0, 0.5002, 0.5000])
    original_floor = e7b.FP16_AGREEMENT_FLOOR
    try:
        e7b.FP16_AGREEMENT_FLOOR = 0
        record = e7b.reproduction_gate("reasoner", near_serial, near_cached)
        check("a documented near tie is explained for the fp16 class",
              record["disagreements"][0]["explained"]
              and record["disagreements"][0]["max_abs_logit_delta"] > 0)
        far_serial, far_cached = _rows(2, 1, [0.0, 0.1, 5.0],
                                       [0.0, 5.0, 0.1])
        must_fail("an unexplained large-margin flip halts even for fp16",
                  lambda: e7b.reproduction_gate("reasoner", far_serial,
                                                far_cached))
    finally:
        e7b.FP16_AGREEMENT_FLOOR = original_floor

    many_serial = {f"q{i}": {"index": 1, "answer": "a", "imageId": "x",
                             "logits": [0.0, 1.0]} for i in range(64)}
    many_cached = {f"q{i}": {"index": 0, "logits": [1.0, 0.0]}
                   for i in range(64)}
    must_fail("falling below the agreement floor halts",
              lambda: e7b.reproduction_gate("reasoner", many_serial,
                                            many_cached))


# --- Timing and maths ---------------------------------------------------------

def test_timing_helpers() -> None:
    floor = e7b.timed_calls(lambda: None, 5, 20)
    check("timed_calls returns the full statistic set",
          all(k in floor for k in ("median_ms", "mean_ms", "std_ms",
                                   "p5_ms", "p95_ms", "n")))
    check("a no-op times at the floor", floor["median_ms"] < 0.05,
          str(floor["median_ms"]))
    check("common-denominator arithmetic",
          round(4462 / e7b.RAW_DEV_QUESTIONS, 5) == 0.44602)

    rng_a = np.random.default_rng(e7b.ORDER_SEED)
    rng_b = np.random.default_rng(e7b.ORDER_SEED)
    orders_a = [list(rng_a.permutation(8)) for _ in range(3)]
    orders_b = [list(rng_b.permutation(8)) for _ in range(3)]
    check("pass-order randomisation is pinned and reproducible",
          orders_a == orders_b and len({tuple(o) for o in orders_a}) > 1)


# --- Rows and immutability ----------------------------------------------------

def test_rows_and_outputs() -> None:
    if e7b.ROWS_FILE.exists():
        record = json.loads(e7b.ROWS_FILE.read_text())
        check("pinned row file has the pinned count and seed",
              record["n_rows"] == e7b.N_ROWS
              and record["row_seed"] == e7b.ROW_SEED)
        check("row digest is stable",
              e7b.rows_digest(record) == e7b.rows_digest(
                  json.loads(e7b.ROWS_FILE.read_text())))
    else:
        check("row file not yet built (preflight will build it)", True)

    with tempfile.TemporaryDirectory() as scratch:
        target = Path(scratch) / "record.json"
        e7b.atomic_write_json(target, {"a": 1})
        check("atomic write produces the record",
              json.loads(target.read_text()) == {"a": 1})
        must_fail("overwriting an existing record is refused",
                  lambda: e7b.atomic_write_json(target, {"a": 2}))
        check("the refused overwrite left the original intact",
              json.loads(target.read_text()) == {"a": 1})


def run() -> None:
    test_registry()
    test_stage_names()
    test_reproduction_gate()
    test_timing_helpers()
    test_rows_and_outputs()
    failed = [name for name, passed in _CHECKS if not passed]
    if failed:
        raise AssertionError(f"{len(failed)} check(s) failed: {failed}")
    print(f"  test_e7b: {len(_CHECKS)} checks passed")


if __name__ == "__main__":
    VERBOSE = True
    run()
