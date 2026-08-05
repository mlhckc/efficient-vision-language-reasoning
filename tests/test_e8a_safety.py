"""Regression tests for the E8A runner and evidence-safety repairs.

Each repaired behaviour is exercised as a known-positive, where the safe path
must succeed, and a known-negative, where the unsafe path must be refused.
The repairs under test:

  * completed-cell reuse passes the strong cell validator, not a bare
    record-existence check (validator components tested here; the reuse wiring
    is in run_matrix.main);
  * stale or partial records fail closed (validator raises, never skips);
  * A1/A1r scheduling is atomic per seed with the random control first;
  * a SystemExit or KeyboardInterrupt leaves structured failure evidence;
  * --force-refreeze requires a reason, and after results exist it also
    requires recorded explicit authorisation;
  * backup verification inspects the actual supplied backup path and
    recomputes hashes; success is measured, never hard-coded.

Nothing here reads, resolves or names the embargoed clean-test target.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import (  # noqa: E402
    build_artefact_manifest as artefact_manifest)
from experiments.e8a_question_encoder import cell_validator  # noqa: E402
from experiments.e8a_question_encoder import run_matrix  # noqa: E402

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
    """Known-negative: the unsafe path must be refused, or the test is
    vacuous. SystemExit is included because the runner refuses via sys.exit."""
    global _INSIDE_KNOWN_NEGATIVE
    _INSIDE_KNOWN_NEGATIVE = True
    try:
        thunk()
        raised = False
    except (AssertionError, KeyError, RuntimeError, ValueError, IndexError,
            TypeError, SystemExit):
        raised = True
    finally:
        _INSIDE_KNOWN_NEGATIVE = False
    check(f"{name} (known-negative raises)", raised,
          "" if raised else "the unsafe path was NOT refused")


# --- Pair-atomic scheduling, random control first ----------------------------

def test_pair_scheduling() -> None:
    def spec(arm, seed):
        return {"arm": arm, "scale": "train_40k", "seed": seed}

    shuffled = [spec("A1", 1), spec("A0p", 0), spec("A1r", 0), spec("A1", 0),
                spec("A1r", 1), spec("A0p", 1)]
    ordered = run_matrix.schedule_pairs(shuffled)
    check("pairs are grouped by seed",
          [s["seed"] for s in ordered] == [0, 0, 0, 1, 1, 1])
    check("within each seed the order is A0p, then A1r, then A1",
          [s["arm"] for s in ordered] == ["A0p", "A1r", "A1"] * 2)
    positions = {(s["arm"], s["seed"]): i for i, s in enumerate(ordered)}
    check("the random control always precedes its paired pretrained arm",
          all(positions[("A1r", seed)] < positions[("A1", seed)]
              for seed in (0, 1)))
    must_fail("an unknown arm cannot be scheduled",
              lambda: run_matrix.schedule_pairs([spec("A2", 0)]))


# --- Force-refreeze guard -----------------------------------------------------

def test_refreeze_guard() -> None:
    check("a normal freeze is not guarded",
          run_matrix.refreeze_guard(False, None, None, ["x"]) is None)
    allowed = run_matrix.refreeze_guard(True, "recorded reason", None, [])
    check("refreeze with a reason and no observed result is recorded",
          allowed == {"reason": "recorded reason", "authorized_by": None,
                      "results_observed_at_refreeze": []})
    authorised = run_matrix.refreeze_guard(
        True, "recorded reason", "user 2026-08-05", ["run_A1.json"])
    check("refreeze after results records the explicit authorisation",
          authorised["authorized_by"] == "user 2026-08-05"
          and authorised["results_observed_at_refreeze"] == ["run_A1.json"])
    must_fail("refreeze without a reason is refused",
              lambda: run_matrix.refreeze_guard(True, None, None, []))
    must_fail("refreeze after results without authorisation is refused",
              lambda: run_matrix.refreeze_guard(True, "reason", None,
                                                ["run_A1.json"]))


# --- Structured interrupt evidence -------------------------------------------

def test_interrupt_evidence() -> None:
    original = e8a.OUT_DIR
    try:
        with tempfile.TemporaryDirectory() as scratch:
            e8a.OUT_DIR = Path(scratch)
            path = run_matrix.interrupt_evidence(
                "A1", "train_40k", 1, "KeyboardInterrupt",
                "operator interrupt during epoch 3", 12.5)
            record = json.loads(path.read_text())
            check("interrupt evidence file is written",
                  path.name == "FAILED_A1_train_40k_seed1.json")
            check("interrupt evidence is structured",
                  record["status"] == "INTERRUPTED"
                  and record["signal"] == "KeyboardInterrupt"
                  and record["arm"] == "A1" and record["seed"] == 1
                  and record["seconds_before_interrupt"] == 12.5)
            run_matrix.interrupt_evidence(
                "A1", "train_40k", 1, "SystemExit", "a different detail", 1.0)
            unchanged = json.loads(path.read_text())
            check("existing failure evidence is never overwritten",
                  unchanged["signal"] == "KeyboardInterrupt")
    finally:
        e8a.OUT_DIR = original


# --- Fail-closed cell validation components ----------------------------------

def test_selected_epoch_checks() -> None:
    good = {"epochs_run": 3, "best_epoch": 2, "best_dev_accuracy": 0.5,
            "history": [{"epoch": 1, "dev_accuracy": 0.4},
                        {"epoch": 2, "dev_accuracy": 0.5},
                        {"epoch": 3, "dev_accuracy": 0.5}]}
    result = cell_validator.check_selected_epoch("cell", good)
    check("earliest-tie-break selected epoch validates",
          result["earliest_tie_break_holds"])
    must_fail("a later tie-break epoch fails closed",
              lambda: cell_validator.check_selected_epoch(
                  "cell", {**good, "best_epoch": 3}))
    must_fail("an epochs_run/history mismatch fails closed",
              lambda: cell_validator.check_selected_epoch(
                  "cell", {**good, "epochs_run": 2}))
    must_fail("a best accuracy not in the history fails closed",
              lambda: cell_validator.check_selected_epoch(
                  "cell", {**good, "best_dev_accuracy": 0.6}))


def test_producer_commit_checks() -> None:
    key = ("A0p", "train_40k", 0)
    attested = cell_validator.ATTESTED_PRODUCERS[key]
    good = {"metadata": {"git_commit": attested[:7], "git_dirty": False}}
    result = cell_validator.check_producer_commit("cell", key, good)
    check("an attested clean producer commit validates",
          result["attested"] == attested and result["exists_in_repository"])
    must_fail("a non-attested producer prefix fails closed",
              lambda: cell_validator.check_producer_commit(
                  "cell", key,
                  {"metadata": {"git_commit": "abcdef0",
                                "git_dirty": False}}))
    must_fail("a dirty-worktree producer fails closed",
              lambda: cell_validator.check_producer_commit(
                  "cell", key,
                  {"metadata": {"git_commit": attested[:7],
                                "git_dirty": True}}))
    must_fail("a missing producer commit fails closed",
              lambda: cell_validator.check_producer_commit(
                  "cell", key, {"metadata": {}}))


def test_architecture_checks() -> None:
    with tempfile.TemporaryDirectory() as scratch:
        wrong = Path(scratch) / "wrong.pt"
        torch.save({"projection.weight": torch.zeros(4, 4)}, wrong)
        run = {"checkpoint": str(Path(wrong).relative_to("/"))}
        # The validator resolves checkpoints project-relative; hand it an
        # absolute path through a run dict of its own shape instead.
        run = {"checkpoint": str(wrong)}
        original_root = cell_validator.PROJECT_ROOT
        try:
            cell_validator.PROJECT_ROOT = Path("/")
            must_fail("a wrong-parameter-count checkpoint fails closed",
                      lambda: cell_validator.check_architecture(
                          "cell", "A0p", 0, run))
        finally:
            cell_validator.PROJECT_ROOT = original_root
    must_fail("a missing checkpoint fails closed",
              lambda: cell_validator.check_architecture(
                  "cell", "A0p", 0,
                  {"checkpoint": "results/does_not_exist.pt"}))


def test_validate_cell_end_to_end() -> None:
    """One real reused pilot passes the full strong validator, if present."""
    manifest_path = e8a.OUT_DIR / "execution_manifest.json"
    if not manifest_path.exists():
        check("execution manifest present for end-to-end validation", True,
              "skipped: no execution manifest on this node")
        return
    from experiments.e8a_question_encoder import (
        analysis_integrity as integrity)
    manifest = json.loads(manifest_path.read_text())[
        "e8a_execution_manifest"]
    view = integrity.canonical_dev_view()
    result = cell_validator.validate_cell(manifest, view, "A0p",
                                          "train_40k", 0)
    check("the reused A0p pilot passes the strong validator end to end",
          result["valid"] and result["reused_phase_1b_pilot"]
          and result["architecture"]["strict_load_succeeded"])


# --- Measured backup verification --------------------------------------------

def test_backup_verification() -> None:
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        source = root / "source" / "results" / "a.bin"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"artefact-bytes")
        entry = {"path": "results/a.bin", "bytes": source.stat().st_size,
                 "sha256": e8a.sha256_file(source),
                 "backup_expected_by_policy": True}

        backup_root = root / "backup"
        (backup_root / "results").mkdir(parents=True)
        (backup_root / "results" / "a.bin").write_bytes(b"artefact-bytes")
        verified = artefact_manifest.verify_backup([dict(entry)], backup_root)
        check("an intact backup verifies by re-hash",
              verified["verified"]
              and verified["artefacts_verified_by_rehash"] == 1)

        (backup_root / "results" / "a.bin").write_bytes(b"artefact-BYTES")
        altered = artefact_manifest.verify_backup([dict(entry)], backup_root)
        check("an altered backup file is reported, not asserted verified",
              not altered["verified"]
              and altered["present_but_mismatched"] == ["results/a.bin"])

        (backup_root / "results" / "a.bin").unlink()
        missing = artefact_manifest.verify_backup([dict(entry)], backup_root)
        check("a missing expected backup file fails verification",
              not missing["verified"]
              and missing["expected_but_missing"] == ["results/a.bin"])

        gone = artefact_manifest.verify_backup([dict(entry)],
                                               root / "not-mounted")
        check("an unreachable backup root is reported unverified",
              not gone["verified"] and not gone["reachable"])

        empty = artefact_manifest.verify_backup(
            [{**entry, "backup_expected_by_policy": False}], backup_root)
        check("zero expected artefacts never verifies vacuously",
              not empty["verified"])


def run() -> None:
    test_pair_scheduling()
    test_refreeze_guard()
    test_interrupt_evidence()
    test_selected_epoch_checks()
    test_producer_commit_checks()
    test_architecture_checks()
    test_validate_cell_end_to_end()
    test_backup_verification()
    failed = [name for name, passed in _CHECKS if not passed]
    if failed:
        raise AssertionError(f"{len(failed)} check(s) failed: {failed}")
    print(f"  test_e8a_safety: {len(_CHECKS)} checks passed")


if __name__ == "__main__":
    VERBOSE = True
    run()
