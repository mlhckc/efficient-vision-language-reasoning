"""CPU-only contract tests for the E10 Phase-1 execution guardrails.

Nothing here trains, evaluates or authorises a scientific cell. Every test that
needs an authorised code path fabricates that state inside a temporary
directory and restores it afterwards; the repository authorization constant is
never changed.
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from experiments.e10_capacity_360m import e10_common as e10
from experiments.e10_capacity_360m import phase1
from experiments.e10_capacity_360m import run as e10_run
from tests.test_e10 import (
    _must_raise,
    _strict_state_restored,
    _temporary_shared_paths,
    _write_ledgers,
    write_retry_authorization,
)


FROZEN_ORDER = [
    ("B4", "train_40k", 0), ("B4r", "train_40k", 0),
    ("B4", "train_40k", 1), ("B4r", "train_40k", 1),
    ("B4", "train_40k", 2), ("B4r", "train_40k", 2),
    ("B4", "train_250k", 0), ("B4r", "train_250k", 0),
    ("B4", "train_250k", 1), ("B4r", "train_250k", 1),
    ("B4", "train_250k", 2), ("B4r", "train_250k", 2),
]
WALL_NS = 12 * 3600 * 1_000_000_000


class _SpyOptimizer:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


def _authorized_core(stack_paths: dict):
    """Patch exactly what a scientific cell would need, and nothing more."""
    return (
        mock.patch.object(e10, "E10_TRAINING_AUTHORIZED",
                          e10.CORE_AUTHORIZATION_TOKEN),
        mock.patch.object(e10, "assert_shared_state_healthy",
                          return_value={"calibration_complete": True}),
        mock.patch.object(e10, "OUT_DIR", stack_paths["OUT_DIR"]),
    )


# 1. The scientific core still refuses.

def test_scientific_core_still_refuses() -> None:
    assert e10.E10_TRAINING_AUTHORIZED is None
    assert phase1.assert_scientific_core_refused() == {
        "refused": True,
        "refusal": "E10 CORE REFUSED: scientific authorization is None",
        "e10_training_authorized": None,
    }
    for arm, scale, seed in FROZEN_ORDER:
        _must_raise(
            SystemExit,
            lambda a=arm, s=scale, d=seed: e10.assert_core_entry_authorized(a, s, d),
            "scientific authorization",
        )
    _must_raise(
        SystemExit,
        lambda: e10_run.core_cell("B4", "train_40k", 0),
        "scientific authorization",
    )
    _must_raise(
        SystemExit,
        lambda: e10.authorize_cell_execution("B4", "train_40k", 0),
        "scientific authorization",
    )


# 2. The 12 h wall comes from the authoritative constant.

def test_wall_is_loaded_from_the_authoritative_constant() -> None:
    assert config.E10_PER_CELL_WALL_CLOCK_HOURS == 12.0
    assert e10.per_cell_wall_hours() == 12.0
    assert e10.per_cell_wall_seconds() == 12.0 * 3600.0
    assert phase1.per_cell_wall_hours() == 12.0
    policy = e10.resource_policy()
    assert policy["per_cell_wall_clock_hours"] == 12.0
    assert policy["wall_source"] == "config.E10_PER_CELL_WALL_CLOCK_HOURS"
    # The value is read, not hard-coded: moving the constant moves the wall.
    with mock.patch.object(config, "E10_PER_CELL_WALL_CLOCK_HOURS", 5.0):
        assert e10.per_cell_wall_hours() == 5.0
        assert phase1.resource_policy()["per_cell_wall_clock_hours"] == 5.0
    assert e10.per_cell_wall_hours() == 12.0
    # The frozen recipe still carries no resource field; the wall is imposed
    # by the permit instead, which is the reviewer-identified repair.
    reimposition = phase1.wall_reimposition_record()
    assert reimposition["recipe_carries_wall_field"] is False
    assert reimposition["per_cell_wall_clock_hours"] == 12.0
    stored = json.loads(
        (e10.OUT_DIR / "recipe_contract_20260810.json").read_text()
    )["e10_recipe_contract"]["pairs"]
    for scale in e10.CORE_SCALES:
        for seed in e10.CORE_SEEDS:
            pair = stored[f"{scale}_seed{seed}"]
            assert reimposition["recipe_digests"][f"B4_{scale}_seed{seed}"] == (
                pair["B4_full_sha256"]
            )
            assert reimposition["recipe_digests"][f"B4r_{scale}_seed{seed}"] == (
                pair["B4r_full_sha256"]
            )


# 3. An unset or invalid wall fails closed.

def test_unset_or_invalid_wall_fails_closed() -> None:
    for bad in (None, 0, 0.0, -1.0, "12", True, float("nan"), float("inf")):
        with mock.patch.object(config, "E10_PER_CELL_WALL_CLOCK_HOURS", bad):
            _must_raise(AssertionError, e10.per_cell_wall_hours, "E10 REFUSED")
            _must_raise(AssertionError, e10.resource_policy, "E10 REFUSED")
            _must_raise(
                SystemExit,
                lambda: e10.assert_core_entry_authorized("B4", "train_40k", 0),
                "REFUSED",
            )
    for bad in (None, 0.0, -1.0, "40", float("nan")):
        with mock.patch.object(config, "E10_PER_IDENTITY_CEILING_HOURS", bad):
            _must_raise(
                AssertionError, e10.effective_identity_ceiling_hours, "E10 REFUSED"
            )
    # A wall wider than the effective identity ceiling is also refused.
    with mock.patch.object(config, "E10_PER_CELL_WALL_CLOCK_HOURS", 41.0):
        _must_raise(AssertionError, e10.resource_policy, "exceeds the effective")


# 4 and 5. The effective ceiling is the minimum and never more permissive.

def test_effective_identity_ceiling_is_the_bounded_minimum() -> None:
    assert config.E10_PER_IDENTITY_CEILING_HOURS == 40.0
    assert config.PER_MODEL_IDENTITY_CEILING_HOURS == 40.0
    assert e10.effective_identity_ceiling_hours() == 40.0
    cases = (
        (40.0, 40.0, 40.0),
        (50.0, 40.0, 40.0),
        (30.0, 40.0, 30.0),
        (40.0, 35.0, 35.0),
        (1000.0, 40.0, 40.0),
    )
    for e10_ceiling, programme, expected in cases:
        with (
            mock.patch.object(
                config, "E10_PER_IDENTITY_CEILING_HOURS", e10_ceiling
            ),
            mock.patch.object(
                config, "PER_MODEL_IDENTITY_CEILING_HOURS", programme
            ),
        ):
            effective = e10.effective_identity_ceiling_hours()
            assert effective == expected
            assert effective <= programme, "E10 became more permissive"
            assert phase1.effective_identity_ceiling_hours() == expected
    # The programme-wide ceiling is read in exactly one E10 module, and E10
    # never declares its own copy of that value.
    readers = []
    package = PROJECT_ROOT / "experiments" / "e10_capacity_360m"
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) \
                    and node.attr == "PER_MODEL_IDENTITY_CEILING_HOURS" \
                    and isinstance(node.value, ast.Name) \
                    and node.value.id == "config":
                readers.append(path.name)
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name)
                and target.id == "PER_MODEL_IDENTITY_CEILING_HOURS"
                for target in node.targets
            ):
                raise AssertionError(f"{path.name} declares its own copy")
    assert sorted(set(readers)) == ["e10_common.py"]
    assert readers, "no E10 module reads the programme-wide ceiling"


def test_e10_cannot_become_more_permissive_in_the_retry_fit() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with (
            _temporary_shared_paths(root) as paths,
            mock.patch.object(config, "E10_PER_IDENTITY_CEILING_HOURS", 400.0),
            mock.patch.object(config, "PER_MODEL_IDENTITY_CEILING_HOURS", 40.0),
        ):
            _write_ledgers(paths["SPEND_LEDGER"], paths["RETRY_LEDGER"])
            fit = e10.retry_admissible(
                "B4r", 1.0, e10._initial_spend_ledger(),
                e10._initial_retry_ledger(), remaining_mandatory_hours=38.5,
            )
            assert fit["ceiling_hours"] == 40.0
            assert fit["e10_ceiling_hours"] == 400.0
            assert fit["programme_wide_ceiling_hours"] == 40.0
            assert fit["admissible"] is False, "the wider E10 value was used"
            assert fit["automatic_retries_permitted"] == 0
            assert fit["requires_fresh_authorization"] is True


# 6. An automatic retry is impossible.

def test_automatic_retry_is_impossible() -> None:
    assert config.E10_AUTOMATIC_RETRIES_PER_IDENTITY == 0
    assert e10.AUTOMATIC_RETRIES_PER_IDENTITY == 0
    scan = phase1.assert_no_automatic_retry()
    assert scan["automatic_retry_writers"] == 0
    assert scan["automatic_retry_call_sites"] == 0
    assert scan["files_scanned"] > 50
    with mock.patch.object(config, "E10_AUTOMATIC_RETRIES_PER_IDENTITY", 1):
        _must_raise(
            phase1.GuardrailError, phase1.assert_no_automatic_retry, "must be 0"
        )
        _must_raise(AssertionError, e10.resource_policy, "automatic retries")
        _must_raise(
            SystemExit,
            lambda: e10.assert_core_entry_authorized("B4", "train_40k", 0),
            "REFUSED",
        )


# 7. A retry needs a fresh authorization record.

def test_retry_without_fresh_authorization_refuses() -> None:
    with tempfile.TemporaryDirectory(
        prefix=".e10-phase1-retry-", dir=e10.PROJECT_ROOT
    ) as directory:
        root = Path(directory)
        with (
            _temporary_shared_paths(root) as paths,
            mock.patch.object(config, "E10_PER_IDENTITY_CEILING_HOURS", 10.0),
            mock.patch.object(config, "PER_MODEL_IDENTITY_CEILING_HOURS", 10.0),
            mock.patch.object(config, "E10_PER_CELL_WALL_CLOCK_HOURS", 2.0),
            mock.patch.object(config, "E10_HEADROOM_FLOOR_HOURS", 1.0),
        ):
            _write_ledgers(paths["SPEND_LEDGER"], paths["RETRY_LEDGER"])
            evidence = root / "evidence.json"
            e10.atomic_write_json(evidence, {"status": "NODE_FAILURE"})
            e10.atomic_write_json(paths["GATE1_PATH"], {
                "status": "GATE1_READY_TEST",
                "binding": e10.binding_record(),
                "per_cell_projections": {
                    "train_40k": {"measured_times_1_25_stress_hours": 1.0},
                    "train_250k": {"measured_times_1_25_stress_hours": 2.0},
                },
            })
            relative_evidence = str(evidence.relative_to(e10.PROJECT_ROOT))

            def attempt():
                return e10.record_forced_retry(
                    "B4r", "train_40k", 0, "node_failure", relative_evidence, 1.0
                )

            _must_raise(AssertionError, attempt, "no fresh retry authorization")

            # An authorization for a different cell does not transfer.
            write_retry_authorization(root, "B4r", "train_250k", 1,
                                      projected_hours=2.0, nonce="a" * 64)
            _must_raise(AssertionError, attempt, "exactly one fresh authorization")

            # A record that claims to be automatic, or that is not signed by
            # the user or an independent reviewer, is refused outright.
            e10.retry_authorization_path("a" * 64).unlink()
            write_retry_authorization(root, "B4r", "train_40k", 0,
                                      automatic=True, nonce="b" * 64)
            _must_raise(AssertionError, attempt, "automatic retry authorization")
            e10.retry_authorization_path("b" * 64).unlink()
            write_retry_authorization(root, "B4r", "train_40k", 0,
                                      authorized_by="runner", nonce="c" * 64)
            _must_raise(AssertionError, attempt, "independent reviewer")
            e10.retry_authorization_path("c" * 64).unlink()

            # A cost that disagrees with the authorization is refused.
            write_retry_authorization(root, "B4r", "train_40k", 0,
                                      projected_hours=2.0, nonce="d" * 64)
            _must_raise(
                AssertionError,
                lambda: e10.record_forced_retry(
                    "B4r", "train_40k", 0, "node_failure", relative_evidence, 1.0
                ),
                "differs from the requested cost",
            )
            e10.retry_authorization_path("d" * 64).unlink()

            write_retry_authorization(root, "B4r", "train_40k", 0, nonce="e" * 64)
            ledger = attempt()
            assert len(ledger["retries"]) == 1
            entry = ledger["retries"][0]
            assert entry["authorization_nonce"] == "e" * 64
            assert entry["authorized_by"] == "user"
            # The consumed nonce is single-use: the same record cannot be
            # replayed, and a second retry needs a second authorization.
            _must_raise(AssertionError, attempt, "exactly one fresh authorization")


# 8 and 9. A reached wall halts, is accounted for, and cannot continue.

def test_wall_halt_accounts_and_cannot_silently_continue() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "out"
        output.mkdir()
        with (
            _strict_state_restored(),
            _temporary_shared_paths(root) as paths,
        ):
            _write_ledgers(paths["SPEND_LEDGER"], paths["RETRY_LEDGER"])
            patches = _authorized_core({"OUT_DIR": output})
            with patches[0], patches[1], patches[2]:
                e10.enable_strict_determinism()
                optimizer = _SpyOptimizer()
                expired = time.monotonic_ns() - WALL_NS - 1
                guard = phase1.ScientificCellGuard(
                    "B4r", "train_40k", 0, started_monotonic_ns=expired
                )
                with guard:
                    _must_raise(e10.CellWallExceeded, guard.check, "per-cell wall")
                    # Latched: no later call may continue the cell.
                    _must_raise(e10.CellWallExceeded, guard.check)
                    _must_raise(
                        e10.CellWallExceeded,
                        lambda: guard.guarded_optimizer_step(optimizer),
                    )
                assert guard.halted is True
                assert optimizer.steps == 0

                # Accounting happened once, atomically, with the halt outcome.
                ledger = e10.read_spend_ledger()
                core = [entry for entry in ledger["entries"]
                        if entry["context"] == "e10_core_cell"]
                assert len(core) == 1
                assert core[0]["outcome"] == "wall_clock_halted"
                assert core[0]["cell"] == ["B4r", "train_40k", 0]
                assert core[0]["identity"] == "random_smollm2_360m"
                assert core[0]["claim_sha256"] is None
                assert core[0]["gpu_occupancy_ns"] >= WALL_NS
                assert e10.identity_hours("random_smollm2_360m") >= 12.0

                records = sorted(output.glob("cell_wall_clock_halted_*.json"))
                assert len(records) == 1
                halt = json.loads(records[0].read_text())
                assert halt["status"] == "WALL_CLOCK_HALTED"
                assert halt["per_cell_wall_clock_hours"] == 12.0
                assert halt["effective_identity_ceiling_hours"] == 40.0
                assert halt["automatic_retry_available"] is False
                assert halt["retry_requires_fresh_authorization"] is True
                assert halt["spend_entry_id"] == core[0]["entry_id"]

                # Positive control: inside the wall the same gate permits one
                # step, so the refusals above are not vacuous.
                fresh = phase1.ScientificCellGuard("B4", "train_40k", 0)
                with fresh:
                    fresh.check("entry")
                    fresh.guarded_optimizer_step(optimizer)
                assert optimizer.steps == 1


# 10 and 11. The frozen order and B4/B4r pair preservation are exact.

def test_frozen_twelve_cell_order_and_pair_preservation() -> None:
    assert list(e10.CORE_CELLS) == FROZEN_ORDER
    assert phase1.execution_plan() == FROZEN_ORDER
    assert e10_run.core_order() == FROZEN_ORDER
    assert phase1.assert_frozen_execution_plan() == {
        "cells": 12, "pair_preserving": True, "adaptive": False,
    }
    assert e10.CORE_ARMS == ("B4", "B4r")
    assert e10.CORE_SCALES == ("train_40k", "train_250k")
    assert e10.CORE_SEEDS == (0, 1, 2)
    # The order is not conditioned on any recorded state.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with _temporary_shared_paths(root) as paths:
            for outcome in ("completed", "wall_clock_halted", "terminated"):
                spend = e10._initial_spend_ledger()
                spend["entries"] = []
                e10.atomic_replace_json(paths["SPEND_LEDGER"], spend)
                assert phase1.execution_plan() == FROZEN_ORDER, outcome
    for scale in e10.CORE_SCALES:
        for seed in e10.CORE_SEEDS:
            _must_raise(
                AssertionError,
                lambda s=scale, d=seed: e10.pair_preserving_order([("B4", s, d)]),
                "unpaired",
            )
            _must_raise(
                AssertionError,
                lambda s=scale, d=seed: e10.pair_preserving_order([("B4r", s, d)]),
                "unpaired",
            )
            left = e10.build_recipe("B4", scale, seed)
            right = e10.build_recipe("B4r", scale, seed)
            assert e10.recipe_sha256(left) != e10.recipe_sha256(right)
            assert e10.paired_recipe_sha256(left) == e10.paired_recipe_sha256(right)


# 12. The frozen E8A, E8B and E9 result trees are unchanged.

def test_frozen_result_trees_are_unchanged() -> None:
    for name, expected in e10.FROZEN_RESULT_BASELINES.items():
        root = config.RESULTS_DIR / "experiments" / name
        rows = sorted(
            (item.relative_to(e10.PROJECT_ROOT).as_posix(), e10.sha256_file(item))
            for item in root.rglob("*") if item.is_file()
        )
        stream = "".join(f"{digest}  {relative}\n" for relative, digest in rows)
        assert {
            "files": len(rows),
            "tree_sha256": e10.sha256_bytes(stream.encode("utf-8")),
        } == expected, name


# 13. The embargo and output guards still pass.

def test_embargo_and_output_guards_still_pass() -> None:
    assert e10.assert_no_embargo_reference()["passed"] is True
    assert e10.assert_no_scientific_outputs() == {
        "checkpoint_files": 0,
        "forbidden_metric_fields": 0,
        "embargo_source_scan_passed": True,
    }
    # The Phase-1 sources resolve no development or embargoed data path at all.
    suffix = bytes([46, 99, 115, 118]).decode()
    for path in (Path(phase1.__file__), Path(__file__)):
        assert suffix not in path.read_text(), path.name
    assert e10.assert_no_embargo_reference(
        [Path(phase1.__file__), Path(__file__)]
    ) == {"files_checked": 2, "passed": True}


# 14. This authorization task created no scientific artefact.

def test_no_scientific_artifact_was_created() -> None:
    assert sorted(e10.OUT_DIR.rglob("*.pt")) == []
    assert sorted(e10.OUT_DIR.glob("cell_*.json")) == []
    assert e10.E10_TRAINING_AUTHORIZED is None
    assert e10.E10_CALIBRATION_AUTHORIZED is None
    ledger = e10.read_spend_ledger()
    assert [entry for entry in ledger["entries"]
            if entry["context"] == "e10_core_cell"] == []
    assert e10.read_retry_ledger()["retries"] == []
    assert not e10.RETRY_AUTHORIZATION_DIR.exists()
    contract = json.loads(
        (e10.OUT_DIR / "recipe_contract_20260810.json").read_text()
    )["e10_recipe_contract"]
    assert contract["scientific_cells_authorized"] == 0
    assert contract["e10_training_authorized"] is None


TESTS = (
    test_scientific_core_still_refuses,
    test_wall_is_loaded_from_the_authoritative_constant,
    test_unset_or_invalid_wall_fails_closed,
    test_effective_identity_ceiling_is_the_bounded_minimum,
    test_e10_cannot_become_more_permissive_in_the_retry_fit,
    test_automatic_retry_is_impossible,
    test_retry_without_fresh_authorization_refuses,
    test_wall_halt_accounts_and_cannot_silently_continue,
    test_frozen_twelve_cell_order_and_pair_preservation,
    test_frozen_result_trees_are_unchanged,
    test_embargo_and_output_guards_still_pass,
    test_no_scientific_artifact_was_created,
)


def run() -> None:
    for test in TESTS:
        test()
    print(f"E10 Phase-1 tests passed ({len(TESTS)} checks)")


if __name__ == "__main__":
    run()
