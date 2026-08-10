"""CPU-only contract tests for the E10 Phase-3 authorization binding.

Nothing here authorises, trains or evaluates a real scientific cell. Every test
that needs an authorised code path fabricates one synthetic external grant
inside a temporary directory that ``_temporary_shared_paths`` has redirected
``CORE_AUTHORIZATION_DIR`` into, and removes it again. No source file is edited
to authorise anything, the repository authorization constant is never accepted
as a route, and the real shared-state authorization directory is never written.

The central invariant is the one the whole phase exists for: creating a grant
must not move the live source digest, because the grant binds that digest.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from experiments.e10_capacity_360m import e10_common as e10
from experiments.e10_capacity_360m import phase1
from experiments.e10_capacity_360m import phase2
from experiments.e10_capacity_360m import phase3
from experiments.e10_capacity_360m import run as e10_run
from experiments.e10_capacity_360m import science
from tests.test_e10 import (
    _must_raise,
    _strict_state_restored,
    _temporary_shared_paths,
    _write_ledgers,
    write_core_authorization_grant,
)


FROZEN_ORDER = [
    ("B4", "train_40k", 0), ("B4r", "train_40k", 0),
    ("B4", "train_40k", 1), ("B4r", "train_40k", 1),
    ("B4", "train_40k", 2), ("B4r", "train_40k", 2),
    ("B4", "train_250k", 0), ("B4r", "train_250k", 0),
    ("B4", "train_250k", 1), ("B4r", "train_250k", 1),
    ("B4", "train_250k", 2), ("B4r", "train_250k", 2),
]
LEGACY_TOKEN = "e10-core-matrix-approved"
UNREGISTERED_CELL = ("B4", "train_100k", 0)


class _SpyOptimizer:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


@contextmanager
def _temporary_authorization(root: Path, *, grant: bool = True, **overrides):
    """Redirect every shared path, then optionally write one synthetic grant."""
    with ExitStack() as stack:
        paths = stack.enter_context(_temporary_shared_paths(root))
        _write_ledgers(paths["SPEND_LEDGER"], paths["RETRY_LEDGER"])
        stack.enter_context(mock.patch.object(
            e10, "assert_shared_state_healthy",
            lambda **_: {"calibration_complete": True}))
        if grant:
            write_core_authorization_grant(**overrides)
        yield paths


def _refuses(message: str, **overrides) -> None:
    """One malformed grant, refused by the validator and by the entry gate."""
    with tempfile.TemporaryDirectory() as directory:
        with _temporary_authorization(Path(directory), **overrides):
            _must_raise(AssertionError,
                        e10.validate_core_authorization_grant, message)
            _must_raise(
                SystemExit,
                lambda: e10.assert_core_entry_authorized("B4", "train_40k", 0),
                "E10 CORE REFUSED",
            )
            _must_raise(
                SystemExit,
                lambda: e10.authorize_cell_execution("B4", "train_40k", 0),
                "E10 CORE REFUSED",
            )


# 1. A missing grant refuses, everywhere that matters.

def test_missing_grant_refuses_every_authoritative_path() -> None:
    with tempfile.TemporaryDirectory() as directory:
        with _temporary_authorization(Path(directory), grant=False):
            assert not e10.core_authorization_grant_path().exists()
            _must_raise(AssertionError, e10.validate_core_authorization_grant,
                        "scientific authorization grant is absent")
            for arm, scale, seed in FROZEN_ORDER:
                _must_raise(
                    SystemExit,
                    lambda a=arm, s=scale, d=seed:
                        e10.assert_core_entry_authorized(a, s, d),
                    "scientific authorization grant is absent",
                )
                _must_raise(
                    SystemExit,
                    lambda a=arm, s=scale, d=seed:
                        e10.authorize_cell_execution(a, s, d),
                    "E10 CORE REFUSED",
                )
            _must_raise(SystemExit,
                        lambda: e10_run.core_cell("B4", "train_40k", 0),
                        "E10 CORE REFUSED")
            _must_raise(SystemExit,
                        lambda: science.run_core_cell("B4", "train_40k", 0),
                        "E10 CORE REFUSED")
            _must_raise(
                SystemExit,
                lambda: phase1.assert_cell_entry_guarded("B4", "train_40k", 0),
                "E10 CORE REFUSED",
            )


# 2. One valid exact grant validates.

def test_valid_exact_synthetic_grant_validates() -> None:
    with tempfile.TemporaryDirectory() as directory:
        with _temporary_authorization(Path(directory)):
            grant = e10.validate_core_authorization_grant()
            assert grant["record_type"] == "e10_core_authorization_grant"
            assert grant["status"] == "SCIENTIFIC_CORE_AUTHORIZED"
            assert grant["authorized_by"] == "user"
            assert grant["authorization_token"] == LEGACY_TOKEN
            assert grant["task_id"] == e10.PHASE3_TASK_ID
            assert grant["experiment"] == "e10_capacity_360m"
            assert grant["protocol_family"] == e10.PROTOCOL_FAMILY
            assert grant["authorized_arms"] == ["B4", "B4r"]
            assert grant["authorized_scales"] == ["train_40k", "train_250k"]
            assert grant["authorized_seeds"] == [0, 1, 2]
            assert grant["per_cell_wall_clock_hours"] == 12.0
            assert grant["effective_identity_ceiling_hours"] == 40.0
            assert grant["automatic_retries"] == 0
            assert grant["model"]["repo"] == e10.MODEL_REPO
            assert grant["model"]["revision"] == e10.MODEL_REVISION
            assert grant["scope_exclusions"] == dict(
                e10.CORE_GRANT_SCOPE_EXCLUSIONS)
            assert all(value is False
                       for value in grant["scope_exclusions"].values())
            assert grant["binding"]["source_digest"] == e10.source_digest()[0]
            # The grant carries no observed scientific quantity: the field set
            # is exact, so a result field cannot even be present.
            assert "accuracy" not in json.dumps(grant).lower()


# 3. Creating a grant does not move the source digest.

def test_creating_a_grant_does_not_move_the_source_digest() -> None:
    before_digest, before_files = e10.source_digest()
    before_config = e10.config_digest()[0]
    with tempfile.TemporaryDirectory() as directory:
        with _temporary_authorization(Path(directory)):
            grant = e10.validate_core_authorization_grant("B4", "train_40k", 0)
            opened = e10.assert_core_entry_authorized("B4", "train_40k", 0)
            assert opened["grant_id"] == grant["grant_id"]
            during_digest, during_files = e10.source_digest()
            assert during_digest == before_digest
            assert during_files == before_files
            assert e10.config_digest()[0] == before_config
            # The reviewed provenance still validates while a grant is live.
            assert e10.validate_phase3_amendment()["status"] == (
                "PHASE2_EVIDENCE_CARRIED_FORWARD")
    after_digest, after_files = e10.source_digest()
    assert after_digest == before_digest
    assert after_files == before_files
    assert e10.config_digest()[0] == before_config
    # And the location is structurally outside every bound digest.
    outside = phase3.assert_grant_cannot_move_source_digest()
    assert outside["inside_repository"] is False
    assert outside["source_digest"] == before_digest


# 4. Exactly the twelve frozen cells become authorisable.

def test_exact_twelve_cells_are_authorized() -> None:
    with tempfile.TemporaryDirectory() as directory:
        with _temporary_authorization(Path(directory)):
            grant = e10.validate_core_authorization_grant()
            assert grant["authorized_cells"] == [
                list(cell) for cell in FROZEN_ORDER]
            assert len(grant["authorized_cells"]) == 12
            for arm, scale, seed in FROZEN_ORDER:
                assert e10.validate_core_authorization_grant(
                    arm, scale, seed)["grant_id"] == grant["grant_id"]
                assert e10.assert_core_entry_authorized(arm, scale, seed)
                permit = e10.authorize_cell_execution(arm, scale, seed)
                assert (permit.arm, permit.scale, permit.seed) == (
                    arm, scale, seed)
            assert e10.pair_preserving_order() == FROZEN_ORDER


# 5. A thirteenth or unregistered cell refuses.

def test_unregistered_cell_refuses() -> None:
    arm, scale, seed = UNREGISTERED_CELL
    with tempfile.TemporaryDirectory() as directory:
        with _temporary_authorization(Path(directory)):
            _must_raise(
                AssertionError,
                lambda: e10.validate_core_authorization_grant(arm, scale, seed),
                "is not an E10 core cell",
            )
            _must_raise(
                SystemExit,
                lambda: e10.assert_core_entry_authorized(arm, scale, seed),
                "unknown cell",
            )
            _must_raise(SystemExit,
                        lambda: e10_run.core_cell(arm, scale, seed),
                        "unknown cell")
    # A grant that tries to add a thirteenth cell is itself refused.
    _refuses("not the frozen twelve-cell matrix",
             authorized_cells=[list(cell) for cell in FROZEN_ORDER]
             + [list(UNREGISTERED_CELL)])
    # ... and so is one that drops half of a pair.
    _refuses("not the frozen twelve-cell matrix",
             authorized_cells=[list(cell) for cell in FROZEN_ORDER
                               if cell != ("B4r", "train_250k", 2)])
    _refuses("cell list is malformed", authorized_cells=["B4"])


# 6-8. A wrong arm, scale or seed refuses.

def test_wrong_arm_scale_or_seed_refuses() -> None:
    _refuses("mismatch for authorized_arms", authorized_arms=["B4"])
    _refuses("mismatch for authorized_arms", authorized_arms=["B4", "B4r", "B3"])
    _refuses("mismatch for authorized_scales", authorized_scales=["train_40k"])
    _refuses("mismatch for authorized_scales",
             authorized_scales=["train_40k", "train_100k", "train_250k"])
    _refuses("mismatch for authorized_seeds", authorized_seeds=[0, 1])
    _refuses("mismatch for authorized_seeds", authorized_seeds=[0, 1, 2, 3])
    # A valid grant still refuses an arm, scale or seed outside the matrix.
    with tempfile.TemporaryDirectory() as directory:
        with _temporary_authorization(Path(directory)):
            for cell in (("B3", "train_40k", 0), ("B4", "train_100k", 0),
                         ("B4", "train_40k", 3)):
                _must_raise(
                    AssertionError,
                    lambda c=cell: e10.validate_core_authorization_grant(*c),
                    "is not an E10 core cell",
                )


# 9. A wrong order refuses.

def test_wrong_matrix_order_refuses() -> None:
    reversed_order = [list(cell) for cell in reversed(FROZEN_ORDER)]
    _refuses("order is not pair preserving", authorized_cells=reversed_order)
    # Swapping B4 and B4r inside one pair breaks the frozen order too.
    swapped = [list(cell) for cell in FROZEN_ORDER]
    swapped[0], swapped[1] = swapped[1], swapped[0]
    _refuses("order is not pair preserving", authorized_cells=swapped)


# 10. A wrong recipe, source or config digest refuses.

def test_wrong_recipe_or_source_digest_refuses() -> None:
    digests = dict(e10.core_authorization_grant_expectations()["recipe_digests"])
    digests["B4_train_40k_seed0"] = "0" * 64
    _refuses("mismatch for recipe_digests", recipe_digests=digests)
    _refuses("mismatch for recipe_contract",
             recipe_contract={"path": "results/experiments/e10_capacity_360m/"
                                      "recipe_contract_20260810.json",
                              "sha256": "1" * 64})
    stale_source = {**e10.binding_record(),
                    "source_digest": e10.PHASE2_R2_SOURCE_DIGEST}
    _refuses("stale source_digest", binding=stale_source)
    stale_config = {**e10.binding_record(), "config_digest": "2" * 64}
    _refuses("stale config_digest", binding=stale_config)
    _refuses("stale protocol_family",
             binding={**e10.binding_record(), "protocol_family": "e8b"})
    _refuses("mismatch for protocol_family", protocol_family="e8b-family")
    _refuses("mismatch for experiment", experiment="e8b_readout_generation")
    _refuses("mismatch for model",
             model={**e10.core_authorization_grant_expectations()["model"],
                    "revision": "0" * 40})


# 11. A modified or tampered grant refuses.

def test_tampered_grant_refuses() -> None:
    with tempfile.TemporaryDirectory() as directory:
        with _temporary_authorization(Path(directory)):
            path = e10.core_authorization_grant_path()
            grant = json.loads(path.read_text())
            grant["purpose"] = grant["purpose"] + " and anything else"
            path.write_text(json.dumps(grant))
            _must_raise(AssertionError, e10.validate_core_authorization_grant,
                        "do not match its grant_id")
            _must_raise(
                SystemExit,
                lambda: e10.assert_core_entry_authorized("B4", "train_40k", 0),
                "E10 CORE REFUSED",
            )
    # A grant whose id does not cover its own body refuses on the same rule.
    _refuses("do not match its grant_id", grant_id="3" * 64)
    with tempfile.TemporaryDirectory() as directory:
        with _temporary_authorization(Path(directory)):
            path = e10.core_authorization_grant_path()
            grant = json.loads(path.read_text())
            grant.pop("scope_exclusions")
            body = {k: v for k, v in grant.items() if k != "grant_id"}
            grant["grant_id"] = e10.sha256_bytes(
                e10.canonical_json_bytes(body))
            path.write_text(json.dumps(grant))
            _must_raise(AssertionError, e10.validate_core_authorization_grant,
                        "schema keys mismatch")
    # A second, competing authorization record refuses rather than being ranked.
    with tempfile.TemporaryDirectory() as directory:
        with _temporary_authorization(Path(directory)):
            competing = e10.CORE_AUTHORIZATION_DIR / "another-grant.json"
            competing.write_text(json.dumps({"status": "GRANTED"}))
            _must_raise(AssertionError, e10.validate_core_authorization_grant,
                        "competing E10 core authorization records")


# 12. A non-zero retry allowance refuses.

def test_nonzero_retries_refuse() -> None:
    _refuses("mismatch for automatic_retries", automatic_retries=1)
    _refuses("mismatch for automatic_retries", automatic_retries=None)
    assert config.E10_AUTOMATIC_RETRIES_PER_IDENTITY == 0
    assert e10.AUTOMATIC_RETRIES_PER_IDENTITY == 0


# 13. Scope broadening refuses.

def test_scope_broadening_refuses() -> None:
    for key in sorted(e10.CORE_GRANT_SCOPE_EXCLUSIONS):
        broadened = {**e10.CORE_GRANT_SCOPE_EXCLUSIONS, key: True}
        _refuses("mismatch for scope_exclusions", scope_exclusions=broadened)
    _refuses("mismatch for scope_exclusions",
             scope_exclusions={**e10.CORE_GRANT_SCOPE_EXCLUSIONS,
                               "clean_test_authorized": True})
    partial = dict(e10.CORE_GRANT_SCOPE_EXCLUSIONS)
    partial.pop("f2_clean_test_evaluation_authorized")
    _refuses("mismatch for scope_exclusions", scope_exclusions=partial)
    _refuses("mismatch for authorized_by", authorized_by="independent_reviewer")
    _refuses("mismatch for authorized_by", authorized_by="codex")
    _refuses("mismatch for status", status="PARTIALLY_AUTHORIZED")
    _refuses("mismatch for authorization_token", authorization_token="approved")
    _refuses("mismatch for per_cell_wall_clock_hours",
             per_cell_wall_clock_hours=24.0)
    _refuses("mismatch for effective_identity_ceiling_hours",
             effective_identity_ceiling_hours=80.0)


# 14. Stale predecessor provenance refuses.

def test_stale_predecessor_provenance_refuses() -> None:
    provenance = e10.core_authorization_grant_expectations()["provenance"]
    _refuses("mismatch for provenance",
             provenance={**provenance,
                         "phase3_binding_amendment": {
                             **provenance["phase3_binding_amendment"],
                             "sha256": "4" * 64}})
    _refuses("mismatch for provenance",
             provenance={**provenance,
                         "phase2_binding_amendment": {
                             **provenance["phase2_binding_amendment"],
                             "sha256": "5" * 64}})
    # A Phase-2 record altered under Phase 3 breaks the amendment itself, so
    # nothing downstream of it can validate either.
    with tempfile.TemporaryDirectory() as directory:
        governance = Path(directory) / "governance"
        shutil.copytree(e10.OUT_DIR, governance)
        target = governance / "phase2_scientific_pipeline_20260810.json"
        target.write_text(target.read_text() + "\n")
        with mock.patch.object(e10, "OUT_DIR", governance):
            _must_raise(AssertionError, e10.validate_phase3_amendment,
                        "changed under Phase 3")


# 15. The source constant is never an accepted route.

def test_source_constant_mutation_is_not_a_grant_route() -> None:
    assert e10.E10_TRAINING_AUTHORIZED is None
    contract = phase3.assert_source_constant_is_not_a_route()
    assert contract["value"] is None
    assert contract["accepted_as_authorization"] is False
    for present in (False, True):
        with tempfile.TemporaryDirectory() as directory:
            with _temporary_authorization(Path(directory), grant=present), \
                    mock.patch.object(e10, "E10_TRAINING_AUTHORIZED",
                                      LEGACY_TOKEN):
                _must_raise(
                    SystemExit,
                    lambda: e10.assert_core_entry_authorized(
                        "B4", "train_40k", 0),
                    "must remain None",
                )
                _must_raise(SystemExit,
                            lambda: e10_run.core_cell("B4", "train_40k", 0),
                            "must remain None")
                _must_raise(
                    SystemExit,
                    lambda: e10.authorize_cell_execution("B4", "train_40k", 0),
                    "must remain None",
                )
    # An environment variable is not a route either.
    with tempfile.TemporaryDirectory() as directory:
        with _temporary_authorization(Path(directory), grant=False):
            os.environ["E10_TRAINING_AUTHORIZED"] = LEGACY_TOKEN
            try:
                _must_raise(
                    SystemExit,
                    lambda: e10.assert_core_entry_authorized(
                        "B4", "train_40k", 0),
                    "scientific authorization grant is absent",
                )
            finally:
                os.environ.pop("E10_TRAINING_AUTHORIZED", None)
    # No production source may write a grant for itself.
    scan = phase3.assert_no_self_granted_authorization()
    assert scan["grant_writers"] == 0
    assert scan["files_scanned"] > 50


# 16. The permit path works with a valid external grant.

def test_permit_path_works_with_the_valid_external_grant() -> None:
    with tempfile.TemporaryDirectory() as directory, _strict_state_restored():
        with _temporary_authorization(Path(directory)):
            e10.enable_strict_determinism()
            grant = e10.validate_core_authorization_grant("B4", "train_40k", 0)
            permit = e10.authorize_cell_execution("B4", "train_40k", 0)
            assert permit.token == LEGACY_TOKEN
            assert permit.grant_id == grant["grant_id"]
            assert permit.grant_sha256 == e10.sha256_file(
                e10.core_authorization_grant_path())
            assert permit.source_digest == e10.source_digest()[0]
            assert permit.per_cell_wall_hours == 12.0
            state = e10.validate_cell_permit(permit, revalidate_grant=True)
            assert state["remaining_seconds"] > 0
            spy = _SpyOptimizer()
            e10.guarded_optimizer_step(spy, permit)
            assert spy.steps == 1
            # A grant that changes under a running cell stops the next stage.
            path = e10.core_authorization_grant_path()
            path.write_text(path.read_text() + "\n")
            _must_raise(
                AssertionError,
                lambda: e10.validate_cell_permit(permit, revalidate_grant=True),
                "changed under a running cell",
            )
            path.unlink()
            _must_raise(
                AssertionError,
                lambda: e10.validate_cell_permit(permit, revalidate_grant=True),
                "removed under a running cell",
            )
            # The mutated source constant is refused on the permit path too.
            write_core_authorization_grant()
            with mock.patch.object(e10, "E10_TRAINING_AUTHORIZED",
                                   LEGACY_TOKEN):
                _must_raise(AssertionError,
                            lambda: e10.validate_cell_permit(permit),
                            "must remain None")
            assert spy.steps == 1


# 17. No real scientific grant exists.

def test_no_real_scientific_grant_exists() -> None:
    proof = e10.assert_no_core_authorization_grant()
    assert proof["grant_present"] is False
    assert proof["records_in_authorization_directory"] == 0
    assert proof["e10_training_authorized"] is None
    assert not e10.core_authorization_grant_path().exists()
    assert str(e10.CORE_AUTHORIZATION_DIR).startswith(str(e10.SHARED_STATE_DIR))
    refusal = phase1.assert_scientific_core_refused()
    assert refusal["core_authorization_grant_present"] is False
    proof_cells = e10.assert_no_scientific_cells()
    assert proof_cells == {"published_cell_records": 0, "checkpoint_files": 0,
                           "charged_core_cells": 0,
                           "e10_training_authorized": None}


# --- the Phase-3 amendment chain ----------------------------------------------

def test_phase3_amendment_is_the_newest_chain_link() -> None:
    chain = e10._amendment_chain()
    assert chain[0][0] == e10.PHASE3_AMENDMENT_PATH
    assert [path.name for path, _ in chain] == [
        "phase3_binding_amendment_20260811.json",
        "phase2_binding_amendment_20260810.json",
        "phase1_whole_cell_wall_repair_20260810.json",
        "phase1_binding_amendment_20260810.json",
        "source_correction_20260810.json",
    ]
    amendment = e10.validate_phase3_amendment()
    assert amendment["record_type"] == "e10_phase3_binding_amendment"
    assert amendment["status"] == "PHASE2_EVIDENCE_CARRIED_FORWARD"
    assert amendment["NON_SCIENTIFIC"] is True
    assert amendment["binding"]["source_digest"] == e10.source_digest()[0]
    assert amendment["phase2_r2_source_digest"] == e10.PHASE2_R2_SOURCE_DIGEST
    assert set(amendment["preserved_records"]) == set(
        e10.PHASE2_R2_RECORD_SHA256)
    assert amendment["change_scope"]["scientific_authorization_granted"] is False
    assert amendment["change_scope"]["scientific_recipe_changed"] is False
    assert amendment["change_scope"]["core_matrix_changed"] is False
    assert amendment["scientific_execution"]["scientific_cells_executed"] == 0
    # The Phase-2 pair is now an accepted historical binding, and every earlier
    # link still validates through the amendment that carries it.
    assert (e10.PHASE2_R2_SOURCE_DIGEST, e10.PHASE2_R2_CONFIG_DIGEST) in \
        e10.accepted_historical_bindings()
    assert e10.validate_phase2_amendment()["record_type"] == (
        "e10_phase2_binding_amendment")
    assert e10.validate_phase1_repair()["status"] == "WHOLE_CELL_WALL_ENFORCED"
    assert e10.validate_phase1_amendment()["status"] == (
        "PHASE0_EVIDENCE_CARRIED_FORWARD")
    assert e10.validate_source_correction()["status"] == (
        "VALIDATOR_CORRECTION_RECORDED")


def test_older_link_cannot_carry_itself_or_a_newer_one() -> None:
    """Without the Phase-3 link, the superseded Phase-2 record is not admitted."""
    with tempfile.TemporaryDirectory() as directory:
        absent = Path(directory) / "no-phase3-amendment.json"
        with mock.patch.object(e10, "PHASE3_AMENDMENT_PATH", absent):
            _must_raise(AssertionError, e10.validate_phase2_amendment,
                        "no amendment binds this record")
            _must_raise(AssertionError, e10.frozen_recipe_contract,
                        "no amendment binds this record")
    # Restored, the same records validate again.
    assert e10.frozen_recipe_contract()["protocol_family"] == e10.PROTOCOL_FAMILY


def test_phase3_records_are_immutable_and_write_once() -> None:
    for path in e10.PHASE3_PATHS:
        assert path.is_file(), path
    _must_raise(FileExistsError, phase3.write_phase3_records,
                "immutable E10 Phase-3 records exist")
    record = phase3._validated_authorization_record()
    assert record["status"] == (
        "AUTHORIZATION_MECHANISM_IMPLEMENTED_CORE_STILL_UNAUTHORIZED")
    assert record["phase2_closure_head"] == phase3.PHASE2_CLOSURE_HEAD
    assert record["state_at_handback"]["core_authorization_grant_present"] \
        is False
    assert record["state_at_handback"]["scientific_cells_executed"] == 0
    assert record["clean_test_accessed"] is False
    assert record["contract"]["defect"]["config_digest_moved"] is False
    assert record["contract"]["validator"]["partial_authorization"] is False


# --- no scientific regression -------------------------------------------------

def test_frozen_scientific_protocol_is_unchanged() -> None:
    assert e10.CORE_ARMS == ("B4", "B4r")
    assert e10.CORE_SCALES == ("train_40k", "train_250k")
    assert e10.CORE_SEEDS == (0, 1, 2)
    assert list(e10.CORE_CELLS) == FROZEN_ORDER
    assert e10.pair_preserving_order() == FROZEN_ORDER
    assert e10.DEV_EVALUATION_EPOCHS == (1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 19, 22)
    assert science.CANONICAL_EPOCH == 22
    assert e10.SCHEDULER_HORIZON_EPOCHS == 100
    assert config.E10_PER_CELL_WALL_CLOCK_HOURS == 12.0
    assert config.E10_PER_IDENTITY_CEILING_HOURS == 40.0
    assert e10.effective_identity_ceiling_hours() == 40.0
    assert config.E10_AUTOMATIC_RETRIES_PER_IDENTITY == 0
    assert e10.MODEL_REPO == "HuggingFaceTB/SmolLM2-360M"
    assert e10.MODEL_REVISION == "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
    assert e10.EXPECTED_TRAINABLE_PARAMETERS == 21_540_800
    # The recipe digests the grant binds are the frozen Phase-0 contract's.
    frozen = phase2.assert_frozen_recipe_reproduced()
    assert frozen["cells"] == 12
    expectations = e10.core_authorization_grant_expectations()
    assert expectations["recipe_digests"] == frozen["digests"]
    assert "wall_clock_halt_hours" not in e10.build_recipe("B4", "train_40k", 0)


def test_no_e10_source_assigns_the_authorization_constant() -> None:
    """Only the one sentinel assignment exists, and it is None."""
    package = PROJECT_ROOT / "experiments" / "e10_capacity_360m"
    assignments = []
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name)
                and target.id == "E10_TRAINING_AUTHORIZED"
                for target in node.targets
            ):
                assert isinstance(node.value, ast.Constant)
                assert node.value.value is None, path.name
                assignments.append(path.name)
    assert assignments == ["e10_common.py"]


def test_phase3_verify_reports_the_real_state() -> None:
    contract = phase3.authorization_contract()
    assert contract["entry_gate"]["cells_refused"] == 12
    assert contract["entry_gate"]["cells_authorised"] == 0
    assert contract["entry_gate"]["external_grant_present"] is False
    assert contract["scientific_core"]["refused"] is True
    assert contract["no_core_authorization_grant"]["grant_present"] is False
    assert contract["no_scientific_execution"]["charged_core_cells"] == 0
    assert contract["grant_schema"]["partial_authorization"] is False
    assert contract["grant_schema"]["contains_scientific_results"] is False
    assert contract["clean_test_accessed"] is False
    assert contract["resource_policy"]["per_cell_wall_clock_hours"] == 12.0
    assert contract["resource_policy"]["automatic_retries_per_identity"] == 0
    assert set(contract["non_routes"]) == {
        "source_constant", "environment_variable", "config_mutation",
        "runtime_monkeypatch"}
    scan = e10.assert_no_embargo_reference()
    assert scan["passed"] is True


TESTS = (
    test_missing_grant_refuses_every_authoritative_path,
    test_valid_exact_synthetic_grant_validates,
    test_creating_a_grant_does_not_move_the_source_digest,
    test_exact_twelve_cells_are_authorized,
    test_unregistered_cell_refuses,
    test_wrong_arm_scale_or_seed_refuses,
    test_wrong_matrix_order_refuses,
    test_wrong_recipe_or_source_digest_refuses,
    test_tampered_grant_refuses,
    test_nonzero_retries_refuse,
    test_scope_broadening_refuses,
    test_stale_predecessor_provenance_refuses,
    test_source_constant_mutation_is_not_a_grant_route,
    test_permit_path_works_with_the_valid_external_grant,
    test_no_real_scientific_grant_exists,
    test_phase3_amendment_is_the_newest_chain_link,
    test_older_link_cannot_carry_itself_or_a_newer_one,
    test_phase3_records_are_immutable_and_write_once,
    test_frozen_scientific_protocol_is_unchanged,
    test_no_e10_source_assigns_the_authorization_constant,
    test_phase3_verify_reports_the_real_state,
)


def run() -> None:
    for test in TESTS:
        test()
    print(f"E10 Phase-3 tests passed ({len(TESTS)} checks)")


if __name__ == "__main__":
    run()
