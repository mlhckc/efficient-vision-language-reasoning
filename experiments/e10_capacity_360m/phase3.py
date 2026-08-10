"""E10 Phase-3 scientific authorization binding.

Phase 2 closed with an independent PASS and one gate left: explicit scientific
authorization. The only mechanism that existed for it was the source constant
``e10_common.E10_TRAINING_AUTHORIZED``, and that mechanism is self-defeating.
``e10_common.py`` is inside ``SOURCE_PATHS``, so setting the constant moves the
live source digest, which immediately invalidates the reviewed provenance chain
the authorization has to bind. The reproduction is recorded in ``DEFECT`` below.

Phase 3 changes only how an authorization is expressed. The frozen scientific
recipe, the twelve-cell matrix, the model pin, the seeds, the cadence, the
epoch-22 rule, the scoring, the 12-hour wall, the 40-hour effective identity
ceiling and the zero-retry rule are untouched. The core is now opened by an
immutable EXTERNAL grant in shared state, validated by one canonical validator,
and the source constant is retained only as a legacy refusal sentinel that must
remain None.

Phase 3 authorises nothing. No grant is created here, ``run core-cell B4
train_40k 0`` still refuses before any model, dataset, CUDA context, optimizer
or output directory is touched, and no scientific cell has run.

    python -B -m experiments.e10_capacity_360m.run phase3-record
    python -B -m experiments.e10_capacity_360m.run phase3-verify
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from experiments.e10_capacity_360m import e10_common as e10  # noqa: E402
from experiments.e10_capacity_360m import phase1  # noqa: E402
from experiments.e10_capacity_360m import phase2  # noqa: E402


TASK_ID = e10.PHASE3_TASK_ID

# The approved Phase-2 closure state this task starts from.
PHASE2_CLOSURE_HEAD = "5d79bf5fb435ce365335593aff6912e37071f65c"
PHASE2_REVIEWED_SOURCE_HEAD = "45986fc49a8117ed951339457099f5d3debda3a3"

# The defect, reproduced against the frozen scientific executor at the closure
# HEAD before any Phase-3 source existed. Changing only the one constant from
# None to the approval token moved the live source digest and made every
# provenance consumer refuse, so the only implemented authorization mechanism
# could not be used without invalidating the source it was meant to open.
DEFECT = {
    "mechanism": (
        "assert_core_entry_authorized required "
        "e10_common.E10_TRAINING_AUTHORIZED == CORE_AUTHORIZATION_TOKEN"
    ),
    "why_self_defeating": (
        "e10_common.py is inside SOURCE_PATHS, so the constant is part of the "
        "live source digest that every immutable E10 record binds"
    ),
    "reproduced_at_head": PHASE2_CLOSURE_HEAD,
    "source_digest_before": e10.PHASE2_R2_SOURCE_DIGEST,
    "source_digest_after_setting_the_constant": (
        "214fcc39f99c146609a8a863be3ad3f64d4879382bd565c06083ab2e9b462094"
    ),
    "observed_consequences": [
        "phase1-verify failed: E10 Phase-2 amendment: stale source_digest",
        "phase2-verify failed: E10 Phase-2 amendment: stale source_digest",
        "run core-cell B4 train_40k 0 failed on stale provenance rather than "
        "reaching the authorization decision at all",
    ],
    "config_digest_moved": False,
    "repair": (
        "the runtime grant is an immutable external record outside the "
        "repository, outside SOURCE_PATHS and outside CONFIG_PATHS, so "
        "creating it cannot move a digest that a reviewed record binds"
    ),
}

# Every authoritative path that must require the external grant. Each one is
# checked by assert_entry_gate_requires_grant below.
AUTHORITATIVE_ENTRY_PATHS = (
    "experiments.e10_capacity_360m.e10_common.assert_core_entry_authorized",
    "experiments.e10_capacity_360m.e10_common.authorize_cell_execution",
    "experiments.e10_capacity_360m.e10_common.charge_core_cell_hours",
    "experiments.e10_capacity_360m.phase1.assert_cell_entry_guarded",
    "experiments.e10_capacity_360m.phase1.ScientificCellGuard.__enter__",
    "experiments.e10_capacity_360m.science.run_core_cell",
    "experiments.e10_capacity_360m.run.core_cell",
)

# Routes that are NOT authorization, stated once so a reviewer never has to
# infer them from a diff.
NON_ROUTES = {
    "source_constant": (
        "E10_TRAINING_AUTHORIZED must remain None; any other value is itself a "
        "refusal, so editing scientific source can never open the core"
    ),
    "environment_variable": (
        "no E10 authorization path reads an environment variable"
    ),
    "config_mutation": (
        "config carries resource bounds only; setting or widening them cannot "
        "produce a grant, and the entry gate refuses without one"
    ),
    "runtime_monkeypatch": (
        "a patched constant is not accepted, and a cell permit is signed, "
        "process-bound and carries the grant bytes it was issued under"
    ),
}

# Sources that must never write an external authorization grant. Tests are
# excluded by design: a test fabricates a grant inside a temporary directory,
# which is exactly what production sources may not do.
_SCANNED_ROOTS = ("experiments", "src")
_EXCLUDED_PARTS = frozenset({"vendor", "__pycache__"})
_WRITE_CALLS = frozenset({
    "atomic_write_json", "atomic_replace_json", "write_text", "write_bytes",
    "open", "touch", "mkdir", "rename", "replace", "dump", "symlink_to",
})
_GRANT_TOKENS = (
    "core_authorization_grant", "CORE_AUTHORIZATION_GRANT",
    "CORE_AUTHORIZATION_DIR",
)


class AuthorizationError(AssertionError):
    """A Phase-3 authorization-binding contract refused."""


def _relative(path: Path) -> str:
    path = Path(path)
    if path.is_relative_to(PROJECT_ROOT):
        return str(path.relative_to(PROJECT_ROOT))
    return str(path)


def _dotted_name(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _mentions_grant(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, (ast.Name, ast.Attribute)):
            text = _dotted_name(child)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            text = child.value
        else:
            continue
        if any(token in text for token in _GRANT_TOKENS):
            return True
    return False


# --- executable contract checks ----------------------------------------------

def assert_no_self_granted_authorization() -> dict:
    """No project source may write an external authorization grant.

    Together with the validator, which refuses anything that is not the exact
    frozen matrix under the reviewed provenance, this makes a self-granted
    authorization impossible rather than merely discouraged.
    """
    writers = []
    scanned = []
    paths = [PROJECT_ROOT / "config.py"]
    for root in _SCANNED_ROOTS:
        paths.extend(sorted((PROJECT_ROOT / root).rglob("*.py")))
    for path in paths:
        if not path.is_file() or _EXCLUDED_PARTS.intersection(
                path.relative_to(PROJECT_ROOT).parts):
            continue
        scanned.append(str(path.relative_to(PROJECT_ROOT)))
        tree = ast.parse(path.read_text(), filename=str(path))
        for call in (node for node in ast.walk(tree)
                     if isinstance(node, ast.Call)):
            name = _dotted_name(call.func).split(".")[-1]
            if name not in _WRITE_CALLS:
                continue
            arguments = list(call.args) + [
                keyword.value for keyword in call.keywords
            ]
            if any(_mentions_grant(argument) for argument in arguments):
                writers.append(str(path.relative_to(PROJECT_ROOT)))
    if writers:
        raise AuthorizationError(
            f"project source writes an E10 core authorization grant: {writers}"
        )
    return {
        "files_scanned": len(scanned),
        "grant_writers": 0,
        "grant_is_created_by": "the user, out of band, after Phase-3 review",
    }


def assert_grant_cannot_move_source_digest() -> dict:
    """The grant location is structurally outside every bound digest.

    This is the invariant the whole task exists for: creating an authorization
    must not change the source the authorization binds.
    """
    grant_path = e10.core_authorization_grant_path()
    if grant_path.is_relative_to(PROJECT_ROOT):
        raise AuthorizationError(
            "the E10 core authorization grant would live inside the repository"
        )
    bound = {path.resolve() for path in e10.SOURCE_PATHS + e10.CONFIG_PATHS}
    if grant_path.resolve() in bound or e10.CORE_AUTHORIZATION_DIR.resolve() in {
        path.parent.resolve() for path in e10.SOURCE_PATHS + e10.CONFIG_PATHS
    }:
        raise AuthorizationError(
            "the E10 core authorization grant is inside a bound digest path"
        )
    return {
        "grant_path": str(grant_path),
        "inside_repository": False,
        "inside_source_paths": False,
        "inside_config_paths": False,
        "source_digest": e10.source_digest()[0],
        "config_digest": e10.config_digest()[0],
    }


def assert_source_constant_is_not_a_route() -> dict:
    """The legacy sentinel is refused as an authorization, in source and live."""
    if e10.E10_TRAINING_AUTHORIZED is not None:
        raise AuthorizationError(
            "E10_TRAINING_AUTHORIZED is set; it is a refusal sentinel and must "
            "remain None"
        )
    source = Path(e10.__file__).read_text()
    tree = ast.parse(source, filename=e10.__file__)
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == "E10_TRAINING_AUTHORIZED"
            for target in node.targets
        )
    ]
    if len(assignments) != 1:
        raise AuthorizationError(
            "E10_TRAINING_AUTHORIZED is not assigned exactly once"
        )
    value = assignments[0].value
    if not isinstance(value, ast.Constant) or value.value is not None:
        raise AuthorizationError("E10_TRAINING_AUTHORIZED is not None in source")
    return {
        "value": e10.E10_TRAINING_AUTHORIZED,
        "assignments": 1,
        "accepted_as_authorization": False,
        "non_routes": NON_ROUTES,
    }


def assert_entry_gate_requires_grant() -> dict:
    """No frozen cell may be authorised while no external grant exists.

    Stated as the invariant rather than as "everything refuses", so the check
    stays truthful once a real grant is created: what it forbids is a cell
    opening without one.
    """
    grant_present = e10.core_authorization_grant_path().is_file()
    outcomes = {}
    for arm, scale, seed in phase1.execution_plan():
        key = f"{arm}_{scale}_seed{seed}"
        try:
            e10.assert_core_entry_authorized(arm, scale, seed)
        except SystemExit as error:
            outcomes[key] = {"authorised": False, "refusal": str(error)}
        except AssertionError as error:
            outcomes[key] = {"authorised": False,
                             "refusal": f"E10 CORE REFUSED: {error}"}
        else:
            outcomes[key] = {"authorised": True, "refusal": None}
    authorised = sorted(key for key, value in outcomes.items()
                        if value["authorised"])
    if not grant_present and authorised:
        raise AuthorizationError(
            f"cells were authorised with no external grant: {authorised}"
        )
    if len(outcomes) != len(e10.CORE_CELLS):
        raise AuthorizationError("the frozen plan did not cover twelve cells")
    return {
        "external_grant_present": grant_present,
        "cells_checked": len(outcomes),
        "cells_authorised": len(authorised),
        "cells_refused": len(outcomes) - len(authorised),
        "authoritative_entry_paths": list(AUTHORITATIVE_ENTRY_PATHS),
        "example_refusal": outcomes["B4_train_40k_seed0"]["refusal"],
    }


def grant_schema() -> dict:
    """The external grant's field contract, without creating one."""
    expectations = e10.core_authorization_grant_expectations()
    return {
        "path": str(e10.core_authorization_grant_path()),
        "record_type": expectations["record_type"],
        "required_fields": sorted(
            set(expectations) | {"grant_id", "utc", "binding"}
        ),
        "authorized_by": expectations["authorized_by"],
        "authorization_token": expectations["authorization_token"],
        "purpose": expectations["purpose"],
        "authorized_arms": expectations["authorized_arms"],
        "authorized_scales": expectations["authorized_scales"],
        "authorized_seeds": expectations["authorized_seeds"],
        "authorized_cells": expectations["authorized_cells"],
        "recipe_digests": expectations["recipe_digests"],
        "recipe_contract": expectations["recipe_contract"],
        "model": expectations["model"],
        "per_cell_wall_clock_hours": expectations["per_cell_wall_clock_hours"],
        "effective_identity_ceiling_hours": expectations[
            "effective_identity_ceiling_hours"
        ],
        "automatic_retries": expectations["automatic_retries"],
        "scope_exclusions": expectations["scope_exclusions"],
        "provenance": expectations["provenance"],
        "grant_id_rule": (
            "SHA-256 of the canonical JSON of every field except grant_id, so "
            "any modified byte refuses"
        ),
        "binding_rule": (
            "the LIVE Phase-3 source and config digests; a grant is created "
            "only after the review of the source it opens, so no historical "
            "binding is admissible"
        ),
        "contains_scientific_results": False,
        "partial_authorization": False,
    }


def validator_contract() -> dict:
    """What the one canonical validator refuses, stated for review."""
    return {
        "validator": (
            "experiments.e10_capacity_360m.e10_common."
            "validate_core_authorization_grant"
        ),
        "fails_closed_on": [
            "missing grant",
            "competing grant records in the authorization directory",
            "malformed grant (missing, extra or mistyped fields)",
            "wrong protocol family or experiment identity",
            "wrong source or config digest (stale reviewed source)",
            "wrong recipe-contract reference or recipe digest",
            "wrong model repository, revision, weight digest or identity",
            "wrong arm, scale or seed set",
            "wrong matrix, wrong order, missing pair or extra cell",
            "changed per-cell wall",
            "changed effective identity ceiling",
            "automatic retries other than zero",
            "authorised_by other than the user",
            "tampered bytes (grant_id mismatch)",
            "stale predecessor provenance",
            "clean-test, F1, F2, A5 or A8c scope broadening",
        ],
        "grants_exactly": "the frozen, pair-preserved twelve-cell B4/B4r matrix",
        "partial_authorization": False,
    }


def authorization_contract() -> dict:
    """What Phase 3 implements, checked rather than asserted in prose."""
    return {
        "task_id": TASK_ID,
        "defect": DEFECT,
        "grant_schema": grant_schema(),
        "validator": validator_contract(),
        "non_routes": NON_ROUTES,
        "source_constant": assert_source_constant_is_not_a_route(),
        "no_self_granted_authorization": assert_no_self_granted_authorization(),
        "grant_outside_bound_digests": assert_grant_cannot_move_source_digest(),
        "entry_gate": assert_entry_gate_requires_grant(),
        "scientific_core": phase1.assert_scientific_core_refused(),
        "no_scientific_execution": e10.assert_no_scientific_cells(),
        "no_core_authorization_grant": e10.assert_no_core_authorization_grant(),
        "frozen_recipe": phase2.assert_frozen_recipe_reproduced(),
        "frozen_plan": phase1.assert_frozen_execution_plan(),
        "resource_policy": phase1.resource_policy(),
        "no_automatic_retry": phase1.assert_no_automatic_retry(),
        "clean_test_accessed": False,
    }


# --- immutable records --------------------------------------------------------

def _phase2_preserved_records() -> dict:
    preserved = {}
    for filename, expected in e10.PHASE2_R2_RECORD_SHA256.items():
        path = e10.OUT_DIR / filename
        if not path.is_file():
            raise AuthorizationError(f"Phase-2 record is missing: {filename}")
        digest = e10.sha256_file(path)
        if digest != expected:
            raise AuthorizationError(f"Phase-2 record changed: {filename}")
        preserved[filename] = {"location": "repository", "sha256": digest}
    return preserved


def write_phase3_records() -> dict:
    """Write the two immutable Phase-3 records, once."""
    existing = [str(path) for path in e10.PHASE3_PATHS if path.exists()]
    if existing:
        raise FileExistsError(f"immutable E10 Phase-3 records exist: {existing}")
    e10.assert_no_embargo_reference()
    phase1.assert_scientific_core_refused()
    e10.assert_no_scientific_cells()
    e10.assert_no_core_authorization_grant()
    preserved = _phase2_preserved_records()

    amendment = {
        "schema_version": 1,
        "record_type": "e10_phase3_binding_amendment",
        "task_id": TASK_ID,
        "status": "PHASE2_EVIDENCE_CARRIED_FORWARD",
        "NON_SCIENTIFIC": True,
        "utc": e10.utc_now(),
        "binding": e10.binding_record(),
        "reason": (
            "Phase 3 adds the external scientific authorization grant, its "
            "canonical validator and the Phase-3 module and test sources, so "
            "the live source digest moves. config.py and requirements.lock.txt "
            "are untouched, so the config digest does not move. The Phase-2 "
            "records are preserved byte for byte and are carried forward by "
            "this record instead of being rewritten."
        ),
        "change_scope": {
            "authorization_binding_implemented": True,
            "resource_constants_changed": False,
            "scientific_recipe_changed": False,
            "core_matrix_changed": False,
            "model_pin_changed": False,
            "dev_evaluation_cadence_changed": False,
            "scientific_authorization_granted": False,
            "phase0_record_rewritten": False,
            "phase1_record_rewritten": False,
            "phase2_record_rewritten": False,
        },
        "phase2_r2_source_digest": e10.PHASE2_R2_SOURCE_DIGEST,
        "phase2_r2_config_digest": e10.PHASE2_R2_CONFIG_DIGEST,
        "preserved_records": preserved,
        "scientific_execution": {
            "optimizer_steps": 0,
            "scientific_cells_executed": 0,
            "gpu_hours_charged": 0.0,
            "e10_training_authorized": None,
        },
    }
    amendment_sha = e10.atomic_write_json(e10.PHASE3_AMENDMENT_PATH, amendment)

    binding = {
        "schema_version": 1,
        "record_type": "e10_phase3_authorization_binding",
        "task_id": TASK_ID,
        "status": "AUTHORIZATION_MECHANISM_IMPLEMENTED_CORE_STILL_UNAUTHORIZED",
        "NON_SCIENTIFIC": True,
        "utc": e10.utc_now(),
        "git_commit": e10.current_git_commit(),
        "binding": e10.binding_record(),
        "phase2_closure_head": PHASE2_CLOSURE_HEAD,
        "phase2_reviewed_source_head": PHASE2_REVIEWED_SOURCE_HEAD,
        "contract": authorization_contract(),
        "state_at_handback": {
            "core_authorization_grant_present": False,
            "e10_training_authorized": e10.E10_TRAINING_AUTHORIZED,
            "scientific_cells_executed": 0,
            "checkpoints": 0,
            "core_gpu_hours_charged": 0.0,
            "next_gate": (
                "one explicit user scientific authorization grant, created out "
                "of band after an independent Phase-3 review returns PASS"
            ),
        },
        "clean_test_accessed": False,
    }
    binding_sha = e10.atomic_write_json(e10.PHASE3_AUTHORIZATION_PATH, binding)
    return {"phase3_binding_amendment": amendment_sha,
            "phase3_authorization_binding": binding_sha}


def _validated_authorization_record() -> dict:
    record = e10.read_json_mapping(e10.PHASE3_AUTHORIZATION_PATH)
    e10.assert_recorded_binding(record, "E10 Phase-3 authorization binding",
                                e10.PHASE3_AUTHORIZATION_PATH)
    if record.get("record_type") != "e10_phase3_authorization_binding" \
            or record.get("task_id") != TASK_ID \
            or record.get("NON_SCIENTIFIC") is not True \
            or record.get("status") != (
                "AUTHORIZATION_MECHANISM_IMPLEMENTED_CORE_STILL_UNAUTHORIZED"):
        raise AuthorizationError(
            "Phase-3 authorization binding record semantics mismatch"
        )
    contract = record.get("contract", {})
    if contract.get("validator", {}).get("partial_authorization") is not False:
        raise AuthorizationError("the record does not forbid partial authorisation")
    if contract.get("scientific_core", {}).get("refused") is not True:
        raise AuthorizationError("the Phase-3 record does not record the refusal")
    return record


def verify() -> dict:
    """Validate every Phase-1 and Phase-2 contract and the Phase-3 binding."""
    result = {
        "phase2": phase2.verify(),
        "phase3_binding_amendment": {
            "sha256": e10.sha256_file(e10.PHASE3_AMENDMENT_PATH),
            "preserved_records": len(
                e10.validate_phase3_amendment()["preserved_records"]),
        } if e10.PHASE3_AMENDMENT_PATH.exists() else {"written": False},
        "authorization_contract": authorization_contract(),
    }
    if e10.PHASE3_AUTHORIZATION_PATH.exists():
        record = _validated_authorization_record()
        result["phase3_authorization_binding"] = {
            "sha256": e10.sha256_file(e10.PHASE3_AUTHORIZATION_PATH),
            "status": record["status"],
        }
    return result


def main(argv: Any = None) -> int:  # pragma: no cover - thin CLI wrapper
    import json

    print(json.dumps(verify(), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
