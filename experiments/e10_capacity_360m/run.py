"""E10 Phase-0 governance, provenance and refusal entry points.

Examples, from the project root with the project environment sourced:

    python -B -m experiments.e10_capacity_360m.run governance
    python -B -m experiments.e10_capacity_360m.run provenance --prepare
    python -B -m experiments.e10_capacity_360m.run source-correction
    python -B -m experiments.e10_capacity_360m.run provenance --download-verify
    python -B -m experiments.e10_capacity_360m.run ledgers --initialize
    python -B -m experiments.e10_capacity_360m.run calibration-grant
    python -B -m experiments.e10_capacity_360m.run core-order
    python -B -m experiments.e10_capacity_360m.run core-cell B4 train_40k 0

The final command is a required known-negative in Phase 0. It refuses before
model, data, CUDA, result-directory or optimizer initialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import snapshot_download  # noqa: E402

import config  # noqa: E402
from src import utils  # noqa: E402
from experiments.e10_capacity_360m import e10_common as e10  # noqa: E402


FROZEN_BASELINES = e10.FROZEN_RESULT_BASELINES


def _record(body: dict, *, seed: int = 0) -> dict:
    return {
        "metadata": {
            **utils.run_metadata(seed=seed),
            "task_id": e10.TASK_ID,
            "protocol_family": e10.PROTOCOL_FAMILY,
            "binding": e10.binding_record(),
        },
        **body,
    }


def write_governance() -> dict:
    e10.assert_no_embargo_reference()
    existing = [str(path) for path in e10.GOVERNANCE_PATHS if path.exists()]
    if existing:
        raise FileExistsError(
            f"immutable E10 governance is already present or partial: {existing}"
        )
    if e10.sha256_file(e10.U1_PATH) != e10.U1_SHA256:
        raise AssertionError("frozen U1 preregistration no longer matches")

    disposition = _record({
        "e10_a5_a8c_disposition": {
            "record_type": "pre_result_governance",
            "dated_utc": e10.utc_now(),
            "status": "RE_DEFERRED_NOT_CANCELLED",
            "A5": {
                "scientifically_owed": True,
                "outside_e10": True,
                "deferred_behind_e10": True,
                "cancelled": False,
                "superseded": False,
                "role": "input-side frozen 360M question encoder and question-only head",
            },
            "A8c": {
                "scientifically_owed": True,
                "outside_e10": True,
                "deferred_behind_e10": True,
                "cancelled": False,
                "superseded": False,
                "role": "input-side frozen 360M question features with a global image-fusion head",
            },
            "e10_B4_role": (
                "output-side answer readout: token-level reasoner, frozen causal "
                "LM generator and teacher-forced answer-token NLL"
            ),
            "scientific_non_overlap": (
                "A5/A8c change the input-side question representation and head; "
                "B4 changes the output-side frozen answer readout and objective"
            ),
            "budget_treatment": {
                "charged_to_e10_usage": False,
                "gate1_view_1": "E10-only with A5/A8c deferred",
                "gate1_view_2": (
                    "E10 plus a reserved A5/A8c term; no tracked authoritative "
                    "numeric reserve exists, so the term remains unquantified"
                ),
            },
            "A2_A2r": {
                "status": "UNRESOLVED_NOT_CANCELLED",
                "definition_invented": False,
                "phase0_blocker": False,
            },
            "pre_result": True,
        }
    })
    disposition_sha = e10.atomic_write_json(e10.A5_A8C_DISPOSITION_PATH, disposition)

    discharge = _record({
        "e10_u1_discharge": {
            "record_type": "pre_result_governance",
            "dated_utc": e10.utc_now(),
            "status": "U1_EXTENSION_INSTANTIATED_IN_E10_PHASE0",
            "immutable_source": {
                "path": str(e10.U1_PATH.relative_to(PROJECT_ROOT)),
                "sha256": e10.U1_SHA256,
                "modified": False,
            },
            "registered_names_retained": ["B4", "B4r"],
            "restored_size_sensitivity_scales": list(e10.CORE_SCALES),
            "pair_clause_mapping": {
                "pretrained": "B4",
                "architecture_matched_random": "B4r",
                "executable_pair_order": "B4 then B4r within every scale and seed",
            },
            "scientific_cells_authorized": 0,
            "pre_result": True,
        }
    })
    discharge_sha = e10.atomic_write_json(e10.U1_DISCHARGE_PATH, discharge)

    recipe_pairs = {}
    for scale in e10.CORE_SCALES:
        for seed in e10.CORE_SEEDS:
            b4 = e10.build_recipe("B4", scale, seed)
            b4r = e10.build_recipe("B4r", scale, seed)
            pair_b4 = e10.paired_recipe_sha256(b4)
            pair_b4r = e10.paired_recipe_sha256(b4r)
            if pair_b4 != pair_b4r:
                raise AssertionError("B4/B4r pairing-normalized recipes differ")
            recipe_pairs[f"{scale}_seed{seed}"] = {
                "B4_full_sha256": e10.recipe_sha256(b4),
                "B4r_full_sha256": e10.recipe_sha256(b4r),
                "pairing_normalized_sha256": pair_b4,
            }
    recipe_record = _record({
        "e10_recipe_contract": {
            "status": "FROZEN_FOR_FUTURE_DECISION",
            "protocol_family": e10.PROTOCOL_FAMILY,
            "core_arms": list(e10.CORE_ARMS),
            "core_scales": list(e10.CORE_SCALES),
            "core_seeds": list(e10.CORE_SEEDS),
            "future_cells": len(e10.CORE_CELLS),
            "scientific_cells_authorized": 0,
            "dev_evaluation_epochs": list(e10.DEV_EVALUATION_EPOCHS),
            "pair_normalization_keys": list(e10.PAIR_NORMALIZATION_KEYS),
            "pairs": recipe_pairs,
            "inheritance": {
                arm: e10.verify_recipe_inheritance(arm, "train_40k", 0)
                for arm in e10.CORE_ARMS
            },
            "e10_training_authorized": e10.E10_TRAINING_AUTHORIZED,
            "cross_size_interpretation": {
                "trainable_parameter_delta": 196_992,
                "relative_delta_percent": 0.923,
                "wording": (
                    "The cross-size comparison changes trainable capacity by "
                    "only 0.923%, substantially less than earlier controlled "
                    "capacity variations; nevertheless, it is interpreted as "
                    "whole-system capacity sensitivity rather than an isolated "
                    "frozen-LM-size effect."
                ),
            },
        }
    })
    recipe_sha = e10.atomic_write_json(e10.RECIPE_CONTRACT_PATH, recipe_record)

    frozen_record = _record({
        "e10_frozen_artifact_baseline": {
            "status": "RECORDED_BEFORE_IMPLEMENTATION",
            "method": (
                "for each root: sorted file paths, each file SHA-256, then "
                "SHA-256 of the complete sha256sum stream"
            ),
            "roots": FROZEN_BASELINES,
            "must_match_after_phase0": True,
        }
    })
    frozen_sha = e10.atomic_write_json(e10.FROZEN_BASELINE_PATH, frozen_record)
    return {
        "a5_a8c_disposition": disposition_sha,
        "u1_discharge": discharge_sha,
        "recipe_contract": recipe_sha,
        "frozen_artifact_baseline": frozen_sha,
    }


def prepare_provenance() -> dict:
    from experiments.e10_capacity_360m.projection import (
        validate_governance_records,
    )
    validate_governance_records()
    digest = e10.write_predownload_provenance()
    return {"path": str(e10.PREDOWNLOAD_PATH), "sha256": digest}


def download_and_verify() -> dict:
    e10.validate_predownload_provenance()
    snapshot = Path(snapshot_download(
        repo_id=e10.MODEL_REPO,
        revision=e10.MODEL_REVISION,
        cache_dir=str(e10.model_snapshot_dir().parents[2]),
    ))
    if snapshot.name != e10.MODEL_REVISION:
        raise AssertionError(
            f"download resolved {snapshot.name!r}, expected {e10.MODEL_REVISION}"
        )
    record = e10.model_verification_record(snapshot)
    digest = e10.atomic_write_json(e10.MODEL_VERIFICATION_PATH, record)
    return {
        "path": str(e10.MODEL_VERIFICATION_PATH),
        "sha256": digest,
        "resolved_snapshot": str(snapshot),
        "parameter_count": record["loaded_model"]["parameter_count"],
    }


def write_source_correction() -> dict:
    return e10.write_source_correction()


def initialize_ledgers() -> dict:
    e10.validate_model_verification()
    return e10.initialize_ledgers(e10.CALIBRATION_TOKEN)


def write_calibration_grant() -> dict:
    record = e10.calibration_grant_record()
    digest = e10.atomic_write_json(e10.CALIBRATION_GRANT_PATH, record)
    return {"path": str(e10.CALIBRATION_GRANT_PATH), "sha256": digest}


def core_order() -> list:
    order = e10.pair_preserving_order()
    if len(order) != 12:
        raise AssertionError("E10 core order does not contain twelve cells")
    return order


def core_cell(arm: str, scale: str, seed: int) -> int:
    # Deliberately the first operation: no output directory, model, data, CUDA,
    # ledger or imported E8B authorization can be reached before this refusal.
    e10.assert_core_entry_authorized(arm, scale, seed)
    raise AssertionError("unreachable in Phase 0")


def _frozen_tree_inventory(root: Path) -> dict:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        rows.append((relative, e10.sha256_file(path)))
    stream = "".join(f"{digest}  {relative}\n" for relative, digest in rows)
    return {
        "files": len(rows),
        "tree_sha256": hashlib.sha256(stream.encode("utf-8")).hexdigest(),
    }


def _run_validation_command(command: list[str], expected_success: bool) -> dict:
    print(f"[E10 validation] {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    passed = (completed.returncode == 0) is expected_success
    if not passed:
        expectation = "zero" if expected_success else "non-zero"
        raise AssertionError(
            f"validation command returned {completed.returncode}; expected {expectation}: "
            f"{command}"
        )
    return {
        "command": command,
        "exit_status": completed.returncode,
        "expected": "zero" if expected_success else "non-zero",
        "passed": True,
    }


def write_validation_record() -> dict:
    if e10.VALIDATION_PATH.exists():
        raise FileExistsError("immutable E10 validation record already exists")
    commands = [
        _run_validation_command(
            [sys.executable, "-B", "tests/test_e10.py"],
            True,
        ),
        _run_validation_command(
            [sys.executable, "-B", "tests/run_all.py"], True,
        ),
        _run_validation_command(
            [sys.executable, "-B", "-m", "experiments.e10_capacity_360m.run",
             "core-order"],
            True,
        ),
        _run_validation_command(
            [sys.executable, "-B", "-m", "experiments.e10_capacity_360m.run",
             "core-cell", "B4", "train_40k", "0"],
            False,
        ),
    ]

    model = e10.validate_model_verification()
    shared = e10.assert_shared_state_healthy(require_calibration_complete=True)
    if not e10.calibration_is_revoked():
        raise AssertionError("calibration revocation is not complete")
    source_scan = e10.assert_no_embargo_reference()
    output_scan = e10.assert_no_scientific_outputs()
    if e10.E10_TRAINING_AUTHORIZED is not None \
            or e10.E10_CALIBRATION_AUTHORIZED is not None:
        raise AssertionError("an E10 authorization constant is live")
    if config.E10_PER_IDENTITY_CEILING_HOURS is not None \
            or config.E10_PER_CELL_WALL_CLOCK_HOURS is not None:
        raise AssertionError("an E10 scientific resource constant is set")

    frozen = {}
    for name, expected in FROZEN_BASELINES.items():
        root = config.RESULTS_DIR / "experiments" / name
        observed = _frozen_tree_inventory(root)
        if observed != expected:
            raise AssertionError(
                f"frozen {name} result tree changed: {observed} != {expected}"
            )
        frozen[name] = {**observed, "matches_preimplementation": True}
    requirement_path = PROJECT_ROOT / "requirements.lock.txt"
    requirement_sha = e10.sha256_file(requirement_path)
    if requirement_sha != "644a7e9db3cfe8632f2802314ca0296cf07dc5e3a41abf6728f3a7904c8a04be":
        raise AssertionError("requirements.lock.txt changed")
    tracked_frozen = subprocess.run(
        [
            "git", "diff", "--exit-code",
            "6592f2f5d30899ca9474426de08a14043fc18b70", "--",
            "results/experiments/e8a_question_encoder",
            "results/experiments/e8b_readout_generation",
            "results/experiments/e9_compact_vlm",
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if tracked_frozen.returncode != 0:
        raise AssertionError("a tracked frozen scientific artifact changed")

    record = {
        "schema_version": 1,
        "record_type": "e10_phase0_validation",
        "task_id": e10.TASK_ID,
        "status": "PASS",
        "NON_SCIENTIFIC": True,
        "utc": e10.utc_now(),
        "git_commit": e10.current_git_commit(),
        "binding": e10.binding_record(),
        "commands": commands,
        "checks": {
            "model_live_reverified": model["all_hard_checks_passed"],
            "shared_calibration_complete": shared["calibration_complete"],
            "calibration_authorization_revoked": True,
            "scientific_authorization_unset": True,
            "scientific_budget_constants_unset": True,
            "source_embargo_scan": source_scan,
            "non_scientific_output_scan": output_scan,
            "frozen_result_trees": frozen,
            "tracked_frozen_diff_clean": True,
            "requirements_lock_sha256": requirement_sha,
        },
        "clean_test_accessed": False,
    }
    digest = e10.atomic_write_json(e10.VALIDATION_PATH, record)
    return {"path": str(e10.VALIDATION_PATH), "sha256": digest, "status": "PASS"}


def status() -> dict:
    return {
        "task_id": e10.TASK_ID,
        "git_commit": e10.current_git_commit(),
        "model_snapshot_present": e10.model_snapshot_dir().is_dir(),
        "pre_download_record": e10.PREDOWNLOAD_PATH.exists(),
        "model_verification": e10.MODEL_VERIFICATION_PATH.exists(),
        "spend_ledger": e10.SPEND_LEDGER.exists(),
        "retry_ledger": e10.RETRY_LEDGER.exists(),
        "calibration_claim": e10.CALIBRATION_CLAIM_PATH.exists(),
        "calibration_completion": e10.CALIBRATION_COMPLETION_PATH.exists(),
        "shared_failure_latch": e10.LEDGER_FAILURE_PATH.exists(),
        "calibration_grant": e10.CALIBRATION_GRANT_PATH.exists(),
        "calibration_revocation": e10.CALIBRATION_REVOCATION_PATH.exists(),
        "validation": e10.VALIDATION_PATH.exists(),
        "e10_training_authorized": e10.E10_TRAINING_AUTHORIZED,
        "scientific_identity_ceiling": getattr(
            __import__("config"), "E10_PER_IDENTITY_CEILING_HOURS"
        ),
        "scientific_cell_wall": getattr(
            __import__("config"), "E10_PER_CELL_WALL_CLOCK_HOURS"
        ),
        "free_scratch_bytes": shutil.disk_usage(PROJECT_ROOT).free,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("governance")

    provenance = subparsers.add_parser("provenance")
    provenance_group = provenance.add_mutually_exclusive_group(required=True)
    provenance_group.add_argument("--prepare", action="store_true")
    provenance_group.add_argument("--download-verify", action="store_true")

    subparsers.add_parser("source-correction")
    ledgers = subparsers.add_parser("ledgers")
    ledgers.add_argument("--initialize", action="store_true", required=True)
    subparsers.add_parser("calibration-grant")
    subparsers.add_parser("core-order")
    cell = subparsers.add_parser("core-cell")
    cell.add_argument("arm")
    cell.add_argument("scale")
    cell.add_argument("seed", type=int)
    subparsers.add_parser("verify")
    subparsers.add_parser("validation")
    subparsers.add_parser("status")
    args = parser.parse_args(argv)

    if args.command == "governance":
        result = write_governance()
    elif args.command == "provenance":
        result = prepare_provenance() if args.prepare else download_and_verify()
    elif args.command == "source-correction":
        result = write_source_correction()
    elif args.command == "ledgers":
        result = initialize_ledgers()
    elif args.command == "calibration-grant":
        result = write_calibration_grant()
    elif args.command == "core-order":
        result = core_order()
    elif args.command == "core-cell":
        return core_cell(args.arm, args.scale, args.seed)
    elif args.command == "verify":
        result = {
            "embargo": e10.assert_no_embargo_reference(),
            "outputs": e10.assert_no_scientific_outputs(),
        }
    elif args.command == "validation":
        result = write_validation_record()
    elif args.command == "status":
        result = status()
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
