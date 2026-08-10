"""Generate the E10 Gate-1 resource projection from measured Phase-0 records."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import tokens_data  # noqa: E402
from experiments.e10_capacity_360m import e10_common as e10  # noqa: E402


E8B_CALIBRATION_PATH = (
    config.RESULTS_DIR
    / "experiments"
    / "e8b_readout_generation"
    / "throughput_calibration_20260807.json"
)
E8B_STEP_SECONDS = {
    "train_40k": 0.235130,
    "train_250k": 0.236435,
}
STRESS_MULTIPLIER = 1.25
FLAT_STRESS_MULTIPLIER = 3.0
WALL_RECOMMENDATION_MULTIPLIER = 1.30
WALL_CANDIDATES_HOURS = (12.0, 13.0, 14.0)
DEV_ROWS = 7714
G14_PRE_ROWS = 64
G14_POST_MAX_ROWS = 160
G14_MAX_ROWS_PER_CELL = G14_PRE_ROWS + G14_POST_MAX_ROWS
E8B_TRAINABLE_PARAMETERS = 21_343_808
MIB = 2 ** 20
GIB = 2 ** 30
E8B_RESUME_MIB = 325.9
E8B_CANONICAL_MIB = 81.5
FUTURE_CELL_STORAGE_RESERVE_BYTES = 1 * GIB
FUTURE_RETRY_STORAGE_RESERVE_BYTES = 2 * GIB


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _finite(value, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssertionError(f"{context} is not numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise AssertionError(f"{context} is not finite and non-negative")
    return number


def _git_state() -> dict:
    def command(*args):
        return subprocess.check_output(
            list(args), cwd=PROJECT_ROOT, text=True
        ).strip()

    return {
        "branch": command("git", "branch", "--show-current"),
        "head": command("git", "rev-parse", "HEAD"),
        "worktree_porcelain": command("git", "status", "--porcelain"),
        "local_vs_origin": command(
            "git", "rev-list", "--left-right", "--count",
            "HEAD...origin/v2-protocol",
        ),
        "scope": (
            "state at Gate-1 generation; final pushed state is recorded in "
            "the collaboration implementation manifest"
        ),
    }


def _validated_validation() -> dict:
    validation = e10.read_json_mapping(e10.VALIDATION_PATH)
    e10.assert_current_binding(validation, "Phase-0 validation")
    if validation.get("schema_version") != 1 \
            or validation.get("record_type") != "e10_phase0_validation" \
            or validation.get("task_id") != e10.TASK_ID \
            or validation.get("status") != "PASS" \
            or validation.get("NON_SCIENTIFIC") is not True:
        raise AssertionError("Gate 1 requires a bound PASS validation record")
    commands = validation.get("commands")
    expected_tails = (
        ["-B", "tests/test_e10.py"],
        ["-B", "tests/run_all.py"],
        ["-B", "-m", "experiments.e10_capacity_360m.run", "core-order"],
        ["-B", "-m", "experiments.e10_capacity_360m.run",
         "core-cell", "B4", "train_40k", "0"],
    )
    if not isinstance(commands, list) or len(commands) != len(expected_tails):
        raise AssertionError("Phase-0 validation command evidence is incomplete")
    for index, (command, expected_tail) in enumerate(zip(commands, expected_tails)):
        if command.get("command", [])[1:] != expected_tail \
                or command.get("passed") is not True:
            raise AssertionError(f"Phase-0 validation command {index} mismatch")
    checks = validation.get("checks", {})
    required_true = (
        "model_live_reverified", "shared_calibration_complete",
        "calibration_authorization_revoked", "scientific_authorization_unset",
        "scientific_budget_constants_unset", "tracked_frozen_diff_clean",
    )
    if any(checks.get(name) is not True for name in required_true):
        raise AssertionError("Phase-0 validation hard checks are incomplete")
    if checks.get("requirements_lock_sha256") != (
        "644a7e9db3cfe8632f2802314ca0296cf07dc5e3a41abf6728f3a7904c8a04be"
    ):
        raise AssertionError("Phase-0 validation dependency-lock hash mismatch")
    if checks.get("source_embargo_scan", {}).get("passed") is not True \
            or checks.get("non_scientific_output_scan", {}).get(
                "checkpoint_files"
            ) != 0:
        raise AssertionError("Phase-0 validation source/output scans mismatch")
    frozen = checks.get("frozen_result_trees", {})
    for name, expected in e10.FROZEN_RESULT_BASELINES.items():
        if frozen.get(name) != {**expected, "matches_preimplementation": True}:
            raise AssertionError(f"Phase-0 frozen-tree proof mismatch for {name}")
    return validation


def validate_governance_records() -> dict:
    specifications = (
        (e10.A5_A8C_DISPOSITION_PATH, "e10_a5_a8c_disposition"),
        (e10.U1_DISCHARGE_PATH, "e10_u1_discharge"),
        (e10.RECIPE_CONTRACT_PATH, "e10_recipe_contract"),
        (e10.FROZEN_BASELINE_PATH, "e10_frozen_artifact_baseline"),
    )
    records = {}
    references = {}
    for path, body_key in specifications:
        record = e10.read_json_mapping(path)
        if set(record) != {"metadata", body_key}:
            raise AssertionError(f"governance root schema mismatch for {path.name}")
        metadata = record["metadata"]
        if not isinstance(metadata, dict) \
                or metadata.get("task_id") != e10.TASK_ID \
                or metadata.get("protocol_family") != e10.PROTOCOL_FAMILY:
            raise AssertionError(f"governance metadata mismatch for {path.name}")
        e10.assert_current_or_precorrection_binding(
            {"binding": metadata.get("binding")},
            f"governance {path.name}",
            path,
        )
        body = record[body_key]
        if not isinstance(body, dict):
            raise AssertionError(f"governance body is malformed for {path.name}")
        records[body_key] = body
        references[body_key] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": e10.sha256_file(path),
        }
    correction = e10.validate_source_correction()
    references["e10_source_correction"] = {
        "path": str(e10.SOURCE_CORRECTION_PATH.relative_to(PROJECT_ROOT)),
        "sha256": e10.sha256_file(e10.SOURCE_CORRECTION_PATH),
        "status": correction["status"],
    }

    disposition = records["e10_a5_a8c_disposition"]
    if disposition.get("status") != "RE_DEFERRED_NOT_CANCELLED" \
            or disposition.get("pre_result") is not True:
        raise AssertionError("A5/A8c disposition status mismatch")
    for name in ("A5", "A8c"):
        item = disposition.get(name, {})
        if item.get("scientifically_owed") is not True \
                or item.get("deferred_behind_e10") is not True \
                or item.get("cancelled") is not False \
                or item.get("superseded") is not False:
            raise AssertionError(f"{name} disposition semantics mismatch")
    if disposition.get("budget_treatment", {}).get("charged_to_e10_usage") is not False:
        raise AssertionError("A5/A8c budget treatment mismatch")

    discharge = records["e10_u1_discharge"]
    if discharge.get("status") != "U1_EXTENSION_INSTANTIATED_IN_E10_PHASE0" \
            or discharge.get("immutable_source") != {
                "path": str(e10.U1_PATH.relative_to(PROJECT_ROOT)),
                "sha256": e10.U1_SHA256,
                "modified": False,
            } \
            or discharge.get("registered_names_retained") != ["B4", "B4r"] \
            or discharge.get("scientific_cells_authorized") != 0:
        raise AssertionError("U1 discharge semantics mismatch")

    recipe = records["e10_recipe_contract"]
    if recipe.get("status") != "FROZEN_FOR_FUTURE_DECISION" \
            or recipe.get("core_arms") != list(e10.CORE_ARMS) \
            or recipe.get("core_scales") != list(e10.CORE_SCALES) \
            or recipe.get("core_seeds") != list(e10.CORE_SEEDS) \
            or recipe.get("future_cells") != len(e10.CORE_CELLS) \
            or recipe.get("scientific_cells_authorized") != 0 \
            or recipe.get("dev_evaluation_epochs") != list(e10.DEV_EVALUATION_EPOCHS) \
            or recipe.get("pair_normalization_keys") != list(e10.PAIR_NORMALIZATION_KEYS) \
            or recipe.get("e10_training_authorized") is not None \
            or len(recipe.get("pairs", {})) != 6:
        raise AssertionError("recipe governance semantics mismatch")

    baseline = records["e10_frozen_artifact_baseline"]
    if baseline.get("status") != "RECORDED_BEFORE_IMPLEMENTATION" \
            or baseline.get("roots") != e10.FROZEN_RESULT_BASELINES \
            or baseline.get("must_match_after_phase0") is not True:
        raise AssertionError("frozen-artifact governance semantics mismatch")
    return references


def _validated_calibration() -> tuple[dict, dict, dict]:
    calibration = e10.read_json_mapping(e10.CALIBRATION_PATH)
    e10.assert_current_binding(calibration, "Phase-0 calibration")
    if calibration.get("schema_version") != 1 \
            or calibration.get("record_type") != "e10_phase0_calibration" \
            or calibration.get("task_id") != e10.TASK_ID \
            or calibration.get("NON_SCIENTIFIC") is not True:
        raise AssertionError("wrong calibration schema/type/task/class")
    allowed_statuses = {
        "COMPLETED_WITHIN_CAP", "CAP_REACHED_PARTIAL_ACCEPTED",
        "HEADROOM_INSUFFICIENT_SAFETY_STOP",
        "INTERRUPTED_AND_RECORDED", "HARD_GATE_HALTED",
        "HARD_GATE_HALTED_CAP_OVERRUN",
    }
    if calibration.get("status") not in allowed_statuses:
        raise AssertionError("unknown calibration outcome status")
    state = e10.assert_shared_state_healthy(require_calibration_complete=True)
    completion = e10._validated_calibration_completion()
    entries = [
        entry for entry in state["spend"]["entries"]
        if entry["context"] == "phase0_calibration"
    ]
    if len(entries) != 1 or len(state["spend"]["entries"]) != 1:
        raise AssertionError("Gate 1 requires exactly one Phase-0 ledger charge")
    entry = entries[0]
    occupancy_ns = calibration.get("gpu_occupancy_ns")
    if not isinstance(occupancy_ns, int) or isinstance(occupancy_ns, bool) \
            or occupancy_ns < 0:
        raise AssertionError("calibration occupancy nanoseconds are invalid")
    charge = calibration.get("resource_charge")
    if not isinstance(charge, dict) \
            or charge.get("entry_id") != entry["entry_id"] \
            or charge.get("gpu_occupancy_ns") != occupancy_ns \
            or entry["gpu_occupancy_ns"] != occupancy_ns \
            or completion["spend_entry_id"] != entry["entry_id"]:
        raise AssertionError("calibration record/ledger/completion do not reconcile")
    expected_outcome = {
        "COMPLETED_WITHIN_CAP": "completed",
        "CAP_REACHED_PARTIAL_ACCEPTED": "cap_reached",
        "HEADROOM_INSUFFICIENT_SAFETY_STOP": "headroom_insufficient",
        "INTERRUPTED_AND_RECORDED": "terminated",
        "HARD_GATE_HALTED": "hard_gate_halted",
        "HARD_GATE_HALTED_CAP_OVERRUN": "cap_overrun",
    }[calibration["status"]]
    if occupancy_ns == 0:
        expected_outcome = "aborted_pre_gpu"
    if entry["outcome"] != expected_outcome or charge.get("outcome") != expected_outcome:
        raise AssertionError("calibration status and charged outcome disagree")
    expected_hours = occupancy_ns / 3_600_000_000_000.0
    if not math.isclose(_finite(charge.get("gpu_hours"), "charged hours"),
                        expected_hours, rel_tol=0.0, abs_tol=1e-15):
        raise AssertionError("calibration charged hours do not match occupancy")
    if calibration.get("shared_claim", {}).get("sha256") != \
            entry["claim_sha256"]:
        raise AssertionError("calibration shared claim does not match the ledger")
    if not e10.calibration_is_revoked():
        raise AssertionError("calibration authorization is not revoked")
    return calibration, state["spend"], completion


def _measured_segment(calibration: dict, name: str) -> dict | None:
    segment = calibration.get("segments", {}).get(name)
    if not isinstance(segment, dict) or segment.get("status") != "MEASURED":
        return None
    return segment


def _validate_calibration_inputs(calibration: dict) -> None:
    setup = calibration.get("segments", {}).get("setup", {})
    if setup.get("input_bindings_reverified_after_load") is not True:
        raise AssertionError("calibration inputs were not reverified after loading")
    bindings = setup.get("input_bindings")
    expected_paths = {
        "train_40k_manifest": e10.V2_DIR / "train_40k.csv",
        "train_250k_manifest": e10.V2_DIR / "train_250k.csv",
        "development_manifest": e10.V2_DIR / "dev.csv",
        "answer_vocabulary": e10.VOCAB_PATH,
        "image_token_store": tokens_data.TOKEN_DIR / "image_tokens.h5",
        "question_token_store": tokens_data.TOKEN_DIR / "question_tokens.h5",
    }
    if not isinstance(bindings, dict) or set(bindings) != set(expected_paths):
        raise AssertionError("calibration input binding set mismatch")
    for name, path in expected_paths.items():
        reference = bindings[name]
        if not isinstance(reference, dict) \
                or set(reference) != {"path", "sha256", "size_bytes"} \
                or reference["path"] != str(path.relative_to(PROJECT_ROOT)) \
                or reference["sha256"] != e10.sha256_file(path) \
                or reference["size_bytes"] != path.stat().st_size:
            raise AssertionError(f"calibration input {name} is stale or malformed")

    invariants = setup.get("loaded_input_invariants", {})
    label_ranges = invariants.get("training_manifest_label_ranges", {})
    if invariants.get("image_tokens", {}).get("dtype") != "float16" \
            or invariants.get("image_tokens", {}).get("shape", [None])[-2:] != [50, 512] \
            or invariants.get("question_tokens", {}).get("dtype") != "float16" \
            or invariants.get("question_tokens", {}).get("shape", [None])[-1:] != [512] \
            or invariants.get("image_ids_unique") is not True \
            or invariants.get("question_ids_and_spans_valid") is not True \
            or invariants.get("development_rows") != DEV_ROWS \
            or invariants.get("development_label_columns_loaded") != 0:
        raise AssertionError("calibration loaded-input invariants mismatch")
    for scale, rows in (("train_40k", 40_000), ("train_250k", 250_000)):
        labels = label_ranges.get(scale, {})
        if labels.get("rows") != rows \
                or labels.get("within_answer_vocabulary") is not True \
                or labels.get("answers_match_vocabulary_labels") is not True:
            raise AssertionError(f"calibration {scale} label invariants mismatch")


def _validate_memory_record(memory: dict, context: str) -> None:
    required = {
        "peak_allocated_bytes", "peak_reserved_bytes", "device_total_bytes",
        "reserved_fraction", "ceiling_fraction", "fires", "gate_basis",
    }
    if not isinstance(memory, dict) or set(memory) != required:
        raise AssertionError(f"{context} memory schema mismatch")
    allocated = _finite(memory["peak_allocated_bytes"], f"{context} allocated")
    reserved = _finite(memory["peak_reserved_bytes"], f"{context} reserved")
    total = _finite(memory["device_total_bytes"], f"{context} total")
    fraction = _finite(memory["reserved_fraction"], f"{context} fraction")
    if total <= 0 or allocated > reserved \
            or not math.isclose(fraction, reserved / total, rel_tol=0.0, abs_tol=1e-15) \
            or memory["ceiling_fraction"] != 0.80 \
            or memory["fires"] is not (fraction >= 0.80):
        raise AssertionError(f"{context} memory arithmetic mismatch")


def _validate_calibration_segments(calibration: dict) -> None:
    segments = calibration.get("segments")
    if not isinstance(segments, dict) \
            or set(segments) != {"setup", "S1", "S2", "S3", "S4"}:
        raise AssertionError("calibration segment set mismatch")
    setup = segments["setup"]
    if setup.get("training_lm_parameter_dtype") != "torch.float32" \
            or setup.get("answer_cache_matches_verification") is not True:
        raise AssertionError("calibration setup precision/cache mismatch")
    if setup.get("environment") != {
        **e10.E8B_CALIBRATION_ENVIRONMENT,
        "matches_e8b_calibration": True,
    }:
        raise AssertionError("calibration environment does not match E8B")
    exclusivity = setup.get("gpu_exclusivity_preflight", {})
    if exclusivity.get("exit_status") != 0 \
            or exclusivity.get("active_compute_processes") != 0 \
            or exclusivity.get("exclusive_at_preflight") is not True:
        raise AssertionError("calibration GPU exclusivity evidence is invalid")
    promotion = setup.get("fp32_lm_promotion", {})
    precision = promotion.get("precision", {})
    if promotion.get("source_dtype") != "torch.bfloat16" \
            or promotion.get("destination_dtype") != "torch.float32" \
            or promotion.get("position") != (
                "before S1/S2, matching the inherited E8B path"
            ) \
            or precision.get("cuda_matmul_allow_tf32") is not False \
            or precision.get("cudnn_allow_tf32") is not False \
            or precision.get("float32_matmul_precision") != "highest":
        raise AssertionError("training FP32-promotion/precision contract mismatch")
    _finite(promotion.get("cpu_seconds"), "FP32 LM promotion")
    _validate_memory_record(
        setup.get("fp32_lm_residency_memory"), "FP32 LM residency"
    )
    if calibration.get("memory", {}).get("fp32_lm_residency") != \
            setup.get("fp32_lm_residency_memory"):
        raise AssertionError("FP32 LM residency memory copies disagree")

    for name, scale, warmup, measured in (
        ("S1", "train_40k", 20, 25),
        ("S2", "train_250k", 20, 60),
    ):
        segment = _measured_segment(calibration, name)
        if segment is None:
            continue
        steps = warmup + measured
        if segment.get("scale") != scale \
                or segment.get("warmup_steps") != warmup \
                or segment.get("measured_steps") != measured \
                or segment.get("optimizer_steps") != steps \
                or segment.get("scheduler_steps") != steps \
                or segment.get("scheduler_horizon_epochs") != (
                    e10.SCHEDULER_HORIZON_EPOCHS
                ) \
                or segment.get("scheduler_horizon_steps") != (
                    e10.SCHEDULER_HORIZON_EPOCHS
                    * e10.STEPS_PER_EPOCH[scale]
                ) \
                or segment.get("steps_per_epoch") != e10.STEPS_PER_EPOCH[scale] \
                or segment.get("actual_loader_batches") != e10.STEPS_PER_EPOCH[scale] \
                or segment.get("epoch_completed") is not False:
            raise AssertionError(f"{name} fixed bounds mismatch")
        recipe = e10.build_recipe("B4r", scale, 0)
        if segment.get("recipe_sha256") != e10.recipe_sha256(recipe) \
                or segment.get("pairing_normalized_recipe_sha256") != (
                    e10.paired_recipe_sha256(recipe)
                ):
            raise AssertionError(f"{name} recipe binding mismatch")
        _finite(segment.get("timing", {}).get("mean_seconds"), f"{name} mean")
        _validate_memory_record(segment.get("memory"), f"{name} training")

    s3 = _measured_segment(calibration, "S3")
    if s3 is not None:
        canonical_precision = setup.get("canonical_fp32_precision_reimposed", {})
        if canonical_precision.get("cuda_matmul_allow_tf32") is not False \
                or canonical_precision.get("cudnn_allow_tf32") is not False \
                or canonical_precision.get("float32_matmul_precision") != "highest":
            raise AssertionError("S3 fp32 precision contract mismatch")
        if s3.get("source_rows") != DEV_ROWS \
                or s3.get("selected_rows") != 1024 \
                or s3.get("selection_rng") != "numpy.default_rng(20260810)" \
                or s3.get("selection_sha256") != (
                    "75c51e94e8dd161922d9b3b4946f7f456bd24318ed909952b5583bfb8132f533"
                ) \
                or s3.get("label_columns_loaded") != 0 \
                or s3.get("warmup_batches_discarded") != 2 \
                or s3.get("measured_batches") != 6 \
                or s3.get("measured_rows") != 768 \
                or s3.get("model_outputs_retained") is not False:
            raise AssertionError("S3 fixed label-free bounds mismatch")
        raw = _finite(
            s3.get("full_dev_raw_projection_seconds"), "S3 raw projection"
        )
        allowanced = _finite(
            s3.get("full_dev_allowanced_projection_seconds"), "S3 allowance"
        )
        if s3.get("allowance_multiplier") != 1.066 \
                or not math.isclose(
                    allowanced, raw * 1.066, rel_tol=0.0, abs_tol=1e-12
                ):
            raise AssertionError("S3 allowance arithmetic mismatch")
        _validate_memory_record(s3.get("memory"), "S3 evaluation")

    s4 = _measured_segment(calibration, "S4")
    if s4 is not None:
        rows = s4.get("per_row")
        if s4.get("rows_requested") != 6 or s4.get("rows_completed") != 6 \
                or s4.get("all_r1_argmax_equal") is not True \
                or s4.get("all_r2_outputs_equal") is not True \
                or s4.get("all_r3_loops_bounded") is not True \
                or s4.get("model_outputs_retained") is not False \
                or not isinstance(rows, list) or len(rows) != 6:
            raise AssertionError("S4 fixed bounded sample mismatch")
        for row in rows:
            if row.get("r1_argmax_equal") is not True \
                    or row.get("r2_output_equal") is not True \
                    or row.get("r3_within_fixed_cap") is not True \
                    or row.get("r3_terminated_or_reached_cap") is not True:
                raise AssertionError("S4 row correctness/bound mismatch")
        _validate_memory_record(s4.get("memory"), "S4 G14")

    bounded = calibration.get("bounded", {})
    if bounded.get("complete_epochs") != 0 \
            or bounded.get("complete_scientific_cells") != 0 \
            or bounded.get("checkpoint_files") != 0 \
            or bounded.get("performance_values_emitted") != 0 \
            or bounded.get("model_outputs_retained") is not False \
            or bounded.get("hyperparameter_selection") is not False:
        raise AssertionError("calibration bounded-execution contract mismatch")


def _g14_projection(s3: dict | None, s4: dict | None) -> dict:
    if s4 is None or s4.get("rows_completed") != 6:
        return {
            "status": "NOT_MEASURED_CAP_OR_HALT",
            "measured_sample_rows": 0 if s4 is None else s4.get("rows_completed", 0),
            "future_per_cell_seconds": None,
            "reason": "the six-row S4 gate did not complete; zero was not substituted",
        }
    booleans = (
        s4.get("all_r1_argmax_equal") is True,
        s4.get("all_r2_outputs_equal") is True,
        s4.get("all_r3_loops_bounded") is True,
    )
    if not all(booleans):
        raise AssertionError("S4 completed without every correctness clause passing")
    rows = s4.get("per_row")
    if not isinstance(rows, list) or len(rows) != 6:
        raise AssertionError("S4 per-row timing evidence is incomplete")
    reference_per_row = []
    r3_per_row = []
    for index, row in enumerate(rows):
        timing = row.get("timing_seconds", {})
        reference_per_row.append(sum(
            _finite(timing.get(key), f"S4 row {index} {key}")
            for key in (
                "r1_reference", "r1_cached", "r2_reference", "r2_cached",
            )
        ))
        r3_per_row.append(_finite(timing.get("r3_bounded"), f"S4 row {index} R3"))
    if s3 is None:
        return {
            "status": "S4_MEASURED_S3_MISSING",
            "measured_sample_rows": 6,
            "future_per_cell_seconds": None,
            "reason": "the canonical batched component cannot be inferred without S3",
        }
    full_dev_seconds = _finite(
        s3.get("full_dev_allowanced_projection_seconds"), "allowanced dev timing"
    )
    canonical_component = full_dev_seconds * G14_MAX_ROWS_PER_CELL / DEV_ROWS
    reference_component = (
        sum(reference_per_row) / len(reference_per_row) * G14_MAX_ROWS_PER_CELL
    )
    return {
        "status": "INFERRED_FROM_MEASURED_S3_S4",
        "measured_sample_rows": 6,
        "measured_sample_total_seconds": _finite(
            s4.get("total_seconds"), "S4 total"
        ),
        "measured_r3_mean_seconds": sum(r3_per_row) / len(r3_per_row),
        "future_schedule": {
            "pre_selection_rows": G14_PRE_ROWS,
            "post_selection_max_union_rows": G14_POST_MAX_ROWS,
            "maximum_rows_per_cell": G14_MAX_ROWS_PER_CELL,
            "invocations_per_cell": 2,
        },
        "canonical_batched_component_seconds": canonical_component,
        "r1_r2_reference_component_seconds": reference_component,
        "future_per_cell_seconds": canonical_component + reference_component,
        "qualification": (
            "inferred upper-bound proxy: inherited P64 pre-selection plus at most "
            "P64/L64/O32 post-selection; S4 R3 is correctness evidence and is not "
            "charged into the inherited R1/R2 G14 schedule"
        ),
    }


def _per_cell(scale: str, mean_step_seconds: float,
              full_dev_seconds: float, g14_seconds: float,
              fixed_overhead_seconds: float) -> dict:
    training = mean_step_seconds * e10.STEPS_PER_EPOCH[scale] * 22
    evaluations = full_dev_seconds * len(e10.DEV_EVALUATION_EPOCHS)
    intervention = full_dev_seconds
    components = {
        "training_seconds": training,
        "pinned_12_development_evaluations_seconds": evaluations,
        "g14_seconds_inferred": g14_seconds,
        "other_fixed_overhead_seconds": fixed_overhead_seconds,
        "final_r1_intervention_seconds": intervention,
    }
    expected_seconds = sum(components.values())
    return {
        "scale": scale,
        "components": components,
        "expected_seconds": expected_seconds,
        "expected_hours": expected_seconds / 3600.0,
        "measured_times_1_25_stress_hours": (
            expected_seconds * STRESS_MULTIPLIER / 3600.0
        ),
        "flat_3x_context_hours": (
            expected_seconds * FLAT_STRESS_MULTIPLIER / 3600.0
        ),
        "evidence_class": "INFERRED_FROM_PHASE0_MEASUREMENTS",
        "future_scientific_execution_authorized": False,
    }


def _storage_projection() -> dict:
    scratch = shutil.disk_usage(PROJECT_ROOT)
    shared = shutil.disk_usage(e10.SHARED_STATE_DIR)
    snapshot_bytes = _tree_bytes(e10.model_snapshot_dir())
    phase0_records_bytes = _tree_bytes(e10.OUT_DIR)
    ratio = e10.EXPECTED_TRAINABLE_PARAMETERS / E8B_TRAINABLE_PARAMETERS
    central_per_cell = int(math.ceil(
        (E8B_RESUME_MIB + 2 * E8B_CANONICAL_MIB) * MIB * ratio
    ))
    central_twelve_cells = central_per_cell * len(e10.CORE_CELLS)
    conservative_core = (
        FUTURE_CELL_STORAGE_RESERVE_BYTES * len(e10.CORE_CELLS)
    )
    future_additional = conservative_core + FUTURE_RETRY_STORAGE_RESERVE_BYTES
    total_retained = snapshot_bytes + phase0_records_bytes + future_additional
    shared_ledger_allowance = 1 * MIB
    physical_fit = (
        future_additional <= scratch.free
        and shared_ledger_allowance <= shared.free
    )
    return {
        "phase0_model_snapshot_bytes_actual": snapshot_bytes,
        "phase0_result_records_bytes_actual": phase0_records_bytes,
        "central_checkpoint_estimate": {
            "E8B_resume_mib": E8B_RESUME_MIB,
            "E8B_each_canonical_or_secondary_mib": E8B_CANONICAL_MIB,
            "E10_over_E8B_trainable_ratio": ratio,
            "per_cell_bytes": central_per_cell,
            "twelve_cells_bytes": central_twelve_cells,
        },
        "conservative_future_core_reservation_bytes": conservative_core,
        "one_retained_retry_per_identity_reservation_bytes": (
            FUTURE_RETRY_STORAGE_RESERVE_BYTES
        ),
        "future_additional_scratch_bytes_required": future_additional,
        "projected_total_retained_e10_bytes": total_retained,
        "scratch_free_bytes": scratch.free,
        "shared_state_free_bytes": shared.free,
        "shared_ledger_allowance_bytes": shared_ledger_allowance,
        "physical_fit": physical_fit,
        "formal_project_allocation": "no established project allocation",
        "qualification": (
            "free filesystem space is a physical-fit observation, not evidence "
            "of an approved project storage allocation"
        ),
    }


def build_gate1_record() -> dict:
    governance_references = validate_governance_records()
    validation = _validated_validation()
    verification = e10.validate_model_verification()
    calibration, spend_ledger, completion = _validated_calibration()
    _validate_calibration_inputs(calibration)
    _validate_calibration_segments(calibration)
    e10.assert_no_scientific_outputs()

    s1 = _measured_segment(calibration, "S1")
    s2 = _measured_segment(calibration, "S2")
    s3 = _measured_segment(calibration, "S3")
    s4 = _measured_segment(calibration, "S4")
    g14 = _g14_projection(s3, s4)
    storage = _storage_projection()

    e8b_record = e10.read_json_mapping(E8B_CALIBRATION_PATH)
    e8b_body = e8b_record["e8b_throughput_calibration"]
    observed_e8b = {
        "train_40k": e8b_body["empirical_equivalence"]["B2"]["mean_s_per_step"],
        "train_250k": e8b_body["calibration_250k"]["mean_s_per_step"],
    }
    if observed_e8b != E8B_STEP_SECONDS:
        raise AssertionError(f"authoritative E8B timing changed: {observed_e8b}")

    cap_pass = (
        calibration.get("accounting_cap_respected") is True
        and calibration.get("gpu_occupancy_ns", math.inf)
        <= int(config.E10_CALIBRATION_BUDGET_HOURS * 3_600_000_000_000)
    )
    calibration_outcome_pass = calibration.get("status") in {
        "COMPLETED_WITHIN_CAP", "CAP_REACHED_PARTIAL_ACCEPTED",
    }
    lm_residency_memory_pass = (
        calibration.get("memory", {}).get(
            "fp32_lm_residency", {}
        ).get("fires") is False
    )
    g14_memory_pass = (
        s4 is not None and s4.get("memory", {}).get("fires") is False
    )

    training_memory_pass = (
        s1 is not None and s2 is not None
        and s1.get("memory", {}).get("fires") is False
        and s2.get("memory", {}).get("fires") is False
    )
    evaluation_memory_pass = (
        s3 is not None and s3.get("memory", {}).get("fires") is False
    )
    g14_pass = (
        s4 is not None
        and s4.get("rows_completed") == 6
        and s4.get("all_r1_argmax_equal") is True
        and s4.get("all_r2_outputs_equal") is True
        and s4.get("all_r3_loops_bounded") is True
    )
    complete_projection = (
        s1 is not None and s2 is not None and s3 is not None
        and g14.get("future_per_cell_seconds") is not None
    )
    hard_gates = {
        "implementation_and_tests": "PASS",
        "model_identity_and_hashes": (
            "PASS" if verification.get("all_hard_checks_passed") is True else "FAIL"
        ),
        "tokenizer_byte_identity": (
            "PASS" if verification.get("tokenizer_comparison", {}).get(
                "no_answer_tokenisation_confound"
            ) is True else "FAIL"
        ),
        "shared_ledger_reconciled": "PASS",
        "calibration_outcome_acceptable": (
            "PASS" if calibration_outcome_pass else "FAIL"
        ),
        "calibration_cap_enforced": "PASS" if cap_pass else "FAIL",
        "training_reserved_memory_below_0_80": (
            "PASS" if training_memory_pass else "FAIL"
        ),
        "evaluation_reserved_memory_below_0_80": (
            "PASS" if evaluation_memory_pass else "FAIL"
        ),
        "fp32_lm_residency_reserved_memory_below_0_80": (
            "PASS" if lm_residency_memory_pass else "FAIL"
        ),
        "g14_reserved_memory_below_0_80": (
            "PASS" if g14_memory_pass else "FAIL"
        ),
        "g14_reference_equality": "PASS" if g14_pass else "FAIL",
        "resource_projection_complete": "PASS" if complete_projection else "FAIL",
        "zero_complete_epochs": (
            "PASS" if calibration.get("bounded", {}).get("complete_epochs") == 0
            else "FAIL"
        ),
        "zero_checkpoints": (
            "PASS" if calibration.get("bounded", {}).get("checkpoint_files") == 0
            else "FAIL"
        ),
        "calibration_authorization_revoked": (
            "PASS" if e10.calibration_is_revoked() else "FAIL"
        ),
        "scientific_authorization_unset": (
            "PASS" if e10.E10_TRAINING_AUTHORIZED is None else "FAIL"
        ),
        "scientific_budget_constants_unset": (
            "PASS" if config.E10_PER_IDENTITY_CEILING_HOURS is None
            and config.E10_PER_CELL_WALL_CLOCK_HOURS is None else "FAIL"
        ),
        "storage_physical_fit": "PASS" if storage["physical_fit"] else "FAIL",
    }

    step_seconds = {}
    ratios = {}
    cells = {}
    walls = {}
    identities = {}
    expected_total = None
    stress_total = None
    wall_recommendation = None
    recommended_identity_ceiling = None
    fixed_overhead_basis = None
    if complete_projection:
        step_seconds = {
            "train_40k": _finite(s1["timing"]["mean_seconds"], "S1 mean"),
            "train_250k": _finite(s2["timing"]["mean_seconds"], "S2 mean"),
        }
        ratios = {
            scale: step_seconds[scale] / E8B_STEP_SECONDS[scale]
            for scale in e10.CORE_SCALES
        }
        full_dev = _finite(
            s3["full_dev_allowanced_projection_seconds"], "full dev allowance"
        )
        g14_seconds = _finite(g14["future_per_cell_seconds"], "G14 projection")
        setup = calibration["segments"]["setup"]
        fixed_overhead_basis = {
            "cpu_setup_seconds": _finite(
                setup["cpu_seconds_before_gpu_occupancy"], "CPU setup"
            ),
            "fp32_lm_promotion_cpu_seconds_included_in_cpu_setup": _finite(
                setup["fp32_lm_promotion"]["cpu_seconds"], "FP32 LM promotion"
            ),
            "gpu_transfer_seconds": _finite(
                setup["gpu_transfer_seconds"], "GPU transfer"
            ),
            "gpu_cleanup_seconds": _finite(
                setup["gpu_cleanup_seconds"], "GPU cleanup"
            ),
        }
        shared_fixed_overhead = sum(
            fixed_overhead_basis[key]
            for key in (
                "cpu_setup_seconds", "gpu_transfer_seconds",
                "gpu_cleanup_seconds",
            )
        )
        cells = {
            scale: _per_cell(
                scale, step_seconds[scale], full_dev, g14_seconds,
                shared_fixed_overhead + _finite(
                    segment.get("segment_setup_seconds", 0.0),
                    f"{scale} setup",
                ),
            )
            for scale, segment in (("train_40k", s1), ("train_250k", s2))
        }
        wall_recommendation = (
            cells["train_250k"]["expected_hours"]
            * WALL_RECOMMENDATION_MULTIPLIER
        )
        for wall in WALL_CANDIDATES_HOURS:
            walls[str(int(wall))] = {
                "fits_expected": cells["train_250k"]["expected_hours"] <= wall,
                "fits_1_25_stress": (
                    cells["train_250k"]["measured_times_1_25_stress_hours"] <= wall
                ),
                "expected_margin_hours": (
                    wall - cells["train_250k"]["expected_hours"]
                ),
                "stress_margin_hours": (
                    wall - cells["train_250k"]["measured_times_1_25_stress_hours"]
                ),
            }
        calibration_hours = (
            calibration["gpu_occupancy_ns"] / 3_600_000_000_000.0
        )
        per_identity_expected = 3 * sum(
            cells[scale]["expected_hours"] for scale in e10.CORE_SCALES
        )
        per_identity_stress = 3 * sum(
            cells[scale]["measured_times_1_25_stress_hours"]
            for scale in e10.CORE_SCALES
        )
        identities = {
            "pretrained_smollm2_360m": {
                "projected_e10_hours": per_identity_expected,
                "stress_hours": per_identity_stress,
                "phase0_hours_already_charged": 0.0,
            },
            "random_smollm2_360m": {
                "projected_e10_hours": per_identity_expected + calibration_hours,
                "stress_hours": per_identity_stress + calibration_hours,
                "phase0_hours_already_charged": calibration_hours,
            },
        }
        largest_cell_stress = max(
            cell["measured_times_1_25_stress_hours"] for cell in cells.values()
        )
        recommended_identity_ceiling = math.ceil(
            max(identity["stress_hours"] for identity in identities.values())
            + largest_cell_stress + config.E10_HEADROOM_FLOOR_HOURS
        )
        for identity in identities.values():
            identity["headroom_floor_hours"] = config.E10_HEADROOM_FLOOR_HOURS
            identity["largest_cell_forced_retry_stress_hours"] = largest_cell_stress
            identity["margin_after_retry_and_floor_hours"] = (
                recommended_identity_ceiling - identity["stress_hours"]
                - largest_cell_stress - config.E10_HEADROOM_FLOOR_HOURS
            )
            identity["one_largest_cell_forced_retry_fits_recommendation"] = (
                identity["margin_after_retry_and_floor_hours"] >= 0
            )
        expected_total = sum(
            identity["projected_e10_hours"] for identity in identities.values()
        )
        stress_total = sum(
            identity["stress_hours"] for identity in identities.values()
        )

    any_failure = "FAIL" in hard_gates.values()
    status = (
        "GATE1_BLOCKED_INCOMPLETE_OR_FAILED_PHASE0"
        if any_failure
        else "GATE1_READY_FOR_USER_DECISION_WITH_UNQUANTIFIED_A5_A8C_RESERVE"
    )
    return {
        "schema_version": 1,
        "record_type": "e10_gate1_projection",
        "task_id": e10.TASK_ID,
        "status": status,
        "NON_SCIENTIFIC": True,
        "utc": e10.utc_now(),
        "git_commit": e10.current_git_commit(),
        "binding": e10.binding_record(),
        "inputs": {
            "governance": governance_references,
            "model_verification": {
                "path": str(e10.MODEL_VERIFICATION_PATH.relative_to(PROJECT_ROOT)),
                "sha256": e10.sha256_file(e10.MODEL_VERIFICATION_PATH),
            },
            "calibration": {
                "path": str(e10.CALIBRATION_PATH.relative_to(PROJECT_ROOT)),
                "sha256": e10.sha256_file(e10.CALIBRATION_PATH),
            },
            "validation": {
                "path": str(e10.VALIDATION_PATH.relative_to(PROJECT_ROOT)),
                "sha256": e10.sha256_file(e10.VALIDATION_PATH),
            },
            "shared_spend_ledger": {
                "path": str(e10.SPEND_LEDGER),
                "sha256": e10.sha256_file(e10.SPEND_LEDGER),
                "phase0_entry_id": completion["spend_entry_id"],
                "entries": len(spend_ledger["entries"]),
            },
            "shared_calibration_completion": {
                "path": str(e10.CALIBRATION_COMPLETION_PATH),
                "sha256": e10.sha256_file(e10.CALIBRATION_COMPLETION_PATH),
            },
            "e8b_calibration": {
                "path": str(E8B_CALIBRATION_PATH.relative_to(PROJECT_ROOT)),
                "sha256": e10.sha256_file(E8B_CALIBRATION_PATH),
            },
        },
        "implementation_status": {
            "phase0_modules": "implemented",
            "final_test_status": validation["status"],
            "scientific_cells_executed": 0,
        },
        "model_identity": {
            "repo_id": e10.MODEL_REPO,
            "revision": e10.MODEL_REVISION,
            "weight_sha256": e10.MODEL_WEIGHT_SHA256,
            "weight_bytes": e10.MODEL_WEIGHT_BYTES,
            "parameter_count": verification["loaded_model"]["parameter_count"],
            "verified_config": verification["files"]["config"],
            "verified_blob_files": verification["files"]["blob_files"],
            "tokenizer_artifacts_vs_135m": verification["files"][
                "tokenizer_artifacts_vs_135m"
            ],
            "v2_vocabulary": verification["tokenizer_comparison"][
                "v2_vocabulary"
            ],
            "frozen_e8b_answer_contract": verification[
                "tokenizer_comparison"
            ]["frozen_e8b_answer_contract"],
            "tokenizer_has_no_cross_size_confound": True,
        },
        "a5_a8c_disposition": {
            "status": "scientifically owed and re-deferred behind E10",
            "charged_to_actual_e10_usage": False,
        },
        "throughput": {
            "E8B_reference_seconds_per_step": E8B_STEP_SECONDS,
            "E10_measured_seconds_per_step": step_seconds or None,
            "E10_over_E8B_ratio": ratios or None,
        },
        "canonical_fp32_development_timing": (
            None if s3 is None else {
                "raw_full_dev_projection_seconds": s3[
                    "full_dev_raw_projection_seconds"
                ],
                "allowanced_full_dev_projection_seconds": s3[
                    "full_dev_allowanced_projection_seconds"
                ],
                "allowance_multiplier": s3["allowance_multiplier"],
            }
        ),
        "g14_cost": g14,
        "memory": {
            "training_40k": None if s1 is None else s1["memory"],
            "training_250k": None if s2 is None else s2["memory"],
            "canonical_fp32_evaluation": None if s3 is None else s3["memory"],
            "fp32_lm_residency": calibration.get("memory", {}).get(
                "fp32_lm_residency"
            ),
            "g14_sample": None if s4 is None else s4["memory"],
        },
        "per_cell_fixed_overhead_basis": fixed_overhead_basis,
        "per_cell_projections": cells or None,
        "recommended_per_cell_wall_hours": wall_recommendation,
        "candidate_wall_analysis": walls or None,
        "identity_budgets": identities or None,
        "recommended_future_per_identity_ceiling_hours": (
            recommended_identity_ceiling
        ),
        "recommended_ceiling_scope": (
            "conditional E10-only recommendation; excludes the unquantified "
            "future A5/A8c reserve and is not written to config"
        ),
        "view_1_e10_only": {
            "expected_total_gpu_hours": expected_total,
            "stress_total_gpu_hours": stress_total,
            "includes_phase0_calibration_charge": True,
        },
        "view_2_e10_plus_reserved_a5_a8c": {
            "e10_expected_gpu_hours": expected_total,
            "e10_stress_gpu_hours": stress_total,
            "reserved_a5_a8c_gpu_hours": None,
            "combined_total_gpu_hours": None,
            "decision_status": "OPEN_UNQUANTIFIED_RESERVE",
            "reason": (
                "no authoritative tracked numeric A5/A8c projection exists; "
                "the reserve is explicit but not fabricated from E10"
            ),
        },
        "storage": storage,
        "hard_gates": hard_gates,
        "resource_decision": {
            "recommendations_only": True,
            "config_identity_ceiling_remains": config.E10_PER_IDENTITY_CEILING_HOURS,
            "config_cell_wall_remains": config.E10_PER_CELL_WALL_CLOCK_HOURS,
            "e10_training_authorized_remains": e10.E10_TRAINING_AUTHORIZED,
            "open_decisions": [
                "user/Claude decision on E10 scientific resource constants",
                "authoritative numeric reserve for future A5/A8c work",
                "formal project storage allocation remains unestablished",
            ],
            "next_action": "review Gate 1; do not start scientific cell 1",
        },
        "clean_test_accessed": False,
        "repository_state_at_generation": _git_state(),
        "final_repository_state_evidence": {
            "path": (
                ".agent-bridge/tasks/e10-phase0-calibration/"
                "30-implementation.json"
            ),
            "timing": "written after the final non-force push",
            "qualification": (
                "an artefact committed inside a Git tree cannot contain its "
                "own final commit identity without a circular self-reference"
            ),
        },
    }


def write_gate1() -> dict:
    record = build_gate1_record()
    digest = e10.atomic_write_json(e10.GATE1_PATH, record)
    return {
        "path": str(e10.GATE1_PATH),
        "sha256": digest,
        "status": record["status"],
    }


def main() -> int:
    result = write_gate1()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
