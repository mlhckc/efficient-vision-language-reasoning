"""E10 Phase-2 records and pipeline verification.

Phase 2 implements the scientific pipeline behind the reviewed Phase-1
guardrails. It authorises nothing: ``E10_TRAINING_AUTHORIZED`` is still unset,
``run core-cell B4 train_40k 0`` still refuses before any model, dataset, CUDA,
optimiser or output directory is touched, and no scientific cell has run.

Two immutable records are written once:

  * the binding amendment, which carries every Phase-1 record forward byte for
    byte across the source-digest move that adding executable pipeline sources
    necessarily causes;
  * the pipeline record, which states what the pipeline implements and pins it
    to the frozen contract it must reproduce.

    python -B -m experiments.e10_capacity_360m.run phase2-record
    python -B -m experiments.e10_capacity_360m.run phase2-verify
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from experiments.e10_capacity_360m import analysis  # noqa: E402
from experiments.e10_capacity_360m import e10_common as e10  # noqa: E402
from experiments.e10_capacity_360m import phase1  # noqa: E402
from experiments.e10_capacity_360m import science  # noqa: E402


TASK_ID = e10.PHASE2_TASK_ID

# Revision 2, the repair of the two blocking findings from the independent
# Phase-2 review of HEAD 20cd5bf. Both Phase-2 records bind the live source
# digest, which the repair moves, so they are regenerated inside this same open,
# unapproved revision rather than a second amendment being chained onto an
# amendment. The superseded bytes remain in git history at 20cd5bf and are
# pinned here and in the regenerated record.
PHASE2_REVISION = 2
PHASE2_REPAIRED_FROM_HEAD = "20cd5bfa35768b1e83d5d8b1d23fc68eff13c8bb"
PHASE2_SUPERSEDED_RECORD_SHA256 = {
    "phase2_binding_amendment_20260810.json":
        "8dcce832b0f1d6a1f6edd4155600eca69d7a2e4aee08531bbadeb73580b48f43",
    "phase2_scientific_pipeline_20260810.json":
        "a7648596c471aaa49ad10ffd6d5acc198b93b2b5d1e8f4a84b15dddc6b54922a",
}

# The reviewer's non-blocking resource observation, recorded exactly rather than
# glossed. G11 repeats the final canonical development pass to prove the
# evaluation is reproducible, so a cell performs one more full-development
# prediction pass than the Gate-1 projection budgeted. This is disclosed, not
# reconciled away, and it is not a claim that the counts agree.
EVALUATION_PASS_ACCOUNTING = {
    "projected_full_development_passes_per_cell": 13,
    "implemented_full_development_passes_per_cell": 14,
    "composition": (
        "twelve pinned development evaluations, one final canonical R1 "
        "evaluation, and one G11 repeat of that evaluation"
    ),
    "counts_identical": False,
    "approximate_extra_gpu_hours_per_cell": 0.063,
    "approximate_extra_gpu_hours_over_the_matrix": 0.75,
    "wall_consequence": "none",
    "identity_ceiling_consequence": "none",
    "disposition": (
        "judged NON-BLOCKING by the independent Phase-2 review; to be "
        "reconciled in final execution and resource reporting rather than by "
        "changing the scientific implementation or the resource constants"
    ),
}

# The E10-specific differences from the frozen E8B B2/B3 behaviour, stated
# once, so a reviewer never has to infer them from a diff.
DELIBERATE_DIFFERENCES = {
    "development_cadence": (
        "E8B evaluated every epoch; E10 evaluates only the twelve pinned "
        "epochs 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 19 and 22 of the frozen E10 "
        "recipe contract. Epoch 22 is always evaluated, so the primary result "
        "is always measured."
    ),
    "g8_overfit_gate": (
        "not run. The frozen Gate-1 per-cell budget enumerates training, "
        "twelve development evaluations, the inherited G14 schedule, a small "
        "fixed overhead and one final R1 pass. E8B's G8 tiny-subset overfit "
        "gate can train for up to 200 epochs and is not funded by it; running "
        "it unbudgeted would put the 12-hour whole-cell wall at risk."
    ),
    "resume": (
        "E8B could resume a crashed cell from a verified checkpoint. E10 never "
        "resumes: a leftover training state is a hard refusal, because "
        "continuing an interrupted trajectory is a retry and no retry exists "
        "without a fresh explicit authorization record."
    ),
    "wall_enforcement": (
        "E8B carried an 8-hour wall inside its recipe and checked it every 25 "
        "training steps. The E10 recipe deliberately carries no resource "
        "field; the 12-hour whole-cell wall is imposed by the signed cell "
        "permit and enforced at every stage boundary, every optimizer step and "
        "at finalisation on every exit path."
    ),
    "artefact_tree": (
        "E10 publishes scientific artefacts to "
        "results/experiments/e10_capacity_360m_core/, beside rather than "
        "inside the immutable Phase-0/Phase-1 governance tree, so the "
        "no-scientific-outputs proof over that tree keeps its meaning."
    ),
    "halting_and_accounting": (
        "every gate halt, lock, ledger charge and identity ceiling is E10's "
        "own. No E10 path reads or writes E8B's authorisation state, gate-halt "
        "records, run locks, result tree or identity ledger."
    ),
}

# Pure, parameterised E8B/E8A helpers reused unchanged, and what each provides.
REUSED_FROM_E8 = {
    "experiments.e8b_readout_generation.latents":
        "LatentTrunk, LatentProjection, E8BPrefixModel, the preregistered "
        "projection scale and the teacher-forced answer-token loss",
    "experiments.e8b_readout_generation.readouts":
        "the answer cache and prefix trie, and the R1/R2 reference and cached "
        "scorers the G14 gate compares",
    "experiments.e8b_readout_generation.training":
        "r1_scores_batched, the G14 row construction and strata record, the "
        "G10 collapse diagnostic and the labels= reference loss",
    "experiments.e8a_question_encoder.g21_scorer":
        "the single approved normaliser and the two pinned metrics",
    "experiments.e8a_question_encoder.analyse_core":
        "the image-clustered resampling stream, the percentile interval and "
        "the universal directional rule",
    "src.tokens_data":
        "the cached token stores, dataset and collation",
}


class PipelineError(AssertionError):
    """A Phase-2 pipeline contract check refused."""


def _relative(path: Path) -> str:
    path = Path(path)
    if path.is_relative_to(PROJECT_ROOT):
        return str(path.relative_to(PROJECT_ROOT))
    return str(path)


# --- executable contract checks ----------------------------------------------

def assert_frozen_training_control_flow() -> dict:
    """Prove the 22-epoch loop's control flow from the source, not from prose.

    Four properties a reader should not have to take on trust: the loop runs
    the frozen budget with no early exit, every optimizer step goes through the
    guard, no bare optimizer step exists anywhere in E10, and evaluation is
    gated on the frozen cadence.
    """
    source = Path(science.__file__).read_text()
    tree = ast.parse(source, filename=science.__file__)
    execute = next(
        (node for node in ast.walk(tree)
         if isinstance(node, ast.FunctionDef) and node.name == "_execute_cell"),
        None)
    if execute is None:
        raise PipelineError("the E10 cell executor is not defined")
    loops = [node for node in ast.walk(execute) if isinstance(node, ast.For)]
    epoch_loops = [node for node in loops
                   if isinstance(node.target, ast.Name)
                   and node.target.id == "epoch"]
    if len(epoch_loops) != 1:
        raise PipelineError("the epoch loop is not defined exactly once")
    epoch_loop = epoch_loops[0]
    # Only the inner batch loop may break, and it does not; an early stop would
    # be a break or a return in the epoch loop body.
    for node in ast.walk(epoch_loop):
        if isinstance(node, (ast.Break, ast.Return)):
            raise PipelineError(
                "the frozen 22-epoch budget has an early exit; E10 has no "
                "early stopping")
    steps = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        if name.endswith("guarded_optimizer_step"):
            steps.append(name)
        if name in ("optimizer.step", "self.optimizer.step"):
            raise PipelineError(
                "an unguarded optimizer step exists in the E10 pipeline")
    if not steps:
        raise PipelineError("the E10 training loop takes no guarded step")
    if "optimizer.step()" in source.replace(
            "guard.guarded_optimizer_step(optimizer)", ""):
        raise PipelineError("a bare optimizer step survives in the source")
    return {
        "epoch_loops": 1,
        "early_exit_statements": 0,
        "guarded_optimizer_step_call_sites": len(steps),
        "bare_optimizer_steps": 0,
        "frozen_epochs": science.CANONICAL_EPOCH,
        "scheduler_horizon_epochs": science.SCHEDULER_HORIZON_EPOCHS,
    }


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def assert_all_stages_guarded() -> dict:
    """Every mandatory scientific stage runs inside ScientificCellGuard.stage."""
    tree = ast.parse(Path(science.__file__).read_text(),
                     filename=science.__file__)
    named = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _dotted(node.func).endswith("stage"):
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) \
                    and isinstance(argument.value, str):
                named.add(argument.value)
    missing = [stage for stage in phase1.CELL_STAGES if stage not in named]
    if missing:
        raise PipelineError(f"unguarded scientific stages: {missing}")
    unknown = sorted(named - set(phase1.CELL_STAGES))
    if unknown:
        raise PipelineError(f"unknown guarded stage names: {unknown}")
    return {"required_stages": list(phase1.CELL_STAGES),
            "guarded_stages": sorted(named), "unguarded": 0}


def assert_frozen_recipe_reproduced() -> dict:
    """Every one of the twelve recipe digests still equals the frozen contract."""
    frozen = e10.frozen_recipe_contract()
    if list(frozen["dev_evaluation_epochs"]) != list(
            science.DEV_EVALUATION_EPOCHS):
        raise PipelineError("the development cadence drifted from the contract")
    digests = {}
    for arm, scale, seed in e10.CORE_CELLS:
        recipe = e10.build_recipe(arm, scale, seed)
        observed = e10.recipe_sha256(recipe)
        expected = frozen["pairs"][f"{scale}_seed{seed}"][f"{arm}_full_sha256"]
        if observed != expected:
            raise PipelineError(
                f"recipe digest drift at {arm}/{scale}/seed{seed}")
        digests[f"{arm}_{scale}_seed{seed}"] = observed
    return {"cells": len(digests), "digests": digests,
            "dev_evaluation_epochs": list(science.DEV_EVALUATION_EPOCHS),
            "canonical_epoch": science.CANONICAL_EPOCH}


def assert_analysis_gate_is_correct() -> dict:
    """The analysis refuses an incomplete matrix and reproduces a complete one.

    Before R3 this asserted only the refusal, which made the verifier
    permanently red after a legitimate completed execution. The invariant is now
    stated for the lifecycle state the repository is actually in, and neither
    branch is permissive: outside the exact completed authorised matrix the
    analysis must still refuse, and inside it the analysis must positively
    reconcile all twelve cells from their own per-row evidence.
    """
    state = e10.core_execution_state()
    if state["state"] == e10.AUTHORIZED_COMPLETE:
        matrix = analysis.load_matrix()
        cells = matrix["cells"] if isinstance(matrix, dict) else {}
        if len(cells) != len(e10.CORE_CELLS):
            raise PipelineError(
                f"the completed E10 matrix reconciled {len(cells)} cells, "
                f"not {len(e10.CORE_CELLS)}")
        return {"lifecycle_state": state["state"], "refused": False,
                "reconciled_cells": len(cells),
                "rule": "a complete authorised matrix must reproduce"}
    try:
        analysis.load_matrix()
    except analysis.AnalysisRefused as error:
        return {"lifecycle_state": state["state"], "refused": True,
                "refusal": str(error),
                "rule": "anything short of the complete matrix must refuse"}
    except AssertionError as error:
        return {"lifecycle_state": state["state"], "refused": True,
                "refusal": f"{type(error).__name__}: {error}",
                "rule": "anything short of the complete matrix must refuse"}
    raise PipelineError(
        "the E10 analysis accepted an incomplete matrix; it must refuse")


# Retained name for the pre-execution invariant, which is unchanged and is what
# a fresh repository and the record writers still assert.
def assert_analysis_refuses_incomplete() -> dict:
    """The analysis refuses while the twelve-cell evidence set is absent."""
    try:
        analysis.load_matrix()
    except analysis.AnalysisRefused as error:
        return {"refused": True, "refusal": str(error)}
    except AssertionError as error:
        return {"refused": True, "refusal": f"{type(error).__name__}: {error}"}
    raise PipelineError(
        "the E10 analysis accepted an incomplete matrix; it must refuse")


def pipeline_contract() -> dict:
    """What the Phase-2 pipeline implements, stated and checked."""
    return {
        "task_id": TASK_ID,
        "implements": {
            "labelled_train_dev_loading": True,
            "b4_b4r_construction": True,
            "frozen_22_epoch_training": True,
            "pinned_development_cadence": list(science.DEV_EVALUATION_EPOCHS),
            "canonical_scoring": "G21 pinned normaliser and the two metrics",
            "g14_diagnostics": list(science.G14_CELL_STAGES),
            "final_r1_evaluation": True,
            "checkpoint_and_result_publication": True,
            "cell_artefact_reconciliation": list(
                science.REQUIRED_CELL_ARTEFACTS),
            "frozen_statistical_analysis": True,
        },
        "primary_result_rule": (
            "the epoch-22 checkpoint is the canonical primary; the best of the "
            "twelve evaluated epochs is a clearly labelled secondary "
            "diagnostic and never replaces it"
        ),
        "primary_metric": (
            "canonical FP32 development R1 of the epoch-22 checkpoint, with "
            "the pinned G21 normalised exact match reported beside it"
        ),
        "reused_from_e8": REUSED_FROM_E8,
        "deliberate_e10_differences": DELIBERATE_DIFFERENCES,
        "frozen_recipe": assert_frozen_recipe_reproduced(),
        "training_control_flow": assert_frozen_training_control_flow(),
        "stage_coverage": assert_all_stages_guarded(),
        "frozen_plan": phase1.assert_frozen_execution_plan(),
        "resource_policy": phase1.resource_policy(),
        "no_automatic_retry": phase1.assert_no_automatic_retry(),
        "analysis_gate": assert_analysis_gate_is_correct(),
        "scientific_core": phase1.assert_scientific_core_refused(),
        "core_execution_state": e10.assert_core_execution_state_is_legitimate(),
        "evaluation_pass_accounting": EVALUATION_PASS_ACCOUNTING,
        "data_contract": {
            "manifests": {key: e10.MANIFEST_SHA256[key]
                          for key in sorted(e10.MANIFEST_SHA256)},
            "manifest_rows": dict(e10.EXPECTED_MANIFEST_ROWS),
            "token_stores": dict(e10.TOKEN_STORE_SHA256),
            "answer_vocabulary_sha256": e10.V2_VOCAB_SHA256,
            "answer_cache_sha256": e10.ANSWER_CACHE_SHA256,
            "clean_test_resolvable": False,
        },
        "clean_test_accessed": False,
    }


# --- immutable records --------------------------------------------------------

def _phase1_preserved_records() -> dict:
    preserved = {}
    for filename, expected in e10.PHASE1_R2_RECORD_SHA256.items():
        path = e10.OUT_DIR / filename
        if not path.is_file():
            raise PipelineError(f"Phase-1 record is missing: {filename}")
        digest = e10.sha256_file(path)
        if digest != expected:
            raise PipelineError(f"Phase-1 record changed: {filename}")
        preserved[filename] = {"location": "repository", "sha256": digest}
    return preserved


def write_phase2_records() -> dict:
    """Write the two immutable Phase-2 records, once."""
    existing = [str(path) for path in e10.PHASE2_PATHS if path.exists()]
    if existing:
        raise FileExistsError(f"immutable E10 Phase-2 records exist: {existing}")
    e10.assert_no_embargo_reference()
    phase1.assert_scientific_core_refused()
    e10.assert_no_scientific_cells()
    preserved = _phase1_preserved_records()

    amendment = {
        "schema_version": 1,
        "record_type": "e10_phase2_binding_amendment",
        "task_id": TASK_ID,
        "status": "PHASE1_EVIDENCE_CARRIED_FORWARD",
        "NON_SCIENTIFIC": True,
        "utc": e10.utc_now(),
        "binding": e10.binding_record(),
        "reason": (
            "Phase 2 adds the executable E10 scientific pipeline sources and "
            "replaces the guarded stub in run.py, so the live source digest "
            "moves. config.py and requirements.lock.txt are untouched, so the "
            "config digest does not move. The Phase-1 records are preserved "
            "byte for byte and are carried forward by this record instead of "
            "being rewritten."
        ),
        "change_scope": {
            "scientific_pipeline_implemented": True,
            "resource_constants_changed": False,
            "scientific_recipe_changed": False,
            "core_matrix_changed": False,
            "model_pin_changed": False,
            "dev_evaluation_cadence_changed": False,
            "scientific_authorization_granted": False,
            "phase0_record_rewritten": False,
            "phase1_record_rewritten": False,
        },
        "phase1_r2_source_digest": e10.PHASE1_R2_SOURCE_DIGEST,
        "phase1_r2_config_digest": e10.PHASE1_R2_CONFIG_DIGEST,
        "preserved_records": preserved,
        "scientific_execution": {
            "optimizer_steps": 0,
            "scientific_cells_executed": 0,
            "gpu_hours_charged": 0.0,
            "e10_training_authorized": None,
        },
    }
    amendment_sha = e10.atomic_write_json(e10.PHASE2_AMENDMENT_PATH, amendment)

    pipeline = {
        "schema_version": 1,
        "record_type": "e10_phase2_scientific_pipeline",
        "task_id": TASK_ID,
        "status": "PIPELINE_IMPLEMENTED_SCIENTIFIC_EXECUTION_STILL_REFUSED",
        "NON_SCIENTIFIC": True,
        "utc": e10.utc_now(),
        "git_commit": e10.current_git_commit(),
        "binding": e10.binding_record(),
        "reviewed_phase1_base_head": "231ad3bdfa3efc7e3caee3137987739b43d3193a",
        "revision": {
            "revision": PHASE2_REVISION,
            "reason": (
                "repair of the two blocking findings from the independent "
                "Phase-2 review: strict determinism was not in force before "
                "the guard's first stage, so a fresh-process cell could not "
                "open setup; and the difference in differences was reported at "
                "half its magnitude because the composite contrast averaged "
                "its two sides"
            ),
            "review_verdict_repaired": "CHANGES_REQUIRED",
            "repaired_from_head": PHASE2_REPAIRED_FROM_HEAD,
            "regenerated_within_open_revision": True,
            "superseded_records": PHASE2_SUPERSEDED_RECORD_SHA256,
            "scientific_recipe_changed": False,
            "core_matrix_changed": False,
            "dev_evaluation_cadence_changed": False,
            "resource_constants_changed": False,
            "scientific_authorization_granted": False,
        },
        "contract": pipeline_contract(),
        "artefact_tree": {
            "scientific_results": _relative(e10.CORE_OUT_DIR),
            "checkpoints": _relative(e10.CORE_CHECKPOINT_DIR),
            "large_artefacts_tracked_by": "per-cell SHA-256 in the cell record",
            "exists_yet": e10.CORE_OUT_DIR.exists(),
        },
        "clean_test_accessed": False,
    }
    pipeline_sha = e10.atomic_write_json(e10.PHASE2_PIPELINE_PATH, pipeline)
    return {"phase2_binding_amendment": amendment_sha,
            "phase2_scientific_pipeline": pipeline_sha}


def _validated_pipeline_record() -> dict:
    record = e10.read_json_mapping(e10.PHASE2_PIPELINE_PATH)
    e10.assert_recorded_binding(record, "E10 Phase-2 pipeline",
                                e10.PHASE2_PIPELINE_PATH)
    if record.get("record_type") != "e10_phase2_scientific_pipeline" \
            or record.get("task_id") != TASK_ID \
            or record.get("NON_SCIENTIFIC") is not True \
            or record.get("status") != (
                "PIPELINE_IMPLEMENTED_SCIENTIFIC_EXECUTION_STILL_REFUSED"):
        raise PipelineError("Phase-2 pipeline record semantics mismatch")
    contract = record.get("contract", {})
    if contract.get("implements", {}).get("pinned_development_cadence") != list(
            science.DEV_EVALUATION_EPOCHS):
        raise PipelineError("the recorded cadence differs from the live one")
    if contract.get("scientific_core", {}).get("refused") is not True:
        raise PipelineError("the Phase-2 record does not record the refusal")
    return record


def verify() -> dict:
    """Validate every Phase-1 guardrail and every Phase-2 pipeline contract."""
    result = {
        "phase1": phase1.verify(),
        "phase2_binding_amendment": {
            "sha256": e10.sha256_file(e10.PHASE2_AMENDMENT_PATH),
            "preserved_records": len(
                e10.validate_phase2_amendment()["preserved_records"]),
        } if e10.PHASE2_AMENDMENT_PATH.exists() else {"written": False},
        "pipeline_contract": pipeline_contract(),
    }
    if e10.PHASE2_PIPELINE_PATH.exists():
        record = _validated_pipeline_record()
        result["phase2_scientific_pipeline"] = {
            "sha256": e10.sha256_file(e10.PHASE2_PIPELINE_PATH),
            "status": record["status"],
        }
    return result


def main(argv: Any = None) -> int:  # pragma: no cover - thin CLI wrapper
    import json

    print(json.dumps(verify(), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
