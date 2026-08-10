"""CPU-only contract tests for the E10 Phase-2 scientific pipeline.

Nothing here trains, evaluates or authorises a scientific cell, and nothing
here performs a real optimisation step on real E10 data. Every test that needs
an authorised code path or a published cell fabricates that state inside a
temporary directory and restores it afterwards; the repository authorization
constant is never changed and the scientific artefact tree is never written.

The synthetic fixtures deliberately use tiny tensors and a scripted stand-in
for the frozen language model, so the control flow, the guard integration, the
publication contract, the reconciliation rules and the analysis arithmetic are
all exercised without a GPU and without the 360M checkpoint.
"""

from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.e8a_question_encoder import analyse_core as e8a_stats
from experiments.e10_capacity_360m import analysis
from experiments.e10_capacity_360m import e10_common as e10
from experiments.e10_capacity_360m import phase1
from experiments.e10_capacity_360m import phase2
from experiments.e10_capacity_360m import run as e10_run
from experiments.e10_capacity_360m import science
from tests.test_e10 import (
    _must_raise,
    _strict_state_restored,
    _temporary_shared_paths,
    _write_ledgers,
)


FROZEN_ORDER = [
    ("B4", "train_40k", 0), ("B4r", "train_40k", 0),
    ("B4", "train_40k", 1), ("B4r", "train_40k", 1),
    ("B4", "train_40k", 2), ("B4r", "train_40k", 2),
    ("B4", "train_250k", 0), ("B4r", "train_250k", 0),
    ("B4", "train_250k", 1), ("B4r", "train_250k", 1),
    ("B4", "train_250k", 2), ("B4r", "train_250k", 2),
]
FROZEN_CADENCE = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 19, 22)
SYNTHETIC_ROWS = 20
SYNTHETIC_IMAGES = 5
AUTH = "e10-core-matrix-approved"


# --- helpers ------------------------------------------------------------------

def _source(module) -> str:
    return Path(module.__file__).read_text()


def _patched_core_tree(stack: ExitStack, root: Path) -> dict:
    """Redirect the scientific artefact tree into a temporary directory."""
    paths = {
        "CORE_OUT_DIR": root / "core",
        "CORE_CHECKPOINT_DIR": root / "core" / "checkpoints",
    }
    for name, path in paths.items():
        stack.enter_context(mock.patch.object(e10, name, path))
    stack.enter_context(mock.patch.object(science, "CELL_LOCK_DIR",
                                          root / "cell-locks"))
    return paths


def _synthetic_dev_frame():
    import pandas as pd

    return pd.DataFrame({
        "questionId": [f"q{index:04d}" for index in range(SYNTHETIC_ROWS)],
        "imageId": [f"i{index % SYNTHETIC_IMAGES}" for index in range(SYNTHETIC_ROWS)],
        "label": [index % 4 for index in range(SYNTHETIC_ROWS)],
        "answer": ["yes", "no", "left", "right"] * (SYNTHETIC_ROWS // 4),
    })


def _synthetic_cell_record(arm, scale, seed, correct_mask, frame, root: Path):
    """A published cell record whose per-row evidence is real and consistent."""
    recipe = e10.build_recipe(arm, scale, seed)
    labels = frame["label"].to_numpy("int64")
    predictions = np.where(correct_mask, labels, (labels + 1) % 4)
    raw = (predictions == labels).astype(np.uint8)
    core = root / "core"
    checkpoints = core / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    name = e10.core_run_name(arm, scale, seed)
    per_row = core / f"{name}_per_row.npz"
    arrays = {
        "question_id": np.asarray(frame["questionId"].tolist(), dtype=np.str_),
        "image_id": np.asarray(frame["imageId"].tolist(), dtype=np.str_),
        "canonical_predictions": predictions.astype(np.int16),
        "labels": labels.astype(np.int16),
        "raw_correct": raw,
        "normalized_correct": raw,
        "canonical_margins": np.full(len(labels), 0.5, dtype=np.float64),
    }
    np.savez_compressed(per_row, **arrays)
    references = {}
    for role in ("canonical_ep22", "secondary_best_of22", "resume"):
        path = checkpoints / f"{name}_{role}.pt"
        path.write_bytes(role.encode())
        references[role] = {"path": str(path.relative_to(PROJECT_ROOT))
                            if path.is_relative_to(PROJECT_ROOT) else str(path),
                            "sha256": e10.sha256_file(path)}
    record = {
        "metadata": {"binding": e10.binding_record()},
        "e10_core_cell": {
            "schema_version": 1,
            "record_type": "e10_scientific_cell_result",
            "run": name,
            "cell": [arm, scale, seed],
            "identity": e10.model_identity(arm),
            "recipe_sha256": e10.recipe_sha256(recipe),
            "paired_recipe_sha256": e10.paired_recipe_sha256(recipe),
            "canonical_epoch": 22,
            "canonical_checkpoint_rule": "epoch_22",
            "early_stopping": False,
            "epochs_run": 22,
            "dev_evaluation_epochs": list(FROZEN_CADENCE),
            "primary_canonical_fp32_dev_accuracy": round(float(raw.mean()), 5),
            "secondary_diagnostic": {"label": "SECONDARY DIAGNOSTIC ONLY",
                                     "best_of_evaluated_epochs": 22,
                                     "best_dev_accuracy": round(
                                         float(raw.mean()), 5),
                                     "never_primary": True},
            "checkpoints": {
                "canonical": {**references["canonical_ep22"],
                              "role": "CANONICAL PRIMARY"},
                "secondary": {**references["secondary_best_of22"],
                              "role": "SECONDARY DIAGNOSTIC ONLY"},
                "training_state": {**references["resume"],
                                   "role": "DURABILITY AND FORENSICS ONLY - "
                                           "NEVER RESUMED"},
            },
            "per_row": {
                "path": str(per_row.relative_to(PROJECT_ROOT))
                if per_row.is_relative_to(PROJECT_ROOT) else str(per_row),
                "sha256": e10.sha256_file(per_row),
                "arrays": sorted(arrays),
            },
            "final_r1_evaluation": {"condition": "normal"},
            "peak_memory": {"reserved_fraction": 0.21},
            "wall": {"per_cell_wall_clock_hours": 12.0,
                     "elapsed_seconds_at_publication": 1234.5,
                     "guarded_stages_completed": list(phase1.CELL_STAGES),
                     "required_stages": list(phase1.CELL_STAGES)},
        },
    }
    e10.atomic_write_json(core / f"{name}.json", record)
    return record


def _core_spend_entry(arm, scale, seed, outcome="completed",
                      occupancy_ns=3_600_000_000_000):
    body = {
        "identity": e10.model_identity(arm),
        "arm": arm,
        "context": "e10_core_cell",
        "cell": [arm, scale, seed],
        "gpu_occupancy_ns": occupancy_ns,
        "measurement": "process_monotonic_ns",
        "outcome": outcome,
        "host": "unit-test-host",
        "pid": 1,
        "started_utc": "2026-08-10T00:00:00Z",
        "ended_utc": "2026-08-10T01:00:00Z",
        "source_digest": e10.source_digest()[0],
        "config_digest": e10.config_digest()[0],
        "protocol_family": e10.PROTOCOL_FAMILY,
        "claim_sha256": None,
        "recipe_digests": {
            f"{arm}_{scale}_seed{seed}": e10.recipe_sha256(
                e10.build_recipe(arm, scale, seed))},
    }
    return {"entry_id": e10.sha256_bytes(e10.canonical_json_bytes(body)), **body}


# --- frozen matrix, recipe and cadence ----------------------------------------

def test_frozen_twelve_cell_order_is_unchanged() -> None:
    order = e10.pair_preserving_order()
    assert order == FROZEN_ORDER, order
    assert e10_run.core_order() == FROZEN_ORDER
    assert phase1.execution_plan() == FROZEN_ORDER


def test_b4_b4r_pair_identity_is_provable() -> None:
    for scale in e10.CORE_SCALES:
        for seed in e10.CORE_SEEDS:
            b4 = e10.build_recipe("B4", scale, seed)
            b4r = e10.build_recipe("B4r", scale, seed)
            assert e10.paired_recipe_sha256(b4) == e10.paired_recipe_sha256(b4r)
            assert e10.recipe_sha256(b4) != e10.recipe_sha256(b4r)
            assert b4["frozen_model_identity"] == "pretrained_smollm2_360m"
            assert b4r["frozen_model_identity"] == "random_smollm2_360m"
            for key in ("lr", "warmup_frac", "dropout", "batch_size",
                        "grad_clip", "weight_decay", "max_epochs",
                        "canonical_checkpoint_rule", "d_lm",
                        "dev_evaluation_epochs"):
                assert b4[key] == b4r[key], key


def test_frozen_recipe_identity_matches_the_contract() -> None:
    frozen = phase2.assert_frozen_recipe_reproduced()
    assert frozen["cells"] == 12
    assert frozen["canonical_epoch"] == 22
    assert frozen["dev_evaluation_epochs"] == list(FROZEN_CADENCE)
    recipe = e10.build_recipe("B4", "train_40k", 0)
    assert recipe["lr"] == 3e-4 and recipe["warmup_frac"] == 0.0
    assert recipe["dropout"] == 0.1 and recipe["weight_decay"] == 0.01
    assert recipe["weight_decay_on"] == "parameters with ndim >= 2 only"
    assert recipe["batch_size"] == 128 and recipe["grad_clip"] == 1.0
    assert recipe["max_epochs"] == 22 and recipe["early_stopping"] is False
    assert recipe["canonical_checkpoint_rule"] == "epoch_22"
    assert recipe["canonical_evaluation_precision"] == "fp32"
    assert recipe["d_lm"] == 960
    assert "wall_clock_halt_hours" not in recipe
    assert "cosine" in recipe["scheduler"] and "100 x steps_per_epoch" in recipe[
        "scheduler"]


def test_frozen_dev_cadence_is_read_from_the_contract() -> None:
    assert science.DEV_EVALUATION_EPOCHS == FROZEN_CADENCE
    assert e10.DEV_EVALUATION_EPOCHS == FROZEN_CADENCE
    assert science.DEV_EVALUATION_EPOCHS[-1] == science.CANONICAL_EPOCH == 22
    contract = e10.read_json_mapping(e10.RECIPE_CONTRACT_PATH)
    recorded = contract["e10_recipe_contract"]["dev_evaluation_epochs"]
    assert list(recorded) == list(FROZEN_CADENCE)
    # The cadence is not redeclared inside the pipeline: it is an alias.
    assert science.DEV_EVALUATION_EPOCHS is e10.DEV_EVALUATION_EPOCHS


def test_scheduler_horizon_is_one_hundred_epochs() -> None:
    assert science.SCHEDULER_HORIZON_EPOCHS == e10.SCHEDULER_HORIZON_EPOCHS == 100
    tree = ast.parse(_source(science))
    call = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "make_scheduler")
    horizon = call.args[1]
    assert isinstance(horizon, ast.BinOp) and isinstance(horizon.op, ast.Mult)
    assert horizon.left.id == "SCHEDULER_HORIZON_EPOCHS"


def test_optimizer_recipe_is_weight_decay_on_ndim_two_only() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    optimizer = science.make_optimizer(model, 3e-4)
    assert isinstance(optimizer, torch.optim.AdamW)
    decay, no_decay = optimizer.param_groups
    assert decay["weight_decay"] == 0.01 and no_decay["weight_decay"] == 0.0
    assert all(parameter.ndim >= 2 for parameter in decay["params"])
    assert all(parameter.ndim < 2 for parameter in no_decay["params"])
    assert optimizer.param_groups[0]["lr"] == 3e-4


def test_cosine_scheduler_covers_only_the_first_part_of_its_horizon() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = science.make_optimizer(model, 3e-4)
    scheduler = science.make_scheduler(optimizer, 100 * 313, 0.0)
    start = optimizer.param_groups[0]["lr"]
    for _ in range(22 * 313):
        scheduler.step()
    end = optimizer.param_groups[0]["lr"]
    # Warmup is zero, so the very first step is already at the peak, and after
    # 22 of 100 epochs the cosine has decayed but is nowhere near zero.
    assert start == 3e-4
    assert 0.5 * start < end < start


# --- control flow: 22 epochs, no early stopping, guarded steps only -----------

def test_training_control_flow_is_the_frozen_budget() -> None:
    flow = phase2.assert_frozen_training_control_flow()
    assert flow["epoch_loops"] == 1
    assert flow["early_exit_statements"] == 0
    assert flow["bare_optimizer_steps"] == 0
    assert flow["guarded_optimizer_step_call_sites"] >= 1
    assert flow["frozen_epochs"] == 22


def test_no_early_stopping_anywhere_in_the_pipeline() -> None:
    """No patience state exists to stop on, and the recipe says so too."""
    tree = ast.parse(_source(science))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    for forbidden in ("patience", "epochs_without_improvement", "best_metric",
                      "epochs_since_improvement"):
        assert forbidden not in names, forbidden
    recipe = e10.build_recipe("B4", "train_40k", 0)
    assert recipe["early_stopping"] is False
    assert recipe["patience"] == ("disabled - no early stopping; fixed "
                                  "22-epoch budget")
    assert recipe["primary_checkpoint_selection"] == "fixed_endpoint"


def test_epoch_twenty_two_is_the_primary_not_the_best_epoch() -> None:
    source = _source(science)
    assert "CANONICAL PRIMARY" in source
    assert "SECONDARY DIAGNOSTIC ONLY" in source
    tree = ast.parse(source)
    publish = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef)
                   and node.name == "publish_cell")
    keys = {node.value for node in ast.walk(publish)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert "primary_canonical_fp32_dev_accuracy" in keys
    assert "best_of_evaluated_epochs" in keys
    # The published primary is the final history entry, which the schedule gate
    # forces to be epoch 22; the best state is only ever the secondary file.
    assert 'canonical_accuracy = history[-1]["canonical_fp32_dev_accuracy"]' \
        in source


def test_every_scientific_stage_runs_inside_the_guard() -> None:
    coverage = phase2.assert_all_stages_guarded()
    assert coverage["unguarded"] == 0
    assert set(coverage["guarded_stages"]) == set(phase1.CELL_STAGES)
    assert coverage["required_stages"] == list(phase1.CELL_STAGES)


def test_optimizer_steps_only_occur_through_the_guarded_path() -> None:
    """A spy optimizer proves the guard is the only door to a step."""
    class _Spy:
        def __init__(self):
            self.steps = 0

        def step(self):
            self.steps += 1

    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory)
        paths = stack.enter_context(_temporary_shared_paths(root))
        _write_ledgers(paths["SPEND_LEDGER"], paths["RETRY_LEDGER"])
        stack.enter_context(mock.patch.object(
            e10, "E10_TRAINING_AUTHORIZED", e10.CORE_AUTHORIZATION_TOKEN))
        stack.enter_context(mock.patch.object(
            e10, "assert_shared_state_healthy", lambda **_: {}))
        stack.enter_context(mock.patch.object(
            e10, "assert_strict_determinism", lambda: {"stub": True}))
        permit = e10.authorize_cell_execution("B4", "train_40k", 0)
        spy = _Spy()
        e10.guarded_optimizer_step(spy, permit)
        assert spy.steps == 1
        # An unsigned or foreign permit can never reach optimizer.step().
        _must_raise(AssertionError,
                    lambda: e10.guarded_optimizer_step(spy, object()))
        assert spy.steps == 1


# --- data, scoring and model contracts ----------------------------------------

def test_labelled_manifest_contract_is_pinned() -> None:
    assert set(e10.MANIFEST_BASENAMES) == {"train_40k", "train_250k", "dev"}
    assert set(e10.MANIFEST_SHA256) == set(e10.MANIFEST_BASENAMES)
    assert e10.EXPECTED_MANIFEST_ROWS == {"train_40k": 40_000,
                                          "train_250k": 250_000,
                                          "dev": 7_714}
    for key, basename in e10.MANIFEST_BASENAMES.items():
        path = science.manifest_path(key)
        assert path.name == basename
        assert basename in e10.APPROVED_DEVELOPMENT_CSV_BASENAMES
        assert e10.sha256_file(path) == e10.MANIFEST_SHA256[key]
    _must_raise(AssertionError, lambda: science.manifest_path("test"))


def test_embargoed_target_is_unreachable() -> None:
    scan = e10.assert_no_embargo_reference()
    assert scan["passed"] is True and scan["files_checked"] > 0
    embargoed = "test_" + "clean_targets"
    for module in (science, analysis, phase2):
        assert embargoed not in _source(module)
    for module in (science, analysis):
        tree = ast.parse(_source(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert ".csv" not in node.value or node.value in \
                    e10.APPROVED_DEVELOPMENT_CSV_BASENAMES


def test_token_store_identity_is_pinned() -> None:
    assert set(e10.TOKEN_STORE_SHA256) == {"image_tokens", "question_tokens"}
    for name, expected in e10.TOKEN_STORE_SHA256.items():
        assert len(expected) == 64 and int(expected, 16) >= 0
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory)
        _patched_core_tree(stack, root)
        stack.enter_context(mock.patch.object(
            science.tokens_data, "TOKEN_DIR", root / "tokens"))
        (root / "tokens").mkdir()
        (root / "tokens" / "image_tokens.h5").write_bytes(b"not the store")
        (root / "tokens" / "question_tokens.h5").write_bytes(b"nor this")
        _must_raise(science.ScientificHalt,
                    lambda: science.assert_token_store_identity("unit-test"),
                    "not the frozen E10 cache")


def test_g21_scorer_is_used_and_not_reimplemented() -> None:
    source = _source(science)
    assert "g21.score_rows(" in source
    assert "g21.scorer_provenance()" in source
    assert "def normalize_answer" not in source
    answers, provenance = science.g21.load_index_to_answer(e10.VOCAB_PATH)
    assert provenance["sha256"] == e10.V2_VOCAB_SHA256
    scored = science.g21.score_rows([0, 1], [0, 0], answers)
    assert scored["raw_correct"] == [True, False]
    assert scored["normalized_correct"][0] is True


def test_b4_b4r_construction_contract_is_asserted_before_training() -> None:
    source = _source(science)
    # The audited construction order and the frozen parameter counts.
    assert "e10.load_frozen_causal_lm(arm, device=None)" in source
    assert "e10.promote_lm_to_fp32(lm)" in source
    assert "e10.build_trainable(arm, seed, dropout, lm)" in source
    assert "e10.EXPECTED_TRAINABLE_PARAMETERS" in source
    assert "e10.MODEL_PARAMETERS" in source
    assert e10.EXPECTED_TRAINABLE_PARAMETERS == 21_540_800
    assert e10.EXPECTED_LATENT_PARAMETERS == 21_047_296
    assert e10.EXPECTED_PROJECTION_PARAMETERS == 493_504
    assert e10.MODEL_PARAMETERS == 361_821_120
    assert e10.D_LM == 960
    assert e10.ARMS["B4"]["pretrained"] is True
    assert e10.ARMS["B4r"]["pretrained"] is False
    from experiments.e8a_question_encoder.e8a_common import RANDOM_INIT_SEED
    assert RANDOM_INIT_SEED == 20260802


def test_trainable_and_frozen_parameter_counts_are_validated() -> None:
    """A model with the wrong split is refused rather than trained."""
    class _Frozen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2, 2))

    frozen = _Frozen()
    trainable = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(trainable.parameters())
    record = e10.assert_optimizer_excludes_lm(optimizer, trainable, frozen)
    assert record["frozen_lm_overlap"] == 0
    leaky = torch.optim.AdamW(
        list(trainable.parameters()) + list(frozen.parameters()))
    _must_raise(AssertionError,
                lambda: e10.assert_optimizer_excludes_lm(leaky, trainable,
                                                         frozen))


# --- publication, reconciliation and staleness --------------------------------

def test_atomic_writers_publish_or_leave_nothing() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        target = root / "checkpoint.pt"
        digest = science._atomic_torch_save(target, {"state": 1})
        assert target.exists() and digest == e10.sha256_file(target)
        assert not list(root.glob("*.tmp"))
        array_path = root / "rows.npz"
        array_digest = science._atomic_npz_save(
            array_path, values=np.arange(4))
        assert array_path.exists()
        assert array_digest == e10.sha256_file(array_path)
        assert not list(root.glob("*.tmp.npz"))
        # The immutable JSON writer refuses a second publication outright.
        record_path = root / "cell.json"
        e10.atomic_write_json(record_path, {"a": 1})
        _must_raise(FileExistsError,
                    lambda: e10.atomic_write_json(record_path, {"a": 2}))


def test_stale_artefacts_and_completed_cells_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory)
        paths = _patched_core_tree(stack, root)
        core = paths["CORE_OUT_DIR"]
        core.mkdir(parents=True)
        cell = e10.core_cell_paths("B4", "train_40k", 0)
        # A completed cell is never overwritten.
        cell["result"].write_text("{}")
        _must_raise(SystemExit,
                    lambda: science._preflight_refusals("B4", "train_40k", 0),
                    "already exists")
        cell["result"].unlink()
        # A recorded failure is never silently retried.
        cell["failed"].write_text("{}")
        _must_raise(SystemExit,
                    lambda: science._preflight_refusals("B4", "train_40k", 0),
                    "recorded failure")
        cell["failed"].unlink()
        # A recorded halt is never silently retried.
        halt_path = core / f"HALT_{cell['run_name']}_G14_20260810T000000Z.json"
        halt_path.write_text("{}")
        _must_raise(SystemExit,
                    lambda: science._preflight_refusals("B4", "train_40k", 0),
                    "recorded gate halt")
        halt_path.unlink()
        # A leftover training state is never resumed.
        cell["resume_checkpoint"].parent.mkdir(parents=True, exist_ok=True)
        cell["resume_checkpoint"].write_bytes(b"state")
        _must_raise(SystemExit,
                    lambda: science._preflight_refusals("B4", "train_40k", 0),
                    "never resumes")
        cell["resume_checkpoint"].unlink()
        # A stale binary with no published result blocks re-entry.
        cell["canonical_checkpoint"].write_bytes(b"stale")
        _must_raise(SystemExit,
                    lambda: science._preflight_refusals("B4", "train_40k", 0),
                    "stale canonical_checkpoint")


def test_partial_cell_cannot_reconcile_as_complete() -> None:
    frame = _synthetic_dev_frame()
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory)
        _patched_core_tree(stack, root)
        stack.enter_context(mock.patch.object(
            e10, "EXPECTED_MANIFEST_ROWS",
            {**e10.EXPECTED_MANIFEST_ROWS, "dev": SYNTHETIC_ROWS}))
        _synthetic_cell_record("B4", "train_40k", 0,
                               np.ones(SYNTHETIC_ROWS, dtype=bool), frame, root)
        run_name = e10.core_run_name("B4", "train_40k", 0)
        # A cell whose per-row evidence has been removed is not complete.
        e10.core_cell_paths("B4", "train_40k", 0)["per_row"].unlink()
        _must_raise(science.ScientificHalt,
                    lambda: science.reconcile_cell("B4", "train_40k", 0,
                                                   run_name),
                    "missing required artefacts")


def test_missing_required_artefact_prevents_completion() -> None:
    """The guard refuses a completed outcome while any stage stayed unguarded."""
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory)
        paths = stack.enter_context(_temporary_shared_paths(root))
        _write_ledgers(paths["SPEND_LEDGER"], paths["RETRY_LEDGER"])
        _patched_core_tree(stack, root)
        stack.enter_context(mock.patch.object(
            e10, "E10_TRAINING_AUTHORIZED", e10.CORE_AUTHORIZATION_TOKEN))
        stack.enter_context(mock.patch.object(
            e10, "assert_shared_state_healthy", lambda **_: {}))
        stack.enter_context(mock.patch.object(
            e10, "assert_strict_determinism", lambda: {"stub": True}))
        governance = root / "governance"
        governance.mkdir()
        stack.enter_context(mock.patch.object(e10, "OUT_DIR", governance))
        with phase1.ScientificCellGuard("B4", "train_40k", 0) as guard:
            with guard.stage("setup"):
                pass
        assert guard._finalized["outcome"] == "terminated"
        written = json.loads(
            (governance / Path(guard._finalized["record"]["path"]).name
             ).read_text())
        assert written["scientific_success"] is False
        assert written["guarded_stages_completed"] == ["setup"]
        assert written["required_stages"] == list(phase1.CELL_STAGES)
        assert written["automatic_retry_available"] is False


# --- the frozen analysis ------------------------------------------------------

def test_analysis_refuses_an_incomplete_matrix() -> None:
    result = phase2.assert_analysis_refuses_incomplete()
    assert result["refused"] is True
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory)
        _patched_core_tree(stack, root)
        _must_raise(analysis.AnalysisRefused, analysis.load_matrix,
                    "no published result")


def test_analysis_refuses_a_matrix_that_is_not_pair_preserved() -> None:
    frame = _synthetic_dev_frame()
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory)
        _patched_core_tree(stack, root)
        stack.enter_context(mock.patch.object(
            e10, "EXPECTED_MANIFEST_ROWS",
            {**e10.EXPECTED_MANIFEST_ROWS, "dev": SYNTHETIC_ROWS}))
        stack.enter_context(mock.patch.object(
            analysis, "_dev_reference", lambda: _synthetic_reference(frame)))
        # Eleven of twelve cells: the missing B4r partner must refuse.
        for arm, scale, seed in FROZEN_ORDER[1:]:
            _synthetic_cell_record(arm, scale, seed,
                                   np.ones(SYNTHETIC_ROWS, dtype=bool), frame,
                                   root)
        _must_raise(analysis.AnalysisRefused, analysis.load_matrix)


def _synthetic_reference(frame) -> dict:
    images = sorted(frame["imageId"].unique().tolist())
    image_of = {image: index for index, image in enumerate(images)}
    return {
        "path": "synthetic",
        "sha256": "0" * 64,
        "rows": len(frame),
        "question_ids": frame["questionId"].astype(str).to_numpy(),
        "labels": frame["label"].to_numpy("int64"),
        "image_index": np.array([image_of[image] for image in frame["imageId"]],
                                dtype="int64"),
        "n_images": len(images),
    }


def test_analysis_reproduces_expected_arithmetic_on_fixtures() -> None:
    """A synthetic matrix with known accuracies must give known contrasts."""
    frame = _synthetic_dev_frame()
    # B4 is right on every row; B4r is right on three quarters of them, at both
    # scales, so the pretraining effect is exactly +0.25 at each scale and the
    # difference in differences is exactly zero.
    b4_mask = np.ones(SYNTHETIC_ROWS, dtype=bool)
    b4r_mask = np.array([index % 4 != 0 for index in range(SYNTHETIC_ROWS)])
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory)
        paths = stack.enter_context(_temporary_shared_paths(root))
        _write_ledgers(paths["SPEND_LEDGER"], paths["RETRY_LEDGER"])
        _patched_core_tree(stack, root)
        stack.enter_context(mock.patch.object(
            analysis, "_dev_reference", lambda: _synthetic_reference(frame)))
        ledger = e10.read_json_mapping(paths["SPEND_LEDGER"])
        for arm, scale, seed in FROZEN_ORDER:
            _synthetic_cell_record(
                arm, scale, seed,
                b4_mask if arm == "B4" else b4r_mask, frame, root)
            ledger["entries"].append(_core_spend_entry(arm, scale, seed))
        e10.atomic_replace_json(paths["SPEND_LEDGER"], ledger)
        matrix = analysis.load_matrix()
        assert len(matrix["cells"]) == 12
        summaries = analysis.seed_summaries(matrix)
        assert summaries["B4_train_40k"]["canonical_r1_correct"]["mean"] == 1.0
        assert summaries["B4r_train_40k"]["canonical_r1_correct"]["mean"] == 0.75
        contrasts = {}
        for name, definition in analysis._frozen_contrasts().items():
            contrasts[name] = analysis._paired_interval(
                matrix, "canonical_r1_correct", definition["positive"],
                definition["negative"])
        assert contrasts["pretraining_effect_train_40k"][
            "per_seed_differences"] == [0.25, 0.25, 0.25]
        assert contrasts["pretraining_effect_train_250k"][
            "mean_difference"] == 0.25
        assert contrasts["scale_effect_B4"]["mean_difference"] == 0.0
        did = contrasts["pretraining_by_scale_difference_in_differences"]
        assert did["mean_difference"] == 0.0
        assert did["interval_95"] == [0.0, 0.0]
        assert did["directional_rule"] == "uncertain or mixed evidence"
        assert contrasts["pretraining_effect_train_40k"][
            "directional_rule"] == "positive directional evidence"
        reliance = analysis.visual_reliance_summary(matrix)
        assert reliance["conditions_present"] == ["normal"]
        assert reliance["contrasts"] == {}
        assert reliance["status"].startswith("NO_INTERVENTION_EVIDENCE")


def test_analysis_refuses_when_accounting_does_not_reconcile() -> None:
    frame = _synthetic_dev_frame()
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory)
        paths = stack.enter_context(_temporary_shared_paths(root))
        _write_ledgers(paths["SPEND_LEDGER"], paths["RETRY_LEDGER"])
        _patched_core_tree(stack, root)
        stack.enter_context(mock.patch.object(
            analysis, "_dev_reference", lambda: _synthetic_reference(frame)))
        ledger = e10.read_json_mapping(paths["SPEND_LEDGER"])
        for arm, scale, seed in FROZEN_ORDER:
            _synthetic_cell_record(arm, scale, seed,
                                   np.ones(SYNTHETIC_ROWS, dtype=bool), frame,
                                   root)
            # One cell is charged as halted: the evidence and the authoritative
            # accounting disagree, so nothing may be analysed.
            outcome = "wall_clock_halted" if (arm, scale, seed) == \
                FROZEN_ORDER[5] else "completed"
            ledger["entries"].append(
                _core_spend_entry(arm, scale, seed, outcome=outcome))
        e10.atomic_replace_json(paths["SPEND_LEDGER"], ledger)
        _must_raise(analysis.AnalysisRefused, analysis.load_matrix)


def test_analysis_refuses_tampered_per_row_evidence() -> None:
    frame = _synthetic_dev_frame()
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory)
        _patched_core_tree(stack, root)
        stack.enter_context(mock.patch.object(
            analysis, "_dev_reference", lambda: _synthetic_reference(frame)))
        _synthetic_cell_record("B4", "train_40k", 0,
                               np.ones(SYNTHETIC_ROWS, dtype=bool), frame, root)
        per_row = e10.core_cell_paths("B4", "train_40k", 0)["per_row"]
        per_row.write_bytes(b"tampered")
        _must_raise(
            analysis.AnalysisRefused,
            lambda: analysis._load_cell("B4", "train_40k", 0,
                                        _synthetic_reference(frame)),
            "does not match its recorded digest")


# --- guardrails that Phase 2 must not have weakened ---------------------------

def test_twelve_hour_wall_and_forty_hour_ceiling_are_unchanged() -> None:
    policy = phase1.resource_policy()
    assert policy["per_cell_wall_clock_hours"] == 12.0
    assert policy["effective_identity_ceiling_hours"] == 40.0
    assert policy["e10_per_identity_ceiling_hours"] == 40.0
    assert policy["programme_wide_identity_ceiling_hours"] == 40.0
    assert policy["automatic_retries_per_identity"] == 0
    assert e10.per_cell_wall_hours() == 12.0
    assert e10.effective_identity_ceiling_hours() == min(
        40.0, 40.0)


def test_automatic_retry_remains_impossible() -> None:
    scan = phase1.assert_no_automatic_retry()
    assert scan["automatic_retry_writers"] == 0
    assert scan["automatic_retry_call_sites"] == 0
    assert scan["automatic_retries_per_identity"] == 0
    for module in (science, analysis, phase2):
        source = _source(module)
        assert "record_forced_retry" not in source
        assert "retry_authorization" not in source


def test_no_e8_frozen_result_is_modified() -> None:
    import config

    for name, expected in e10.FROZEN_RESULT_BASELINES.items():
        root = config.RESULTS_DIR / "experiments" / name
        rows = sorted((item.relative_to(e10.PROJECT_ROOT).as_posix(),
                       e10.sha256_file(item))
                      for item in root.rglob("*") if item.is_file())
        stream = "".join(f"{digest}  {path}\n" for path, digest in rows)
        observed = {"files": len(rows),
                    "tree_sha256": e10.sha256_bytes(stream.encode("utf-8"))}
        assert observed == expected, name


def test_scientific_authorization_remains_closed() -> None:
    assert e10.E10_TRAINING_AUTHORIZED is None
    assert e10.E10_CALIBRATION_AUTHORIZED is None
    refusal = phase1.assert_scientific_core_refused()
    assert refusal["refused"] is True
    assert "scientific authorization is None" in refusal["refusal"]
    _must_raise(SystemExit,
                lambda: e10_run.core_cell("B4", "train_40k", 0),
                "E10 CORE REFUSED")
    _must_raise(SystemExit,
                lambda: e10.authorize_cell_execution("B4", "train_40k", 0),
                "E10 CORE REFUSED")


def test_no_scientific_cell_has_executed() -> None:
    proof = e10.assert_no_scientific_cells()
    assert proof == {"published_cell_records": 0, "checkpoint_files": 0,
                     "charged_core_cells": 0, "e10_training_authorized": None}
    outputs = e10.assert_no_scientific_outputs()
    assert outputs["checkpoint_files"] == 0
    assert outputs["forbidden_metric_fields"] == 0


def test_core_cell_refuses_before_any_scientific_side_effect() -> None:
    """The refusal precedes model, data, CUDA, optimizer and output creation."""
    calls = []
    with ExitStack() as stack:
        for module, name in ((e10, "load_frozen_causal_lm"),
                             (e10, "build_trainable"),
                             (science, "build_scientific_loaders"),
                             (science, "build_answer_cache"),
                             (science, "_preflight_refusals"),
                             (science, "_acquire_cell_lock")):
            stack.enter_context(mock.patch.object(
                module, name,
                mock.Mock(side_effect=AssertionError(f"{name} was reached"))))
            calls.append(name)
        stack.enter_context(mock.patch.object(
            torch.cuda, "is_available", lambda: True))
        _must_raise(SystemExit,
                    lambda: e10_run.core_cell("B4", "train_40k", 0),
                    "E10 CORE REFUSED")
    assert not e10.CORE_OUT_DIR.exists(), e10.CORE_OUT_DIR


def test_phase2_pipeline_contract_reports_the_real_state() -> None:
    contract = phase2.pipeline_contract()
    assert contract["implements"]["pinned_development_cadence"] == list(
        FROZEN_CADENCE)
    assert contract["frozen_recipe"]["cells"] == 12
    assert contract["stage_coverage"]["unguarded"] == 0
    assert contract["scientific_core"]["refused"] is True
    assert contract["no_scientific_execution"]["charged_core_cells"] == 0
    assert contract["clean_test_accessed"] is False
    assert contract["data_contract"]["clean_test_resolvable"] is False
    assert "g8_overfit_gate" in contract["deliberate_e10_differences"]


# --- B1: strict determinism before the first guard stage ----------------------

def _non_strict_process_state() -> None:
    """Put the process in the state a freshly started real cell starts in."""
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def _authorized(stack: ExitStack) -> None:
    stack.enter_context(mock.patch.object(
        e10, "E10_TRAINING_AUTHORIZED", AUTH))
    stack.enter_context(mock.patch.object(
        e10, "assert_shared_state_healthy", lambda **_: {}))


def test_first_stage_refuses_without_strict_determinism() -> None:
    """The defect B1 names: the guard's first stage check asserts determinism.

    This is the mechanism, exercised directly and without stubbing, so the
    repair below is pinned to a real requirement rather than to a comment.
    """
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack, \
            _strict_state_restored():
        root = Path(directory)
        paths = stack.enter_context(_temporary_shared_paths(root))
        _write_ledgers(paths["SPEND_LEDGER"], paths["RETRY_LEDGER"])
        governance = root / "governance"
        governance.mkdir()
        stack.enter_context(mock.patch.object(e10, "OUT_DIR", governance))
        _authorized(stack)
        _non_strict_process_state()
        _must_raise(AssertionError, e10.assert_strict_determinism,
                    "strict determinism is not in force")
        error = None
        try:
            with phase1.ScientificCellGuard("B4", "train_40k", 0) as guard:
                with guard.stage("setup"):
                    pass
        except AssertionError as raised:
            error = raised
        assert error is not None, "the first stage opened without determinism"
        assert "strict determinism is not in force" in str(error)


def test_production_entry_enables_determinism_before_the_first_stage() -> None:
    """B1 regression: the production runner, from a non-strict fresh state.

    Nothing here stubs assert_strict_determinism. The process is put into the
    state a freshly started real cell starts in, the real production entry
    sequence runs, and the real guard's first setup stage must then open.
    """
    observed: dict = {}

    def _only_setup(guard, arm, scale, seed, run_name, preflight):
        observed["at_entry"] = preflight["determinism_at_entry"]
        with guard.stage("setup"):
            observed["inside_setup"] = e10.assert_strict_determinism()
        observed["permit"] = guard.permit
        observed["stages"] = guard.stages_completed

    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack, \
            _strict_state_restored():
        root = Path(directory)
        paths = stack.enter_context(_temporary_shared_paths(root))
        _write_ledgers(paths["SPEND_LEDGER"], paths["RETRY_LEDGER"])
        governance = root / "governance"
        governance.mkdir()
        stack.enter_context(mock.patch.object(e10, "OUT_DIR", governance))
        _patched_core_tree(stack, root)
        _authorized(stack)
        stack.enter_context(mock.patch.object(science, "_execute_cell",
                                              _only_setup))
        _non_strict_process_state()
        _must_raise(AssertionError, e10.assert_strict_determinism,
                    "strict determinism is not in force")

        assert science.run_core_cell("B4", "train_40k", 0) == 0

        # The production entry established it, and the first stage opened.
        assert observed["at_entry"]["deterministic_algorithms"] is True
        assert observed["at_entry"]["warn_only"] is False
        assert observed["inside_setup"]["deterministic_algorithms"] is True
        assert observed["stages"] == ("setup",)
        # The permit is not poisoned and the wall semantics are untouched.
        permit = observed["permit"]
        assert permit.per_cell_wall_hours == 12.0
        assert permit.effective_identity_ceiling_hours == 40.0
        assert permit.deadline_monotonic_ns - permit.started_monotonic_ns == \
            12 * 3600 * 1_000_000_000
        # Exactly one charge, and it is the harness stopping after setup, not a
        # determinism failure and not a wall halt.
        ledger = e10.read_json_mapping(paths["SPEND_LEDGER"])
        entries = [entry for entry in ledger["entries"]
                   if entry["context"] == "e10_core_cell"]
        assert len(entries) == 1
        assert entries[0]["outcome"] == "terminated"
        record = json.loads(next(governance.glob("cell_*.json")).read_text())
        assert record["scientific_success"] is False
        assert "unguarded scientific stages" in record["detail"]
        assert "determinism" not in record["detail"]
        assert record["wall_reached_before_finalisation"] is False


# --- the production synthetic end-to-end harness ------------------------------

HARNESS_TOKENS = {
    "yes": [1], "no": [2], "left": [3, 4], "right": [5, 6],
    "blue": [7, 8, 9], "green": [10, 11, 12],
    "table": [13, 14, 15, 1], "chair": [2, 3, 4, 5],
}
HARNESS_ANSWERS = list(HARNESS_TOKENS)
HARNESS_VOCAB = 16
HARNESS_IMAGE_TOKENS = 8
HARNESS_G14_ROWS = 8


class _HarnessDeviceProperties:
    """Stand-in for the single-device properties the cell reads."""

    total_memory = 20 * 2 ** 30
    major = 8
    minor = 9
    name = "harness"


class _HarnessTokenizer:
    """A deterministic reverse-lookup tokenizer for the synthetic support.

    The real answer cache and trie builders run over it unchanged, so the G1
    round-trip, duplicate-sequence and prefix inventory are the production ones.
    """

    def __call__(self, answer, add_special_tokens=False):
        return {"input_ids": list(HARNESS_TOKENS[answer])}

    def decode(self, ids, skip_special_tokens=False,
               clean_up_tokenization_spaces=False):
        for answer, sequence in HARNESS_TOKENS.items():
            if list(sequence) == list(ids):
                return answer
        raise AssertionError(f"unknown ids {ids}")


def _harness_lm():
    from transformers import LlamaConfig, LlamaForCausalLM

    configuration = LlamaConfig(
        hidden_size=e10.D_LM, num_hidden_layers=1, num_attention_heads=15,
        num_key_value_heads=5, head_dim=64, intermediate_size=64,
        vocab_size=HARNESS_VOCAB, tie_word_embeddings=True,
        rope_theta=100000.0, rms_norm_eps=1e-5, max_position_embeddings=512)
    torch.manual_seed(0)
    model = LlamaForCausalLM(configuration).to(torch.bfloat16)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def _write_harness_inputs(root: Path) -> dict:
    """Synthetic manifests, token stores and vocabulary in the real formats."""
    import h5py
    import pandas as pd

    data = root / "v2"
    tokens = root / "tokens"
    data.mkdir(parents=True)
    tokens.mkdir(parents=True)
    rng = np.random.default_rng(0)

    images = [f"i{index}" for index in range(SYNTHETIC_IMAGES)]
    with h5py.File(tokens / "image_tokens.h5", "w") as store:
        store.create_dataset("ids", data=np.array(images, dtype="S8"))
        store.create_dataset(
            "tokens",
            data=rng.normal(size=(len(images), HARNESS_IMAGE_TOKENS, 512)
                            ).astype(np.float16))

    question_ids, lengths = [], []
    for index in range(2 * SYNTHETIC_ROWS):
        question_ids.append(f"q{index:04d}")
        lengths.append(3 + index % 4)
    offsets = np.cumsum([0] + lengths[:-1]).astype("int64")
    with h5py.File(tokens / "question_tokens.h5", "w") as store:
        store.create_dataset("ids", data=np.array(question_ids, dtype="S8"))
        store.create_dataset(
            "tokens",
            data=rng.normal(size=(int(sum(lengths)), 512)).astype(np.float16))
        store.create_dataset("offsets", data=offsets)
        store.create_dataset("lengths", data=np.array(lengths, dtype="int64"))

    def frame(start):
        rows = []
        for index in range(SYNTHETIC_ROWS):
            label = index % len(HARNESS_ANSWERS)
            rows.append({"questionId": question_ids[start + index],
                         "imageId": images[index % len(images)],
                         "label": label, "answer": HARNESS_ANSWERS[label]})
        return pd.DataFrame(rows)

    frame(0).to_csv(data / "dev.csv", index=False)
    frame(SYNTHETIC_ROWS).to_csv(data / "train_40k.csv", index=False)
    vocabulary = data / "answer_vocab_v2.json"
    vocabulary.write_text(json.dumps(
        {"answer_to_index": {answer: index
                             for index, answer in enumerate(HARNESS_ANSWERS)}}))
    return {
        "v2": data,
        "tokens": tokens,
        "vocabulary": vocabulary,
        "manifest_sha256": {
            "train_40k": e10.sha256_file(data / "train_40k.csv"),
            "train_250k": e10.sha256_file(data / "train_40k.csv"),
            "dev": e10.sha256_file(data / "dev.csv"),
        },
        "store_sha256": {
            "image_tokens": e10.sha256_file(tokens / "image_tokens.h5"),
            "question_tokens": e10.sha256_file(tokens / "question_tokens.h5"),
        },
        "vocabulary_sha256": e10.sha256_file(vocabulary),
    }


def _harness_answer_cache(run_name: str):
    from experiments.e8b_readout_generation import readouts

    cache = readouts.build_answer_cache(_HarnessTokenizer(), HARNESS_ANSWERS)
    if cache["sha256"] != e10.ANSWER_CACHE_SHA256:
        raise AssertionError("harness answer-support tokenisation drifted")
    return (None, list(HARNESS_ANSWERS),
            {"path": "harness", "sha256": e10.V2_VOCAB_SHA256,
             "n_classes": len(HARNESS_ANSWERS)},
            cache, readouts.build_trie(cache))


def _install_harness(stack: ExitStack, root: Path, inputs: dict) -> dict:
    """Point the production runner at synthetic inputs on CPU.

    Everything patched here is an INPUT, a device or a machine measurement.
    No rule, gate, threshold, cadence, budget, selection rule, accounting path
    or publication step is replaced; those all run as production code.
    """
    from experiments.e8b_readout_generation import readouts
    from experiments.e8b_readout_generation import training as e8b_training

    ledgers = {
        "SPEND_LEDGER": root / "spend.json",
        "RETRY_LEDGER": root / "retry.json",
        "GOVERNANCE_LOCK": root / "governance.lock",
        "INITIALIZATION_PENDING": root / "initialization-pending.json",
        "LEDGER_FAILURE_PATH": root / "ledger-failure.json",
    }
    for name, path in ledgers.items():
        stack.enter_context(mock.patch.object(e10, name, path))
    _write_ledgers(ledgers["SPEND_LEDGER"], ledgers["RETRY_LEDGER"])
    # The guard writes its cell outcome into OUT_DIR, so OUT_DIR is redirected;
    # the immutable governance records the production validators read are
    # mirrored into it byte for byte rather than being stubbed away.
    import shutil

    governance = root / "governance"
    governance.mkdir()
    for record in sorted(e10.OUT_DIR.glob("*.json")):
        shutil.copy2(record, governance / record.name)
    stack.enter_context(mock.patch.object(e10, "OUT_DIR", governance))
    # The embargo scan reports paths relative to the project root, which the
    # mirrored copies under a temporary directory are not. It is re-pointed at
    # the REAL E10 sources, so the scan itself still runs for real.
    real_scan = e10.assert_no_embargo_reference
    real_sources = list(Path(e10.__file__).parent.rglob("*.py"))
    stack.enter_context(mock.patch.object(
        e10, "assert_no_embargo_reference",
        lambda paths=None: real_scan(real_sources)))
    _patched_core_tree(stack, root)
    _authorized(stack)

    cache = readouts.build_answer_cache(_HarnessTokenizer(), HARNESS_ANSWERS)
    replacements = {
        # inputs
        (e10, "V2_DIR"): inputs["v2"],
        (e10, "VOCAB_PATH"): inputs["vocabulary"],
        (e10, "V2_VOCAB_SHA256"): inputs["vocabulary_sha256"],
        (e10, "MANIFEST_SHA256"): inputs["manifest_sha256"],
        (e10, "TOKEN_STORE_SHA256"): inputs["store_sha256"],
        (e10, "EXPECTED_MANIFEST_ROWS"): {"train_40k": SYNTHETIC_ROWS,
                                          "train_250k": SYNTHETIC_ROWS,
                                          "dev": SYNTHETIC_ROWS},
        (e10, "STEPS_PER_EPOCH"): {"train_40k": 1, "train_250k": 1},
        (e10, "ANSWER_CACHE_SHA256"): cache["sha256"],
        (e10, "ANSWER_CACHE_FACTS"): {"answers": len(HARNESS_ANSWERS)},
        (science.tokens_data, "TOKEN_DIR"): inputs["tokens"],
        (science, "build_answer_cache"): _harness_answer_cache,
        # frozen model identity: a two-layer-free stand-in at the real width
        (e10, "load_frozen_causal_lm"):
            lambda arm, device=None: (_harness_lm(), {"arm": arm,
                                                      "harness": True}),
        (e10, "pretrained_embed_weight"):
            lambda: torch.full((HARNESS_VOCAB, e10.D_LM), 0.02),
        (science, "assert_model_identity"):
            lambda arm, run_name: {"harness": True, "arm": arm,
                                   "d_lm": e10.D_LM},
        (e10, "MODEL_PARAMETERS"): sum(p.numel() for p in _harness_lm().parameters()),
        # device and machine measurements
        (science, "SCIENTIFIC_DEVICE"): "cpu",
        (e10, "gpu_exclusivity_preflight"):
            lambda: {"exclusive_at_preflight": True, "harness": True},
        # G14 row counts scale to the synthetic development set; the P/L/O
        # construction, the C1/C2/C3 clauses and the halting rule are unchanged
        (e8b_training, "G14_N_EXAMPLES"): HARNESS_G14_ROWS,
        (e8b_training, "G14_FP32_LOWEST"): HARNESS_G14_ROWS,
        (e8b_training, "G14_FP32_ORDINARY_PER_BAND"): 2,
    }
    for (module, name), value in replacements.items():
        stack.enter_context(mock.patch.object(module, name, value))

    for name in ("reset_peak_memory_stats", "empty_cache"):
        stack.enter_context(mock.patch.object(torch.cuda, name, lambda: None))
    stack.enter_context(mock.patch.object(
        torch.cuda, "max_memory_allocated", lambda: 2 * 2 ** 30))
    stack.enter_context(mock.patch.object(
        torch.cuda, "max_memory_reserved", lambda: 4 * 2 ** 30))
    stack.enter_context(mock.patch.object(
        torch.cuda, "get_device_properties",
        lambda index=0: _HarnessDeviceProperties()))

    # The synthetic model cannot be expected to spread its predictions over the
    # answer support, so the collapse diagnostic still RUNS and is still
    # recorded in full; only its halting outcome is neutralised for the harness.
    real_collapse = e8b_training.g10_collapse_record

    def _recorded_collapse(arm, predictions, scores):
        record = real_collapse(arm, predictions, scores)
        return {**record, "fires": False,
                "harness_note": "computed in full; halt neutralised because a "
                                "synthetic model's answer distribution is not "
                                "a scientific observation"}

    stack.enter_context(mock.patch.object(
        e8b_training, "g10_collapse_record", _recorded_collapse))
    return {"ledgers": ledgers, "governance": governance}


def test_production_synthetic_cell_reaches_complete() -> None:
    """End to end through the real runner: setup to a reconciled COMPLETE cell.

    This drives experiments.e10_capacity_360m.science.run_core_cell itself. The
    determinism repair is NOT compensated for anywhere: the process is left in
    a non-strict state and only the production entry establishes it.
    """
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack, \
            _strict_state_restored():
        root = Path(directory)
        inputs = _write_harness_inputs(root)
        installed = _install_harness(stack, root, inputs)
        _non_strict_process_state()
        _must_raise(AssertionError, e10.assert_strict_determinism,
                    "strict determinism is not in force")

        assert science.run_core_cell("B4", "train_40k", 0) == 0

        paths = e10.core_cell_paths("B4", "train_40k", 0)
        record = e10.read_json_mapping(paths["result"])["e10_core_cell"]
        # the frozen budget and cadence
        assert record["epochs_run"] == 22
        assert record["canonical_epoch"] == 22
        assert record["canonical_checkpoint_rule"] == "epoch_22"
        assert record["early_stopping"] is False
        assert record["resumed"] is False
        assert record["dev_evaluation_epochs"] == list(FROZEN_CADENCE)
        assert record["scheduler_horizon_epochs"] == 100
        evaluated = [row["epoch"] for row in record["history"] if row["evaluated"]]
        assert evaluated == list(FROZEN_CADENCE)
        assert [row["epoch"] for row in record["history"]] == list(range(1, 23))
        assert record["history"][-1]["canonical_fp32_dev_accuracy"] == \
            record["primary_canonical_fp32_dev_accuracy"]
        # the frozen diagnostics
        assert record["g14_pre_selection"]["stage"] == "pre-selection"
        assert record["g14_post_selection"]["stage"] == "post-selection"
        for stage in ("g14_pre_selection", "g14_post_selection"):
            assert record[stage]["clauses"] == {
                "C1_canonical_vs_brute": "all identical",
                "C2_cached_vs_canonical": "all identical",
                "C3_r2_brute_vs_cached": "all identical"}
        final = record["final_r1_evaluation"]
        assert final["condition"] == "normal"
        assert final["g9_reload_identical"] and final["g11_repeat_identical"]
        assert final["canonical_epoch"] == 22
        assert 0.0 <= final["g21_normalized_exact_accuracy"] <= 1.0
        # all six Phase-1 stages, inside the wall. The cell record is written
        # while result_publication is still open, so the complete set lives in
        # the guard's own outcome record, which is what gates the charge.
        assert set(record["wall"]["guarded_stages_completed"]) == set(
            phase1.CELL_STAGES) - {"result_publication"}
        outcome = json.loads(
            next(installed["governance"].glob("cell_completed_*.json")
                 ).read_text())
        assert outcome["scientific_success"] is True
        assert set(outcome["guarded_stages_completed"]) == set(
            phase1.CELL_STAGES)
        assert outcome["wall_reached_before_finalisation"] is False
        assert outcome["automatic_retry_available"] is False
        assert record["wall"]["per_cell_wall_clock_hours"] == 12.0
        assert record["resource_policy"]["effective_identity_ceiling_hours"] == 40.0
        # atomic publication and reconciliation
        for name in science.REQUIRED_CELL_ARTEFACTS:
            assert paths[name].exists(), name
        for reference in record["checkpoints"].values():
            assert e10.sha256_file(PROJECT_ROOT / reference["path"]) == \
                reference["sha256"]
        assert e10.sha256_file(PROJECT_ROOT / record["per_row"]["path"]) == \
            record["per_row"]["sha256"]
        assert science.reconcile_cell("B4", "train_40k", 0,
                                      record["run"])["reconciled"] is True
        # completed accounting, inside the wall, charged once
        ledger = e10.read_json_mapping(installed["ledgers"]["SPEND_LEDGER"])
        entries = [entry for entry in ledger["entries"]
                   if entry["context"] == "e10_core_cell"]
        assert len(entries) == 1
        assert entries[0]["outcome"] == "completed"
        assert entries[0]["cell"] == ["B4", "train_40k", 0]
        assert entries[0]["gpu_occupancy_ns"] < 12 * 3600 * 1_000_000_000
        assert not list(installed["governance"].glob("cell_terminated*.json"))
        # a completed cell is immutable and is never overwritten
        _must_raise(SystemExit,
                    lambda: science.run_core_cell("B4", "train_40k", 0),
                    "already exists")


# --- B2: the difference in differences is on the full scale -------------------

def _did_matrix(stack: ExitStack, root: Path, frame, masks: dict) -> dict:
    _patched_core_tree(stack, root)
    stack.enter_context(mock.patch.object(
        analysis, "_dev_reference", lambda: _synthetic_reference(frame)))
    ledgers = {"SPEND_LEDGER": root / "spend.json",
               "RETRY_LEDGER": root / "retry.json",
               "GOVERNANCE_LOCK": root / "governance.lock",
               "INITIALIZATION_PENDING": root / "initialization-pending.json",
               "LEDGER_FAILURE_PATH": root / "ledger-failure.json"}
    for name, path in ledgers.items():
        stack.enter_context(mock.patch.object(e10, name, path))
    _write_ledgers(ledgers["SPEND_LEDGER"], ledgers["RETRY_LEDGER"])
    ledger = e10.read_json_mapping(ledgers["SPEND_LEDGER"])
    for arm, scale, seed in FROZEN_ORDER:
        _synthetic_cell_record(arm, scale, seed, masks[(arm, scale)], frame,
                               root)
        ledger["entries"].append(_core_spend_entry(arm, scale, seed))
    e10.atomic_replace_json(ledgers["SPEND_LEDGER"], ledger)
    return analysis.load_matrix()


def test_difference_in_differences_is_on_the_full_scale() -> None:
    """B2 regression: a fixture whose true DiD is independently known to be 0.2.

    B4 is right everywhere. B4r is right on 90 per cent at train_40k and on 70
    per cent at train_250k, so the pretraining effect is 0.1 and 0.3 and the
    difference in differences is exactly 0.2. The halved implementation
    returned 0.1.
    """
    frame = _synthetic_dev_frame()
    index = np.arange(SYNTHETIC_ROWS)
    masks = {
        ("B4", "train_40k"): np.ones(SYNTHETIC_ROWS, dtype=bool),
        ("B4", "train_250k"): np.ones(SYNTHETIC_ROWS, dtype=bool),
        ("B4r", "train_40k"): index % 10 != 0,
        ("B4r", "train_250k"): index % 10 >= 3,
    }
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        matrix = _did_matrix(stack, Path(directory), frame, masks)
        correct = {cell: payload["correct"]["canonical_r1_correct"]
                   for cell, payload in matrix["cells"].items()}
        # Independently computed from the arrays, without touching analysis.py.
        effect_40k = float(correct[("B4", "train_40k", 0)].mean()
                           - correct[("B4r", "train_40k", 0)].mean())
        effect_250k = float(correct[("B4", "train_250k", 0)].mean()
                            - correct[("B4r", "train_250k", 0)].mean())
        assert round(effect_40k, 10) == 0.1
        assert round(effect_250k, 10) == 0.3
        true_did = effect_250k - effect_40k
        assert round(true_did, 10) == 0.2

        definitions = analysis._frozen_contrasts()
        did = analysis._paired_interval(
            matrix, "canonical_r1_correct",
            definitions["pretraining_by_scale_difference_in_differences"][
                "positive"],
            definitions["pretraining_by_scale_difference_in_differences"][
                "negative"])
        assert did["per_seed_differences"] == [0.2, 0.2, 0.2]
        assert abs(did["mean_difference"] - 0.2) < 1e-9
        assert abs(did["mean_difference"] - 0.5 * true_did) > 1e-9

        # The interval is on the same full scale: recompute it here with an
        # explicit two-term statistic and require exact agreement.
        reference = matrix["reference"]

        def statistic(rows):
            return float(np.mean([
                (correct[("B4", "train_250k", seed)][rows].mean()
                 - correct[("B4r", "train_250k", seed)][rows].mean())
                - (correct[("B4", "train_40k", seed)][rows].mean()
                   - correct[("B4r", "train_40k", seed)][rows].mean())
                for seed in e10.CORE_SEEDS]))

        expected = e8a_stats.percentile_interval(e8a_stats.clustered_draws(
            statistic, reference["image_index"], reference["n_images"]))
        assert did["interval_95"] == list(expected)
        assert did["aggregation"].startswith("sum")

        # The three one-element contrasts are numerically identical to a direct
        # computation, so the repair changed nothing outside the composite.
        simple = analysis._paired_interval(
            matrix, "canonical_r1_correct",
            definitions["pretraining_effect_train_250k"]["positive"],
            definitions["pretraining_effect_train_250k"]["negative"])
        assert simple["per_seed_differences"] == [0.3, 0.3, 0.3]
        assert abs(simple["mean_difference"] - effect_250k) < 1e-9
        scale_effect = analysis._paired_interval(
            matrix, "canonical_r1_correct",
            definitions["scale_effect_B4r"]["positive"],
            definitions["scale_effect_B4r"]["negative"])
        assert abs(scale_effect["mean_difference"] - (0.7 - 0.9)) < 1e-9


TESTS = (
    test_frozen_twelve_cell_order_is_unchanged,
    test_b4_b4r_pair_identity_is_provable,
    test_frozen_recipe_identity_matches_the_contract,
    test_frozen_dev_cadence_is_read_from_the_contract,
    test_scheduler_horizon_is_one_hundred_epochs,
    test_optimizer_recipe_is_weight_decay_on_ndim_two_only,
    test_cosine_scheduler_covers_only_the_first_part_of_its_horizon,
    test_training_control_flow_is_the_frozen_budget,
    test_no_early_stopping_anywhere_in_the_pipeline,
    test_epoch_twenty_two_is_the_primary_not_the_best_epoch,
    test_every_scientific_stage_runs_inside_the_guard,
    test_optimizer_steps_only_occur_through_the_guarded_path,
    test_labelled_manifest_contract_is_pinned,
    test_embargoed_target_is_unreachable,
    test_token_store_identity_is_pinned,
    test_g21_scorer_is_used_and_not_reimplemented,
    test_b4_b4r_construction_contract_is_asserted_before_training,
    test_trainable_and_frozen_parameter_counts_are_validated,
    test_atomic_writers_publish_or_leave_nothing,
    test_stale_artefacts_and_completed_cells_fail_closed,
    test_partial_cell_cannot_reconcile_as_complete,
    test_missing_required_artefact_prevents_completion,
    test_first_stage_refuses_without_strict_determinism,
    test_production_entry_enables_determinism_before_the_first_stage,
    test_production_synthetic_cell_reaches_complete,
    test_analysis_refuses_an_incomplete_matrix,
    test_analysis_refuses_a_matrix_that_is_not_pair_preserved,
    test_analysis_reproduces_expected_arithmetic_on_fixtures,
    test_difference_in_differences_is_on_the_full_scale,
    test_analysis_refuses_when_accounting_does_not_reconcile,
    test_analysis_refuses_tampered_per_row_evidence,
    test_twelve_hour_wall_and_forty_hour_ceiling_are_unchanged,
    test_automatic_retry_remains_impossible,
    test_no_e8_frozen_result_is_modified,
    test_scientific_authorization_remains_closed,
    test_no_scientific_cell_has_executed,
    test_core_cell_refuses_before_any_scientific_side_effect,
    test_phase2_pipeline_contract_reports_the_real_state,
)


def run() -> None:
    for test in TESTS:
        test()
    print(f"E10 Phase-2 tests passed ({len(TESTS)} checks)")


if __name__ == "__main__":
    run()
