"""The E10 scientific pipeline for one B4/B4r cell.

Phase 0 established model identity, provenance and the frozen recipe contract.
Phase 1 set the mandatory resource bounds and made them enforceable through a
signed, process-bound cell permit and ``phase1.ScientificCellGuard``. Neither
phase implemented the science: ``run.core_cell`` opened the guard and raised.

This module is that missing implementation. It builds the labelled development
inputs, constructs the frozen 360M readout arm, trains the frozen 22-epoch
budget, evaluates on the pinned E10 development cadence, runs the inherited
G14-FP32 scorer-identity gate, performs the final canonical R1 evaluation of the
epoch-22 checkpoint and publishes one reconciled cell.

Nothing here authorises anything. Every entry point runs behind
``e10_common.assert_core_entry_authorized``, which refuses while
``E10_TRAINING_AUTHORIZED`` is unset, and every scientifically relevant stage
runs inside ``ScientificCellGuard.stage()`` so the 12-hour whole-cell wall
governs setup, training, the development evaluations, the G14 diagnostics, the
final R1 evaluation and publication alike.

Relationship to E8B. The scientific behaviour is the frozen E8B B2/B3 behaviour
at d_lm 960. Pure, parameterised E8B helpers are imported and reused unchanged
(the optimiser and scheduler builders, the canonical FP32 evaluation graph, the
batched R1 scorer, the G14 row construction, the collapse diagnostic, the
readout implementations). The E8B core runner itself is hard-bound to E8B's
authorisation state, gate-halt records, run locks, result tree and identity
ledger, so the parts of it that E10 needs are ported here against E10's own
guard, ledger and artefact tree rather than reused across that boundary. The
deliberate E10 differences are the pinned 12-epoch development cadence, the
absence of the expensive G8 overfit gate (which the frozen Gate-1 per-cell
budget does not fund), and the refusal to resume.

    python -B -m experiments.e10_capacity_360m.run core-cell B4 train_40k 0
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import tokens_data, utils  # noqa: E402
from experiments.e8a_question_encoder import g21_scorer as g21  # noqa: E402
from experiments.e8b_readout_generation import latents as e8b_latents  # noqa: E402
from experiments.e8b_readout_generation import readouts  # noqa: E402
from experiments.e10_capacity_360m import e10_common as e10  # noqa: E402
from experiments.e10_capacity_360m import phase1  # noqa: E402


# --- frozen execution constants ----------------------------------------------

# The canonical primary checkpoint and result. Read from the frozen recipe at
# use; this constant exists so a test can assert the two agree.
CANONICAL_EPOCH = 22
# The pinned E10 development-evaluation cadence, read from the frozen contract
# and never redeclared here.
DEV_EVALUATION_EPOCHS = e10.DEV_EVALUATION_EPOCHS
SCHEDULER_HORIZON_EPOCHS = e10.SCHEDULER_HORIZON_EPOCHS
# One cell writes one durability checkpoint, one canonical checkpoint, one
# secondary checkpoint and one per-row array. The frozen Gate-1 storage
# projection reserves 1 GiB per cell; the same figure gates the start.
CELL_STORAGE_BYTES = 1 * 2 ** 30
CELL_LOCK_DIR = e10.SHARED_STATE_DIR / "e10-cell-locks"
CANONICAL_EVAL_BATCH = 128
G14_CELL_STAGES = ("pre-selection", "post-selection")

# Every artefact a cell must have durably on disk before it may be called
# complete. A cell missing any one of these is never published as a result.
REQUIRED_CELL_ARTEFACTS = (
    "canonical_checkpoint", "secondary_checkpoint", "per_row", "result",
)


class ScientificHalt(AssertionError):
    """A hard E10 scientific gate refused. The cell never completes."""


def _now_stamp() -> str:
    return e10.utc_now().replace("-", "").replace(":", "")


def halt(run_name: str, gate: str, reason: str, detail: dict | None = None):
    """Record a permanent, cell-specific gate halt and stop the cell.

    E10's analogue of E8B's ``gate_halt``, written against E10's own artefact
    tree. A halt record is permanent: it blocks re-entry to the same cell, and
    clearing it is a human decision that needs a fresh authorization, exactly as
    the no-automatic-retry policy requires.
    """
    e10.CORE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = e10.CORE_OUT_DIR / f"HALT_{run_name}_{gate}_{_now_stamp()}.json"
    payload = {
        "schema_version": 1,
        "record_type": "e10_scientific_gate_halt",
        "task_id": e10.PHASE2_TASK_ID,
        "status": "HALTED",
        "utc": e10.utc_now(),
        "binding": e10.binding_record(),
        "run_name": run_name,
        "gate": gate,
        "reason": reason,
        "detail": detail or {},
        "scientific_success": False,
        "automatic_retry_available": False,
        "retry_requires_fresh_authorization": True,
        "clean_test_accessed": False,
    }
    try:
        e10.atomic_write_json(path, payload)
    except FileExistsError:  # pragma: no cover - stamp collision only
        pass
    raise ScientificHalt(f"E10 {gate} HALT [{run_name}]: {reason}")


def sha256_file(path: Path) -> str:
    return e10.sha256_file(Path(path))


# --- the labelled development inputs -----------------------------------------

def manifest_path(key: str) -> Path:
    """The one place an E10 scientific manifest path is formed."""
    if key not in e10.MANIFEST_BASENAMES:
        raise AssertionError(f"{key!r} is not an approved E10 manifest key")
    return e10.V2_DIR / e10.MANIFEST_BASENAMES[key]


def assert_manifest_identity(key: str, run_name: str) -> dict:
    """Pin one manifest by basename, existence and SHA-256, fail-closed.

    The digests were cross-checked against the frozen E8B core-cell records, so
    an E10 cell trained on a rebuilt or substituted manifest cannot be compared
    with anything and stops here instead.
    """
    path = manifest_path(key)
    if not path.is_file():
        halt(run_name, "G15", f"the {key} manifest is missing",
             {"path": str(path)})
    digest = sha256_file(path)
    if digest != e10.MANIFEST_SHA256[key]:
        halt(run_name, "G15",
             f"the {key} manifest is not the frozen E10 input",
             {"path": str(path), "observed_sha256": digest,
              "expected_sha256": e10.MANIFEST_SHA256[key]})
    return {"key": key, "path": str(path), "sha256": digest}


def assert_token_store_identity(run_name: str) -> dict:
    """Pin the cached token stores. Token-store drift invalidates every cell."""
    observed = {}
    for name, expected in e10.TOKEN_STORE_SHA256.items():
        path = tokens_data.TOKEN_DIR / f"{name}.h5"
        if not path.is_file():
            halt(run_name, "G15", f"the {name} store is missing",
                 {"path": str(path)})
        digest = sha256_file(path)
        if digest != expected:
            halt(run_name, "G15",
                 f"the {name} store is not the frozen E10 cache",
                 {"path": str(path), "observed_sha256": digest,
                  "expected_sha256": expected})
        observed[name] = digest
    return observed


def load_vocabulary(run_name: str) -> tuple:
    """The V2 answer vocabulary, pinned by the Phase-0 digest."""
    answers, provenance = g21.load_index_to_answer(e10.VOCAB_PATH)
    if provenance["sha256"] != e10.V2_VOCAB_SHA256:
        halt(run_name, "G18", "V2 answer-vocabulary digest mismatch",
             {"observed": provenance["sha256"],
              "expected": e10.V2_VOCAB_SHA256})
    if provenance["n_classes"] != e10.ANSWER_CACHE_FACTS["answers"]:
        halt(run_name, "G18", "V2 answer-vocabulary size mismatch",
             {"observed": provenance["n_classes"]})
    return answers, provenance


def g1_label_binding(key: str, answers: list, run_name: str) -> dict:
    """G1, ported: assert the answer-string to label-index binding per row.

    A V1-indexed manifest would train on silently wrong targets and score a
    silently wrong metric, so the binding is asserted for every row rather than
    assumed. Duplicate question identifiers are refused in the same pass.
    """
    import pandas as pd

    path = manifest_path(key)
    frame = pd.read_csv(path, dtype={"questionId": str, "imageId": str},
                        keep_default_na=False)
    for column in ("answer", "label", "questionId", "imageId"):
        if column not in frame.columns:
            halt(run_name, "G1", f"the {key} manifest lacks a {column} column",
                 {"columns": list(frame.columns)})
    identifiers = frame["questionId"].astype(str).tolist()
    if len(set(identifiers)) != len(identifiers):
        halt(run_name, "G1", f"the {key} manifest has duplicate questionIds",
             {"rows": len(identifiers), "distinct": len(set(identifiers))})
    labels = frame["label"].to_numpy()
    strings = frame["answer"].astype(str).tolist()
    mismatches, out_of_range = [], []
    for position, (label, answer) in enumerate(zip(labels, strings)):
        index = int(label)
        if index < 0 or index >= len(answers):
            out_of_range.append(position)
            if len(out_of_range) > 5:
                break
        elif answers[index] != answer:
            mismatches.append({"row": position, "label": index,
                               "manifest_answer": answer,
                               "vocabulary_answer": answers[index]})
            if len(mismatches) > 5:
                break
    if mismatches or out_of_range:
        halt(run_name, "G1",
             f"answer-string to label-index binding broken in the {key} "
             f"manifest",
             {"first_mismatches": mismatches[:5],
              "first_out_of_range_rows": out_of_range[:5],
              "rows_checked": int(len(labels))})
    return {"manifest": str(path), "rows_checked": int(len(labels)),
            "distinct_question_ids": len(set(identifiers)),
            "mismatches": 0, "out_of_range": 0,
            "vocabulary_size": len(answers)}


def g15_manifest_record(key: str, dataset_rows: int, run_name: str) -> dict:
    """G15, ported: the row count is MEASURED from the live dataset."""
    expected = e10.EXPECTED_MANIFEST_ROWS[key]
    if dataset_rows != expected:
        halt(run_name, "G15",
             f"the {key} manifest has {dataset_rows} rows, expected {expected}",
             {"measured_rows": dataset_rows, "expected_rows": expected})
    return {**assert_manifest_identity(key, run_name),
            "rows": int(dataset_rows),
            "rows_source": "measured from the live dataset and asserted"}


def build_scientific_loaders(scale: str, seed: int, batch_size: int,
                             run_name: str) -> tuple:
    """The labelled train/dev loaders for one scientific cell.

    Only the three allowlisted development manifests are reachable from here;
    the embargoed clean-test target has no path into this module and the
    E10 source scan proves it.
    """
    stores = tokens_data.TokenStores()
    train_loader, dev_loader = tokens_data.make_token_loaders(
        manifest_path(scale), manifest_path("dev"), stores=stores,
        batch_size=batch_size)
    train_loader.generator.manual_seed(seed)
    steps_per_epoch = len(train_loader)
    if steps_per_epoch != e10.STEPS_PER_EPOCH[scale]:
        halt(run_name, "G15",
             f"steps per epoch {steps_per_epoch} does not match the pinned "
             f"{e10.STEPS_PER_EPOCH[scale]}",
             {"observed": steps_per_epoch})
    record = {
        "train_manifest": g15_manifest_record(
            scale, len(train_loader.dataset), run_name),
        "dev_manifest": g15_manifest_record(
            "dev", len(dev_loader.dataset), run_name),
        "token_stores": assert_token_store_identity(run_name),
        "steps_per_epoch": steps_per_epoch,
        "batch_size": batch_size,
        "clean_test_accessed": False,
    }
    return train_loader, dev_loader, stores, record


# --- B4 / B4r construction ----------------------------------------------------

def assert_model_identity(arm: str, run_name: str) -> dict:
    """Re-verify the pinned frozen-model identity before a cell may train.

    The Phase-0 verification record already pins the repository, the revision,
    the weight digest, the tokenizer blob digests and the configuration; it is
    revalidated live here so a cell can never train against a snapshot that has
    since been replaced.
    """
    try:
        verification = e10.validate_model_verification()
    except AssertionError as error:
        halt(run_name, "G2", f"frozen-model verification failed: {error}", {})
    snapshot = e10.model_snapshot_dir()
    if not snapshot.is_dir():
        halt(run_name, "G2", "the pinned model snapshot is absent",
             {"snapshot": str(snapshot)})
    files = e10.verify_snapshot_files(snapshot)
    return {
        "repository": e10.MODEL_REPO,
        "revision": e10.MODEL_REVISION,
        "arm": arm,
        "identity": e10.model_identity(arm),
        "pretrained": e10.ARMS[arm]["pretrained"],
        "random_init_seed": None if e10.ARMS[arm]["pretrained"] else 20260802,
        "verification_record_status": verification["status"],
        "snapshot_files": files,
        "d_lm": e10.D_LM,
    }


def build_cell_model(arm: str, seed: int, dropout: float, device,
                     run_name: str) -> tuple:
    """Construct the frozen 360M readout arm and its trainable interface.

    Order matters and is the audited G13 order: load and freeze the language
    model FIRST, promote it losslessly to fp32, then seed and construct the
    trunk and the projection with nothing between the seed and the two
    constructors. B4 and B4r therefore share bitwise-identical trainable
    initial states at a given seed, which is what makes the pair a controlled
    pretrained-versus-random contrast rather than two unrelated runs.
    """
    lm, provenance = e10.load_frozen_causal_lm(arm, device=None)
    # The random arm's constructor consumes randomness, so determinism is
    # re-imposed after it, exactly as after every other reseeding point.
    e10.enable_strict_determinism()
    promotion = e10.promote_lm_to_fp32(lm)
    lm = lm.to(device)
    if int(lm.config.hidden_size) != e10.D_LM:
        halt(run_name, "G2",
             f"frozen LM hidden size {lm.config.hidden_size} is not "
             f"{e10.D_LM}", {"arm": arm})
    model, trainable = e10.build_trainable(arm, seed, dropout, lm)
    e10.enable_strict_determinism()
    if any(parameter.requires_grad for parameter in lm.parameters()):
        halt(run_name, "G3", "a frozen LM parameter requires gradients",
             {"arm": arm})
    if not all(parameter.requires_grad for parameter in model.parameters()):
        halt(run_name, "G3", "a trainable parameter is frozen", {"arm": arm})
    frozen_count = int(sum(p.numel() for p in lm.parameters()))
    trainable_count = int(sum(p.numel() for p in model.parameters()))
    if (frozen_count, trainable_count) != (
            e10.MODEL_PARAMETERS, e10.EXPECTED_TRAINABLE_PARAMETERS):
        halt(run_name, "G3",
             "frozen/trainable parameter counts are not the frozen contract",
             {"frozen": frozen_count, "trainable": trainable_count,
              "expected_frozen": e10.MODEL_PARAMETERS,
              "expected_trainable": e10.EXPECTED_TRAINABLE_PARAMETERS})
    model = model.to(device)
    record = {
        **trainable,
        "lm_provenance": dict(provenance),
        "fp32_promotion": promotion,
        "frozen_parameters": frozen_count,
        "trainable_parameters": trainable_count,
        "trainable_init_sha256": _state_dict_digest(model.state_dict()),
    }
    return model, lm, record


def _state_dict_digest(state: dict) -> str:
    parts = []
    for key, value in sorted(state.items()):
        parts.append(f"{key}:" + hashlib.sha256(
            value.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def build_answer_cache(run_name: str) -> tuple:
    """The tokenised answer support, pinned by the Phase-0 cache digest.

    The tokenizer comes from the pinned 360M snapshot, never from the 135M
    helper. Phase 0 established that the two tokenizer artefacts are byte
    identical, but the E10 model identity is the 360M checkpoint and the
    tokenizer it is loaded with has to be that one.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        e10.model_snapshot_dir(), local_files_only=True)
    answers, provenance = load_vocabulary(run_name)
    cache = readouts.build_answer_cache(tokenizer, answers)
    if cache["sha256"] != e10.ANSWER_CACHE_SHA256:
        halt(run_name, "G1_CACHE", "answer-support tokenisation drifted",
             {"observed": cache["sha256"],
              "expected": e10.ANSWER_CACHE_SHA256})
    trie = readouts.build_trie(cache)
    return tokenizer, answers, provenance, cache, trie


# --- the frozen recipe --------------------------------------------------------

def assert_recipe_contract(recipe: dict, arm: str, scale: str, seed: int,
                           run_name: str) -> dict:
    """Verify the executable recipe against the immutable E10 contract.

    The twelve recipe digests were frozen in Phase 0 and reproduced in the
    Phase-1 whole-cell-wall repair record. A cell whose recipe does not hash to
    its frozen digest is not the registered cell and never runs.
    """
    frozen = e10.frozen_recipe_contract()
    key = f"{scale}_seed{seed}"
    expected = frozen["pairs"][key][f"{arm}_full_sha256"]
    observed = e10.recipe_sha256(recipe)
    if observed != expected:
        halt(run_name, "G0",
             "the executable recipe does not match its frozen digest",
             {"observed": observed, "expected": expected})
    if list(frozen["dev_evaluation_epochs"]) != list(DEV_EVALUATION_EPOCHS) \
            or list(recipe["dev_evaluation_epochs"]) != list(
                DEV_EVALUATION_EPOCHS):
        halt(run_name, "G0", "the development-evaluation cadence drifted",
             {"contract": frozen["dev_evaluation_epochs"],
              "recipe": recipe.get("dev_evaluation_epochs"),
              "module": list(DEV_EVALUATION_EPOCHS)})
    problems = []
    if recipe["max_epochs"] != CANONICAL_EPOCH:
        problems.append(f"max_epochs {recipe['max_epochs']}")
    if recipe["early_stopping"] is not False:
        problems.append("early stopping is enabled")
    if recipe["canonical_checkpoint_rule"] != "epoch_22":
        problems.append(f"rule {recipe['canonical_checkpoint_rule']}")
    if recipe["primary_checkpoint_selection"] != "fixed_endpoint":
        problems.append(f"selection {recipe['primary_checkpoint_selection']}")
    if (recipe["lr"], recipe["warmup_frac"], recipe["dropout"]) != (
            3e-4, 0.0, 0.1):
        problems.append("hyperparameters differ from the frozen values")
    if (recipe["batch_size"], recipe["grad_clip"], recipe["weight_decay"]) != (
            128, 1.0, 0.01):
        problems.append("optimisation constants differ from the frozen values")
    if recipe["weight_decay_on"] != "parameters with ndim >= 2 only":
        problems.append("the weight-decay scope differs")
    if recipe["canonical_evaluation_precision"] != "fp32":
        problems.append("the canonical evaluation precision is not fp32")
    if recipe["d_lm"] != e10.D_LM:
        problems.append(f"d_lm {recipe['d_lm']}")
    if recipe["frozen_model_identity"] != e10.model_identity(arm):
        problems.append("the frozen model identity does not match the arm")
    if "wall_clock_halt_hours" in recipe:
        problems.append("the frozen recipe must not carry a resource field")
    if problems:
        halt(run_name, "G0", "frozen recipe violation", {"problems": problems})
    return {
        "recipe_sha256": observed,
        "paired_recipe_sha256": e10.paired_recipe_sha256(recipe),
        "contract_digest_matched": True,
        "inheritance": e10.verify_recipe_inheritance(arm, scale, seed),
        "dev_evaluation_epochs": list(DEV_EVALUATION_EPOCHS),
    }


def make_optimizer(model, lr: float):
    """AdamW with weight decay on ndim >= 2 only (the frozen recipe)."""
    decay, no_decay = [], []
    for _, parameter in model.named_parameters():
        (no_decay if parameter.ndim < 2 else decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def make_scheduler(optimizer, total_steps: int, warmup_frac: float):
    """Cosine after warmup, stepped per optimizer step.

    The horizon is 100 x steps_per_epoch, the inherited v3_01 horizon. Training
    covers only the first 22 epochs of it, so the learning-rate trajectory is
    the pre-result one and is not re-tuned to the shorter budget.
    """
    warmup_steps = int(round(warmup_frac * total_steps))

    def factor(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# --- the canonical FP32 evaluation graph --------------------------------------

CANONICAL_EVAL_DTYPE = torch.float32


def assert_canonical_dtype(tensor, where: str) -> None:
    if tensor.dtype != CANONICAL_EVAL_DTYPE:
        raise AssertionError(
            f"E10 DTYPE GATE FAILED at {where}: canonical scientific "
            f"evaluation requires {CANONICAL_EVAL_DTYPE}, found "
            f"{tensor.dtype}. A prefix computed in bfloat16 and cast to fp32 "
            f"is forbidden; the evaluation graph is fp32 from the trainable "
            f"trunk onward.")


def canonical_prefix(model, lm, images, questions, mask):
    """Build the soft prefix DIRECTLY in fp32, never by casting a bf16 result."""
    if (torch.is_autocast_enabled() or torch.is_autocast_enabled("cuda")
            or torch.is_autocast_enabled("cpu")):
        raise AssertionError(
            "E10 DTYPE GATE FAILED: autocast is active on the canonical "
            "evaluation path; the prefix would be computed in bfloat16")
    model_dtypes = {p.dtype for p in model.parameters()}
    if model_dtypes != {CANONICAL_EVAL_DTYPE}:
        raise AssertionError(
            f"E10 DTYPE GATE FAILED: trainable parameter dtypes are "
            f"{sorted(str(d) for d in model_dtypes)}, not float32 only")
    lm_dtypes = {p.dtype for p in lm.parameters()}
    if lm_dtypes != {CANONICAL_EVAL_DTYPE}:
        raise AssertionError(
            f"E10 DTYPE GATE FAILED: frozen language-model parameter dtypes "
            f"are {sorted(str(d) for d in lm_dtypes)}, not float32 only; "
            f"promote_lm_to_fp32 was not applied")
    assert_canonical_dtype(images, "cached image tokens as stored")
    assert_canonical_dtype(questions, "cached question tokens as stored")
    prefix = model.prefix_embeddings(lm, images, questions, mask)
    assert_canonical_dtype(prefix, "constructed soft prefix")
    return prefix


def assert_canonical_scores(scores) -> None:
    assert_canonical_dtype(scores, "R1 candidate scores")
    if not torch.isfinite(scores).all():
        raise AssertionError(
            "E10 GATE FAILED: non-finite R1 candidate scores on the canonical "
            "evaluation path")


@torch.no_grad()
def canonical_dev_predictions(model, lm, loader, cache: dict, device) -> tuple:
    """The canonical fp32 development pass.

    Returns predictions, labels, the fp32 score matrix and the per-row
    top1-minus-top2 canonical margin. This is the one evaluation that produces
    the reported R1 accuracy and stratum L of the G14 gate.
    """
    from experiments.e8b_readout_generation import training as e8b_training

    model.eval()
    predictions, labels, scores_all = [], [], []
    for images, questions, _, mask, batch_labels in loader:
        prefix = canonical_prefix(model, lm, images.to(device),
                                  questions.to(device), mask.to(device))
        scores = e8b_training.r1_scores_batched(lm, prefix, cache)
        assert_canonical_scores(scores)
        predictions.append(scores.argmax(dim=1).cpu())
        scores_all.append(scores.cpu())
        labels.append(batch_labels)
    matrix = torch.cat(scores_all).numpy().astype(np.float64)
    order = np.argsort(-matrix, axis=1, kind="stable")
    index = np.arange(matrix.shape[0])
    margins = matrix[index, order[:, 0]] - matrix[index, order[:, 1]]
    return (torch.cat(predictions).numpy(), torch.cat(labels).numpy(),
            matrix, margins)


def evaluation_binding(model, predictions, margins, recipe,
                       recipe_hash: str) -> dict:
    """Bind an evaluation's outputs to the exact state that produced them."""
    return {
        "model_state_digest": _state_dict_digest(model.state_dict()),
        "recipe_sha256": recipe_hash,
        "protocol_family": recipe["protocol_family"],
        "canonical_evaluation_precision":
            recipe["canonical_evaluation_precision"],
        "predictions_sha256": hashlib.sha256(
            np.ascontiguousarray(predictions).tobytes()).hexdigest(),
        "margins_sha256": hashlib.sha256(
            np.ascontiguousarray(margins).tobytes()).hexdigest(),
        "n_rows": int(len(predictions)),
    }


# --- G14-FP32: scorer identity ------------------------------------------------

@torch.no_grad()
def g14_fp32_gate(model, lm, dev_dataset, cache: dict, trie: dict, device,
                  stage: str, run_name: str, margins=None, dev_frame=None,
                  binding=None) -> dict:
    """The inherited G14-FP32 gate, ported against E10's halting path.

    One fp32 prefix per row, SHARED by the canonical batched scorer, the
    single-example cached scorer and the single-sequence unpadded uncached
    brute force, so the gate isolates the scorer implementation and never
    confounds it with batch composition. Three hard halts and no numerical
    exemption of any kind; score deltas are diagnostics only.

    Rows come from the frozen P + L + O construction. At the pre-selection stage
    the weights are untrained, so stratum P alone is used.
    """
    from experiments.e8b_readout_generation import training as e8b_training

    if stage not in G14_CELL_STAGES:
        raise AssertionError(f"G14-FP32: unknown stage {stage!r}")
    n_dev = len(dev_dataset)
    if stage == "post-selection":
        if margins is None or len(margins) != n_dev:
            halt(run_name, "G14",
                 "post-selection G14 requires this checkpoint's canonical "
                 "margins for stratum L",
                 {"stage": stage,
                  "margins": None if margins is None else len(margins),
                  "dev_rows": n_dev})
        finite = np.isfinite(np.asarray(margins, dtype=np.float64))
        if not finite.all():
            halt(run_name, "G14",
                 f"{int((~finite).sum())} non-finite canonical margins; a "
                 f"numerical failure must not escape stratum L",
                 {"stage": stage,
                  "non_finite_rows": np.nonzero(~finite)[0][:20].tolist()})
        if binding is None:
            halt(run_name, "G14",
                 "post-selection G14 requires an evaluation binding proving "
                 "the margins belong to the evaluated state", {"stage": stage})
        live = _state_dict_digest(model.state_dict())
        if live != binding["model_state_digest"]:
            halt(run_name, "G14",
                 "the supplied margins do not belong to the model being "
                 "evaluated (model state digest mismatch)",
                 {"stage": stage, "live_digest": live,
                  "binding_digest": binding["model_state_digest"]})
        live_margins = hashlib.sha256(
            np.ascontiguousarray(margins).tobytes()).hexdigest()
        if live_margins != binding["margins_sha256"]:
            halt(run_name, "G14",
                 "the supplied margins do not match the binding's digest",
                 {"stage": stage, "live": live_margins,
                  "binding": binding["margins_sha256"]})
    selection = e8b_training.g14_fp32_validation_rows(
        margins if margins is not None else np.zeros(n_dev), n_dev, stage)
    rows = selection["rows"]
    model.eval()

    deltas_brute, deltas_cached, checked = [], [], 0
    for start in range(0, len(rows), CANONICAL_EVAL_BATCH):
        chunk = rows[start:start + CANONICAL_EVAL_BATCH]
        batch = tokens_data.collate_tokens([dev_dataset[i] for i in chunk])
        prefixes = canonical_prefix(model, lm, batch[0].to(device),
                                    batch[1].to(device), batch[3].to(device))
        batched = e8b_training.r1_scores_batched(lm, prefixes, cache)
        assert_canonical_scores(batched)
        batched_argmax = batched.argmax(dim=1).cpu().tolist()
        for position, row in enumerate(chunk):
            shared = prefixes[position:position + 1]
            assert_canonical_dtype(shared, f"shared prefix, row {row}")
            brute = readouts.r1_brute_force(lm, shared, cache)
            cached = readouts.r1_cached(lm, shared, cache)
            canonical = batched_argmax[position]
            detail = _g14_halt_detail(row, canonical, brute, cached,
                                      batched[position], stage, cache,
                                      margins, dev_frame)
            if brute["argmax"] != canonical:                      # C1
                halt(run_name, "G14",
                     f"C1 canonical-versus-brute R1 argmax disagreement "
                     f"({stage}) at development row {row}", detail)
            if cached["argmax"] != canonical:                     # C2
                halt(run_name, "G14",
                     f"C2 cached-versus-canonical R1 argmax disagreement "
                     f"({stage}) at development row {row}", detail)
            r2_brute = readouts.r2_brute_force(lm, shared, cache, trie)
            r2_cached = readouts.r2_cached(lm, shared, cache, trie)
            if r2_brute != r2_cached:                             # C3
                halt(run_name, "G14_R2",
                     f"C3 R2 brute-versus-cached disagreement ({stage}) at "
                     f"development row {row}",
                     {"stage": stage, "row": int(row),
                      "brute": r2_brute, "cached": r2_cached})
            canon_scores = batched[position].cpu().numpy().astype(np.float64)
            deltas_brute.append(float(np.abs(
                np.asarray(brute["scores"], dtype=np.float64)
                - canon_scores).max()))
            deltas_cached.append(float(np.abs(
                np.asarray(cached["scores"], dtype=np.float64)
                - canon_scores).max()))
            checked += 1

    def quantiles(values):
        array = np.asarray(values, dtype=np.float64)
        return {"max": float(array.max()), "p99": float(np.quantile(array, .99)),
                "p50": float(np.quantile(array, .50))}

    record = {"gate": "G14-FP32", "stage": stage,
              "evaluation_precision": "fp32",
              "rows_checked": checked,
              "clauses": {"C1_canonical_vs_brute": "all identical",
                          "C2_cached_vs_canonical": "all identical",
                          "C3_r2_brute_vs_cached": "all identical"},
              "no_numerical_exemption": True,
              "score_deltas_are_diagnostic_only": True,
              "brute_delta_vs_canonical": quantiles(deltas_brute),
              "cached_delta_vs_canonical": quantiles(deltas_cached)}
    record.update(e8b_training.g14_fp32_strata_record(selection))
    record["evaluated_state_binding"] = binding if binding is not None else {
        "model_state_digest": _state_dict_digest(model.state_dict()),
        "note": "pre-selection stage: stratum P only, no margins consumed"}
    if margins is not None:
        array = np.asarray(margins, dtype=np.float64)
        record["dev_margin_quantiles"] = {
            "min": float(array.min()),
            **{name: float(np.quantile(array, value))
               for name, value in (("p01", .01), ("p05", .05), ("p50", .50))}}
        record["dev_rows_below_margin"] = {
            f"{threshold:g}": int((array < threshold).sum())
            for threshold in (1e-5, 1e-4, 1e-3)}
    return record


def _g14_halt_detail(row, canonical, brute, cached, canon_row, stage, cache,
                     margins, dev_frame=None) -> dict:
    scores = canon_row.cpu().numpy().astype(np.float64)
    order = np.argsort(-scores, kind="stable")
    first, second = int(order[0]), int(order[1])
    detail = {"stage": stage, "row": int(row),
              "canonical_argmax": int(canonical),
              "brute_argmax": int(brute["argmax"]),
              "cached_argmax": int(cached["argmax"]),
              "top2_candidates": {str(first): cache["answers"][first],
                                  str(second): cache["answers"][second]},
              "canonical_scores_top2": [float(scores[first]),
                                        float(scores[second])],
              "canonical_margin": float(scores[first] - scores[second])}
    if margins is not None:
        detail["recorded_margin"] = float(margins[int(row)])
    if dev_frame is not None:
        detail["questionId"] = str(dev_frame.iloc[int(row)]["questionId"])
        detail["imageId"] = str(dev_frame.iloc[int(row)]["imageId"])
    return detail


# --- implementation gates -----------------------------------------------------

def implementation_gates(model, lm, train_loader, cache, device,
                         run_name: str) -> dict:
    """G4-G7 and G12: shapes and finiteness, padding invariance, loss identity,
    gradient isolation, and read-only token stores.

    Ported from the audited E8B implementation with identical arithmetic and
    identical tolerances; only the halting path is E10's. The expensive G8
    tiny-subset overfit gate is deliberately NOT run: the frozen E10 Gate-1
    per-cell budget funds training, twelve development evaluations, the G14
    schedule, a small fixed overhead and one final R1 pass, and nothing else.
    """
    from experiments.e8b_readout_generation import training as e8b_training

    record = {}
    images, questions, _, mask, labels = next(iter(train_loader))
    images, questions, mask = (images.to(device), questions.to(device),
                               mask.to(device))
    answer_ids = [cache["sequences"][int(label)] for label in labels]

    model.eval()
    with torch.no_grad():
        prefix = model.prefix_embeddings(lm, images, questions, mask)
        loss, per_example = model.teacher_forced_loss(
            lm, prefix.to(torch.bfloat16), answer_ids)
    if prefix.shape != (labels.shape[0], 33, e10.D_LM) \
            or not torch.isfinite(loss) \
            or not torch.isfinite(per_example).all():
        halt(run_name, "G4", "non-finite loss or wrong prefix shape",
             {"prefix_shape": list(prefix.shape),
              "expected_width": e10.D_LM,
              "loss_finite": bool(torch.isfinite(loss))})
    record["g4"] = {"prefix_shape": list(prefix.shape), "loss_finite": True,
                    "one_batch_loss": round(float(loss), 4)}

    corrupted = questions.clone()
    corrupted[mask] = 1e4
    with torch.no_grad():
        prefix_corrupt = model.prefix_embeddings(lm, images, corrupted, mask)
        loss_corrupt, _ = model.teacher_forced_loss(
            lm, prefix_corrupt.to(torch.bfloat16), answer_ids)
    delta = abs(float(loss) - float(loss_corrupt))
    if delta >= 1e-4:
        halt(run_name, "G5",
             f"padded-position corruption moved the loss by {delta:.2e}, "
             f"tolerance 1e-4", {"loss_delta": delta, "tolerance": 1e-4})
    record["g5"] = {"loss_delta": delta}

    same_length = [i for i in range(len(answer_ids))
                   if len(answer_ids[i]) == len(answer_ids[0])][:8]
    ids_subset = [answer_ids[i] for i in same_length]
    with torch.no_grad():
        prefix_subset = prefix[same_length].to(torch.bfloat16)
        explicit, _ = model.teacher_forced_loss(lm, prefix_subset, ids_subset)
        reference = e8b_training._labels_path_loss(lm, prefix_subset,
                                                   ids_subset, device)
    g6_delta = abs(float(explicit) - float(reference))
    if g6_delta >= 1e-5:
        halt(run_name, "G6",
             f"explicit loss {float(explicit):.6f} differs from the labels= "
             f"path {float(reference):.6f}",
             {"explicit": float(explicit), "labels_path": float(reference),
              "delta": g6_delta, "tolerance": 1e-5})
    record["g6"] = {"explicit": float(explicit),
                    "labels_path": float(reference), "delta": g6_delta,
                    "note": "equal-length targets, where the per-example and "
                            "per-token reductions coincide"}

    model.train()
    prefix = model.prefix_embeddings(lm, images, questions, mask)
    loss, _ = model.teacher_forced_loss(lm, prefix.to(torch.bfloat16),
                                        answer_ids)
    loss.backward()
    missing = [name for name, parameter in model.named_parameters()
               if parameter.grad is None]
    frozen_hit = any(parameter.grad is not None for parameter in lm.parameters())
    if missing or frozen_hit:
        halt(run_name, "G7", "gradient isolation violated",
             {"trainable_without_grad": missing,
              "frozen_lm_received_grad": bool(frozen_hit)})
    model.zero_grad(set_to_none=True)
    record["g7"] = {"trainable_with_grad": "all", "lm_grads": "none"}

    source = (PROJECT_ROOT / "src" / "tokens_data.py").read_text()
    read_only = all('"r"' in line for line in source.splitlines()
                    if "h5py.File" in line)
    if not read_only:
        halt(run_name, "G12", "tokens_data opens a token store not read-only",
             {"source": "src/tokens_data.py"})
    record["g12"] = {"stores_read_only": True}
    record["g8_overfit_gate"] = {
        "run": False,
        "reason": "not funded by the frozen E10 Gate-1 per-cell budget; the "
                  "budget enumerates training, twelve development "
                  "evaluations, the inherited G14 schedule, a small fixed "
                  "overhead and one final R1 pass",
    }
    return record


# --- durability, never resume -------------------------------------------------

def write_training_state(path: Path, *, model, optimizer, scheduler,
                         epoch: int, global_step: int, recipe_sha256: str,
                         protocol_family: str, history: list) -> str:
    """Atomically persist the in-flight training state for forensics only.

    E10 never resumes. The file exists so that a cell which dies mid-run can be
    inspected, and its presence at the start of a cell is a hard refusal rather
    than an invitation to continue: silently resuming would splice a trajectory
    across two processes with no fresh authorisation, which the no-automatic-
    retry policy forbids.
    """
    state = {
        "role": "DURABILITY AND FORENSICS ONLY - NEVER RESUMED",
        "protocol_family": protocol_family,
        "recipe_sha256": recipe_sha256,
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "history": list(history),
        "utc": e10.utc_now(),
    }
    return _atomic_torch_save(path, state)


def _atomic_torch_save(path: Path, payload: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    with open(temporary, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def _atomic_npz_save(path: Path, **arrays) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
    return sha256_file(path)


# --- the scientific cell ------------------------------------------------------

def _preflight_refusals(arm: str, scale: str, seed: int) -> dict:
    """Cheap, side-effect-free refusals before any GPU or guard work.

    A valid completed cell is never overwritten, a recorded failure or halt is
    never silently retried, and a leftover training state is never resumed.
    """
    paths = e10.core_cell_paths(arm, scale, seed)
    run_name = paths["run_name"]
    if paths["result"].exists():
        raise SystemExit(
            f"E10 CELL REFUSED: {paths['result']} already exists; a completed "
            f"scientific cell is immutable and is never overwritten")
    if paths["failed"].exists():
        raise SystemExit(
            f"E10 CELL REFUSED: a recorded failure exists for {run_name}; "
            f"execution returns to the user for a fresh authorization")
    halts = sorted(e10.CORE_OUT_DIR.glob(f"HALT_{run_name}_*.json")) \
        if e10.CORE_OUT_DIR.exists() else []
    if halts:
        raise SystemExit(
            f"E10 CELL REFUSED: a recorded gate halt exists for {run_name} "
            f"({halts[0].name}); execution returns to the user")
    if paths["resume_checkpoint"].exists():
        raise SystemExit(
            f"E10 CELL REFUSED: a leftover training state exists at "
            f"{paths['resume_checkpoint']}. E10 never resumes: restarting "
            f"this cell needs a fresh explicit authorization, and continuing "
            f"an interrupted trajectory is not authorised by any protocol")
    for stale in ("canonical_checkpoint", "secondary_checkpoint", "per_row"):
        if paths[stale].exists():
            raise SystemExit(
                f"E10 CELL REFUSED: a stale {stale} artefact exists at "
                f"{paths[stale]} with no published result; reconcile it by "
                f"hand before this cell may run")
    free = shutil.disk_usage(PROJECT_ROOT).free
    if free < CELL_STORAGE_BYTES:
        raise SystemExit(
            f"E10 CELL REFUSED: {free} bytes free, {CELL_STORAGE_BYTES} "
            f"required for one cell")
    return {"run_name": run_name, "free_bytes": free,
            "storage_required_bytes": CELL_STORAGE_BYTES}


def identity_headroom_gate(arm: str, scale: str) -> dict:
    """The per-identity ceiling, checked BEFORE any GPU work.

    The projected cost is read verbatim from the frozen Gate-1 record so the
    gate and the published projection can never drift apart. The measured hours
    are charged afterwards by the guard, and that charge is authoritative.
    """
    gate1 = e10.read_json_mapping(e10.GATE1_PATH)
    e10.assert_recorded_binding(gate1, "E10 Gate-1 projection", e10.GATE1_PATH)
    projected = float(
        gate1["per_cell_projections"][scale]["measured_times_1_25_stress_hours"]
    )
    identity = e10.model_identity(arm)
    charged = e10.identity_hours(identity)
    ceiling = e10.effective_identity_ceiling_hours()
    floor = float(config.E10_HEADROOM_FLOOR_HOURS)
    total = charged + projected
    fires = total > (ceiling - floor)
    return {
        "identity": identity,
        "already_charged_hours": charged,
        "projected_cell_hours": projected,
        "projected_total_hours": total,
        "effective_identity_ceiling_hours": ceiling,
        "headroom_floor_hours": floor,
        "fires": bool(fires),
        "basis": "frozen Gate-1 measured_times_1_25_stress_hours",
    }


def _acquire_cell_lock(run_name: str) -> Path:
    """At most one execution of a given cell anywhere on the pool.

    A hard kill leaves this lock held with no owner. Recovery is deliberately
    manual: the lock records host and pid so a human can confirm the process is
    gone. Automatic stale-lock reclamation is not implemented, because a
    wrongly reclaimed lock means two processes training one cell.
    """
    CELL_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    path = CELL_LOCK_DIR / f"{run_name}.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        holder = path.read_text() if path.exists() else "unknown"
        raise SystemExit(
            f"E10 DUPLICATE RUN PREVENTED: {path.name} is held ({holder})"
        ) from None
    os.write(descriptor, json.dumps({
        "host": socket.gethostname(), "pid": os.getpid(),
        "utc": e10.utc_now()}).encode())
    os.close(descriptor)
    return path


def run_core_cell(arm: str, scale: str, seed: int) -> int:
    """Execute one of the twelve frozen scientific cells, end to end.

    The caller has already run ``assert_core_entry_authorized``; it is repeated
    inside the guard and inside accounting, so no path into this function can
    reach scientific work while the core is unauthorised.
    """
    e10.assert_core_entry_authorized(arm, scale, seed)
    preflight = _preflight_refusals(arm, scale, seed)
    run_name = preflight["run_name"]
    lock_path = _acquire_cell_lock(run_name)
    try:
        with phase1.ScientificCellGuard(arm, scale, seed) as guard:
            _execute_cell(guard, arm, scale, seed, run_name, preflight)
    finally:
        lock_path.unlink(missing_ok=True)
    return 0


def _execute_cell(guard, arm: str, scale: str, seed: int, run_name: str,
                  preflight: dict) -> None:
    """The six mandatory scientific stages, each inside the whole-cell wall."""
    paths = e10.core_cell_paths(arm, scale, seed)
    recipe = e10.build_recipe(arm, scale, seed)
    cell_record: dict = {}

    # --- stage 1: setup -------------------------------------------------------
    with guard.stage("setup"):
        headroom = identity_headroom_gate(arm, scale)
        if headroom["fires"]:
            halt(run_name, "G19_IDENTITY",
                 f"the {headroom['identity']} identity has no headroom for "
                 f"this cell", {"identity_gate": headroom})
        e10.assert_no_embargo_reference()
        contract = assert_recipe_contract(recipe, arm, scale, seed, run_name)
        exclusivity = e10.gpu_exclusivity_preflight()
        device = torch.device("cuda")
        e10.reseed_strict(recipe["seed"])
        precision = e10.pin_fp32_precision()
        identity = assert_model_identity(arm, run_name)
        tokenizer, answers, vocabulary, cache, trie = build_answer_cache(
            run_name)
        del tokenizer
        model, lm, construction = build_cell_model(
            arm, recipe["seed"], recipe["dropout"], device, run_name)
        train_loader, dev_loader, stores, data_record = \
            build_scientific_loaders(scale, recipe["seed"],
                                     recipe["batch_size"], run_name)
        label_binding = {
            "train": g1_label_binding(scale, answers, run_name),
            "dev": g1_label_binding("dev", answers, run_name),
        }
        import pandas as pd
        dev_frame = pd.read_csv(manifest_path("dev"),
                                dtype={"questionId": str, "imageId": str},
                                keep_default_na=False)
        torch.cuda.reset_peak_memory_stats()
        gates = implementation_gates(model, lm, train_loader, cache, device,
                                     run_name)
        determinism = e10.assert_strict_determinism()
        cell_record["setup"] = {
            "identity_headroom_gate": headroom,
            "gpu_exclusivity": exclusivity,
            "frozen_recipe": recipe,
            "recipe_contract": contract,
            "model_identity": identity,
            "construction": construction,
            "answer_support": {"sha256": cache["sha256"],
                               "vocabulary": vocabulary,
                               "facts": e10.ANSWER_CACHE_FACTS},
            "data": data_record,
            "g1_label_binding": label_binding,
            "implementation_gates": gates,
            "fp32_precision_pin": precision,
            "determinism_verified_at_use": determinism,
            "storage": {"free_bytes": preflight["free_bytes"],
                        "required_bytes": preflight["storage_required_bytes"]},
        }

    # --- stage 2: G14 pre-selection ------------------------------------------
    with guard.stage("g14_diagnostics"):
        g14_pre = g14_fp32_gate(model, lm, dev_loader.dataset, cache, trie,
                                device, stage="pre-selection",
                                run_name=run_name)
        print(f"[G14-FP32] pre-selection: C1/C2/C3 identity on "
              f"{g14_pre['rows_checked']} rows", flush=True)

    # --- stages 3 and 4: the frozen 22-epoch budget ---------------------------
    optimizer = make_optimizer(model, recipe["lr"])
    e10.assert_optimizer_excludes_lm(optimizer, model, lm)
    steps_per_epoch = data_record["steps_per_epoch"]
    scheduler = make_scheduler(
        optimizer, SCHEDULER_HORIZON_EPOCHS * steps_per_epoch,
        recipe["warmup_frac"])
    # Pair integrity: the paired arms enter the loop on the same global RNG
    # state, so their dropout masks and shuffles differ only through the
    # pretrained-versus-random contrast the pair is meant to isolate.
    e10.reseed_strict(recipe["seed"])
    train_loader.generator.manual_seed(recipe["seed"])
    device_total = torch.cuda.get_device_properties(0).total_memory

    history: list = []
    best_accuracy, best_epoch, best_state = -1.0, -1, None
    step_count = 0
    canonical_margins = canonical_binding = None
    max_epochs = recipe["max_epochs"]
    if max_epochs != CANONICAL_EPOCH:  # pragma: no cover - contract asserted
        halt(run_name, "SCHEDULE", "the frozen budget is not 22 epochs",
             {"max_epochs": max_epochs})

    for epoch in range(1, max_epochs + 1):
        with guard.stage("training"):
            model.train()
            epoch_start = time.time()
            running_loss, seen = 0.0, 0
            for images, questions, _, mask, labels in train_loader:
                optimizer.zero_grad(set_to_none=True)
                answer_ids = [cache["sequences"][int(label)] for label in labels]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prefix = model.prefix_embeddings(
                        lm, images.to(device), questions.to(device),
                        mask.to(device))
                    loss, _ = model.teacher_forced_loss(
                        lm, prefix.to(torch.bfloat16), answer_ids)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               recipe["grad_clip"])
                # The ONLY optimizer gate in E10. It revalidates the signed
                # permit and the whole-cell wall on every single step.
                guard.guarded_optimizer_step(optimizer)
                scheduler.step()
                step_count += 1
                running_loss += float(loss.detach()) * labels.shape[0]
                seen += labels.shape[0]
            train_seconds = time.time() - epoch_start
            if not math.isfinite(running_loss):
                halt(run_name, "NAN",
                     f"non-finite training loss at epoch {epoch}",
                     {"epoch": epoch})

        entry = {"epoch": epoch,
                 "train_loss": round(running_loss / seen, 5),
                 "train_seconds": round(train_seconds, 1),
                 "evaluated": epoch in DEV_EVALUATION_EPOCHS,
                 "canonical_fp32_dev_accuracy": None,
                 "eval_seconds": None}
        # The pinned E10 cadence: twelve of the twenty-two epochs are
        # evaluated, and epoch 22 is always one of them.
        if epoch in DEV_EVALUATION_EPOCHS:
            with guard.stage("development_evaluation"):
                eval_start = time.time()
                predictions, labels_np, _, margins = canonical_dev_predictions(
                    model, lm, dev_loader, cache, device)
                accuracy = float((predictions == labels_np).mean())
                entry["canonical_fp32_dev_accuracy"] = round(accuracy, 5)
                entry["eval_seconds"] = round(time.time() - eval_start, 1)
                if accuracy > best_accuracy:
                    best_accuracy, best_epoch = accuracy, epoch
                    best_state = {key: value.detach().float().cpu().clone()
                                  for key, value in model.state_dict().items()}
                if epoch == CANONICAL_EPOCH:
                    canonical_margins = margins
                    canonical_binding = evaluation_binding(
                        model, predictions, margins, recipe,
                        contract["recipe_sha256"])
        history.append(entry)
        print(f"[{run_name}] epoch {epoch:3d}: loss "
              f"{running_loss / seen:.4f}  "
              f"dev {entry['canonical_fp32_dev_accuracy']}", flush=True)
        write_training_state(
            paths["resume_checkpoint"], model=model, optimizer=optimizer,
            scheduler=scheduler, epoch=epoch, global_step=step_count,
            recipe_sha256=contract["recipe_sha256"],
            protocol_family=recipe["protocol_family"], history=history)
        memory = e10.memory_gate(torch.cuda.max_memory_allocated(),
                                 torch.cuda.max_memory_reserved(),
                                 device_total)
        if memory["fires"]:
            halt(run_name, "MEMORY",
                 f"peak reserved memory {memory['reserved_fraction']:.1%} "
                 f"reached the 80 per cent ceiling",
                 {"memory": memory, "epoch": epoch})
    # No early stopping and no patience: the loop above runs the full frozen
    # budget unconditionally, and there is no break in it.

    evaluated = [row["epoch"] for row in history if row["evaluated"]]
    if evaluated != list(DEV_EVALUATION_EPOCHS):
        halt(run_name, "SCHEDULE",
             "the development-evaluation cadence was not the frozen one",
             {"evaluated": evaluated, "frozen": list(DEV_EVALUATION_EPOCHS)})
    if len(history) != max_epochs or history[-1]["epoch"] != max_epochs:
        halt(run_name, "SCHEDULE",
             f"the fixed budget requires exactly {max_epochs} epochs",
             {"epochs_run": len(history)})
    if canonical_margins is None or canonical_binding is None:
        halt(run_name, "SCHEDULE",
             "epoch 22 produced no canonical evaluation", {})

    canonical_state = {key: value.detach().float().cpu().clone()
                       for key, value in model.state_dict().items()}
    canonical_accuracy = history[-1]["canonical_fp32_dev_accuracy"]

    # --- stage 5: G14 post-selection on the canonical checkpoint --------------
    with guard.stage("g14_diagnostics"):
        g14_post = g14_fp32_gate(model, lm, dev_loader.dataset, cache, trie,
                                 device, stage="post-selection",
                                 run_name=run_name, margins=canonical_margins,
                                 dev_frame=dev_frame, binding=canonical_binding)
        print(f"[G14-FP32] post-selection: C1/C2/C3 identity on "
              f"{g14_post['rows_checked']} rows", flush=True)

    # --- stage 6: the final canonical R1 evaluation ---------------------------
    with guard.stage("final_r1_evaluation"):
        final = final_r1_evaluation(
            model, lm, dev_loader, cache, device, recipe, contract,
            canonical_state, canonical_accuracy, answers, dev_frame, run_name)

    # --- stage 7: publication -------------------------------------------------
    with guard.stage("result_publication"):
        published = publish_cell(
            arm, scale, seed, run_name, recipe, contract, cell_record,
            history=history, canonical_state=canonical_state,
            best_state=best_state, best_epoch=best_epoch,
            best_accuracy=best_accuracy, final=final,
            g14_pre=g14_pre, g14_post=g14_post,
            peak_memory=e10.memory_gate(torch.cuda.max_memory_allocated(),
                                        torch.cuda.max_memory_reserved(),
                                        device_total),
            guard=guard)
    print(f"[E10 CELL DONE] {run_name}: canonical epoch {CANONICAL_EPOCH}, "
          f"dev {final['primary_canonical_fp32_dev_accuracy']:.4f}, "
          f"record {published['result']['sha256'][:12]}", flush=True)


def final_r1_evaluation(model, lm, dev_loader, cache, device, recipe,
                        contract, canonical_state, canonical_accuracy,
                        answers, dev_frame, run_name: str) -> dict:
    """The final canonical R1 evaluation of the epoch-22 checkpoint.

    Three inherited hard checks, then the primary metric:

    G9  reloading the canonical state into a freshly constructed model
        reproduces bit-identical scores;
    G11 repeating the canonical evaluation reproduces identical predictions;
    G10 the prediction distribution has not collapsed.

    The primary result is the epoch-22 checkpoint's canonical fp32 development
    R1 accuracy, and it must reproduce the value the in-loop epoch-22
    evaluation recorded. The best-of-evaluated-epochs figure is retained as a
    clearly labelled secondary diagnostic and never replaces it.
    """
    from experiments.e8b_readout_generation import training as e8b_training

    model.load_state_dict(canonical_state)
    model = model.to(device)
    trunk, projection = e8b_latents.build_trunk_and_projection(
        recipe["seed"], recipe["dropout"], e10.D_LM)
    reloaded = e8b_latents.E8BPrefixModel(trunk, projection)
    e10.enable_strict_determinism()  # the constructor reseeded
    e10.assert_strict_determinism()
    reloaded.load_state_dict(canonical_state)
    reloaded = reloaded.to(device).eval()
    batch = tokens_data.collate_tokens(
        [dev_loader.dataset[i] for i in range(32)])
    with torch.no_grad():
        prefix_a = canonical_prefix(model.eval(), lm, batch[0].to(device),
                                    batch[1].to(device), batch[3].to(device))
        prefix_b = canonical_prefix(reloaded, lm, batch[0].to(device),
                                    batch[1].to(device), batch[3].to(device))
        scores_a = e8b_training.r1_scores_batched(lm, prefix_a, cache)
        scores_b = e8b_training.r1_scores_batched(lm, prefix_b, cache)
    if not torch.equal(scores_a, scores_b):
        halt(run_name, "G9",
             "the reloaded canonical checkpoint does not reproduce identical "
             "scores", {})
    del reloaded, trunk, projection
    torch.cuda.empty_cache()

    predictions_a, labels_np, scores, margins = canonical_dev_predictions(
        model, lm, dev_loader, cache, device)
    predictions_b, _, _, _ = canonical_dev_predictions(
        model, lm, dev_loader, cache, device)
    if not np.array_equal(predictions_a, predictions_b):
        halt(run_name, "G11", "repeated canonical evaluation differs", {})
    primary = float((predictions_a == labels_np).mean())
    if round(primary, 5) != round(float(canonical_accuracy), 5):
        halt(run_name, "SELECTION",
             f"the canonical accuracy {primary:.5f} does not reproduce the "
             f"in-loop epoch-22 value {canonical_accuracy}", {})
    collapse = e8b_training.g10_collapse_record("B4", predictions_a, scores)
    if collapse["fires"]:
        halt(run_name, "G10",
             f"prediction collapse: top-1 {collapse['top1_share']:.3f}, "
             f"entropy {collapse['mean_prediction_entropy_nats']:.3f} nats",
             collapse)

    # The pinned G21 scoring, applied to the classifier-style readout outputs.
    # Nothing is reimplemented here: the project's single approved normaliser
    # and its two metrics are used verbatim.
    scored = g21.score_rows(predictions_a, labels_np, answers)
    raw_correct = np.asarray(scored["raw_correct"], dtype=np.uint8)
    normalized_correct = np.asarray(scored["normalized_correct"],
                                    dtype=np.uint8)
    return {
        "condition": "normal",
        "readout": "R1 length-normalised candidate scoring over the closed "
                   "100-answer support",
        "canonical_epoch": CANONICAL_EPOCH,
        "primary_canonical_fp32_dev_accuracy": round(primary, 5),
        "primary_metric": recipe["primary_metric"],
        "g21_normalized_exact_accuracy": round(
            float(normalized_correct.mean()), 6),
        "g21_raw_exact_accuracy": round(float(raw_correct.mean()), 6),
        "g21_scorer_provenance": g21.scorer_provenance(),
        "denominator": {"rows": int(len(labels_np)),
                        "support": "closed 100-answer V2 vocabulary",
                        "images": int(dev_frame["imageId"].nunique())},
        "g9_reload_identical": True,
        "g11_repeat_identical": True,
        "g10_collapse": collapse,
        "evaluation_binding": evaluation_binding(
            model, predictions_a, margins, recipe, contract["recipe_sha256"]),
        # Fixed-width unicode, not object dtype: the arrays must reload with
        # allow_pickle=False so an independent rescoring never has to trust a
        # pickled artefact.
        "_arrays": {
            "question_id": np.asarray(
                dev_frame["questionId"].astype(str).tolist(), dtype=np.str_),
            "image_id": np.asarray(
                dev_frame["imageId"].astype(str).tolist(), dtype=np.str_),
            "canonical_predictions": predictions_a.astype(np.int16),
            "labels": labels_np.astype(np.int16),
            "raw_correct": raw_correct,
            "normalized_correct": normalized_correct,
            "canonical_margins": margins.astype(np.float64),
        },
    }


def publish_cell(arm, scale, seed, run_name, recipe, contract, cell_record, *,
                 history, canonical_state, best_state, best_epoch,
                 best_accuracy, final, g14_pre, g14_post, peak_memory,
                 guard) -> dict:
    """Publish one cell atomically, then reconcile it before it counts.

    Order matters. Every binary artefact is written and fsynced first, its
    digest is recorded, and the result JSON that references those digests is
    written last through the immutable atomic writer. A cell that dies part way
    leaves binaries with no result record, which the pre-flight refuses to walk
    past on re-entry; it never leaves a result record that points at artefacts
    which are absent or different.
    """
    paths = e10.core_cell_paths(arm, scale, seed)
    e10.CORE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    e10.CORE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    arrays = final.pop("_arrays")

    canonical_sha = _atomic_torch_save(paths["canonical_checkpoint"], {
        "role": "CANONICAL PRIMARY",
        "rule": recipe["canonical_checkpoint_rule"],
        "epoch": CANONICAL_EPOCH,
        "protocol_family": recipe["protocol_family"],
        "recipe_sha256": contract["recipe_sha256"],
        "arm": arm, "scale": scale, "seed": seed,
        "state": canonical_state})
    if best_state is None:  # pragma: no cover - epoch 22 always evaluates
        halt(run_name, "SELECTION", "no evaluated epoch produced a state", {})
    secondary_sha = _atomic_torch_save(paths["secondary_checkpoint"], {
        "role": "SECONDARY DIAGNOSTIC ONLY - best of the twelve evaluated "
                "epochs; never determines the primary comparison, the model "
                "freeze or the clean-test checkpoint",
        "epoch": best_epoch,
        "protocol_family": recipe["protocol_family"],
        "recipe_sha256": contract["recipe_sha256"],
        "arm": arm, "scale": scale, "seed": seed,
        "state": best_state})
    per_row_sha = _atomic_npz_save(paths["per_row"], **arrays)

    checkpoints = {
        "canonical": {"path": _relative(paths["canonical_checkpoint"]),
                      "sha256": canonical_sha, "role": "CANONICAL PRIMARY"},
        "secondary": {"path": _relative(paths["secondary_checkpoint"]),
                      "sha256": secondary_sha,
                      "role": "SECONDARY DIAGNOSTIC ONLY"},
        "training_state": {"path": _relative(paths["resume_checkpoint"]),
                           "sha256": sha256_file(paths["resume_checkpoint"]),
                           "role": "DURABILITY AND FORENSICS ONLY - NEVER "
                                   "RESUMED"},
    }
    record = {
        "metadata": {**utils.run_metadata(seed=seed),
                     "task_id": e10.PHASE2_TASK_ID,
                     "protocol_family": e10.PROTOCOL_FAMILY,
                     "binding": e10.binding_record()},
        "e10_core_cell": {
            "schema_version": 1,
            "record_type": "e10_scientific_cell_result",
            "run": run_name,
            "cell": [arm, scale, seed],
            "identity": e10.model_identity(arm),
            "frozen_order_index": list(e10.CORE_CELLS).index((arm, scale, seed)),
            "protocol_family": recipe["protocol_family"],
            "recipe": recipe,
            "recipe_sha256": contract["recipe_sha256"],
            "paired_recipe_sha256": contract["paired_recipe_sha256"],
            "setup": cell_record["setup"],
            "epochs_run": len(history),
            "resumed": False,
            "early_stopping": False,
            "dev_evaluation_epochs": list(DEV_EVALUATION_EPOCHS),
            "scheduler_horizon_epochs": SCHEDULER_HORIZON_EPOCHS,
            "canonical_checkpoint_rule": recipe["canonical_checkpoint_rule"],
            "canonical_epoch": CANONICAL_EPOCH,
            "primary_canonical_fp32_dev_accuracy":
                final["primary_canonical_fp32_dev_accuracy"],
            "secondary_diagnostic": {
                "label": "SECONDARY DIAGNOSTIC ONLY",
                "best_of_evaluated_epochs": best_epoch,
                "best_dev_accuracy": round(float(best_accuracy), 5),
                "never_primary": True,
            },
            "history": history,
            "g14_pre_selection": g14_pre,
            "g14_post_selection": g14_post,
            "final_r1_evaluation": final,
            "checkpoints": checkpoints,
            "per_row": {"path": _relative(paths["per_row"]),
                        "sha256": per_row_sha,
                        "arrays": sorted(arrays)},
            "peak_memory": peak_memory,
            "wall": {
                "per_cell_wall_clock_hours": guard.policy[
                    "per_cell_wall_clock_hours"],
                "elapsed_seconds_at_publication": guard.elapsed_seconds(),
                "guarded_stages_completed": list(guard.stages_completed),
                "required_stages": list(phase1.CELL_STAGES),
            },
            "resource_policy": guard.policy,
            "automatic_retry_available": False,
            "clean_test_accessed": False,
        },
    }
    result_sha = e10.atomic_write_json(paths["result"], record)
    published = {
        "result": {"path": _relative(paths["result"]), "sha256": result_sha},
        **checkpoints,
        "per_row": {"path": _relative(paths["per_row"]),
                    "sha256": per_row_sha},
    }
    reconcile_cell(arm, scale, seed, run_name)
    return published


def reconcile_cell(arm: str, scale: str, seed: int, run_name: str) -> dict:
    """Prove a published cell is durable, complete and self-consistent.

    Read back from disk, not from memory: every required artefact must exist,
    every recorded digest must match the bytes now on disk, the per-row arrays
    must cover the development denominator, and the result must be the frozen
    recipe's cell. A cell that fails any clause is not complete and the guard
    finalises it as terminated instead of completed.
    """
    paths = e10.core_cell_paths(arm, scale, seed)
    missing = [name for name in REQUIRED_CELL_ARTEFACTS
               if not paths[name].exists()]
    if missing:
        halt(run_name, "RECONCILE",
             f"a published cell is missing required artefacts: {missing}",
             {"missing": missing})
    record = e10.read_json_mapping(paths["result"])
    e10.assert_current_binding(record["metadata"], "E10 cell result")
    cell = record["e10_core_cell"]
    problems = []
    if cell["cell"] != [arm, scale, seed]:
        problems.append("the record names a different cell")
    if cell["recipe_sha256"] != e10.recipe_sha256(
            e10.build_recipe(arm, scale, seed)):
        problems.append("the recorded recipe digest is not the frozen one")
    if cell["canonical_epoch"] != CANONICAL_EPOCH \
            or cell["canonical_checkpoint_rule"] != "epoch_22":
        problems.append("the canonical checkpoint rule is not epoch 22")
    if cell["dev_evaluation_epochs"] != list(DEV_EVALUATION_EPOCHS):
        problems.append("the recorded cadence is not the frozen one")
    if cell["early_stopping"] is not False or cell["epochs_run"] != CANONICAL_EPOCH:
        problems.append("the epoch budget or early-stopping flag is wrong")
    for name, reference in cell["checkpoints"].items():
        path = PROJECT_ROOT / reference["path"]
        if not path.exists() or sha256_file(path) != reference["sha256"]:
            problems.append(f"the {name} checkpoint does not match its digest")
    per_row_path = PROJECT_ROOT / cell["per_row"]["path"]
    if sha256_file(per_row_path) != cell["per_row"]["sha256"]:
        problems.append("the per-row array does not match its digest")
    with np.load(per_row_path, allow_pickle=False) as arrays:
        rows = {name: int(arrays[name].shape[0]) for name in arrays.files}
    expected_rows = e10.EXPECTED_MANIFEST_ROWS["dev"]
    if set(rows.values()) != {expected_rows}:
        problems.append(f"per-row arrays do not all cover {expected_rows} rows")
    stages = set(cell["wall"]["guarded_stages_completed"])
    unguarded = [stage for stage in phase1.CELL_STAGES
                 if stage not in stages and stage != "result_publication"]
    if unguarded:
        problems.append(f"unguarded scientific stages: {unguarded}")
    if problems:
        halt(run_name, "RECONCILE", "a published cell failed reconciliation",
             {"problems": problems})
    return {"reconciled": True, "artefacts": len(REQUIRED_CELL_ARTEFACTS),
            "per_row_rows": expected_rows}


def _relative(path: Path) -> str:
    path = Path(path)
    if path.is_relative_to(PROJECT_ROOT):
        return str(path.relative_to(PROJECT_ROOT))
    return str(path)
