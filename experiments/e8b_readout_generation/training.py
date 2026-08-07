"""E8B search training: the authorised B3/train_40k/seed0 grid points.

Each run is one point of the frozen eight-point section-7.4 grid, under
the complete frozen recipe; only lr, warmup fraction and dropout vary,
and each point carries its own recipe hash so a checkpoint can never
resume under another point's recipe. Grid point 1 additionally served as
the G19 multiplier pilot on 2026-08-06: its measured per-epoch train and
evaluation times, peak memory and storage set the operative section-14
projections. Every core cell, B1, B2, every other arm, scale and seed,
E9, E10, F1 and F2 are refused by the runner; nothing here evaluates,
promotes or selects beyond each run's own preregistered
checkpoint-selection metric (development R1 accuracy).

Gate order per the master protocol: G2/G3 (pinned frozen LM) before
anything; G13 construction order (LM first, then seed, trunk, projection);
G15/G18 manifest, store and vocabulary pinning, with row counts measured
from the live datasets and the G1 answer-to-label binding asserted for
every row; G0 recipe identity; G4-G7 and G12 implementation gates; G8
tiny-subset overfit (halting, principal arm); the binding 64-example G14
BEFORE any checkpoint selection; then the training loop with the 8
GPU-hour operational wall (recorded failure status on halt), complete
19-field resumable checkpoints every epoch, best-on-dev-R1 selection with
patience 10; then G9/G10/G11 on the selected checkpoint, the closing
64-example G14, and the G19 projection and gate evaluation. Every halting
gate writes an atomic JSON record before it exits.

Nothing here reads, resolves or names the embargoed clean-test target.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import tokens_data, utils  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import g21_scorer as g21  # noqa: E402
from experiments.e8b_readout_generation import latents as e8b_latents  # noqa: E402
from experiments.e8b_readout_generation import readouts  # noqa: E402
from experiments.e8b_readout_generation import run as e8b_run  # noqa: E402

V2_DIR = config.DATA_DIR / "v2"
OUT_DIR = e8b_run.OUT_DIR
CHECKPOINT_DIR = OUT_DIR / "checkpoints"

def build_recipe(grid_point: int) -> dict:
    """The complete frozen recipe for one section-7.4 search grid point.

    Only lr, warmup_frac and dropout vary across the eight points; every
    other property is pinned by section 7.4 and is identical in all
    eight recipes. The grid itself lives in run.py and is transcribed
    from the protocol; nothing may be added after results are observed.
    """
    row = next((r for r in e8b_run.SEARCH_GRID
                if r["grid_point"] == grid_point), None)
    if row is None:
        raise AssertionError(
            f"grid point {grid_point} is not one of the eight frozen "
            f"section-7.4 points")
    return {
        "grid_point": row["grid_point"],
        "lr": row["lr"],
        "warmup_frac": row["warmup_frac"],
        "dropout": row["dropout"],
        "objective": "teacher-forced mean token NLL over answer tokens "
                     "plus EOS, per example, then mean over the batch",
        "selection_metric": "development R1 accuracy, EOS included",
        "tie_break": "earliest epoch attaining the best value",
        "max_epochs": 100,
        "patience": 10,
        "batch_size": 128,
        "weight_decay": 0.01,
        "weight_decay_on": "parameters with ndim >= 2 only",
        "grad_clip": 1.0,
        "scheduler": "cosine after warmup, LambdaLR, horizon "
                     "100 x steps_per_epoch, stepped per optimizer step",
        "precision": "bf16 autocast on the training path only; fp32 "
                     "evaluation of the trainable path; the frozen LM "
                     "runs in its pinned bfloat16",
        "wall_clock_halt_hours": 8.0,
        "seed": 0,
        "arm": "B3",
        "scale": "train_40k",
    }


# --- The amended core recipe family (2026-08-07) -----------------------------

CORE_FIXED_HYPER = {"lr": 3e-4, "warmup_frac": 0.0, "dropout": 0.1}

# Projected cost of one cell, used ONLY to check the 35 GPU-hour
# per-identity ceiling BEFORE a cell starts. Sourced verbatim from
# core_resource_projection_20260807.json -> per_cell_hours, so the gate
# and the published projection cannot drift apart. The ceiling is
# re-checked against MEASURED hours after the cell completes, which is
# the authoritative charge.
CORE_CELL_PROJECTED_HOURS = {
    ("B3", "train_40k"): 1.718, ("B3", "train_250k"): 4.226,
    ("B2", "train_40k"): 1.718, ("B2", "train_250k"): 4.226,
    ("B1", "train_40k"): 0.287, ("B1", "train_250k"): 0.751,
}
# One cell writes a resume checkpoint (325.9 MiB), a canonical
# checkpoint (81.5 MiB), a secondary best-of-22 checkpoint and a per-row
# npz. Rounded up to 1 GiB, checked against the space actually free.
CORE_CELL_STORAGE_BYTES = 1 * 2 ** 30
# Finalisation (G9, G11, G10, the canonical passes and G14 post-selection)
# costs roughly 0.5 to 1 h. A cell resumed at the final epoch has already
# done its training, so it is given this bounded allowance ON TOP of the
# 8-hour wall rather than being destroyed for the sake of it. Exceeding
# the allowance still halts with a recorded failure status.
FINALISATION_ALLOWANCE_S = 3600.0
CORE_FIXED_JUSTIFICATION = (
    "retained SOLELY because it was the PRE-RESULT preregistered pilot "
    "and default configuration, which is also the section 7.1 inherited "
    "v3_01 configuration. The eight-point search was permanently "
    "abandoned and NO hyperparameter winner is claimed. No statistic "
    "from grid points 1-3 justifies this choice or supports a "
    "superiority claim in EITHER direction. DISCLOSURE, exploratory and "
    "NON-CANONICAL and SUPERSEDED, offered as support for nothing: this "
    "configuration was also the grid point that showed the highest bf16 "
    "development maximum of the three points that completed. That "
    "observation is invalidated for ranking purposes (the pools it was "
    "computed over differ in length) and the recipe would be frozen at "
    "these values regardless of it, because the freeze is "
    "outcome-independent by construction")


def build_core_recipe(arm: str, scale: str, seed: int) -> dict:
    """The frozen, HASHED, executable recipe for one of the 18 core cells.

    User decision of 2026-08-07 (fixed-22 amendment): the execution rule
    lives INSIDE the hashed recipe. For B2 and B3 that means exactly 22
    training epochs, no early stopping, and the epoch-22 checkpoint as
    the canonical primary. A recipe emitting max_epochs 100 or patience
    10 for an LM arm is invalid by definition. B1 keeps the stored v3_01
    classifier recipe of master protocol section 7.3 verbatim, including
    its own selection rule, because the E8B likelihood treatment never
    applied to B1.

    Every recipe carries the protocol family, both precisions and the
    strict-determinism contract, so its hash separates this family from
    the superseded BF16 search family AND from the earlier core family
    that predated the fixed-22 rule."""
    if arm not in e8b_run.CORE_ARMS:
        raise AssertionError(f"{arm} is not a core arm")
    if scale not in e8b_run.CORE_SCALES:
        raise AssertionError(f"{scale} is not a core scale")
    if seed not in e8b_run.CORE_SEEDS:
        raise AssertionError(f"{seed} is not a core seed")
    common = {
        "protocol_family": e8b_run.PROTOCOL_FAMILY,
        "arm": arm, "scale": scale, "seed": seed,
        "batch_size": 128,
        "grad_clip": 1.0,
        "lr": 3e-4, "warmup_frac": 0.0, "dropout": 0.1,
        "weight_decay": 0.01,
        "weight_decay_on": "parameters with ndim >= 2 only",
        "scheduler": "cosine after warmup, LambdaLR, horizon "
                     "100 x steps_per_epoch, stepped per optimizer step "
                     "(the inherited v3_01 horizon; training covers only "
                     "the FIRST part of it, so the learning-rate "
                     "trajectory is the pre-result one)",
        "training_precision": "bf16 autocast on the training path only; "
                              "fp32 master weights for the trainable "
                              "modules",
        "frozen_lm_instance_dtype": "fp32, promoted losslessly from the "
                                    "pinned bfloat16 values; under bf16 "
                                    "autocast every per-op operand is "
                                    "exactly the pinned bfloat16 value, "
                                    "so the training math equals the "
                                    "pinned-bf16 construction",
        "canonical_evaluation_precision": "fp32",
        "strict_determinism": True,
        "deterministic_backend": {
            "use_deterministic_algorithms": True,
            "warn_only": False,
            "flash_sdp": False,
            "mem_efficient_sdp": False,
            "math_sdp": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cublas_workspace_config": ":4096:8",
            "tf32": "pinned off; canonical fp32 is true IEEE fp32"},
        "deterministic_execution": "STRICT: use_deterministic_algorithms("
                                   "True, warn_only=False); flash and "
                                   "mem-efficient SDPA disabled; math "
                                   "SDPA only; cudnn deterministic, no "
                                   "benchmark; CUBLAS_WORKSPACE_CONFIG "
                                   ":4096:8; TF32 pinned off; enforcement "
                                   "re-imposed after every reseeding "
                                   "point and asserted at the point of "
                                   "use",
        "wall_clock_halt_hours": 8.0,
    }
    if arm == "B1":
        # Master protocol section 7.3: the stored v3_01 classifier
        # recipe, verbatim (section 7.1 table: lr 3e-4, warmup 0.0,
        # dropout 0.1, wd 0.01 on ndim >= 2, batch 128, clip 1.0,
        # patience 10, max 100, cosine over 100 x steps_per_epoch,
        # best-on-dev selection). B1 has no frozen language model.
        common.update({
            "recipe_identifier": "7.3 stored v3_01 classifier recipe",
            "max_epochs": 100,
            "early_stopping": True,
            "patience": 10,
            "primary_checkpoint_selection": "best_dev_accuracy "
                                            "(section 7.3 inherited "
                                            "rule; the fixed-endpoint "
                                            "rule applies to B2/B3 only)",
            "objective": "softmax cross-entropy over the 100-answer "
                         "vocabulary",
            "selection_metric": "development classification accuracy "
                                "under canonical FP32 evaluation",
            "canonical_checkpoint_rule": "best development accuracy, "
                                         "earliest epoch on ties (the "
                                         "v3_01 selection rule)",
            "tie_break": "earliest epoch attaining the best value",
            "frozen_lm_instance_dtype": "not applicable: B1 has no "
                                        "frozen language model",
            "fixed_hyperparameters_justification":
                "inherited verbatim from the stored v3_01 recipe "
                "(section 7.3); the E8B likelihood treatment and the "
                "fixed-22 rule never applied to B1"})
    else:
        common.update({
            "recipe_identifier": "7.4 E8B likelihood recipe, "
                                 "hyperparameters fixed (A3) and "
                                 "execution fixed at 22 epochs "
                                 "(fixed-22 amendment, 2026-08-07)",
            "max_epochs": 22,
            "early_stopping": False,
            "primary_checkpoint_selection": "fixed_endpoint",
            "patience": "disabled - no early stopping; fixed 22-epoch "
                        "budget",
            "canonical_checkpoint_rule": "epoch_22",
            "objective": "teacher-forced mean token NLL over answer "
                         "tokens plus EOS, per example, then mean over "
                         "the batch",
            "primary_metric": "canonical FP32 development R1 accuracy of "
                              "the epoch-22 checkpoint",
            "secondary_diagnostic": "best-of-22 development R1 and its "
                                    "checkpoint, retained ONLY as a "
                                    "clearly labelled secondary "
                                    "diagnostic; it never determines the "
                                    "primary comparison, the model "
                                    "freeze or the clean-test checkpoint",
            "fixed_hyperparameters_justification":
                CORE_FIXED_JUSTIFICATION,
            "epoch_budget_justification":
                "the 22-epoch budget is a DATED POST-DIAGNOSTIC "
                "amendment (2026-08-07), distinct from the "
                "hyperparameters: lr, warmup and dropout were retained "
                "pre-result, while the budget was fixed after the "
                "diagnostic runs showed best epochs 20, 11 and 6 under "
                "patience with unequal lengths 30, 21 and 16. A fixed "
                "budget removes endogenous run-length differences and "
                "the epoch-22 canonical rule removes max-over-epochs "
                "development-selection bias from the primary B3 minus "
                "B2 contrast. 22 coincides with the previously "
                "registered 15/22 resource basis."})
    return common


def core_run_name(arm: str, scale: str, seed: int) -> str:
    """Distinct from the superseded search names, so no path collides."""
    return f"e8b_core_{arm}_{scale}_seed{seed}"


def recipe_sha256(recipe: dict) -> str:
    return hashlib.sha256(
        json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def paired_recipe_sha256(recipe: dict) -> str:
    """A6: B2 and B3 at the same (scale, seed) must share an identical
    recipe apart from the arm label itself. Hashing the recipe with
    `arm` removed makes that pairing provable in the record."""
    stripped = {k: v for k, v in recipe.items() if k != "arm"}
    return hashlib.sha256(
        json.dumps(stripped, sort_keys=True).encode()).hexdigest()


# --- G14-FP32 validation set (frozen 2026-08-07, amendment A7) ---------------

G14_FP32_PINNED = 64        # stratum P, the legacy fixed rows
G14_FP32_LOWEST = 64        # stratum L, the per-checkpoint risk region
G14_FP32_ORDINARY_PER_BAND = 8      # stratum O
G14_FP32_ORDINARY_BANDS = ((1e-3, 1e-2), (1e-2, 1e-1),
                           (1e-1, 1.0), (1.0, float("inf")))
G14_FP32_ORDINARY_SEED = 20260807


def g14_fp32_validation_rows(margins, n_dev: int, stage: str) -> dict:
    """The frozen G14-FP32 sampling rule.

    Stratum P: the 64 legacy pinned rows, fixed and checkpoint
    independent, for continuity with the superseded gate.
    Stratum L: the 64 smallest FP32 canonical margins for THIS
    checkpoint, ties broken by lowest row index. This is the only region
    where an ordering flip is arithmetically possible, and measured tie
    density varies by an order of magnitude between checkpoints, so it
    must adapt per checkpoint.
    Stratum O: 8 rows from each of four ordinary margin bands, drawn with
    a pinned seed, confirming that ordinary margins never disagree.

    At the pre-selection stage the weights are untrained, so L and O are
    not meaningful and stratum P alone is used."""
    pinned = pinned_g14_rows(n_dev)
    if stage == "pre-selection":
        return {"rows": sorted(pinned), "strata": {"P": sorted(pinned)}}
    order = sorted(range(n_dev), key=lambda r: (float(margins[r]), r))
    lowest = order[:G14_FP32_LOWEST]
    ordinary = []
    for index, (low, high) in enumerate(G14_FP32_ORDINARY_BANDS):
        band = [r for r in range(n_dev) if low <= float(margins[r]) < high]
        if not band:
            continue
        rng = np.random.default_rng(G14_FP32_ORDINARY_SEED + index)
        take = rng.choice(np.array(sorted(band)),
                          size=min(G14_FP32_ORDINARY_PER_BAND, len(band)),
                          replace=False)
        ordinary.extend(int(r) for r in take)
    union = sorted(set(pinned) | set(lowest) | set(ordinary))
    return {"rows": union,
            "strata": {"P": sorted(pinned), "L": sorted(lowest),
                       "O": sorted(set(ordinary))}}


def g14_fp32_strata_record(selection: dict) -> dict:
    def digest(rows):
        return hashlib.sha256(json.dumps(sorted(rows)).encode()).hexdigest()
    return {"total_rows": len(selection["rows"]),
            "rows_sha256": digest(selection["rows"]),
            "strata_sizes": {k: len(v)
                             for k, v in selection["strata"].items()},
            "strata_sha256": {k: digest(v)
                              for k, v in selection["strata"].items()},
            "rule": "P = 64 legacy pinned rows; L = 64 lowest FP32 "
                    "canonical margins for this checkpoint; O = 8 rows "
                    "from each of four ordinary margin bands at a pinned "
                    "seed; union deduplicated"}


def run_name_for(grid_point: int) -> str:
    return f"e8b_B3_train_40k_seed0_search{grid_point}"


# Grid point 1, the G19 multiplier pilot, kept as the module default so
# the recorded pilot artefacts stay reproducible from these constants.
RECIPE = build_recipe(1)
RECIPE_SHA256 = recipe_sha256(RECIPE)

VOCAB_SHA256_PIN = ("f92618b2f59939586d5ad79b184a44ed5f3c9d2aaf4d6e10"
                    "f6a3947d90358680")  # G18: the V2 vocabulary
G14_N_EXAMPLES = 64
G14_ROW_SEED = 0
G8_SUBSET = 1000
G8_MAX_EPOCHS = 200
G8_EVAL_EVERY = 10
G8_TARGET = 0.99
WALL_CHECK_EVERY_STEPS = 25

# U1 E8B-min remaining-matrix constants for the G19 projection: 18 core
# runs (B1/B2/B3 x two scales x seeds 0/1/2) of which B3/40k/seed0 comes
# from promotion (U4), 8 search runs of which this pilot is point 1.
def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            hasher.update(block)
    return hasher.hexdigest()


# --- Batched R1 scoring (the checkpoint-selection path) -----------------------

@torch.no_grad()
def r1_scores_batched(lm, prefix, cache: dict) -> torch.Tensor:
    """(B, n_candidates) length-normalised R1 scores in fp32: one batched
    prefix forward, then one incremental forward per multi-token candidate
    over a clone of the shared prefix cache. Candidate order is vocabulary
    order, so argmax ties resolve to the lowest index."""
    prefix_len = prefix.shape[1]
    batch = prefix.shape[0]
    out = lm(inputs_embeds=prefix, use_cache=True)
    past = out.past_key_values
    first = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
    embed = lm.get_input_embeddings()
    scores = torch.empty(batch, len(cache["sequences"]),
                         dtype=torch.float32, device=prefix.device)
    for index, sequence in enumerate(cache["sequences"]):
        target = sequence + [readouts.EOS_ID]
        total = first[:, target[0]].clone()
        if len(target) > 1:
            ids = torch.tensor([target[:-1]],
                               device=prefix.device).expand(batch, -1)
            token_embeds = embed(ids).to(prefix.dtype)
            positions = torch.arange(
                prefix_len, prefix_len + len(target) - 1,
                device=prefix.device).unsqueeze(0).expand(batch, -1)
            step = lm(inputs_embeds=token_embeds,
                      past_key_values=readouts._clone_cache(past),
                      position_ids=positions)
            logprobs = torch.log_softmax(step.logits.float(), dim=-1)
            for offset in range(1, len(target)):
                total = total + logprobs[:, offset - 1, target[offset]]
        scores[:, index] = total / len(target)
    return scores


def mean_prediction_entropy_nats(scores: torch.Tensor) -> float:
    """Master protocol section 11: `H = mean_n( - sum_k p_n[k] * ln
    p_n[k] )` in nats, where `p_n` is the model's predictive distribution
    over the 100 answers for development row `n`. For the R1 readout that
    distribution is the softmax of the same fp32 length-normalised
    candidate scores whose argmax is the recorded prediction, so the
    entropy and the prediction describe one distribution. Accumulated in
    float64 over every development row, matching the E8A implementation
    of the same clause (e8a_common.py:915-918)."""
    probabilities = torch.softmax(scores.double(), dim=-1)
    return float((-(probabilities
                    * torch.log(probabilities.clamp_min(1e-300)))
                  ).sum(dim=-1).mean())


# --- OI6: the canonical FP32 evaluation graph --------------------------------

CANONICAL_EVAL_DTYPE = torch.float32


def assert_canonical_dtype(tensor, where: str) -> None:
    """Fail-closed dtype gate at a scientific evaluation boundary."""
    if tensor.dtype != CANONICAL_EVAL_DTYPE:
        raise AssertionError(
            f"OI6 DTYPE GATE FAILED at {where}: canonical scientific "
            f"evaluation requires {CANONICAL_EVAL_DTYPE}, found "
            f"{tensor.dtype}. A prefix computed in bfloat16 and cast to "
            f"fp32 is forbidden; the evaluation graph must be fp32 from "
            f"the trainable trunk onward.")


def canonical_prefix(model, lm, images, questions, mask):
    """Build the soft prefix DIRECTLY in FP32, never by casting a
    bfloat16 result.

    The cached image and question token values are consumed exactly as
    stored. The trunk and the projection run in fp32 because their
    parameters are fp32 and autocast is refused here; the BOS embedding
    is taken from the fp32-promoted frozen table; the prefix is therefore
    fp32 by construction rather than by conversion. Every precondition is
    asserted fail-closed, so a bfloat16 graph cannot masquerade as fp32."""
    if (torch.is_autocast_enabled() or torch.is_autocast_enabled("cuda")
            or torch.is_autocast_enabled("cpu")):
        raise AssertionError(
            "OI6 DTYPE GATE FAILED: autocast is active on the canonical "
            "evaluation path; the prefix would be computed in bfloat16")
    model_dtypes = {p.dtype for p in model.parameters()}
    if model_dtypes != {CANONICAL_EVAL_DTYPE}:
        raise AssertionError(
            f"OI6 DTYPE GATE FAILED: trainable parameter dtypes are "
            f"{sorted(str(d) for d in model_dtypes)}, not float32 only")
    lm_dtypes = {p.dtype for p in lm.parameters()}
    if lm_dtypes != {CANONICAL_EVAL_DTYPE}:
        raise AssertionError(
            f"OI6 DTYPE GATE FAILED: frozen language-model parameter "
            f"dtypes are {sorted(str(d) for d in lm_dtypes)}, not "
            f"float32 only; promote_lm_to_fp32 was not applied")
    assert_canonical_dtype(images, "cached image tokens as stored")
    assert_canonical_dtype(questions, "cached question tokens as stored")
    prefix = model.prefix_embeddings(lm, images, questions, mask)
    assert_canonical_dtype(prefix, "constructed soft prefix")
    return prefix


def assert_canonical_scores(scores) -> None:
    """dtype AND finiteness: a NaN or Inf entering the selection metric
    would otherwise compare False against every threshold and sail
    through argmax unnoticed."""
    assert_canonical_dtype(scores, "R1 candidate scores")
    if not torch.isfinite(scores).all():
        raise AssertionError(
            "OI6 GATE FAILED: non-finite R1 candidate scores on the "
            "canonical evaluation path")


def model_state_digest(model) -> str:
    """A digest of the exact trainable state being evaluated, so a G14
    record can be proven to belong to the model that produced it."""
    parts = []
    for key, value in sorted(model.state_dict().items()):
        parts.append(f"{key}:" + hashlib.sha256(
            value.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest())
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


@torch.no_grad()
def canonical_dev_predictions(model, lm, loader, cache: dict, device
                              ) -> tuple:
    """The canonical FP32 development pass: predictions, labels, the fp32
    score matrix and the per-row top1-minus-top2 canonical margin.

    This is the ONE evaluation that drives early stopping, best-checkpoint
    selection, the reported R1 accuracy and stratum L of G14-FP32. It is
    fp32 from the trainable trunk onward; nothing here casts to bfloat16."""
    model.eval()
    predictions, labels, scores_all = [], [], []
    for images, questions, _, mask, batch_labels in loader:
        prefix = canonical_prefix(model, lm, images.to(device),
                                  questions.to(device), mask.to(device))
        scores = r1_scores_batched(lm, prefix, cache)
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
    """B-L4: bind an evaluation's outputs to the exact state that
    produced them. A G14 record carrying this binding can be proven to
    belong to its evaluated model; one that cannot is refused."""
    return {
        "model_state_digest": model_state_digest(model),
        "recipe_sha256": recipe_hash,
        "protocol_family": recipe["protocol_family"],
        "canonical_evaluation_precision":
            recipe["canonical_evaluation_precision"],
        "predictions_sha256": hashlib.sha256(
            np.ascontiguousarray(predictions).tobytes()).hexdigest(),
        "margins_sha256": hashlib.sha256(
            np.ascontiguousarray(margins).tobytes()).hexdigest(),
        "n_rows": int(len(predictions)),
        "scorer": {"canonical": "r1_scores_batched at batch 128, fp32",
                   "brute_force": "readouts.r1_brute_force",
                   "cached": "readouts.r1_cached",
                   "source_sha256": hashlib.sha256(
                       (Path(__file__).read_text()
                        + (Path(__file__).parent / "readouts.py").read_text()
                        ).encode()).hexdigest()}}


# --- OI7: G14-FP32, the live gate --------------------------------------------

@torch.no_grad()
def g14_fp32_gate(model, lm, dev_dataset, cache: dict, trie: dict, device,
                  stage: str, run_name: str, margins=None,
                  dev_frame=None, binding=None) -> dict:
    """The reviewed FP32 G14.

    One FP32 prefix per row, SHARED by the canonical batched scorer, the
    single-example cached scorer and the single-sequence unpadded uncached
    brute force, so the gate isolates the scorer implementation and never
    confounds it with batch composition. Three hard halts, no numerical
    exemption of any kind; score deltas are recorded as diagnostics only.

    Rows come from the frozen P + L + O construction: 64 legacy pinned
    rows, the 64 lowest FP32 canonical margins for THIS checkpoint, and 8
    rows from each of four ordinary margin bands. At the pre-selection
    stage the weights are untrained, so stratum P alone is used."""
    n_dev = len(dev_dataset)
    if stage not in ("pre-selection", "post-selection"):
        raise AssertionError(f"G14-FP32: unknown stage {stage!r}")
    if stage == "post-selection":
        if margins is None:
            raise AssertionError(
                "G14-FP32: post-selection requires this checkpoint's "
                "canonical FP32 margins for stratum L")
        if len(margins) != n_dev:
            raise AssertionError(
                f"G14-FP32: margins cover {len(margins)} rows, dev has "
                f"{n_dev}")
        finite = np.isfinite(np.asarray(margins, dtype=np.float64))
        if not finite.all():
            e8b_run.gate_halt(
                run_name, "G14",
                f"{int((~finite).sum())} non-finite canonical margins; a "
                f"numerical failure must not escape stratum L",
                {"stage": stage,
                 "non_finite_rows": np.nonzero(~finite)[0][:20].tolist()})
        # B-L4: the margins driving stratum L must be PROVEN to belong
        # to the model about to be evaluated. Recompute the state digest
        # and the margin digest and refuse on any mismatch.
        if binding is None:
            e8b_run.gate_halt(
                run_name, "G14",
                "post-selection G14 requires an evaluation binding "
                "proving the margins belong to the evaluated state",
                {"stage": stage})
        live = model_state_digest(model)
        if live != binding["model_state_digest"]:
            e8b_run.gate_halt(
                run_name, "G14",
                "the supplied margins do not belong to the model being "
                "evaluated (model state digest mismatch)",
                {"stage": stage, "live_digest": live,
                 "binding_digest": binding["model_state_digest"]})
        live_margins = hashlib.sha256(
            np.ascontiguousarray(margins).tobytes()).hexdigest()
        if live_margins != binding["margins_sha256"]:
            e8b_run.gate_halt(
                run_name, "G14",
                "the supplied margins do not match the binding's margin "
                "digest", {"stage": stage, "live": live_margins,
                           "binding": binding["margins_sha256"]})
    selection = g14_fp32_validation_rows(
        margins if margins is not None else np.zeros(n_dev), n_dev, stage)
    rows = selection["rows"]
    model.eval()

    deltas_brute, deltas_cached, checked = [], [], 0
    # Chunked at the CANONICAL evaluation batch size of 128, per the
    # frozen specification's definition of the canonical scorer.
    for start in range(0, len(rows), 128):
        chunk = rows[start:start + 128]
        batch = tokens_data.collate_tokens([dev_dataset[i] for i in chunk])
        prefixes = canonical_prefix(model, lm, batch[0].to(device),
                                    batch[1].to(device), batch[3].to(device))
        batched = r1_scores_batched(lm, prefixes, cache)
        assert_canonical_scores(batched)
        batched_argmax = batched.argmax(dim=1).cpu().tolist()
        for position, row in enumerate(chunk):
            shared = prefixes[position:position + 1]
            assert_canonical_dtype(shared, f"shared prefix, row {row}")
            brute = readouts.r1_brute_force(lm, shared, cache)
            cached = readouts.r1_cached(lm, shared, cache)
            canonical = batched_argmax[position]
            if brute["argmax"] != canonical:            # C1
                e8b_run.gate_halt(
                    run_name, "G14",
                    f"C1 canonical-versus-brute R1 argmax disagreement "
                    f"({stage}) at development row {row}",
                    _g14_halt_detail(row, canonical, brute, cached,
                                     batched[position], stage, cache,
                                     margins, dev_frame))
            if cached["argmax"] != canonical:           # C2
                e8b_run.gate_halt(
                    run_name, "G14",
                    f"C2 cached-versus-canonical R1 argmax disagreement "
                    f"({stage}) at development row {row}",
                    _g14_halt_detail(row, canonical, brute, cached,
                                     batched[position], stage, cache,
                                     margins, dev_frame))
            r2_brute = readouts.r2_brute_force(lm, shared, cache, trie)
            r2_cached = readouts.r2_cached(lm, shared, cache, trie)
            if r2_brute != r2_cached:                   # C3
                e8b_run.gate_halt(
                    run_name, "G14_R2",
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
        arr = np.asarray(values, dtype=np.float64)
        return {"max": float(arr.max()), "p99": float(np.quantile(arr, .99)),
                "p50": float(np.quantile(arr, .50))}

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
    record.update(g14_fp32_strata_record(selection))
    record["evaluated_state_binding"] = binding if binding is not None else {
        "model_state_digest": model_state_digest(model),
        "note": "pre-selection stage: stratum P only, no margins consumed"}
    if margins is not None:
        arr = np.asarray(margins, dtype=np.float64)
        record["dev_margin_quantiles"] = {
            "min": float(arr.min()),
            **{q: float(np.quantile(arr, v))
               for q, v in (("p01", .01), ("p05", .05), ("p50", .50))}}
        record["dev_rows_below_margin"] = {
            f"{t:g}": int((arr < t).sum())
            for t in (1e-5, 1e-4, 1e-3)}
    return record


def _g14_halt_detail(row, canonical, brute, cached, canon_row, stage,
                     cache, margins, dev_frame=None) -> dict:
    scores = canon_row.cpu().numpy().astype(np.float64)
    order = np.argsort(-scores, kind="stable")
    a, b = int(order[0]), int(order[1])
    detail = {"stage": stage, "row": int(row),
              "canonical_argmax": int(canonical),
              "brute_argmax": int(brute["argmax"]),
              "cached_argmax": int(cached["argmax"]),
              "top2_candidates": {str(a): cache["answers"][a],
                                  str(b): cache["answers"][b]},
              "canonical_scores_top2": [float(scores[a]), float(scores[b])],
              "brute_scores_top2": [float(brute["scores"][a]),
                                    float(brute["scores"][b])],
              "cached_scores_top2": [float(cached["scores"][a]),
                                     float(cached["scores"][b])],
              "canonical_margin": float(scores[a] - scores[b])}
    if margins is not None:
        detail["recorded_margin"] = float(margins[int(row)])
    if dev_frame is not None:
        detail["questionId"] = str(dev_frame.iloc[int(row)]["questionId"])
        detail["imageId"] = str(dev_frame.iloc[int(row)]["imageId"])
    return detail


# --- The binding 64-example G14 gate ------------------------------------------

def pinned_g14_rows(n_dev: int) -> list:
    rng = np.random.default_rng(G14_ROW_SEED)
    return sorted(rng.choice(n_dev, size=G14_N_EXAMPLES,
                             replace=False).tolist())


@torch.no_grad()
def g14_binding_gate(model, lm, dev_dataset, cache: dict, trie: dict,
                     device, stage: str, run_name: str) -> dict:
    """READ-ONLY SUPERSEDED DIAGNOSTIC - NOT THE LIVE G14 GATE.

    Superseded on 2026-08-07 by g14_fp32_gate, which is the gate the
    core cells actually call. This version builds a BF16 prefix and
    scores in bf16, which the FP32 amendment forbids for any scientific
    evaluation, and it has NO caller anywhere in the executable path.
    It is retained only as the historical record of the pre-amendment
    gate. Its name is one word away from the live gate's, so read the
    call site before assuming which one ran.

    Master protocol section 18 G14 at the binding width: on 64 pinned
    development examples, the batched selection scorer AND the
    single-example cached scorer must both agree with the single-sequence
    unpadded uncached brute force on the R1 argmax for every example, and
    the cached R2 walk must equal the brute-force constrained search.
    Any disagreement halts. Score deltas are recorded against the 1e-4
    tolerance, which is informational for the bf16 frozen LM (the fp32
    diagnostic g14_precision_diagnostic.json records why)."""
    rows = pinned_g14_rows(len(dev_dataset))
    model.eval()
    batch = tokens_data.collate_tokens([dev_dataset[i] for i in rows])
    images, questions, _, mask, _ = batch
    prefixes = model.prefix_embeddings(
        lm, images.to(device), questions.to(device), mask.to(device)
    ).to(torch.bfloat16)
    batched = r1_scores_batched(lm, prefixes, cache)
    batched_argmax = batched.argmax(dim=1).cpu().tolist()

    worst_delta = 0.0
    r2_checked = 0
    for position, row in enumerate(rows):
        prefix = prefixes[position:position + 1]
        brute = readouts.r1_brute_force(lm, prefix, cache)
        cached = readouts.r1_cached(lm, prefix, cache)
        if not (brute["argmax"] == cached["argmax"]
                == batched_argmax[position]):
            e8b_run.gate_halt(
                run_name, "G14",
                f"R1 argmax disagreement ({stage}) at pinned row {row}",
                {"stage": stage, "pinned_row": int(row),
                 "brute_argmax": brute["argmax"],
                 "cached_argmax": cached["argmax"],
                 "batched_argmax": batched_argmax[position],
                 "examples_checked_before_failure": position})
        deltas = [abs(a - b) for a, b in zip(
            brute["scores"], batched[position].cpu().tolist())]
        worst_delta = max(worst_delta, max(deltas))
        r2_brute = readouts.r2_brute_force(lm, prefix, cache, trie)
        r2_cached = readouts.r2_cached(lm, prefix, cache, trie)
        if r2_brute != r2_cached:
            e8b_run.gate_halt(
                run_name, "G14_R2",
                f"R2 constrained-walk disagreement ({stage}) at pinned "
                f"row {row}",
                {"stage": stage, "pinned_row": int(row),
                 "brute": r2_brute, "cached": r2_cached,
                 "examples_checked_before_failure": position})
        r2_checked += 1
    return {"stage": stage, "examples": len(rows),
            "pinned_rows_sha256": hashlib.sha256(
                json.dumps(rows).encode()).hexdigest(),
            "r1_argmax_identical_all": True,
            "r2_identical_all": True, "r2_examples": r2_checked,
            "max_abs_score_delta_batched_vs_brute": worst_delta,
            "tolerance_note": "binding clause is argmax identity on all "
                              "64 examples (master protocol sections 18 "
                              "and 20); the 1e-4 score tolerance is "
                              "informational under the bf16 frozen LM, "
                              "per g14_precision_diagnostic.json"}


# --- Implementation gates (G4-G12, G15, G18) ----------------------------------

def implementation_gates(model, lm, train_loader, cache, device,
                         run_name: str) -> dict:
    record = {}
    images, questions, _, mask, labels = next(iter(train_loader))
    images, questions, mask = (images.to(device), questions.to(device),
                               mask.to(device))
    answer_ids = [cache["sequences"][int(label)] for label in labels]

    # G4: one-batch forward, finite outputs, correct shapes.
    model.eval()
    with torch.no_grad():
        prefix = model.prefix_embeddings(lm, images, questions, mask)
        loss, per_example = model.teacher_forced_loss(
            lm, prefix.to(torch.bfloat16), answer_ids)
    if prefix.shape != (labels.shape[0], 33, e8b_latents.D_LM) \
            or not torch.isfinite(loss) \
            or not torch.isfinite(per_example).all():
        e8b_run.gate_halt(run_name, "G4",
                          "non-finite loss or wrong prefix shape",
                          {"prefix_shape": list(prefix.shape),
                           "loss_finite": bool(torch.isfinite(loss))})
    record["g4"] = {"prefix_shape": list(prefix.shape),
                    "loss_finite": True,
                    "one_batch_loss": round(float(loss), 4)}

    # G5: corrupting padded question positions moves the loss < 1e-4.
    corrupted = questions.clone()
    corrupted[mask] = 1e4
    with torch.no_grad():
        prefix_corrupt = model.prefix_embeddings(lm, images, corrupted,
                                                 mask)
        loss_corrupt, _ = model.teacher_forced_loss(
            lm, prefix_corrupt.to(torch.bfloat16), answer_ids)
    delta = abs(float(loss) - float(loss_corrupt))
    if delta >= 1e-4:
        e8b_run.gate_halt(run_name, "G5",
                          f"padded-position corruption moved the loss "
                          f"by {delta:.2e}, tolerance 1e-4",
                          {"loss_delta": delta, "tolerance": 1e-4})
    record["g5"] = {"loss_delta": delta}

    # G6: the explicit indexed loss equals the labels= path within 1e-5,
    # compared on equal-length targets where the two reductions coincide.
    same_length = [i for i in range(len(answer_ids))
                   if len(answer_ids[i]) == len(answer_ids[0])][:8]
    ids_subset = [answer_ids[i] for i in same_length]
    with torch.no_grad():
        prefix_subset = prefix[same_length].to(torch.bfloat16)
        explicit, _ = model.teacher_forced_loss(lm, prefix_subset,
                                                ids_subset)
        hf_loss = _labels_path_loss(lm, prefix_subset, ids_subset, device)
    g6_delta = abs(float(explicit) - float(hf_loss))
    if g6_delta >= 1e-5:
        e8b_run.gate_halt(run_name, "G6",
                          f"explicit loss {float(explicit):.6f} differs "
                          f"from the labels= path {float(hf_loss):.6f}",
                          {"explicit": float(explicit),
                           "labels_path": float(hf_loss),
                           "delta": g6_delta, "tolerance": 1e-5})
    record["g6"] = {"explicit": float(explicit), "labels_path":
                    float(hf_loss), "delta": g6_delta,
                    "note": "equal-length targets, where the per-example "
                            "and per-token reductions coincide"}

    # G7: gradients reach trunk and projection; every LM parameter None.
    model.train()
    prefix = model.prefix_embeddings(lm, images, questions, mask)
    loss, _ = model.teacher_forced_loss(lm, prefix.to(torch.bfloat16),
                                        answer_ids)
    loss.backward()
    missing = [name for name, p in model.named_parameters()
               if p.grad is None]
    frozen_hit = any(p.grad is not None for p in lm.parameters())
    if missing or frozen_hit:
        e8b_run.gate_halt(run_name, "G7",
                          "gradient isolation violated",
                          {"trainable_without_grad": missing,
                           "frozen_lm_received_grad": bool(frozen_hit)})
    model.zero_grad(set_to_none=True)
    record["g7"] = {"trainable_with_grad": "all", "lm_grads": "none"}

    # G12: token stores are opened read-only by the data layer.
    source = (PROJECT_ROOT / "src" / "tokens_data.py").read_text()
    read_only = all('"r"' in line for line in source.splitlines()
                    if "h5py.File" in line)
    if not read_only:
        e8b_run.gate_halt(run_name, "G12",
                          "tokens_data opens a token store not "
                          "read-only",
                          {"source": "src/tokens_data.py"})
    record["g12"] = {"stores_read_only": True}
    return record


@torch.no_grad()
def _labels_path_loss(lm, prefix, answer_ids: list, device):
    """The HuggingFace labels= cross-entropy over the same sequences."""
    targets = [ids + [readouts.EOS_ID] for ids in answer_ids]
    longest = max(len(t) for t in targets)
    if any(len(t) != longest for t in targets):
        raise AssertionError("G6 requires equal-length targets")
    embed = lm.get_input_embeddings()
    token_ids = torch.tensor(targets, device=device)
    token_embeds = embed(token_ids).to(prefix.dtype)
    inputs = torch.cat([prefix, token_embeds], dim=1)
    labels = torch.cat(
        [torch.full((len(targets), prefix.shape[1]), -100,
                    dtype=torch.long, device=device), token_ids], dim=1)
    out = lm(inputs_embeds=inputs, labels=labels)
    return out.loss.float()


def g8_overfit_gate(lm, train_loader, cache, device,
                    run_name: str, arm: str = "B3") -> dict:
    """Tiny-subset overfit. Master protocol section 11 gate treatment:
    HALTING for the principal arms (B1, B3); a RECORDED scientific
    diagnostic, never a halt, for the random control B2, where failing
    to overfit is an expected and informative outcome. Gate settings
    lr 1e-3, dropout 0.0 (not tuned hyperparameters), 1,000 examples,
    target R1 accuracy >= 0.99.

    Internally re-seeds, which downgrades strict determinism; the caller
    MUST re-impose enforcement afterwards (reseed_strict discipline).

    Public module-level optimizer path: carries its own fail-closed
    authorisation rather than relying on its caller."""
    e8b_run.authorize_optimizer_path("core-gate",
                                     f"g8_overfit_gate {run_name}")
    # The gate is itself an optimizer path, so enforcement must hold
    # INSIDE it, not merely be restored for the caller afterwards.
    # utils.set_seed downgrades to warn_only=True and
    # build_trunk_and_projection re-seeds again, so both are re-imposed.
    e8b_run.reseed_strict(0)
    subset = Subset(train_loader.dataset, list(range(G8_SUBSET)))
    loader = DataLoader(subset, batch_size=128,
                        shuffle=True, generator=utils.make_generator(0),
                        collate_fn=tokens_data.collate_tokens)
    trunk, projection = e8b_latents.build_trunk_and_projection(
        0, 0.0, d_lm=e8b_latents.D_LM)
    e8b_latents.apply_projection_scale(projection, trunk,
                                      e8b_run._pretrained_embed_weight())
    model = e8b_latents.E8BPrefixModel(trunk, projection).to(device)
    optimizer = make_optimizer(model, 1e-3)
    e8b_run.enable_strict_determinism()   # build_* re-seeded internally
    gate_determinism = e8b_run.assert_strict_determinism()
    record_determinism = {"verified_at_use": gate_determinism}
    reached, accuracy = None, 0.0    # bound before the loop, as in B1
    for epoch in range(1, G8_MAX_EPOCHS + 1):
        model.train()
        for images, questions, _, mask, labels in loader:
            answer_ids = [cache["sequences"][int(l)] for l in labels]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prefix = model.prefix_embeddings(
                    lm, images.to(device), questions.to(device),
                    mask.to(device))
                loss, _ = model.teacher_forced_loss(
                    lm, prefix.to(torch.bfloat16), answer_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if epoch % G8_EVAL_EVERY == 0 or epoch == G8_MAX_EPOCHS:
            predictions, labels_np, _, _ = canonical_dev_predictions(
                model, lm, loader, cache, device)
            accuracy = float((predictions == labels_np).mean())
            print(f"[G8] epoch {epoch}: subset R1 accuracy "
                  f"{accuracy:.4f}")
            if accuracy >= G8_TARGET:
                reached = (epoch, accuracy)
                break
    del model, trunk, projection, optimizer
    torch.cuda.empty_cache()
    if reached is None:
        if arm == "B2":
            return {"epochs_to_target": None,
                    "subset_accuracy": round(float(accuracy), 5),
                    "target": G8_TARGET, "epochs": G8_MAX_EPOCHS,
                    "gate_treatment": "recorded scientific diagnostic "
                                      "for the random control (section "
                                      "11); NOT a halt",
                    "gate_settings": "lr 1e-3, dropout 0.0, 1000 examples"}
        e8b_run.gate_halt(
            run_name, "G8",
            f"tiny-subset overfit did not reach {G8_TARGET}: subset R1 "
            f"accuracy {accuracy:.4f} after {G8_MAX_EPOCHS} epochs "
            f"(halting for the principal arm {arm})",
            {"subset_accuracy": round(float(accuracy), 5),
             "target": G8_TARGET, "epochs": G8_MAX_EPOCHS,
             "subset_size": G8_SUBSET, "arm": arm})
    return {"epochs_to_target": reached[0],
            "subset_accuracy": round(reached[1], 5),
            "gate_treatment": ("halting" if arm != "B2"
                               else "recorded diagnostic (section 11)"),
            "gate_settings": "lr 1e-3, dropout 0.0, 1000 examples",
            "determinism": record_determinism}


def make_optimizer(model, lr):
    decay, no_decay = [], []
    for _, parameter in model.named_parameters():
        (no_decay if parameter.ndim < 2 else decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def make_scheduler(optimizer, total_steps, warmup_frac):
    warmup_steps = int(round(warmup_frac * total_steps))

    def factor(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = ((step - warmup_steps)
                    / max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# --- G1 and G15: bindings derived from the manifests, never assumed ----------

EXPECTED_ROWS = {"train_40k": 40000, "train_250k": 250000,
                 "dev": 7714}


def g1_label_binding(manifest: Path, answers: list, run_name: str,
                     which: str) -> dict:
    """G1, the answer-string to label-index binding, ASSERTED for every
    row of the manifest rather than assumed. `answers` is the V2
    vocabulary in index order, so answers[label] must equal the row's
    recorded answer string. A V1-indexed manifest would otherwise train
    on silently wrong targets and score a silently wrong metric."""
    import pandas as pd
    frame = pd.read_csv(manifest)
    if "answer" not in frame.columns or "label" not in frame.columns:
        e8b_run.gate_halt(run_name, "G1",
                          f"{which} manifest lacks an answer or label "
                          f"column", {"columns": list(frame.columns)})
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
        e8b_run.gate_halt(
            run_name, "G1",
            f"answer-string to label-index binding broken in the "
            f"{which} manifest",
            {"first_mismatches": mismatches[:5],
             "first_out_of_range_rows": out_of_range[:5],
             "rows_checked": int(len(labels))})
    return {"manifest": str(manifest), "rows_checked": int(len(labels)),
            "mismatches": 0, "out_of_range": 0,
            "vocabulary_size": len(answers)}


def g15_manifest_record(manifest: Path, dataset_rows: int, key: str,
                        run_name: str) -> dict:
    """G15 pins each manifest by path, MEASURED row count and SHA-256.
    The row count is derived from the live dataset and asserted against
    the protocol figure, never recorded as a literal."""
    expected = EXPECTED_ROWS[key]
    if dataset_rows != expected:
        e8b_run.gate_halt(
            run_name, "G15",
            f"{key} manifest has {dataset_rows} rows, expected "
            f"{expected}",
            {"manifest": str(manifest), "measured_rows": dataset_rows,
             "expected_rows": expected})
    return {"path": str(manifest), "rows": int(dataset_rows),
            "rows_source": "measured from the live dataset and asserted",
            "sha256": sha256_file(manifest)}


# --- The 18-cell core matrix (amendment A5, OI2 fixed schedule) ---------------

# OI2 fixes the LM-arm budget only. B1 retains its already-frozen
# section 7.3 classifier recipe, including its own schedule; the user's
# amendment applies the fixed 22-epoch budget to B2 and B3.
CORE_EPOCHS = {"B2": 22, "B3": 22}
CORE_NO_EARLY_STOPPING = ("B2", "B3")
CORE_EPOCH_JUSTIFICATION = (
    "Dated post-diagnostic amendment of 2026-08-07. Exactly 22 epochs, no "
    "patience-based early stopping, for every B2 and B3 core cell. 22 was "
    "chosen because it covers the observed diagnostic best epochs (20, 11 "
    "and 6), because a fixed budget removes the endogenous run-length "
    "differences that made the abandoned grid points incomparable (their "
    "epoch counts were 30, 21 and 16 under patience), and because it "
    "coincides with the previously registered 15/22 resource basis. Every "
    "cell therefore has the SAME 22 evaluation opportunities, so the "
    "maximum-over-epochs statistic is no longer biased by run length.")


def core_epoch_budget(arm: str):
    """22 for the LM arms; B1 keeps its inherited section 7.3 schedule."""
    return CORE_EPOCHS.get(arm)


# --- B1: the section-7.3 classifier path --------------------------------------

class B1Classifier(nn.Module):
    """LatentTrunk plus the 52,836-parameter attention-pool readout: the
    interface-width non-LM control, trained as a 100-way classifier under
    the stored v3_01 recipe (section 7.3). No frozen language model is
    involved anywhere in this arm."""

    def __init__(self, trunk, readout):
        super().__init__()
        self.trunk = trunk
        self.readout = readout

    def forward(self, image_tokens, question_tokens, question_mask):
        """Logits ONLY.

        AttentionPoolReadout returns (logits, attention_weights) so the
        pooling can be inspected. Every B1 consumer -- the loss, the
        dtype gate, the shape gate and the canonical evaluation --
        expects a tensor, so returning the pair unchanged made all six
        B1 cells die on the first forward pass. The weights stay
        reachable through forward_with_weights for diagnostics."""
        logits, _ = self.readout(self.trunk(image_tokens, question_tokens,
                                            question_mask))
        return logits

    def forward_with_weights(self, image_tokens, question_tokens,
                             question_mask):
        """The diagnostic form: (logits, attention_weights)."""
        return self.readout(self.trunk(image_tokens, question_tokens,
                                       question_mask))


@torch.no_grad()
def b1_canonical_predictions(model, loader, device) -> tuple:
    """Canonical FP32 classifier evaluation for B1: fp32 parameters, no
    autocast, fp32 logits. Returns predictions, labels and logits."""
    if (torch.is_autocast_enabled() or torch.is_autocast_enabled("cuda")
            or torch.is_autocast_enabled("cpu")):
        raise AssertionError("OI6 DTYPE GATE FAILED: autocast is active "
                             "on B1's canonical evaluation path")
    dtypes = {p.dtype for p in model.parameters()}
    if dtypes != {CANONICAL_EVAL_DTYPE}:
        raise AssertionError(f"OI6 DTYPE GATE FAILED: B1 parameter "
                             f"dtypes {sorted(str(d) for d in dtypes)}")
    model.eval()
    predictions, labels, logits_all = [], [], []
    for images, questions, _, mask, batch_labels in loader:
        logits = model(images.to(device), questions.to(device),
                       mask.to(device))
        assert_canonical_dtype(logits, "B1 classifier logits")
        if not torch.isfinite(logits).all():
            raise AssertionError("non-finite B1 logits on the canonical "
                                 "evaluation path")
        predictions.append(logits.argmax(dim=1).cpu())
        logits_all.append(logits.cpu())
        labels.append(batch_labels)
    return (torch.cat(predictions).numpy(), torch.cat(labels).numpy(),
            torch.cat(logits_all))


def b1_implementation_gates(model, train_loader, device,
                            run_name: str) -> dict:
    """B1 analogues of G4 and G7: one-batch finiteness and shape, and
    gradient flow to every trainable parameter (there is no frozen
    module to isolate)."""
    images, questions, _, mask, labels = next(iter(train_loader))
    images, questions, mask = (images.to(device), questions.to(device),
                               mask.to(device))
    model.eval()
    with torch.no_grad():
        logits = model(images, questions, mask)
    if logits.shape != (labels.shape[0], 100) \
            or not torch.isfinite(logits).all():
        e8b_run.gate_halt(run_name, "G4",
                          "non-finite B1 logits or wrong shape",
                          {"shape": list(logits.shape)})
    record = {"g4": {"logit_shape": list(logits.shape),
                     "finite": True}}
    model.train()
    loss = F.cross_entropy(model(images, questions, mask),
                           labels.to(device))
    loss.backward()
    missing = [name for name, p in model.named_parameters()
               if p.grad is None]
    if missing:
        e8b_run.gate_halt(run_name, "G7",
                          "gradients missing on trainable parameters",
                          {"missing": missing})
    model.zero_grad(set_to_none=True)
    record["g7"] = {"trainable_with_grad": "all", "frozen_modules": "none"}
    return record


def b1_overfit_gate(train_loader, device, run_name: str) -> dict:
    """B1's G8: tiny-subset overfit under the gate settings (lr 1e-3,
    dropout 0.0, 1,000 examples, target 0.99). HALTING for B1 (section
    11). Re-seeds internally; the caller must re-impose strict
    determinism afterwards.

    Public module-level optimizer path: carries its own fail-closed
    authorisation."""
    e8b_run.authorize_optimizer_path("core-gate",
                                     f"b1_overfit_gate {run_name}")
    e8b_run.reseed_strict(0)   # see g8_overfit_gate: enforcement INSIDE
    subset = Subset(train_loader.dataset, list(range(G8_SUBSET)))
    loader = DataLoader(subset, batch_size=128, shuffle=True,
                        generator=utils.make_generator(0),
                        collate_fn=tokens_data.collate_tokens)
    trunk, readout = e8b_run.build_arm("B1", 0, 0.0, None)
    model = B1Classifier(trunk, readout).to(device)
    optimizer = make_optimizer(model, 1e-3)
    e8b_run.enable_strict_determinism()   # build_arm re-seeded internally
    gate_determinism = e8b_run.assert_strict_determinism()
    record_determinism = {"verified_at_use": gate_determinism}
    reached, accuracy = None, 0.0
    for epoch in range(1, G8_MAX_EPOCHS + 1):
        model.train()
        for images, questions, _, mask, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(
                    model(images.to(device), questions.to(device),
                          mask.to(device)), labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if epoch % G8_EVAL_EVERY == 0 or epoch == G8_MAX_EPOCHS:
            predictions, labels_np, _ = b1_canonical_predictions(
                model, loader, device)
            accuracy = float((predictions == labels_np).mean())
            print(f"[G8/B1] epoch {epoch}: subset accuracy "
                  f"{accuracy:.4f}")
            if accuracy >= G8_TARGET:
                reached = (epoch, accuracy)
                break
    del model, trunk, readout, optimizer
    torch.cuda.empty_cache()
    if reached is None:
        e8b_run.gate_halt(run_name, "G8",
                          f"B1 tiny-subset overfit did not reach "
                          f"{G8_TARGET}: {accuracy:.4f} after "
                          f"{G8_MAX_EPOCHS} epochs (halting for B1)",
                          {"subset_accuracy": round(accuracy, 5)})
    return {"epochs_to_target": reached[0],
            "subset_accuracy": round(reached[1], 5),
            "gate_treatment": "halting",
            "gate_settings": "lr 1e-3, dropout 0.0, 1000 examples",
            "determinism": record_determinism}


# --- The executable 18-cell core trainer --------------------------------------

def train_core_cell(arm: str, scale: str, seed: int) -> int:
    """One of the 18 final core cells under the amended protocol.

    Fail-closed preamble: authorisation, cell membership, immutability of
    any prior result or halt for this cell, the clean-test embargo scan,
    GPU exclusivity and the NFS cell lock. Only then does the locked body
    run. Scientific training stays refused until the user flips the
    authorisation state to 'core-matrix-approved'."""
    if e8b_run.TRAINING_AUTHORIZED != "core-matrix-approved":
        sys.exit(f"E8B CORE TRAINING IS NOT AUTHORISED: the recorded "
                 f"authorisation state is "
                 f"{e8b_run.TRAINING_AUTHORIZED!r}; core execution "
                 f"requires a separate explicit user approval")
    if (arm, scale, seed) not in e8b_run.CORE_CELLS:
        sys.exit(f"E8B CORE TRAINING REFUSED: {arm}/{scale}/seed{seed} "
                 f"is not one of the 18 core cells")
    # G17: the embargo scan over every E8B source, executed, not assumed.
    for source in sorted(Path(__file__).parent.glob("*.py")):
        e8a.assert_no_clean_test_path(source.read_text(), source.name)

    recipe = build_core_recipe(arm, scale, seed)
    run_name = core_run_name(arm, scale, seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUT_DIR / f"{run_name}.json"
    if result_path.exists():
        sys.exit(f"{result_path} already exists; E8B records are "
                 f"immutable and a core cell is never repeated "
                 f"automatically")
    if (OUT_DIR / f"{run_name}_FAILED.json").exists():
        sys.exit("a recorded failure status exists for this cell; "
                 "execution returns to the user")
    # An unrepaired ledger failure anywhere means every per-identity
    # gate from here on under-counts, so no further cell may start until
    # a human repairs it. Without this the warning was written and never
    # read, and the ceiling could be defeated in silence.
    ledger_failures = sorted(OUT_DIR.glob("LEDGER_FAILURE_*.json"))
    if ledger_failures:
        sys.exit(
            f"UNREPAIRED LEDGER FAILURE: {ledger_failures[0].name} "
            f"records GPU hours that were never charged, so the "
            f"per-identity ceiling under-counts and cannot be trusted. "
            f"Repair the ledger by hand, then remove the record. "
            f"Execution returns to the user.")
    exhausted = sorted(OUT_DIR.glob("IDENTITY_EXHAUSTED_*.json"))
    for marker in exhausted:
        if marker.stem.endswith(e8b_run.model_identity(arm)):
            sys.exit(
                f"IDENTITY EXHAUSTED: {marker.name} records that the "
                f"{e8b_run.model_identity(arm)} identity has crossed "
                f"its GPU-hour ceiling. No further cell on this "
                f"identity may start. Execution returns to the user.")
    halts = sorted(OUT_DIR.glob(f"HALT_{run_name}_*.json"))
    if halts:
        sys.exit(f"a recorded gate halt exists for this cell "
                 f"({halts[0].name}); execution returns to the user")

    # The 35 GPU-hour per-model-identity ceiling, checked BEFORE any GPU
    # work: the projected cost of this cell is added to everything
    # already charged to the same frozen-model identity. Previously the
    # ceiling existed only as a constant inside a projection helper, so
    # nothing would have stopped cell 5 of 6 from crossing it.
    projected = CORE_CELL_PROJECTED_HOURS[(arm, scale)]
    identity_gate = e8b_run.per_identity_gate(
        arm, projected, cell=(arm, scale, seed))
    if identity_gate["fires"]:
        e8b_run.gate_halt(run_name, "G19_IDENTITY",
                          f"the 35 GPU-hour ceiling for the "
                          f"{identity_gate['identity']} model identity "
                          f"would be exceeded: "
                          f"{identity_gate['already_charged_hours']} h "
                          f"already charged plus {projected} h projected "
                          f"for this cell",
                          {"identity_gate": identity_gate})
    # The 180 GPU-hour core ceiling, executable for the same reason the
    # 35-hour one now is: it was a constant with no call site.
    remaining = e8b_run.remaining_core_gate(
        *e8b_run.core_remaining_hours())
    if remaining["fires"]:
        e8b_run.gate_halt(run_name, "G19_CORE",
                          f"the 180 GPU-hour core ceiling would be "
                          f"exceeded: "
                          f"{remaining['expected_remaining_hours']} h "
                          f"expected remaining",
                          {"remaining_core_gate": remaining})
    storage = e8b_run.storage_gate(CORE_CELL_STORAGE_BYTES, OUT_DIR)
    if storage["fires"]:
        e8b_run.gate_halt(run_name, "G19_STORAGE",
                          f"this cell needs {storage['required_gib']} GiB "
                          f"and only {storage['free_gib']} GiB are free",
                          {"storage_gate": storage})

    from experiments.e8a_question_encoder import reinfer_g21
    reinfer_g21.assert_gpu_exclusive(f"e8b core {run_name}")
    lock_path = e8b_run.acquire_run_lock(arm, scale, seed)
    print(f"[LOCK] {lock_path.name} acquired")
    # A SIGKILL leaves this lock held with no owner. Recovery is
    # DELIBERATELY manual: the lock records host and pid, so a human can
    # confirm the process is gone before removing it. Automatic stale
    # -lock reclamation is not implemented, because a wrongly reclaimed
    # lock means two processes training one cell.
    started = time.time()
    try:
        return _train_core_locked(
            arm, scale, seed, recipe, run_name, result_path,
            started, identity_gate, storage, remaining)
    finally:
        # H-1: charge THIS PROCESS's hours whatever happened. A cell
        # that halts at a gate or dies mid-epoch burned those hours just
        # as surely as one that completed, and a completion-path-only
        # charge let them vanish from the ledger -- which would have let
        # the 35-hour ceiling green-light the next cell on an identity
        # that had already spent the headroom. Charging per process is
        # what makes this safe to run on every exit path without
        # double-counting a resume.
        #
        # M-1: this block must never destroy the failure that brought us
        # here. read_spend_ledger is deliberately fail-closed and raises
        # `from None`, so an unreadable ledger would otherwise replace a
        # CUDA OOM with a JSONDecodeError and leave the cell lock held.
        # The lock is released FIRST, and every accounting step is
        # contained, with its own failure recorded rather than raised.
        try:
            # missing_ok suppresses only FileNotFoundError; on the
            # shared filesystem ESTALE and EIO propagate, so the unlink
            # belongs INSIDE the containment. Outside it, one stale NFS
            # handle would leak the cell lock, skip the charge and
            # replace the real failure all at once.
            lock_path.unlink(missing_ok=True)
            e8b_run.charge_identity_hours(
                arm, scale, seed, (time.time() - started) / 3600)
            after = e8b_run.per_identity_gate(arm, 0.0)
        except BaseException as accounting_error:   # noqa: BLE001
            # Recorded, never raised: losing the accounting is bad, but
            # masking why the cell actually stopped is worse.
            # NOT a gate halt. record_gate_halt writes
            # HALT_{run_name}_*.json, and train_core_cell refuses any
            # cell for which such a file exists -- so recording an
            # accounting failure that way would turn a transient NFS
            # hiccup into a permanent block on a perfectly resumable
            # trajectory, at 4.2 h a restart against 0.6 h of headroom.
            # This is an operational warning about the LEDGER, not a
            # scientific halt about the CELL, and it is filed as such.
            e8b_run.record_ledger_failure(
                run_name, "the GPU-hour ledger could not be updated "
                          "after this process; the hours it burned are "
                          "NOT recorded and the identity ceiling is "
                          "under-counted until a human repairs the "
                          "ledger",
                {"error": repr(accounting_error)})
            print(f"[LEDGER] FAILED to charge this process: "
                  f"{accounting_error!r}; a LEDGER_FAILURE record was "
                  f"written (NOT a halt, so this cell can still "
                  f"resume) and the original failure, if any, is "
                  f"preserved")
        else:
            print(f"[LEDGER] {after['identity']} identity now at "
                  f"{after['already_charged_hours']} h of "
                  f"{after['ceiling_hours']} h "
                  f"({after['headroom_hours']} h headroom)")
            if after["fires"]:
                # The ceiling is about the IDENTITY, not this cell. This
                # cell may have completed correctly and written a valid
                # result, so it is never stamped FAILED; what is
                # recorded is that the identity is exhausted and the
                # NEXT cell must not start. The next cell's own pre-gate
                # enforces that independently.
                e8b_run.record_identity_exhausted(
                    run_name, after,
                    f"the {after['identity']} identity has EXCEEDED its "
                    f"{after['ceiling_hours']} GPU-hour ceiling. This "
                    f"cell's own result stands or falls on its own "
                    f"record; no further cell may start on this "
                    f"identity without returning to the user.")
                print(f"[G19_IDENTITY] the {after['identity']} identity "
                      f"is at {after['already_charged_hours']} h of "
                      f"{after['ceiling_hours']} h. Recorded; no "
                      f"further cell on this identity may start.")
                # Exit non-zero ONLY if nothing is already propagating:
                # a sys.exit inside finally would swallow the real
                # failure.
                if sys.exc_info()[0] is None:
                    sys.exit(f"G19_IDENTITY: the {after['identity']} "
                             f"identity is exhausted; execution stops "
                             f"and returns to the user")


def _train_core_locked(arm, scale, seed, recipe, run_name, result_path,
                       started, identity_gate, storage,
                       remaining) -> int:
    # Independent fail-closed authorisation. This helper is private by
    # naming convention only and is callable by direct import, so it must
    # NOT rely on train_core_cell's wrapper checks.
    authorisation = e8b_run.authorize_optimizer_path(
        "core-cell", f"_train_core_locked {run_name}")
    device = torch.device("cuda")
    recipe_hash = recipe_sha256(recipe)
    fingerprint = e8b_run.environment_fingerprint()
    if fingerprint["gpu"] != "NVIDIA RTX 4000 Ada Generation" \
            or not fingerprint["python"].startswith("3.12"):
        e8b_run.gate_halt(run_name, "FINGERPRINT",
                          "environment fingerprint mismatch",
                          {"fingerprint": fingerprint})

    # Strict determinism, imposed AFTER the first seeding and re-imposed
    # after every later reseeding point (utils.set_seed unconditionally
    # downgrades warn_only, and build_arm / random-LM loading / G8 all
    # reseed internally).
    e8b_run.reseed_strict(recipe["seed"])
    precision = e8b_run.pin_fp32_precision()

    # G2/G3 and the frozen-model identity.
    lm = lm_provenance = identity = parity = None
    if arm in ("B2", "B3"):
        pretrained = arm == "B3"
        lm, lm_provenance = e8b_run.load_frozen_causal_lm(pretrained,
                                                          device=None)
        e8b_run.enable_strict_determinism()  # random load reseeds
        identity = e8b_run.promote_lm_to_fp32(lm)
        lm = lm.to(device)
        # OI5 pair integrity: compare against the counterpart arm's
        # buffers, loaded briefly on CPU. Persistent AND non-persistent.
        counterpart, counterpart_prov = e8b_run.load_frozen_causal_lm(
            not pretrained, device=None)
        e8b_run.enable_strict_determinism()
        parity = e8b_run.assert_lm_buffer_parity(
            lm_provenance["buffer_inventory"],
            counterpart_prov["buffer_inventory"],
            label_a=arm, label_b=("B2" if pretrained else "B3"))
        del counterpart
    e8b_run.assert_strict_determinism()
    # Peak memory is measured across the WHOLE cell. Resetting after the
    # gates would have excluded the one combination never measured
    # elsewhere: the fp32-promoted LM with a non-autocast G7 backward.
    torch.cuda.reset_peak_memory_stats()

    tokenizer = e8a.load_tokenizer() if arm != "B1" else None
    answers, vocabulary = g21.load_index_to_answer(
        V2_DIR / "answer_vocab_v2.json")
    if vocabulary["sha256"] != VOCAB_SHA256_PIN:
        e8b_run.gate_halt(run_name, "G18", "V2 vocabulary hash mismatch",
                          {"observed": vocabulary["sha256"]})
    cache = trie = None
    if arm != "B1":
        cache = readouts.build_answer_cache(tokenizer, answers)
        trie = readouts.build_trie(cache)

    # G13 construction order: seed, trunk, projection/readout. build_arm
    # reseeds internally, so enforcement is re-imposed afterwards.
    if arm == "B1":
        trunk, readout = e8b_run.build_arm("B1", recipe["seed"],
                                           recipe["dropout"], None)
        model = B1Classifier(trunk, readout)
        scale_record = {"not_applicable": "B1 has no projection scale"}
    else:
        model, scale_record = e8b_run.build_arm(arm, recipe["seed"],
                                                recipe["dropout"], lm)
    e8b_run.enable_strict_determinism()
    trainable_init_sha256 = e8a.sha256_state_dict(model.state_dict())
    model = model.to(device)

    # G15 and G1, derived from the live inputs.
    train_manifest = V2_DIR / f"{scale}.csv"
    dev_manifest = V2_DIR / "dev.csv"
    stores = tokens_data.TokenStores()
    train_loader, dev_loader = tokens_data.make_token_loaders(
        train_manifest, dev_manifest, stores=stores,
        batch_size=recipe["batch_size"])
    train_loader.generator.manual_seed(recipe["seed"])
    steps_per_epoch = len(train_loader)
    if steps_per_epoch != e8b_run.STEPS_PER_EPOCH[scale]:
        e8b_run.gate_halt(run_name, "G15",
                          f"steps per epoch {steps_per_epoch} does not "
                          f"match the pinned "
                          f"{e8b_run.STEPS_PER_EPOCH[scale]}",
                          {"observed": steps_per_epoch})
    g15 = {"train_manifest": g15_manifest_record(
               train_manifest, len(train_loader.dataset), scale, run_name),
           "dev_manifest": g15_manifest_record(
               dev_manifest, len(dev_loader.dataset), "dev", run_name),
           "image_store_sha256": sha256_file(
               tokens_data.TOKEN_DIR / "image_tokens.h5"),
           "question_store_sha256": sha256_file(
               tokens_data.TOKEN_DIR / "question_tokens.h5")}
    g1_binding = {
        "train": g1_label_binding(train_manifest, answers, run_name,
                                  scale),
        "dev": g1_label_binding(dev_manifest, answers, run_name, "dev")}
    import pandas as pd
    dev_frame = pd.read_csv(dev_manifest, keep_default_na=False)

    gates_record = {
        "g0_recipe": recipe, "g0_recipe_sha256": recipe_hash,
        "recipe_identifier": recipe["recipe_identifier"],
        "g1_label_binding": g1_binding,
        "g13_projection_scale": scale_record,
        "g13_trainable_init_sha256": trainable_init_sha256,
        "g15": g15, "g18_vocabulary_sha256": vocabulary["sha256"]}
    if arm != "B1":
        gates_record["g1_answer_cache_sha256"] = cache["sha256"]
        gates_record["g2_provenance"] = dict(lm_provenance)
        gates_record["fp32_model_identity"] = identity
        gates_record["oi5_buffer_parity"] = parity

    # Implementation gates, then the overfit gate (which reseeds), then
    # re-imposition, then the opening G14 for the LM arms.
    if arm == "B1":
        gates_record.update(b1_implementation_gates(model, train_loader,
                                                    device, run_name))
        gates_record["g8"] = b1_overfit_gate(train_loader, device,
                                             run_name)
    else:
        gates_record.update(implementation_gates(model, lm, train_loader,
                                                 cache, device, run_name))
        gates_record["g8"] = g8_overfit_gate(lm, train_loader, cache,
                                             device, run_name, arm=arm)
    e8b_run.enable_strict_determinism()
    train_loader.generator.manual_seed(recipe["seed"])

    if arm != "B1":
        gates_record["g14_pre_selection"] = g14_fp32_gate(
            model, lm, dev_loader.dataset, cache, trie, device,
            stage="pre-selection", run_name=run_name)
        print(f"[G14-FP32] pre-selection: C1/C2/C3 identity on "
              f"{gates_record['g14_pre_selection']['rows_checked']} rows")

    optimizer = make_optimizer(model, recipe["lr"])
    scheduler = make_scheduler(optimizer, 100 * steps_per_epoch,
                               recipe["warmup_frac"])
    resume_path = CHECKPOINT_DIR / f"resume_{run_name}.pt"
    canonical_path = CHECKPOINT_DIR / (
        f"{run_name}_canonical_ep22.pt" if arm != "B1"
        else f"{run_name}_canonical_best.pt")
    secondary_path = CHECKPOINT_DIR / f"{run_name}_secondary_best_of22.pt"
    wall_seconds = recipe["wall_clock_halt_hours"] * 3600
    device_total = torch.cuda.get_device_properties(0).total_memory
    max_epochs = recipe["max_epochs"]
    fixed_budget = arm != "B1"

    # M-2 PAIR INTEGRITY. The G8 gate trains to a convergence threshold,
    # so it consumes a DIFFERENT number of RNG draws in B2 than in B3:
    # the pretrained and random arms reach the threshold at different
    # epochs. Without this re-seed the paired arms enter the training
    # loop on divergent global RNG states, so their dropout masks and
    # shuffles differ for reasons that have nothing to do with the
    # pretrained-versus-random contrast that B3-B2 is supposed to
    # isolate. Re-seed to the common recipe seed HERE, before the resume
    # block, which legitimately overwrites RNG with the saved state.
    e8b_run.reseed_strict(recipe["seed"])
    train_loader.generator.manual_seed(recipe["seed"])

    best_accuracy, best_epoch, best_state = 0.0, -1, None
    epochs_without_improvement = 0
    history, train_times, eval_times = [], [], []
    step_count, permutation_counter, start_epoch = 0, 0, 1
    resumed_from = None
    if resume_path.exists():
        state = e8b_run.verify_resume_checkpoint(
            resume_path, recipe_sha256=recipe_hash,
            protocol_family=recipe["protocol_family"])
        if state["vocabulary_sha256"] != vocabulary["sha256"] \
                or state["store_sha256s"] != {
                    "image_tokens": g15["image_store_sha256"],
                    "question_tokens": g15["question_store_sha256"]}:
            e8b_run.gate_halt(run_name, "RESUME",
                              "vocabulary or store hash mismatch",
                              {"checkpoint": str(resume_path)})
        restored = e8b_run.restore_resume_state(
            state, model=model, optimizer=optimizer, scheduler=scheduler,
            loader_generator=train_loader.generator)
        best_accuracy = restored["best_metric"]
        best_epoch = restored["best_epoch"]
        best_state = restored["best_model_state"]
        permutation_counter = restored["epoch_permutation_counter"]
        step_count = restored["global_step"]
        start_epoch = restored["epoch"] + 1
        epochs_without_improvement = restored["epoch"] - best_epoch
        resumed_from = restored["epoch"]
        history = restored["history"]
        train_times = restored["train_times"]
        eval_times = restored["eval_times"]
        e8b_run.assert_strict_determinism()
        print(f"[RESUME] verified; continuing from epoch {start_epoch}")

    # C-M2: the wall is per CELL, not per process. A resumed cell keeps
    # the hours its earlier processes already spent, restored from the
    # per-epoch record, so a cell that crashes at 7 h cannot be handed a
    # fresh 8 h by restarting.
    # Every prior process of this cell, read from the append-only
    # ledger: per-epoch train and eval time would omit store hashing,
    # the LM load and promotion, G8 (up to 200 epochs) and G14
    # pre-selection, so a crash-resume cycle could otherwise buy back
    # hours the cell had already burned.
    prior_seconds = 3600.0 * e8b_run.cell_hours(
        e8b_run.read_spend_ledger(), arm, scale, seed)
    if prior_seconds:
        print(f"[WALL] {prior_seconds / 3600:.2f} h carried forward from "
              f"earlier processes of this cell")

    def wall_halt(epoch, step):
        elapsed = prior_seconds + (time.time() - started)
        if elapsed >= wall_seconds:
            failure = {"metadata": utils.run_metadata(),
                       "core_failure": {
                           "status": "FAILED: 8 GPU-hour operational "
                                     "wall-clock halt",
                           "epoch": epoch, "step": step,
                           "elapsed_hours": round(elapsed / 3600, 3),
                           "includes_prior_process_hours":
                               round(prior_seconds / 3600, 3)}}
            e8b_run.atomic_write_json(
                OUT_DIR / f"{run_name}_FAILED.json", failure)
            e8b_run.gate_halt(run_name, "G19_WALL",
                              "the 8 GPU-hour per-run operational wall "
                              "was reached",
                              {"epoch": epoch, "step": step})

    # A resumed cell that has already exhausted its wall halts here
    # rather than one batch into the loop.
    #
    # CORRECTION of 2026-08-07: an earlier comment here claimed this
    # halts "before the LM load, the parity load, G8 and G14
    # pre-selection have run again". THAT WAS FALSE -- all of those run
    # earlier in this function, and this check cannot precede them.
    #
    # It is SKIPPED when the loop will not execute. A cell resumed at
    # the final epoch has already done its training; halting it here
    # would destroy a complete 22-epoch trajectory that only needs
    # finalising, and would leave both a FAILED record and a halt record
    # blocking any re-entry.
    if start_epoch <= max_epochs:
        wall_halt(start_epoch - 1, step_count)
    else:
        # The wall is NOT abandoned here, only relaxed: finalisation
        # gets a bounded allowance on top, because destroying a complete
        # 22-epoch trajectory to save an hour of evaluation is the worse
        # outcome. Exceeding even that is recorded, not ignored.
        finalisation_budget = wall_seconds + FINALISATION_ALLOWANCE_S
        if prior_seconds >= finalisation_budget:
            wall_halt(start_epoch - 1, step_count)
        print(f"[WALL] resuming at epoch {start_epoch} of {max_epochs}: "
              f"the loop will not run, so this cell only needs "
              f"finalising. {prior_seconds / 3600:.2f} h already spent "
              f"against a {finalisation_budget / 3600:.2f} h "
              f"finalisation budget.")
    verified_at_use = e8b_run.assert_strict_determinism()
    first_step_asserted = False
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss, seen = 0.0, 0
        for step, (images, questions, _, mask, labels) in enumerate(
                train_loader):
            if step % WALL_CHECK_EVERY_STEPS == 0:
                wall_halt(epoch, step_count)
            if not first_step_asserted:
                # the user-mandated assertion immediately before the
                # FIRST optimizer step of the run
                e8b_run.assert_strict_determinism()
                first_step_asserted = True
            optimizer.zero_grad(set_to_none=True)
            if arm == "B1":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(
                        model(images.to(device), questions.to(device),
                              mask.to(device)), labels.to(device))
            else:
                answer_ids = [cache["sequences"][int(l)] for l in labels]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prefix = model.prefix_embeddings(
                        lm, images.to(device), questions.to(device),
                        mask.to(device))
                    loss, _ = model.teacher_forced_loss(
                        lm, prefix.to(torch.bfloat16), answer_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           recipe["grad_clip"])
            optimizer.step()
            scheduler.step()
            step_count += 1
            running_loss += float(loss.detach()) * labels.shape[0]
            seen += labels.shape[0]
        train_seconds = time.time() - epoch_start
        if not math.isfinite(running_loss):
            e8b_run.gate_halt(run_name, "NAN",
                              f"non-finite training loss at epoch "
                              f"{epoch}", {"epoch": epoch})

        eval_start = time.time()
        if arm == "B1":
            predictions, labels_np, _ = b1_canonical_predictions(
                model, dev_loader, device)
            margins = None
        else:
            predictions, labels_np, _, margins = \
                canonical_dev_predictions(model, lm, dev_loader, cache,
                                          device)
        accuracy = float((predictions == labels_np).mean())
        eval_seconds = time.time() - eval_start
        train_times.append(train_seconds)
        eval_times.append(eval_seconds)
        permutation_counter += 1
        history.append({"epoch": epoch,
                        "train_loss": round(running_loss / seen, 5),
                        "canonical_fp32_dev_accuracy": round(accuracy, 5),
                        "train_seconds": round(train_seconds, 1),
                        "eval_seconds": round(eval_seconds, 1)})
        print(f"[{run_name}] epoch {epoch:3d}: loss "
              f"{running_loss / seen:.4f}  dev {accuracy:.4f}  "
              f"({train_seconds:.0f}s + {eval_seconds:.0f}s)")

        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            epochs_without_improvement = 0
            best_state = {k: v.detach().float().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        e8b_run.save_resume_checkpoint(
            resume_path, model=model, optimizer=optimizer,
            scheduler=scheduler, epoch=epoch, global_step=step_count,
            best_model_state=best_state, best_metric=best_accuracy,
            best_epoch=best_epoch,
            loader_generator=train_loader.generator,
            epoch_permutation_counter=permutation_counter,
            recipe_sha256=recipe_hash,
            vocabulary_sha256=vocabulary["sha256"],
            store_sha256s={"image_tokens": g15["image_store_sha256"],
                           "question_tokens":
                               g15["question_store_sha256"]},
            protocol_family=recipe["protocol_family"],
            history=history, train_times=train_times,
            eval_times=eval_times)

        memory_check = e8b_run.memory_gate(
            torch.cuda.max_memory_allocated(), device_total,
            torch.cuda.max_memory_reserved())
        if memory_check["fires"]:
            e8b_run.gate_halt(run_name, "MEMORY",
                              f"peak reserved memory "
                              f"{memory_check['reserved_fraction']:.1%} "
                              f"exceeds the 80 per cent ceiling",
                              {"memory": memory_check, "epoch": epoch})
        wall_halt(epoch, step_count)
        if not fixed_budget \
                and epochs_without_improvement >= recipe["patience"]:
            print(f"[{run_name}] early stop at epoch {epoch} "
                  f"(patience {recipe['patience']}; section 7.3 rule)")
            break

    # --- Canonical checkpoint (the amended selection rule) ---
    if fixed_budget:
        # M-5: an empty history means the loop body never ran, which
        # happens when a resume checkpoint already sits at the final
        # epoch. Halt with the reason stated rather than dying on an
        # IndexError inside a resume loop that would then repeat.
        if not history:
            e8b_run.gate_halt(run_name, "SCHEDULE",
                              f"no epoch ran in this process: the resume "
                              f"checkpoint is already at epoch "
                              f"{resumed_from} of the fixed "
                              f"{max_epochs}-epoch budget, so there is no "
                              f"per-epoch record to select from. Finalise "
                              f"from the completed run's own record or "
                              f"restart the cell from scratch.",
                              {"resumed_from": resumed_from,
                               "max_epochs": max_epochs})
        if history[-1]["epoch"] != max_epochs:
            e8b_run.gate_halt(run_name, "SCHEDULE",
                              f"the fixed budget requires exactly "
                              f"{max_epochs} epochs; the loop ended at "
                              f"{history[-1]['epoch']}",
                              {"epochs_run": len(history)})
        canonical_state = {k: v.detach().float().cpu().clone()
                          for k, v in model.state_dict().items()}
        canonical_epoch = max_epochs
        canonical_accuracy = history[-1]["canonical_fp32_dev_accuracy"]
    else:
        canonical_state = best_state
        canonical_epoch = best_epoch
        canonical_accuracy = round(best_accuracy, 5)
        if canonical_state is None:
            e8b_run.gate_halt(run_name, "SELECTION",
                              "no B1 checkpoint ever improved on zero "
                              "accuracy", {})
    tmp = canonical_path.with_name(canonical_path.name + ".tmp")
    torch.save({"role": "CANONICAL PRIMARY",
                "rule": recipe["canonical_checkpoint_rule"],
                "epoch": canonical_epoch,
                "protocol_family": recipe["protocol_family"],
                "recipe_sha256": recipe_hash,
                "state": canonical_state}, tmp)
    os.replace(tmp, canonical_path)
    secondary = None
    if fixed_budget and best_state is not None:
        tmp = secondary_path.with_name(secondary_path.name + ".tmp")
        torch.save({"role": "SECONDARY DIAGNOSTIC ONLY - best-of-22 "
                            "dev checkpoint; never determines the "
                            "primary comparison, the model freeze or "
                            "the clean-test checkpoint",
                    "epoch": best_epoch,
                    "protocol_family": recipe["protocol_family"],
                    "recipe_sha256": recipe_hash,
                    "state": best_state}, tmp)
        os.replace(tmp, secondary_path)
        secondary = {"best_of_22_epoch": best_epoch,
                     "best_of_22_dev_accuracy": round(best_accuracy, 5),
                     "path": str(secondary_path),
                     "sha256": sha256_file(secondary_path),
                     "label": "SECONDARY DIAGNOSTIC ONLY"}

    # --- G9: reload reproduces identical canonical behaviour ---
    model.load_state_dict(canonical_state)
    model = model.to(device)
    if arm == "B1":
        trunk2, readout2 = e8b_run.build_arm("B1", recipe["seed"],
                                             recipe["dropout"], None)
        reloaded = B1Classifier(trunk2, readout2)
    else:
        trunk2, projection2 = e8b_latents.build_trunk_and_projection(
            recipe["seed"], recipe["dropout"])
        reloaded = e8b_latents.E8BPrefixModel(trunk2, projection2)
    e8b_run.enable_strict_determinism()   # build_arm reseeded
    e8b_run.assert_strict_determinism()    # G9/G10/G11/G14-post follow
    reloaded.load_state_dict(torch.load(
        canonical_path, map_location="cpu", weights_only=False)["state"])
    reloaded = reloaded.to(device).eval()
    batch = tokens_data.collate_tokens(
        [dev_loader.dataset[i] for i in range(32)])
    with torch.no_grad():
        if arm == "B1":
            s_a = model.eval()(batch[0].to(device), batch[1].to(device),
                               batch[3].to(device))
            s_b = reloaded(batch[0].to(device), batch[1].to(device),
                           batch[3].to(device))
        else:
            p_a = canonical_prefix(model.eval(), lm, batch[0].to(device),
                                   batch[1].to(device),
                                   batch[3].to(device))
            p_b = canonical_prefix(reloaded, lm, batch[0].to(device),
                                   batch[1].to(device),
                                   batch[3].to(device))
            s_a = r1_scores_batched(lm, p_a, cache)
            s_b = r1_scores_batched(lm, p_b, cache)
    if not torch.equal(s_a, s_b):
        e8b_run.gate_halt(run_name, "G9",
                          "reloaded canonical checkpoint does not "
                          "reproduce identical scores", {})
    del reloaded
    torch.cuda.empty_cache()

    # --- G11 and the primary metric ---
    if arm == "B1":
        predictions_a, labels_np, _ = b1_canonical_predictions(
            model, dev_loader, device)
        predictions_b, _, _ = b1_canonical_predictions(model, dev_loader,
                                                       device)
        margins_a = None
    else:
        predictions_a, labels_np, scores_a, margins_a = \
            canonical_dev_predictions(model, lm, dev_loader, cache,
                                      device)
        g14_binding = evaluation_binding(model, predictions_a, margins_a,
                                         recipe, recipe_hash)
        predictions_b, _, _, _ = canonical_dev_predictions(
            model, lm, dev_loader, cache, device)
    if not np.array_equal(predictions_a, predictions_b):
        e8b_run.gate_halt(run_name, "G11",
                          "repeated canonical evaluation differs", {})
    primary_accuracy = float((predictions_a == labels_np).mean())
    if round(primary_accuracy, 5) != round(canonical_accuracy, 5):
        e8b_run.gate_halt(run_name, "SELECTION",
                          f"canonical accuracy {primary_accuracy:.5f} "
                          f"does not reproduce the in-loop value "
                          f"{canonical_accuracy:.5f}", {})

    # --- G10: collapse (halting for B1/B3; recorded for B2) ---
    counts = np.bincount(predictions_a, minlength=100)
    shares = counts / counts.sum()
    top1_share = float(shares.max())
    distinct = int((counts > 0).sum())
    if arm == "B1":
        probs = torch.softmax(
            b1_canonical_predictions(model, dev_loader, device)[2]
            .double(), dim=-1)
        entropy = float((-(probs * torch.log(probs.clamp_min(1e-300)))
                         ).sum(dim=-1).mean())
    else:
        entropy = mean_prediction_entropy_nats(scores_a)
    collapse = top1_share >= 0.60 or entropy <= 0.30
    g10 = {"distinct_answers": distinct,
           "top1_share": round(top1_share, 4),
           "mean_prediction_entropy_nats": round(entropy, 6),
           "comparability": "P1: an E8B-specific pseudo-probability "
                            "diagnostic; never numerically comparable "
                            "with an E8A entropy" if arm != "B1" else
                            "B1 classifier softmax entropy",
           "fires": bool(collapse),
           "gate_treatment": ("recorded scientific diagnostic (section "
                              "11)" if arm == "B2" else "halting")}
    if collapse and arm != "B2":
        e8b_run.gate_halt(run_name, "G10",
                          f"prediction collapse: top-1 {top1_share:.3f}, "
                          f"entropy {entropy:.3f} nats", g10)

    # --- Closing G14-FP32 on the canonical checkpoint ---
    g14_post = None
    if arm != "B1":
        g14_post = g14_fp32_gate(model, lm, dev_loader.dataset, cache,
                                 trie, device, stage="post-selection",
                                 run_name=run_name, margins=margins_a,
                                 dev_frame=dev_frame,
                                 binding=g14_binding)
        print(f"[G14-FP32] post-selection: C1/C2/C3 identity on "
              f"{g14_post['rows_checked']} rows")

    # --- Per-row prediction dump (OI3) ---
    npz_path = OUT_DIR / f"{run_name}_per_row.npz"
    if npz_path.exists():
        sys.exit(f"{npz_path} already exists; refusing to overwrite")
    np.savez_compressed(
        npz_path, canonical_predictions=predictions_a,
        canonical_correct=(predictions_a == labels_np).astype(np.uint8),
        labels=labels_np)

    # Cumulative across every process that trained this cell, so both
    # the recorded wall_hours and the identity ledger charge the cell's
    # true cost rather than only the final process's share.
    elapsed = prior_seconds + (time.time() - started)
    record = {"metadata": utils.run_metadata(),
              "e8b_core_cell": {
        "run": run_name, "cell": [arm, scale, seed],
        "protocol_family": recipe["protocol_family"],
        "recipe": recipe, "recipe_sha256": recipe_hash,
        "paired_recipe_sha256": paired_recipe_sha256(recipe),
        "gates": gates_record,
        "optimizer_path_authorisation": authorisation,
        "determinism_verified_at_use": verified_at_use,
        "fp32_precision_pin": precision,
        "epochs_run": len(history),
        "resumed_from_epoch": resumed_from,
        "canonical_checkpoint_rule": recipe["canonical_checkpoint_rule"],
        "canonical_epoch": canonical_epoch,
        "primary_canonical_fp32_dev_accuracy": round(primary_accuracy, 5),
        "secondary_diagnostic": secondary,
        "history": history,
        "train_seconds_per_epoch": round(float(np.mean(train_times)), 2),
        "eval_seconds_per_epoch": round(float(np.mean(eval_times)), 2),
        "wall_hours": round(elapsed / 3600, 3),
        "wall_hours_this_process": round(
            (elapsed - prior_seconds) / 3600, 3),
        "wall_hours_carried_from_earlier_processes": round(
            prior_seconds / 3600, 3),
        "identity_gate_before": identity_gate,
        "remaining_core_gate": remaining,
        "storage_gate": storage,
        "gpu": fingerprint,
        "peak_memory": e8b_run.memory_gate(
            torch.cuda.max_memory_allocated(), device_total,
            torch.cuda.max_memory_reserved()),
        "checkpoints": {
            "canonical": {"path": str(canonical_path),
                          "sha256": sha256_file(canonical_path)},
            "resume": {"path": str(resume_path),
                       "sha256": sha256_file(resume_path)}},
        "per_row_dump": str(npz_path),
        "g9_reload_identical": True,
        "g11_repeat_identical": True,
        "g10_collapse": g10,
        "g14_post_selection": g14_post,
        "clean_test_accessed": False}}
    # The identity charge is made by train_core_cell's finally block, on
    # EVERY exit path, so it is not repeated here. What is recorded here
    # is the ledger position as it stands before this process's own
    # charge lands.
    # Contained: the per-row dump is already written, and the result
    # JSON is written a few lines below. An uncontained ledger read here
    # -- deliberately fail-closed, on the shared filesystem -- would
    # leave the npz with no result, and re-entry would then die at the
    # npz-exists guard. A completed cell is never stranded for the sake
    # of a diagnostic field.
    try:
        record["e8b_core_cell"][
            "identity_gate_before_this_process_charge"] = \
            e8b_run.per_identity_gate(arm, 0.0)
    except BaseException as ledger_error:      # noqa: BLE001
        record["e8b_core_cell"][
            "identity_gate_before_this_process_charge"] = {
                "unavailable": repr(ledger_error),
                "note": "the ledger could not be read while writing "
                        "this record; the charge itself is made by the "
                        "caller's finally block and records its own "
                        "failure separately"}
    e8b_run.atomic_write_json(result_path, record)
    print(f"[CORE DONE] {run_name}: canonical epoch {canonical_epoch}, "
          f"dev {primary_accuracy:.4f}, wall {elapsed / 3600:.2f} h")
    # The ledger charge and the post-charge ceiling re-check belong to
    # train_core_cell's finally block, which runs on every exit path
    # including this one. They are deliberately NOT repeated here.
    return 0
