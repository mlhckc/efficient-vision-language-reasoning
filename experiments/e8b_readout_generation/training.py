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
import sys
import time
from pathlib import Path

import numpy as np
import torch
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
CORE_FIXED_JUSTIFICATION = (
    "retained because it was the PRE-RESULT preregistered pilot and "
    "default configuration, not because it achieved the highest observed "
    "development score; the eight-point search was abandoned and no "
    "hyperparameter winner is claimed")


def build_core_recipe(arm: str, scale: str, seed: int) -> dict:
    """The frozen recipe for one of the 18 core cells.

    B2 and B3 take the E8B likelihood recipe of section 7.4 with the
    hyperparameters fixed to the pre-result pilot configuration. B1 keeps
    the stored v3_01 classifier recipe of section 7.3, to which the
    likelihood search never applied.

    Every recipe in this family carries `protocol_family` and an FP32
    canonical-evaluation precision string, so its hash necessarily
    differs from the superseded BF16 search family and a search
    checkpoint can never be resumed by a core run (A8)."""
    if arm not in e8b_run.CORE_ARMS:
        raise AssertionError(f"{arm} is not a core arm")
    if scale not in e8b_run.CORE_SCALES:
        raise AssertionError(f"{scale} is not a core scale")
    if seed not in e8b_run.CORE_SEEDS:
        raise AssertionError(f"{seed} is not a core seed")
    common = {
        "protocol_family": e8b_run.PROTOCOL_FAMILY,
        "arm": arm, "scale": scale, "seed": seed,
        "max_epochs": 100, "patience": 10, "batch_size": 128,
        "grad_clip": 1.0,
        "scheduler": "cosine after warmup, LambdaLR, horizon "
                     "100 x steps_per_epoch, stepped per optimizer step",
        "precision": "bf16 autocast on the training path only; CANONICAL "
                     "FP32 evaluation, the pinned frozen state values "
                     "promoted losslessly to fp32; TF32 pinned off",
        "canonical_evaluation_precision": "fp32",
        "wall_clock_halt_hours": 8.0,
    }
    if arm == "B1":
        # Section 7.3: the stored v3_01 classifier recipe.
        common.update({
            "recipe_identifier": "7.3 stored v3_01 classifier recipe",
            "lr": 1e-3, "warmup_frac": 0.0, "dropout": 0.1,
            "weight_decay": 1e-4, "weight_decay_on": "all parameters",
            "objective": "softmax cross-entropy over the 100-answer "
                         "vocabulary",
            "selection_metric": "development classification accuracy",
            "fixed_hyperparameters_justification":
                "inherited from v3_01; the E8B likelihood search never "
                "applied to B1 (section 7.3)"})
    else:
        common.update({
            "recipe_identifier": "7.4 E8B likelihood recipe, "
                                 "hyperparameters fixed (amendment A3)",
            "lr": CORE_FIXED_HYPER["lr"],
            "warmup_frac": CORE_FIXED_HYPER["warmup_frac"],
            "dropout": CORE_FIXED_HYPER["dropout"],
            "weight_decay": 0.01,
            "weight_decay_on": "parameters with ndim >= 2 only",
            "objective": "teacher-forced mean token NLL over answer "
                         "tokens plus EOS, per example, then mean over "
                         "the batch",
            "selection_metric": "development R1 accuracy, EOS included, "
                                "under canonical FP32 evaluation",
            "fixed_hyperparameters_justification":
                CORE_FIXED_JUSTIFICATION})
    common["tie_break"] = "earliest epoch attaining the best value"
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
TOTAL_SEARCH_RUNS = 8         # the frozen section-7.4 grid
REMAINING_SEARCH_RUNS = 7     # after grid point 1, the G19 pilot
CORE_LM_RUNS_40K = 5          # B2 x3, B3 seeds 1-2 (seed 0 promoted, U4)
CORE_LM_RUNS_250K = 6         # B2 x3, B3 x3
CORE_B1_RUNS = 6              # classifier recipe, both scales x3 seeds
CORE_B1_RUNS_40K = 3          # priced at the 40k rate and epoch count
CORE_B1_RUNS_250K = 3         # priced at the 250k step ratio and epochs
LM_CORE_CHECKPOINTS = 12      # B2/B3 x two scales x three seeds
READOUTS_PER_CHECKPOINT = 3   # R1, R2, R3
INTERVENTION_CONDITIONS = 3   # fixed-image, fixed-question, shuffled
SELECTION_SENSITIVITY_EVALS = 22   # retained checkpoints (U1 halving)
SELECTION_SENSITIVITY_PASSES = 44  # those checkpoints x (R2, R3)

# Protocol section 13.2b: the pretrained SmolLM2-135M identity is
# A1 + A4 + A7c + B3. The stored IDENTITY_BASELINE_HOURS figure is A1
# alone, and its source record (resource_correction_g21.json) states in
# terms that it is a lower bound because the E8B readout arms were then
# unauthorised. A4 and A7c are unconditional core arms that load the same
# pinned pretrained checkpoint, so they belong in the 35 h aggregate.
A4_IDENTITY_HOURS = 1.804     # protocol table line 1186
A7C_IDENTITY_HOURS = 1.864    # protocol table line 1188
# Evaluation-hour split between the two identities, by weighted units,
# derived on the 44-pass selection-sensitivity basis the evaluation total
# actually uses. Readout passes carry weight 1.5, intervention passes
# 1.0, and the selection-sensitivity passes belong wholly to the
# pretrained identity because section 10.1 runs them on two B3
# configurations. B3: 6 checkpoints x 3 readouts x 1.5 = 27, plus
# 6 x 3 x 1.0 = 18 interventions, plus 44 x 1.5 = 66, so 111 of 156
# weighted units. B2 takes the remaining 45.
EVAL_SHARE_PRETRAINED = 111.0 / 156.0   # 0.7115
EVAL_SHARE_RANDOM = 45.0 / 156.0        # 0.2885

# Checkpoint sizes MEASURED from the grid point 1 artefacts on
# 2026-08-06, replacing an earlier 85 MiB placeholder. A resume
# checkpoint carries the model, both AdamW moments and the best state,
# i.e. four copies of the 21,343,808 trainable fp32 parameters.
RESUME_CHECKPOINT_MIB = 325.9
BEST_CHECKPOINT_MIB = 81.5


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


@torch.no_grad()
def dev_r1_predictions(model, lm, loader, cache: dict, device) -> tuple:
    """fp32 evaluation of the trainable path: R1 argmax per dev row, plus
    the fp32 score matrix the argmax is taken over, so section 11's
    prediction entropy is computed from the same distribution that
    produces the prediction."""
    model.eval()
    predictions, labels, scores_all = [], [], []
    for images, questions, _, mask, batch_labels in loader:
        prefix = model.prefix_embeddings(
            lm, images.to(device), questions.to(device), mask.to(device)
        ).to(torch.bfloat16)
        scores = r1_scores_batched(lm, prefix, cache)
        predictions.append(scores.argmax(dim=1).cpu())
        scores_all.append(scores.cpu())
        labels.append(batch_labels)
    return (torch.cat(predictions).numpy(), torch.cat(labels).numpy(),
            torch.cat(scores_all))


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
    if torch.is_autocast_enabled() or torch.is_autocast_enabled("cuda"):
        raise AssertionError(
            "OI6 DTYPE GATE FAILED: autocast is active on the canonical "
            "evaluation path; the prefix would be computed in bfloat16")
    trainable = next(model.parameters())
    assert_canonical_dtype(trainable, "trainable trunk/projection weights")
    frozen = next(lm.parameters())
    assert_canonical_dtype(frozen, "frozen language-model weights")
    assert_canonical_dtype(images, "cached image tokens as stored")
    assert_canonical_dtype(questions, "cached question tokens as stored")
    prefix = model.prefix_embeddings(lm, images, questions, mask)
    assert_canonical_dtype(prefix, "constructed soft prefix")
    return prefix


def assert_canonical_scores(scores) -> None:
    assert_canonical_dtype(scores, "R1 candidate scores")


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


# --- OI7: G14-FP32, the live gate --------------------------------------------

@torch.no_grad()
def g14_fp32_gate(model, lm, dev_dataset, cache: dict, trie: dict, device,
                  stage: str, run_name: str, margins=None) -> dict:
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
    selection = g14_fp32_validation_rows(
        margins if margins is not None else np.zeros(n_dev), n_dev, stage)
    rows = selection["rows"]
    model.eval()

    deltas_brute, deltas_cached, checked = [], [], 0
    for start in range(0, len(rows), 64):
        chunk = rows[start:start + 64]
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
                                     margins))
            if cached["argmax"] != canonical:           # C2
                e8b_run.gate_halt(
                    run_name, "G14",
                    f"C2 cached-versus-canonical R1 argmax disagreement "
                    f"({stage}) at development row {row}",
                    _g14_halt_detail(row, canonical, brute, cached,
                                     batched[position], stage, cache,
                                     margins))
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
    if margins is not None:
        arr = np.asarray(margins, dtype=np.float64)
        record["dev_margin_quantiles"] = {
            q: float(np.quantile(arr, v))
            for q, v in (("p01", .01), ("p05", .05), ("p50", .50))}
        record["dev_rows_below_margin"] = {
            f"{t:g}": int((arr < t).sum())
            for t in (1e-5, 1e-4, 1e-3)}
    return record


def _g14_halt_detail(row, canonical, brute, cached, canon_row, stage,
                     cache, margins) -> dict:
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
    return detail


# --- The binding 64-example G14 gate ------------------------------------------

def pinned_g14_rows(n_dev: int) -> list:
    rng = np.random.default_rng(G14_ROW_SEED)
    return sorted(rng.choice(n_dev, size=G14_N_EXAMPLES,
                             replace=False).tolist())


@torch.no_grad()
def g14_binding_gate(model, lm, dev_dataset, cache: dict, trie: dict,
                     device, stage: str, run_name: str) -> dict:
    """Master protocol section 18 G14 at the binding width: on 64 pinned
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
                    run_name: str) -> dict:
    """Tiny-subset overfit, halting for the principal arm: gate settings
    lr 1e-3, dropout 0.0 (not tuned hyperparameters), 1,000 examples,
    target R1 accuracy >= 0.99."""
    utils.set_seed(0)
    subset = Subset(train_loader.dataset, list(range(G8_SUBSET)))
    loader = DataLoader(subset, batch_size=RECIPE["batch_size"],
                        shuffle=True, generator=utils.make_generator(0),
                        collate_fn=tokens_data.collate_tokens)
    trunk, projection = e8b_latents.build_trunk_and_projection(
        0, 0.0, d_lm=e8b_latents.D_LM)
    e8b_latents.apply_projection_scale(projection, trunk,
                                      e8b_run._pretrained_embed_weight())
    model = e8b_latents.E8BPrefixModel(trunk, projection).to(device)
    optimizer = make_optimizer(model, 1e-3)
    reached = None
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
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           RECIPE["grad_clip"])
            optimizer.step()
        if epoch % G8_EVAL_EVERY == 0 or epoch == G8_MAX_EPOCHS:
            predictions, labels_np, _ = dev_r1_predictions(
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
        e8b_run.gate_halt(
            run_name, "G8",
            f"tiny-subset overfit did not reach {G8_TARGET}: subset R1 "
            f"accuracy {accuracy:.4f} after {G8_MAX_EPOCHS} epochs "
            f"(halting for the principal arm)",
            {"subset_accuracy": round(float(accuracy), 5),
             "target": G8_TARGET, "epochs": G8_MAX_EPOCHS,
             "subset_size": G8_SUBSET})
    return {"epochs_to_target": reached[0],
            "subset_accuracy": round(reached[1], 5),
            "gate_settings": "lr 1e-3, dropout 0.0, 1000 examples"}


def make_optimizer(model, lr):
    decay, no_decay = [], []
    for _, parameter in model.named_parameters():
        (no_decay if parameter.ndim < 2 else decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": RECIPE["weight_decay"]},
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


# --- G19 projections from measured pilot quantities ---------------------------

def g19_projection(train_seconds_per_epoch: float,
                   eval_seconds_per_epoch: float,
                   peak_allocated_bytes: int,
                   peak_reserved_bytes: int,
                   device_total_bytes: int,
                   v3_01_seconds_per_epoch: float,
                   b1_seconds_per_epoch: float,
                   storage_dir: Path,
                   completed_search_runs: int = 1,
                   measured_search_hours: float = 0.0) -> dict:
    """The operative section-14 projection from measured run quantities.

    Per-epoch time at 250k scales the TRAIN component by the step ratio;
    the dev evaluation is a fixed 7,714-row cost at every scale.

    U3 as clarified by the user on 2026-08-07 (P3): the GOVERNING
    projection is the expected 15/22-epoch basis; the 100-epoch
    projection is a mandatory reported stress scenario that does not
    halt by itself. Enforcement remains the caller's g19_halt (U2/P2:
    stop and return, never descope).

    Three arithmetic corrections recorded by the pre-execution audits are
    applied here, all in the conservative direction. The per-identity
    aggregate now includes the A4 and A7c arms that share the pretrained
    checkpoint, per protocol section 13.2b, which the stored A1-only
    baseline explicitly calls a lower bound. B1's three 250k runs are
    priced at the 250k step ratio and epoch count instead of the 40k
    rate. Storage uses the sizes MEASURED from the grid point 1
    artefacts, a 325.9 MiB resume checkpoint (model plus two AdamW
    moments plus the best state) and an 81.5 MiB best checkpoint,
    instead of the earlier 85 MiB placeholder, and the selection
    sensitivity term counts 44 evaluation passes rather than 22
    checkpoints."""
    steps = e8b_run.STEPS_PER_EPOCH
    epoch_40k = train_seconds_per_epoch + eval_seconds_per_epoch
    epoch_250k = (train_seconds_per_epoch
                  * steps["train_250k"] / steps["train_40k"]
                  + eval_seconds_per_epoch)
    multiplier = epoch_40k / v3_01_seconds_per_epoch
    remaining_search = max(0, TOTAL_SEARCH_RUNS - completed_search_runs)

    def hours(seconds, epochs):
        return seconds * epochs / 3600.0

    bases = {"expected_epoch_15_22": e8b_run.EXPECTED_EPOCHS,
             "worst_case_100_epoch": e8b_run.WORST_CASE_EPOCHS}
    wall = e8b_run.WALL_CLOCK_HALT_HOURS
    projections = {}
    for label, epochs in bases.items():
        run_40k = hours(epoch_40k, epochs["train_40k"])
        run_250k = hours(epoch_250k, epochs["train_250k"])
        run_250k_walled = min(run_250k, wall)
        run_40k_walled = min(run_40k, wall)
        search_remaining = remaining_search * run_40k_walled
        core_lm = (CORE_LM_RUNS_40K * run_40k_walled
                   + CORE_LM_RUNS_250K * run_250k_walled)
        b1_epoch_250k = (b1_seconds_per_epoch
                         * steps["train_250k"] / steps["train_40k"])
        core_b1 = (CORE_B1_RUNS_40K
                   * min(hours(b1_seconds_per_epoch, epochs["train_40k"]),
                         wall)
                   + CORE_B1_RUNS_250K
                   * min(hours(b1_epoch_250k, epochs["train_250k"]), wall))
        eval_hours = (LM_CORE_CHECKPOINTS * READOUTS_PER_CHECKPOINT
                      * eval_seconds_per_epoch * 1.5 / 3600
                      + LM_CORE_CHECKPOINTS * INTERVENTION_CONDITIONS
                      * eval_seconds_per_epoch / 3600
                      + SELECTION_SENSITIVITY_PASSES
                      * eval_seconds_per_epoch * 1.5 / 3600)
        remaining_core = (search_remaining + core_lm + core_b1
                          + eval_hours)
        projections[label] = {
            "seconds_per_epoch": {"train_40k": round(epoch_40k, 2),
                                  "train_250k": round(epoch_250k, 2)},
            "run_hours": {"train_40k": round(run_40k, 3),
                          "train_250k": round(run_250k, 3),
                          "train_250k_after_8h_wall":
                              round(run_250k_walled, 3)},
            "largest_250k_run_hours": round(run_250k, 3),
            "remaining_search_runs": remaining_search,
            "search_remaining_hours": round(search_remaining, 3),
            "core_lm_hours": round(core_lm, 3),
            "core_b1_hours": round(core_b1, 3),
            "evaluation_hours": round(eval_hours, 3),
            "remaining_core_hours": round(remaining_core, 3)}

    expected = projections["expected_epoch_15_22"]
    worst = projections["worst_case_100_epoch"]

    # Per-identity aggregates (35 h ceiling), governing expected basis.
    # Search runs already executed are charged at their MEASURED cost;
    # only the runs still to do are projected.
    b3_hours = (e8b_run.IDENTITY_BASELINE_HOURS["pretrained_smollm2_135m"]
                + A4_IDENTITY_HOURS + A7C_IDENTITY_HOURS
                + measured_search_hours
                + expected["search_remaining_hours"]
                + 2 * hours(epoch_40k, 15)          # B3 40k seeds 1-2
                + 3 * min(hours(epoch_250k, 22), wall)
                + expected["evaluation_hours"] * EVAL_SHARE_PRETRAINED)
    b2_hours = (e8b_run.IDENTITY_BASELINE_HOURS["random_smollm2_135m"]
                + 3 * hours(epoch_40k, 15)
                + 3 * min(hours(epoch_250k, 22), wall)
                + expected["evaluation_hours"] * EVAL_SHARE_RANDOM)

    runs_with_checkpoints = (remaining_search + CORE_LM_RUNS_40K
                             + CORE_LM_RUNS_250K + CORE_B1_RUNS)
    storage_needed = int(
        (runs_with_checkpoints * (RESUME_CHECKPOINT_MIB + BEST_CHECKPOINT_MIB)
         + SELECTION_SENSITIVITY_EVALS * BEST_CHECKPOINT_MIB) * 2 ** 20)

    gates = {
        "per_run_8h": {
            "fires": expected["largest_250k_run_hours"] > wall,
            "largest_250k_expected_hours":
                expected["largest_250k_run_hours"],
            "worst_case_hours": worst["largest_250k_run_hours"],
            "note": "the 8 h wall is also a hard operational halt during "
                    "every run, so the worst case is bounded by "
                    "construction"},
        "pretrained_identity_35h": {
            "fires": b3_hours > e8b_run.PER_IDENTITY_CEILING_HOURS,
            "projected_hours": round(b3_hours, 3),
            "includes": "A1 baseline, A4, A7c, every B3 search and core "
                        "run and the pretrained share of evaluation "
                        "(protocol section 13.2b)"},
        "random_identity_35h": {
            "fires": b2_hours > e8b_run.PER_IDENTITY_CEILING_HOURS,
            "projected_hours": round(b2_hours, 3)},
        "remaining_core": e8b_run.remaining_core_gate(
            expected["remaining_core_hours"],
            worst["remaining_core_hours"]),
        "memory_80pct": e8b_run.memory_gate(peak_allocated_bytes,
                                            device_total_bytes,
                                            peak_reserved_bytes),
        "storage": e8b_run.storage_gate(storage_needed, storage_dir),
    }
    fired = [name for name, gate in gates.items() if gate["fires"]]
    return {"multiplier_vs_v3_01": round(multiplier, 3),
            "v3_01_seconds_per_epoch": v3_01_seconds_per_epoch,
            "completed_search_runs": completed_search_runs,
            "measured_search_hours": round(measured_search_hours, 3),
            "projections": projections,
            "governing_basis": "expected_epoch_15_22 (U3 as clarified "
                               "2026-08-07, P3). The 100-epoch "
                               "projection is a mandatory reported "
                               "stress scenario and does not halt by "
                               "itself.",
            "gates": gates, "fired": fired,
            "stop_and_return": bool(fired)}


# --- G1 and G15: bindings derived from the manifests, never assumed ----------

EXPECTED_ROWS = {"train_40k": 40000, "dev": 7714}


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


def completed_search_runs() -> tuple:
    """How many section-7.4 grid points already have an immutable result
    record, and their total MEASURED wall hours. Used so the identity
    aggregate charges executed runs at their real cost and projects only
    the runs still to do."""
    total_hours = 0.0
    count = 0
    for point in range(1, TOTAL_SEARCH_RUNS + 1):
        path = OUT_DIR / f"pilot_{run_name_for(point)}.json"
        if not path.exists():
            continue
        # Deliberately broad: this runs after training and the closing
        # gates but before the record is written, so an unexpected shape
        # in an EARLIER run's file must never cost this run its record.
        try:
            body = json.loads(path.read_text())["e8b_pilot_g19"]
            hours = float(body.get("wall_hours", 0.0))
        except Exception:
            continue
        count += 1
        total_hours += hours
    return count, total_hours


# --- The search runs ----------------------------------------------------------

def train_search_point(grid_point: int) -> int:
    """One authorised B3/train_40k/seed0 search grid point."""
    device = torch.device("cuda")
    recipe = build_recipe(grid_point)
    run_name = run_name_for(grid_point)
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUT_DIR / f"pilot_{run_name}.json"
    if result_path.exists():
        sys.exit(f"{result_path} already exists; E8B records are "
                 f"immutable and a run is never repeated automatically")
    if (OUT_DIR / f"pilot_{run_name}_FAILED.json").exists():
        sys.exit("a recorded failure status exists for this run; a "
                 "failed run's result is not used and is never resumed "
                 "automatically -- execution returns to the user")
    existing_halts = sorted(OUT_DIR.glob(f"HALT_{run_name}_*.json"))
    if existing_halts:
        sys.exit(f"a recorded gate halt exists for this run "
                 f"({existing_halts[0].name}); execution returns to the "
                 f"user rather than silently retrying")

    from experiments.e8a_question_encoder import reinfer_g21
    reinfer_g21.assert_gpu_exclusive(f"e8b search point {grid_point}")
    # The lock is keyed on the CELL, so at most one of the eight grid
    # points can execute anywhere on the pool at a time (one node per
    # run, sequential by construction).
    lock_path = e8b_run.acquire_run_lock(*e8b_run.SEARCH_CELL)
    print(f"[LOCK] {lock_path.name} acquired for grid point {grid_point}")

    try:
        return _train_locked(device, recipe, run_name, result_path,
                             started)
    finally:
        lock_path.unlink(missing_ok=True)


def train_pilot() -> int:
    """Grid point 1, the G19 multiplier pilot."""
    return train_search_point(1)


def _train_locked(device, recipe, run_name, result_path, started) -> int:
    RECIPE_LOCAL_SHA = recipe_sha256(recipe)
    fingerprint = e8b_run.environment_fingerprint()
    if fingerprint["gpu"] != "NVIDIA RTX 4000 Ada Generation" \
            or not fingerprint["python"].startswith("3.12"):
        e8b_run.gate_halt(run_name, "FINGERPRINT",
                          "environment fingerprint does not match the "
                          "pinned execution environment",
                          {"fingerprint": fingerprint})

    # G2/G3 first, then G13 order: seed, trunk, projection.
    lm, lm_provenance = e8b_run.load_frozen_causal_lm(True, device)
    tokenizer = e8a.load_tokenizer()
    answers, vocabulary = g21.load_index_to_answer(
        V2_DIR / "answer_vocab_v2.json")
    if vocabulary["sha256"] != VOCAB_SHA256_PIN:
        e8b_run.gate_halt(run_name, "G18",
                          "V2 vocabulary hash mismatch",
                          {"observed": vocabulary["sha256"],
                           "pinned": VOCAB_SHA256_PIN})
    cache = readouts.build_answer_cache(tokenizer, answers)   # G1
    trie = readouts.build_trie(cache)
    model, scale_record = e8b_run.build_arm(
        "B3", recipe["seed"], recipe["dropout"], lm)
    model = model.to(device)

    # G15: manifests, stores and comparators pinned before first use.
    train_manifest = V2_DIR / "train_40k.csv"
    dev_manifest = V2_DIR / "dev.csv"
    stores = tokens_data.TokenStores()

    train_loader, dev_loader = tokens_data.make_token_loaders(
        train_manifest, dev_manifest, stores=stores,
        batch_size=recipe["batch_size"])
    train_loader.generator.manual_seed(recipe["seed"])
    steps_per_epoch = len(train_loader)
    if steps_per_epoch != e8b_run.STEPS_PER_EPOCH["train_40k"]:
        e8b_run.gate_halt(run_name, "G15",
                          f"steps per epoch {steps_per_epoch} does not "
                          f"match the pinned 313",
                          {"observed": steps_per_epoch})

    # Row counts DERIVED from the live datasets and asserted; the answer
    # to label-index binding asserted for every row of both manifests.
    g15 = {"train_manifest": g15_manifest_record(
               train_manifest, len(train_loader.dataset), "train_40k",
               run_name),
           "dev_manifest": g15_manifest_record(
               dev_manifest, len(dev_loader.dataset), "dev", run_name),
           "image_store_sha256": sha256_file(
               tokens_data.TOKEN_DIR / "image_tokens.h5"),
           "question_store_sha256": sha256_file(
               tokens_data.TOKEN_DIR / "question_tokens.h5"),
           "v3_01_results_sha256": sha256_file(
               config.RESULTS_DIR / "experiments" / "v3_01_reasoner"
               / "results.json")}
    g1_binding = {
        "train": g1_label_binding(train_manifest, answers, run_name,
                                  "train_40k"),
        "dev": g1_label_binding(dev_manifest, answers, run_name, "dev")}

    # Canonical section 20: the record carries the COMPLETE provenance
    # block, not a subset. E8B records are immutable and U4 conditions
    # promotion of this checkpoint on complete provenance, so a partial
    # record would fail its own promotion test.
    gates_record = {"g0_recipe": recipe,
                    "g0_recipe_sha256": RECIPE_LOCAL_SHA,
                    "recipe_identifier": "7.4 E8B likelihood recipe",
                    "g1_answer_cache_sha256": cache["sha256"],
                    "g1_label_binding": g1_binding,
                    "g2_provenance": dict(lm_provenance),
                    "g13_projection_scale": scale_record, "g15": g15,
                    "g18_vocabulary_sha256": vocabulary["sha256"]}
    gates_record.update(implementation_gates(model, lm, train_loader,
                                             cache, device, run_name))
    gates_record["g8"] = g8_overfit_gate(lm, train_loader, cache, device,
                                         run_name)

    # The binding G14 BEFORE any checkpoint selection.
    gates_record["g14_pre_selection"] = g14_binding_gate(
        model, lm, dev_loader.dataset, cache, trie, device,
        stage="pre-selection, initial weights", run_name=run_name)
    print(f"[G14] pre-selection: argmax identity on all "
          f"{G14_N_EXAMPLES} pinned examples (R1 and R2)")

    optimizer = make_optimizer(model, recipe["lr"])
    scheduler = make_scheduler(optimizer,
                               recipe["max_epochs"] * steps_per_epoch,
                               recipe["warmup_frac"])
    resume_path = CHECKPOINT_DIR / f"resume_{run_name}.pt"
    best_path = CHECKPOINT_DIR / f"{run_name}_best.pt"
    wall_seconds = recipe["wall_clock_halt_hours"] * 3600
    device_total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.reset_peak_memory_stats()

    best_accuracy, best_epoch = 0.0, -1
    best_state = None
    epochs_without_improvement = 0
    history, train_times, eval_times = [], [], []
    step_count = 0
    permutation_counter = 0
    start_epoch = 1
    resumed_from = None
    if resume_path.exists():
        state = e8b_run.verify_resume_checkpoint(
            resume_path, recipe_sha256=RECIPE_LOCAL_SHA)
        if state["vocabulary_sha256"] != vocabulary["sha256"] \
                or state["store_sha256s"] != {
                    "image_tokens": g15["image_store_sha256"],
                    "question_tokens": g15["question_store_sha256"]}:
            e8b_run.gate_halt(run_name, "RESUME",
                              "vocabulary or store hash mismatch "
                              "against the live inputs",
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
        print(f"[RESUME] complete state verified; continuing from epoch "
              f"{start_epoch} (best {best_accuracy:.4f} at epoch "
              f"{best_epoch})")

    def wall_halt(epoch, step):
        elapsed = time.time() - started
        if elapsed >= wall_seconds:
            failure = {"metadata": utils.run_metadata(),
                       "pilot_failure": {
                           "status": "FAILED: 8 GPU-hour operational "
                                     "wall-clock halt",
                           "epoch": epoch, "step": step,
                           "elapsed_hours": round(elapsed / 3600, 3),
                           "note": "the run terminates with a recorded "
                                   "failure status and its result is "
                                   "not used (section 13.2c)"}}
            e8b_run.atomic_write_json(
                OUT_DIR / f"pilot_{run_name}_FAILED.json", failure)
            e8b_run.gate_halt(
                run_name, "G19_WALL",
                "the 8 GPU-hour per-run operational wall was reached",
                {"epoch": epoch, "step": step,
                 "elapsed_hours": round(elapsed / 3600, 3)})

    for epoch in range(start_epoch, recipe["max_epochs"] + 1):
        model.train()
        epoch_start = time.time()
        running_loss, seen = 0.0, 0
        for step, (images, questions, _, mask, labels) in enumerate(
                train_loader):
            if step % WALL_CHECK_EVERY_STEPS == 0:
                wall_halt(epoch, step_count)
            answer_ids = [cache["sequences"][int(l)] for l in labels]
            optimizer.zero_grad(set_to_none=True)
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
                              f"{epoch}",
                              {"epoch": epoch,
                               "running_loss": str(running_loss)})

        eval_start = time.time()
        predictions, labels_np, _ = dev_r1_predictions(model, lm, dev_loader,
                                                       cache, device)
        accuracy = float((predictions == labels_np).mean())
        eval_seconds = time.time() - eval_start
        train_times.append(train_seconds)
        eval_times.append(eval_seconds)
        permutation_counter += 1
        history.append({"epoch": epoch,
                        "train_loss": round(running_loss / seen, 5),
                        "dev_r1_accuracy": round(accuracy, 5),
                        "train_seconds": round(train_seconds, 1),
                        "eval_seconds": round(eval_seconds, 1)})
        print(f"[{run_name}] epoch {epoch:3d}: loss "
              f"{running_loss / seen:.4f}  dev R1 {accuracy:.4f}  "
              f"({train_seconds:.0f}s train + {eval_seconds:.0f}s eval)")

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
            best_epoch=best_epoch, loader_generator=train_loader.generator,
            epoch_permutation_counter=permutation_counter,
            recipe_sha256=RECIPE_LOCAL_SHA,
            vocabulary_sha256=vocabulary["sha256"],
            store_sha256s={"image_tokens": g15["image_store_sha256"],
                           "question_tokens":
                               g15["question_store_sha256"]})

        memory_check = e8b_run.memory_gate(
            torch.cuda.max_memory_allocated(), device_total,
            torch.cuda.max_memory_reserved())
        if memory_check["fires"]:
            e8b_run.g19_halt(
                [f"peak reserved memory "
                 f"{memory_check['reserved_fraction']:.1%} exceeds the "
                 f"80 per cent ceiling"],
                run_name, {"memory": memory_check, "epoch": epoch})
        wall_halt(epoch, step_count)
        if epochs_without_improvement >= recipe["patience"]:
            print(f"[{run_name}] early stop at epoch {epoch} "
                  f"(patience {recipe['patience']})")
            break

    # Selected checkpoint: earliest epoch attaining the best value.
    model.load_state_dict(best_state)
    model = model.to(device)
    tmp_best = best_path.with_name(best_path.name + ".tmp")
    torch.save(best_state, tmp_best)
    import os
    os.replace(tmp_best, best_path)

    # G9: save and reload reproduces identical scores on one dev batch.
    reload_trunk, reload_projection = \
        e8b_latents.build_trunk_and_projection(recipe["seed"],
                                               recipe["dropout"])
    reloaded = e8b_latents.E8BPrefixModel(reload_trunk, reload_projection)
    reloaded.load_state_dict(torch.load(best_path, map_location="cpu",
                                        weights_only=False))
    reloaded = reloaded.to(device).eval()
    batch = tokens_data.collate_tokens(
        [dev_loader.dataset[i] for i in range(32)])
    with torch.no_grad():
        p_a = model.eval().prefix_embeddings(
            lm, batch[0].to(device), batch[1].to(device),
            batch[3].to(device)).to(torch.bfloat16)
        p_b = reloaded.prefix_embeddings(
            lm, batch[0].to(device), batch[1].to(device),
            batch[3].to(device)).to(torch.bfloat16)
        s_a = r1_scores_batched(lm, p_a, cache)
        s_b = r1_scores_batched(lm, p_b, cache)
    if not torch.equal(s_a, s_b):
        e8b_run.gate_halt(run_name, "G9",
                          "reloaded checkpoint does not reproduce "
                          "identical scores",
                          {"checkpoint": str(best_path)})

    # G11: repeated deterministic evaluation matches exactly; this pass
    # also provides the final predictions for G10.
    predictions_a, labels_np, scores_a = dev_r1_predictions(
        model, lm, dev_loader, cache, device)
    predictions_b, _, _ = dev_r1_predictions(model, lm, dev_loader, cache,
                                             device)
    if not np.array_equal(predictions_a, predictions_b):
        e8b_run.gate_halt(run_name, "G11",
                          "repeated deterministic evaluation differs",
                          {"differing_rows": int(
                              (predictions_a != predictions_b).sum())})
    final_accuracy = float((predictions_a == labels_np).mean())
    if round(final_accuracy, 5) != round(best_accuracy, 5):
        e8b_run.gate_halt(
            run_name, "SELECTION",
            f"selected-checkpoint accuracy {final_accuracy:.5f} does "
            f"not reproduce the recorded best {best_accuracy:.5f}",
            {"reproduced": final_accuracy, "recorded": best_accuracy})

    # G10: no collapse (halting for the principal arm). Section 11 fixed
    # both thresholds before execution: collapse fires if the mean
    # PREDICTION entropy H <= 0.30 nats or the maximum class share
    # >= 0.60. H is the per-row quantity of section 11, not the entropy
    # of the argmax histogram; the histogram entropy is a second function
    # of the same counts as the class share and is recorded separately,
    # clearly marked as not the gate quantity.
    counts = np.bincount(predictions_a, minlength=100)
    shares = counts / counts.sum()
    top1_share = float(shares.max())
    distinct = int((counts > 0).sum())
    entropy = mean_prediction_entropy_nats(scores_a)
    histogram_entropy = float(
        -(shares[shares > 0] * np.log(shares[shares > 0])).sum())
    if top1_share >= 0.60 or entropy <= 0.30:
        e8b_run.gate_halt(
            run_name, "G10",
            f"prediction collapse (halting, principal arm): top-1 "
            f"share {top1_share:.3f}, mean prediction entropy "
            f"{entropy:.3f} nats",
            {"top1_share": top1_share, "entropy_nats": entropy,
             "distinct_answers": distinct,
             "thresholds": {"entropy_at_or_below": 0.30,
                            "share_at_or_above": 0.60}})

    # Closing binding G14 on the selected checkpoint.
    g14_post = g14_binding_gate(model, lm, dev_loader.dataset, cache,
                                trie, device,
                                stage="post-selection, best checkpoint",
                                run_name=run_name)
    print(f"[G14] post-selection: argmax identity on all "
          f"{G14_N_EXAMPLES} pinned examples (R1 and R2)")

    # G19: the operative projections from this run's measurements.
    v3_results = json.loads(
        (config.RESULTS_DIR / "experiments" / "v3_01_reasoner"
         / "results.json").read_text())["v3_01_reasoner"]
    v3_seconds = float(v3_results["efficiency"]["seconds_per_epoch"])
    completed, measured_hours = completed_search_runs()
    elapsed_now = (time.time() - started) / 3600.0
    projection = g19_projection(
        train_seconds_per_epoch=float(np.mean(train_times)),
        eval_seconds_per_epoch=float(np.mean(eval_times)),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        device_total_bytes=torch.cuda.get_device_properties(0)
        .total_memory,
        v3_01_seconds_per_epoch=v3_seconds,
        b1_seconds_per_epoch=v3_seconds,
        storage_dir=OUT_DIR,
        completed_search_runs=completed + 1,
        measured_search_hours=measured_hours + elapsed_now)

    elapsed = time.time() - started
    record = {"metadata": utils.run_metadata(),
              "e8b_pilot_g19": {
        "label": (f"E8B search grid point {recipe['grid_point']}"
                  + (" and G19 MULTIPLIER PILOT"
                     if recipe["grid_point"] == 1 else "")),
        "grid_point": recipe["grid_point"],
        "run": run_name, "cell": list(e8b_run.SEARCH_CELL),
        "recipe": recipe, "recipe_sha256": RECIPE_LOCAL_SHA,
        "gates": gates_record,
        "epochs_run": len(history),
        "resumed_from_epoch": resumed_from,
        "best_epoch": best_epoch,
        "best_dev_r1_accuracy": round(best_accuracy, 5),
        "selected_checkpoint_reproduced_accuracy":
            round(final_accuracy, 5),
        "history": history,
        "train_seconds_per_epoch": round(float(np.mean(train_times)), 2),
        "eval_seconds_per_epoch": round(float(np.mean(eval_times)), 2),
        "wall_hours": round(elapsed / 3600, 3),
        "gpu": fingerprint,
        "peak_memory": e8b_run.memory_gate(
            torch.cuda.max_memory_allocated(),
            torch.cuda.get_device_properties(0).total_memory,
            torch.cuda.max_memory_reserved()),
        "checkpoints": {
            "best": {"path": str(best_path),
                     "sha256": sha256_file(best_path)},
            "resume": {"path": str(resume_path),
                       "sha256": sha256_file(resume_path)}},
        "g9_reload_identical": True,
        "g11_repeat_identical": True,
        "g10_collapse": {
            "distinct_answers": distinct,
            "top1_share": round(top1_share, 4),
            "mean_prediction_entropy_nats": round(entropy, 6),
            "entropy_definition": "master protocol section 11: H = "
                                  "mean_n(-sum_k p_n[k] ln p_n[k]) in "
                                  "nats, p_n = softmax over the 100 "
                                  "answers of the fp32 length-normalised "
                                  "R1 scores whose argmax is the "
                                  "prediction, over all development rows",
            "thresholds": {"entropy_nats_at_or_below": 0.30,
                           "max_class_share_at_or_above": 0.60},
            "argmax_histogram_entropy_nats": round(histogram_entropy, 6),
            "argmax_histogram_entropy_note": "reported for completeness; "
                                             "NOT the section 11 gate "
                                             "quantity",
            "comparability": "P1 (user decision 2026-08-07): p_n is an "
                             "E8B-specific PSEUDO-PROBABILITY for "
                             "collapse diagnostics only. This entropy "
                             "must NEVER be compared numerically with "
                             "an E8A prediction entropy, in either "
                             "direction."},
        "g14_post_selection": g14_post,
        "g19_projection": projection,
        "clean_test_accessed": False}}
    e8b_run.atomic_write_json(result_path, record)
    print(f"[RUN DONE] grid point {recipe['grid_point']}: "
          f"{len(history)} epochs, best epoch {best_epoch}, dev R1 "
          f"{best_accuracy:.4f}, wall {elapsed / 3600:.2f} h; "
          f"record written")

    if projection["stop_and_return"]:
        e8b_run.g19_halt([f"projection gate(s) fired: "
                          f"{projection['fired']}"],
                         run_name, {"projection": projection})
    return 0


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


def train_core_cell(arm: str, scale: str, seed: int) -> int:
    """One of the 18 final core cells, under the amended protocol.

    Refuses unless the runner's authorisation state names an approved core
    matrix; `run.train` performs that check before dispatching here."""
    raise NotImplementedError(
        "train_core_cell is specified and gated but intentionally not "
        "implemented: the amendment requires the determinism probe (OI1) "
        "to pass under STRICT determinism and the B1 classifier path to "
        "be built before any core cell is runnable. run.train refuses "
        "before reaching this point.")
