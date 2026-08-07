"""E8B runner: arms, gates, checkpoint format, locks, projections, preflight.

    python -B experiments/e8b_readout_generation/run.py --preflight
    python -B experiments/e8b_readout_generation/run.py --core-cell ARM SCALE SEED

Amended protocol phase (user decision of 2026-08-07, recorded in
protocol_amendment_20260807_fp32.json).

The eight-point recipe search is PERMANENTLY STOPPED. Grid points 4-8 are
not authorised and will not run. Grid points 1-3 are retained as
exploratory / protocol-diagnostic evidence only; none is a final core
result and no hyperparameter winner is claimed. U4 is WITHDRAWN: no
search checkpoint is promoted, and every core cell is trained fresh.

The final recipe is fixed to the originally preregistered grid point 1
pilot configuration, lr 3e-4, warmup 0, dropout 0.1, retained because it
was the PRE-RESULT default and not because it scored highest.

Canonical scientific evaluation is FP32: the pinned frozen SmolLM2-135M
state values, which are BF16 on disk, are promoted losslessly to FP32 so
the model identity is unchanged and only the forward-pass arithmetic
differs. FP32 governs per-epoch development evaluation, early stopping,
best-checkpoint selection, final R1/R2/R3 evaluation and G14-FP32. BF16
figures may be reported only as a secondary deployment or efficiency
diagnostic. Training remains BF16 autocast on the training path.

The successor matrix is 18 core cells, B1/B2/B3 x train_40k/train_250k x
seeds 0/1/2, pair-matched so that B3 minus B2 remains a clean paired
contrast. Core execution requires a SEPARATE explicit user approval which
has not been given, so every training entry currently refuses.
E9, E10, F1, F2 and the clean test remain refused.

Nothing here reads, resolves or names the embargoed clean-test target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import g21_scorer as g21  # noqa: E402
from experiments.e8b_readout_generation import latents as e8b_latents  # noqa: E402
from experiments.e8b_readout_generation import readouts  # noqa: E402

OUT_DIR = config.RESULTS_DIR / "experiments" / "e8b_readout_generation"

# Authorisation state for the remaining-search phase (user decision of
# 2026-08-07, recorded in protocol_clarification_20260807.json). The
# authorised runs are the eight B3/train_40k/seed0 search grid points of
# master protocol section 7.4, of which point 1 already ran on
# 2026-08-06 as the G19 multiplier pilot; points 2-8 are now authorised.
# Every core cell, every other arm, scale and seed, B1, B2, E9, E10, F1,
# F2 and the clean test remain refused and require further explicit user
# authorisation.
# The eight-point search was permanently STOPPED by the user on
# 2026-08-07 (protocol_amendment_20260807_fp32.json, A2). Grid points 4-8
# are not authorised and will not run; grid points 1-3 are retained as
# exploratory / protocol-diagnostic evidence only and no hyperparameter
# winner is claimed. The successor is the 18-cell core matrix under the
# fixed pre-result recipe and FP32 canonical evaluation. Core execution
# requires a SEPARATE explicit user approval, which has not been given:
# this state refuses every training entry.
TRAINING_AUTHORIZED = "core-matrix-frozen-pending-approval"
SEARCH_ABANDONED = True

# The amended protocol family. Every recipe hash in this family differs
# from the superseded BF16 search family, so a search checkpoint can
# never be resumed or reused by a core run (A8).
PROTOCOL_FAMILY = "e8b-fp32-core-2026-08-07"
EVALUATION_PRECISION = "fp32"

# The final pair-matched core matrix (A5): 3 arms x 2 scales x 3 seeds.
CORE_ARMS = ("B1", "B2", "B3")
CORE_SCALES = ("train_40k", "train_250k")
CORE_SEEDS = (0, 1, 2)
CORE_CELLS = tuple((arm, scale, seed) for arm in CORE_ARMS
                   for scale in CORE_SCALES for seed in CORE_SEEDS)
PILOT_CELL = ("B3", "train_40k", 0)
PILOT_HYPER = {"lr": 3e-4, "warmup_frac": 0.0, "dropout": 0.1}

# HISTORICAL ONLY. The frozen eight-point grid of master protocol
# section 7.4, transcribed verbatim. The search was permanently stopped
# on 2026-08-07; this table is retained so the abandoned design stays
# auditable and so tests can assert it was never extended. No training
# entry consumes it.
SEARCH_GRID = (
    {"grid_point": 1, "lr": 3e-4, "warmup_frac": 0.0, "dropout": 0.1},
    {"grid_point": 2, "lr": 3e-4, "warmup_frac": 0.0, "dropout": 0.3},
    {"grid_point": 3, "lr": 3e-4, "warmup_frac": 0.03, "dropout": 0.1},
    {"grid_point": 4, "lr": 3e-4, "warmup_frac": 0.03, "dropout": 0.3},
    {"grid_point": 5, "lr": 1e-3, "warmup_frac": 0.0, "dropout": 0.1},
    {"grid_point": 6, "lr": 1e-3, "warmup_frac": 0.0, "dropout": 0.3},
    {"grid_point": 7, "lr": 1e-3, "warmup_frac": 0.03, "dropout": 0.1},
    {"grid_point": 8, "lr": 1e-3, "warmup_frac": 0.03, "dropout": 0.3},
)
SEARCH_CELL = PILOT_CELL
# Section 7.4 selection: highest development R1 accuracy; ties broken by
# the lowest grid index, then the earlier epoch.
SELECTION_METRIC = "development R1 accuracy, EOS included"
SELECTION_TIE_BREAK = "lowest grid index, then the earlier epoch"

# U4 is WITHDRAWN by the amendment of 2026-08-07 (A4). No search
# checkpoint is promoted into the final core matrix; every core cell is
# trained fresh under the amended protocol. The superseded decision is
# preserved verbatim in u4_decision.json as historical record.
U4_DECIDED = "withdrawn-2026-08-07"

MODEL_REPO = e8a.MODEL_REPO
MODEL_REVISION = e8a.MODEL_REVISION
RANDOM_INIT_SEED = e8a.RANDOM_INIT_SEED

# The only E8B-min arms. No 360M identity exists in this registry (U1).
ARMS = {
    "B1": {"kind": "classifier", "lm": None,
           "description": "interface-width attention-pool control"},
    "B2": {"kind": "lm_readout", "lm": "random",
           "identity": "random_smollm2_135m",
           "description": "frozen random SmolLM2-135M readout"},
    "B3": {"kind": "lm_readout", "lm": "pretrained",
           "identity": "pretrained_smollm2_135m",
           "description": "frozen pretrained SmolLM2-135M readout"},
}
SCALES = ("train_40k", "train_250k")
SEEDS = (0, 1, 2)

# Resource constants (canonical section 14 and the accepted design).
WALL_CLOCK_HALT_HOURS = 8.0
PER_IDENTITY_CEILING_HOURS = 35.0
CORE_CEILING_HOURS = 180.0
MEMORY_CEILING_FRACTION = 0.80
IDENTITY_BASELINE_HOURS = {"pretrained_smollm2_135m": 2.29222,
                           "random_smollm2_135m": 1.95811}
EXPECTED_EPOCHS = {"train_40k": 15, "train_250k": 22}   # U3 basis
WORST_CASE_EPOCHS = {"train_40k": 100, "train_250k": 100}
STEPS_PER_EPOCH = {"train_40k": 313, "train_250k": 1954}  # ceil(N/128)

EXECUTION_LOCK_DIR = Path("/user/HS400/mc02623/backups"
                          "/efficient-vision-language-reasoning/e8b-locks")


# --- B1: interface-width attention-pool readout ------------------------------

class AttentionPoolReadout(nn.Module):
    """One learned query; dot-product over the 32 latents scaled by
    1/sqrt(512); softmax; weighted sum; LayerNorm; Linear(512, 100).
    Parameters 512 + 1,024 + 51,300 = 52,836 (canonical section 4)."""

    def __init__(self, n_classes: int = 100):
        super().__init__()
        self.query = nn.Parameter(torch.randn(e8b_latents.D_MODEL) * 0.02)
        self.norm = nn.LayerNorm(e8b_latents.D_MODEL)
        self.readout = nn.Linear(e8b_latents.D_MODEL, n_classes)

    def forward(self, latent_states):
        scores = (latent_states @ self.query) / (e8b_latents.D_MODEL ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        pooled = (weights.unsqueeze(-1) * latent_states).sum(dim=1)
        return self.readout(self.norm(pooled)), weights


# --- G2: pinned model loading -------------------------------------------------

def load_frozen_causal_lm(pretrained: bool, device=None) -> tuple:
    """The frozen SmolLM2-135M as a CAUSAL LM (with the tied LM head),
    pretrained (B3) or pinned-seed random (B2). Offline-safe: everything
    resolves from the local pinned cache; nothing is downloaded."""
    import tokenizers
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    if pretrained:
        lm = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO, revision=MODEL_REVISION, dtype=torch.bfloat16)
        construction = "AutoModelForCausalLM.from_pretrained at the pinned " \
                       "revision"
    else:
        utils.set_seed(RANDOM_INIT_SEED)
        lm = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
        lm = lm.to(torch.bfloat16)
        construction = ("utils.set_seed(20260802); from_config in float32; "
                        "cast to bfloat16; no pretrained weight loaded")
    for parameter in lm.parameters():
        parameter.requires_grad_(False)
    lm.eval()
    if device is not None:
        lm = lm.to(device)
    embed = lm.get_input_embeddings().weight
    head = lm.get_output_embeddings().weight
    if embed.data_ptr() != head.data_ptr():
        raise AssertionError("G2: SmolLM2 embeddings are not tied; the "
                             "causal head is not the embedding table")
    if cfg.bos_token_id != 0 or cfg.eos_token_id != 0:
        raise AssertionError(
            f"G2: pinned BOS/EOS ids are (0, 0) but the loaded config has "
            f"({cfg.bos_token_id}, {cfg.eos_token_id}); the explicit BOS "
            f"embedding and readouts.EOS_ID both depend on this pin")
    assert cfg.hidden_size == e8b_latents.D_LM
    assert not any(p.requires_grad for p in lm.parameters())
    snapshot = e8a.model_snapshot_dir()
    provenance = {
        "arm_model": "pretrained" if pretrained else "random",
        "repo_id": MODEL_REPO, "pinned_revision": MODEL_REVISION,
        "construction": construction,
        "tied_embeddings_verified": True,
        "parameter_count": int(sum(p.numel() for p in lm.parameters())),
        "config_sha256": e8a.sha256_file(snapshot / "config.json"),
        "tokenizer_file_sha256": e8a.sha256_file(snapshot / "tokenizer.json"),
        "state_dict_sha256": e8a.sha256_state_dict(lm.state_dict()),
        # Canonical section 20 requires the resolved revision, both
        # library versions and the loaded dtype beside the pinned
        # revision; E8A records the same set (e8a_common.py:548-551).
        "resolved_revision": getattr(cfg, "_commit_hash", None)
        or snapshot.name,
        "resolved_snapshot_path": str(snapshot),
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
        "loaded_dtype": str(next(lm.parameters()).dtype),
    }
    return lm, provenance


def build_arm(arm: str, seed: int, dropout: float, lm) -> tuple:
    """G13 order: the caller loads and freezes the LM first; then seed,
    trunk, projection (or B1 readout) with nothing in between.

    Both pair members take the projection scale from the SAME cached CPU
    copy of the pretrained embedding table (`_pretrained_embed_weight`),
    never from `lm` directly: a CPU and a CUDA reduction over the same
    values can differ in the last ulp, which would break the G13 bitwise
    identity of the alpha buffer across the B2/B3 pair. The preflight
    asserts that B3's own table equals this cached copy bitwise.
    """
    if arm == "B1":
        utils.set_seed(seed)
        trunk = e8b_latents.LatentTrunk(dropout)
        readout = AttentionPoolReadout()
        return trunk, readout
    trunk, projection = e8b_latents.build_trunk_and_projection(seed, dropout)
    scale_record = e8b_latents.apply_projection_scale(
        projection, trunk, _pretrained_embed_weight())
    model = e8b_latents.E8BPrefixModel(trunk, projection)
    model.projection_scale_record = scale_record
    return model, scale_record


_PRETRAINED_EMBED_CACHE = {}


def _pretrained_embed_weight() -> torch.Tensor:
    """B2 uses the PRETRAINED model's embed RMS for its projection scale
    (canonical section 5: the same fixed scale for both pair members)."""
    if "weight" not in _PRETRAINED_EMBED_CACHE:
        from transformers import AutoModelForCausalLM
        pretrained = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO, revision=MODEL_REVISION, dtype=torch.bfloat16)
        _PRETRAINED_EMBED_CACHE["weight"] = \
            pretrained.get_input_embeddings().weight.detach().clone()
        del pretrained
    return _PRETRAINED_EMBED_CACHE["weight"]


# --- Interventions: trunk-input replacements only ----------------------------

NEUTRAL_IMAGE_SHA_PREFIX = "8cb31f37"
NEUTRAL_QUESTION_SHA_PREFIX = "64589b2e"


def intervention_inputs(kind: str, image_tokens, question_tokens,
                        question_mask, neutral_image, neutral_question,
                        neutral_question_mask, deranged_image_tokens=None):
    """Replace TRUNK INPUTS for the matched interventions; never the
    latents or the LM prefix. `kind` in {normal, fixed_image,
    fixed_question, shuffled_image}."""
    if kind == "normal":
        return image_tokens, question_tokens, question_mask
    if kind == "fixed_image":
        batch = image_tokens.shape[0]
        fixed = neutral_image.unsqueeze(0).expand(batch, -1, -1)
        return fixed, question_tokens, question_mask
    if kind == "fixed_question":
        batch = image_tokens.shape[0]
        fixed = neutral_question.unsqueeze(0).expand(batch, -1, -1)
        mask = neutral_question_mask.unsqueeze(0).expand(batch, -1)
        return image_tokens, fixed, mask
    if kind == "shuffled_image":
        if deranged_image_tokens is None:
            raise AssertionError("shuffled_image needs the deranged batch")
        return deranged_image_tokens, question_tokens, question_mask
    raise AssertionError(f"unknown intervention {kind!r}")


# --- Resumable checkpoint format ----------------------------------------------

RESUME_FIELDS = (
    "model_state", "optimizer_state", "scheduler_state", "epoch",
    "global_step", "best_model_state", "best_metric", "best_epoch",
    "python_rng", "numpy_rng", "torch_cpu_rng", "cuda_rng_all",
    "loader_generator_state", "epoch_permutation_counter", "recipe_sha256",
    "vocabulary_sha256", "store_sha256s", "code_head",
    "environment_fingerprint")


def environment_fingerprint() -> dict:
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                             "--format=csv,noheader"], capture_output=True,
                            text=True).stdout.strip()
    return {"hostname": socket.gethostname(),
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else None),
            "driver": driver, "cuda": torch.version.cuda,
            "python": sys.version.split()[0], "torch": torch.__version__}


def save_resume_checkpoint(path: Path, *, model, optimizer, scheduler,
                           epoch: int, global_step: int, best_model_state,
                           best_metric: float, best_epoch: int,
                           loader_generator, epoch_permutation_counter: int,
                           recipe_sha256: str, vocabulary_sha256: str,
                           store_sha256s: dict) -> None:
    """Atomic write of the complete resumable state. Every RESUME_FIELDS
    entry is present by construction."""
    import random
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch, "global_step": global_step,
        "best_model_state": best_model_state,
        "best_metric": best_metric, "best_epoch": best_epoch,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_cpu_rng": torch.get_rng_state(),
        "cuda_rng_all": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else []),
        "loader_generator_state": loader_generator.get_state(),
        "epoch_permutation_counter": epoch_permutation_counter,
        "recipe_sha256": recipe_sha256,
        "vocabulary_sha256": vocabulary_sha256,
        "store_sha256s": store_sha256s,
        "code_head": utils.run_metadata()["git_commit"],
        "environment_fingerprint": environment_fingerprint(),
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def verify_resume_checkpoint(path: Path, *, same_node_required: bool = False,
                             recipe_sha256: str | None = None) -> dict:
    """Fail-closed verification: ANY missing field prohibits resume and the
    seed must restart from scratch; partial states are never combined."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    missing = [f for f in RESUME_FIELDS if f not in state]
    if missing:
        raise AssertionError(
            f"RESUME PROHIBITED: checkpoint {path.name} is missing "
            f"{missing}; the seed restarts from scratch and partial states "
            f"are never combined")
    if recipe_sha256 is not None and state["recipe_sha256"] != recipe_sha256:
        raise AssertionError("RESUME PROHIBITED: recipe hash mismatch")
    here = environment_fingerprint()
    written = state["environment_fingerprint"]
    if same_node_required and written["hostname"] != here["hostname"]:
        raise AssertionError(
            f"RESUME PROHIBITED: checkpoint written on "
            f"{written['hostname']}, resume attempted on "
            f"{here['hostname']} with same-node required")
    return state


def restore_resume_state(state: dict, *, model, optimizer, scheduler,
                         loader_generator) -> dict:
    import random
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_cpu_rng"])
    if torch.cuda.is_available() and state["cuda_rng_all"]:
        torch.cuda.set_rng_state_all(state["cuda_rng_all"])
    loader_generator.set_state(state["loader_generator_state"])
    return {"epoch": state["epoch"], "global_step": state["global_step"],
            "best_model_state": state["best_model_state"],
            "best_metric": state["best_metric"],
            "best_epoch": state["best_epoch"],
            "epoch_permutation_counter":
                state["epoch_permutation_counter"]}


# --- Locks and atomicity ------------------------------------------------------

def acquire_run_lock(arm: str, scale: str, seed: int,
                     lock_dir: Path = EXECUTION_LOCK_DIR):
    """NFS-backed, identity/scale/seed-specific: at most one execution of a
    given cell anywhere on the pool. O_CREAT|O_EXCL is atomic on NFSv4."""
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"e8b_{arm}_{scale}_seed{seed}.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        holder = lock_path.read_text() if lock_path.exists() else "unknown"
        raise AssertionError(
            f"DUPLICATE RUN PREVENTED: {lock_path.name} held ({holder}); "
            f"this cell is or was executing elsewhere") from None
    os.write(descriptor, json.dumps(
        {"host": socket.gethostname(), "pid": os.getpid(),
         "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    ).encode())
    os.close(descriptor)
    return lock_path


def atomic_write_json(path: Path, payload: dict) -> None:
    if path.exists():
        sys.exit(f"{path} already exists; E8B records are immutable")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(temporary, path)


# --- Resource projections (U3) ------------------------------------------------

def project_resources(seconds_per_epoch_40k: float,
                      eval_hours_range=(5.83, 11.67)) -> dict:
    """Both bases, per U3: expected-epoch 15/22 governs go/no-go; the
    100-epoch worst case is always reported beside it. Stop-and-return on
    any projected gate failure; nothing is descoped automatically (U2)."""
    per_epoch = {"train_40k": seconds_per_epoch_40k,
                 "train_250k": seconds_per_epoch_40k
                 * STEPS_PER_EPOCH["train_250k"]
                 / STEPS_PER_EPOCH["train_40k"]}

    def run_hours(scale, epochs):
        return per_epoch[scale] * epochs[scale] / 3600

    def matrix_hours(epochs):
        search = 8 * run_hours("train_40k", epochs)
        core_lm_one_arm = 3 * run_hours("train_40k", epochs) \
            + 3 * run_hours("train_250k", epochs)
        return search, core_lm_one_arm

    projections = {}
    for label, epochs in (("expected_epoch_15_22", EXPECTED_EPOCHS),
                          ("worst_case_100_epoch", WORST_CASE_EPOCHS)):
        search, core_arm = matrix_hours(epochs)
        largest_250k = run_hours("train_250k", epochs)
        pretrained_total = (IDENTITY_BASELINE_HOURS[
            "pretrained_smollm2_135m"] + search + core_arm
            + eval_hours_range[1] * 80 / 140)
        random_total = (IDENTITY_BASELINE_HOURS["random_smollm2_135m"]
                        + core_arm + eval_hours_range[1] * 36 / 140)
        projections[label] = {
            "seconds_per_epoch": {k: round(v, 1)
                                  for k, v in per_epoch.items()},
            "largest_250k_run_hours": round(largest_250k, 3),
            "search_hours": round(search, 3),
            "core_hours_per_lm_arm": round(core_arm, 3),
            "pretrained_identity_projection_hours":
                round(pretrained_total, 3),
            "random_identity_projection_hours": round(random_total, 3),
            "gates": {
                "per_run_8h": {"fires": largest_250k > WALL_CLOCK_HALT_HOURS,
                               "value": round(largest_250k, 3)},
                "pretrained_identity_35h": {
                    "fires": pretrained_total > PER_IDENTITY_CEILING_HOURS,
                    "value": round(pretrained_total, 3)},
                "random_identity_35h": {
                    "fires": random_total > PER_IDENTITY_CEILING_HOURS,
                    "value": round(random_total, 3)},
            }}
    governing = projections["expected_epoch_15_22"]["gates"]
    any_fires = any(g["fires"] for g in governing.values())
    return {"projections": projections,
            "governing_basis": "expected_epoch_15_22 (U3)",
            "worst_case_always_reported": True,
            "stop_and_return": any_fires,
            "stop_note": ("a fired projection STOPS AND RETURNS to the "
                          "user; no arm is descoped automatically (U2)"
                          if any_fires else None)}


# --- Bounded, non-scientific preflight ---------------------------------------

def preflight(device) -> int:
    """Everything the phase authorises, nothing more. No optimizer.step, no
    parameter update, no epoch, no G19 timing, no accuracy, no selection.
    Every output is labelled NON-SCIENTIFIC."""
    started = time.time()
    tokenizer = e8a.load_tokenizer()
    answers, vocabulary = g21.load_index_to_answer(
        config.DATA_DIR / "v2" / "answer_vocab_v2.json")
    cache = readouts.build_answer_cache(tokenizer, answers)
    trie = readouts.build_trie(cache)
    print(f"[G1] cache sha {cache['sha256'][:16]} max_len "
          f"{cache['max_length']} prefix_pairs "
          f"{cache['strict_prefix_pairs']} (recorded)")

    results = {"label": "NON-SCIENTIFIC PREFLIGHT",
               "g1": {k: cache[k] for k in ("max_length",
                                            "length_histogram",
                                            "strict_prefix_pairs",
                                            "sha256")},
               "vocabulary_sha256": vocabulary["sha256"]}

    lms = {}
    for arm in ("B2", "B3"):
        lm, provenance = load_frozen_causal_lm(
            ARMS[arm]["lm"] == "pretrained", device)
        lms[arm] = lm
        results[f"g2_{arm}"] = {k: provenance[k] for k in (
            "arm_model", "pinned_revision", "tied_embeddings_verified",
            "parameter_count", "config_sha256", "tokenizer_file_sha256")}
        print(f"[G2] {arm}: {provenance['parameter_count']:,} params, "
              f"tied embeddings verified")

    # G13: paired trunk/projection bit identity at one probe seed.
    probe_seed = 0
    models = {}
    for arm in ("B2", "B3"):
        model, scale_record = build_arm(arm, probe_seed, 0.1, lms[arm])
        models[arm] = model
        results[f"projection_scale_{arm}"] = scale_record
    pair_identical = all(
        torch.equal(a, b) for a, b in zip(
            models["B2"].state_dict().values(),
            models["B3"].state_dict().values()))
    if not pair_identical:
        sys.exit("G13 FAILED: B2/B3 trunk+projection initial states differ")
    results["g13_pair_bitwise_identical"] = True
    print("[G13] B2/B3 trunk and projection initial states bitwise "
          "identical at the probe seed")

    trainable = e8b_latents.count_parameters(models["B3"])
    b1_trunk, b1_readout = build_arm("B1", probe_seed, 0.1, None)
    results["parameter_counts"] = {
        "lm_arm_trainable": trainable,
        "trunk": e8b_latents.count_parameters(models["B3"].trunk),
        "projection": e8b_latents.count_parameters(
            models["B3"].projection),
        "b1_readout": e8b_latents.count_parameters(b1_readout),
        "frozen_lm": int(sum(p.numel() for p in lms["B3"].parameters()))}
    assert results["parameter_counts"]["b1_readout"] == 52_836

    # B3's own embedding table must equal the cached CPU copy both pair
    # members took their projection scale from (see build_arm).
    if not torch.equal(
            lms["B3"].get_input_embeddings().weight.detach().cpu(),
            _pretrained_embed_weight()):
        sys.exit("PREFLIGHT FAILED: B3's embedding table differs from the "
                 "cached pretrained copy used for the projection scale")
    results["b3_embed_equals_scale_source"] = True

    results["tokenizer_facts"] = {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "note": "pad_token None in the pinned tokenizer; the unpadded "
                "G14 references never need one"}

    # One forward/backward per LM arm on a tiny fixed sample; gradients must
    # exist exactly on trainable modules and never on the frozen LM. The
    # question store is packed (ids[i] owns tokens[offsets[i]:+lengths[i]]),
    # so the sample question block is sliced by its recorded offset and
    # length, never by a raw row range.
    stores_dir = config.DATA_DIR / "v3" / "tokens"
    import h5py
    with h5py.File(stores_dir / "image_tokens.h5", "r") as store:
        image_store_shape = list(store["tokens"].shape)
        image_tokens = torch.from_numpy(
            store["tokens"][:2].astype(np.float32)).to(device)
    with h5py.File(stores_dir / "question_tokens.h5", "r") as store:
        question_store_shape = list(store["tokens"].shape)
        offset = int(store["offsets"][0])
        length = int(store["lengths"][0])
        block = store["tokens"][offset:offset + length].astype(np.float32)
    question_tokens = torch.from_numpy(block).unsqueeze(0).repeat(
        2, 1, 1).to(device)
    question_mask = torch.zeros(2, length, dtype=torch.bool, device=device)
    results["store_shapes"] = {
        "image_tokens": image_store_shape,
        "question_tokens_packed": question_store_shape,
        "sample_question_length": length}
    answer_ids = [cache["sequences"][0], cache["sequences"][1]]

    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    for arm in ("B2", "B3"):
        model = models[arm].to(device)
        prefix = model.prefix_embeddings(lms[arm], image_tokens,
                                         question_tokens, question_mask)
        loss, per_example = model.teacher_forced_loss(
            lms[arm], prefix.to(torch.bfloat16), answer_ids)
        loss.backward()
        trainable_grads = all(
            p.grad is not None for p in model.parameters())
        frozen_grads = any(
            p.grad is not None for p in lms[arm].parameters())
        if not trainable_grads or frozen_grads:
            sys.exit(f"GRADIENT GATE FAILED for {arm}: trainable_grads="
                     f"{trainable_grads} frozen_grads={frozen_grads}")
        model.zero_grad(set_to_none=True)
        results[f"forward_backward_{arm}"] = {
            "loss": round(float(loss.detach()), 4),
            "per_example_nll": [round(float(v), 4)
                                for v in per_example.detach()],
            "gradients_on_trainable_only": True}
        print(f"[FWD/BWD] {arm}: loss {float(loss.detach()):.4f}, "
              f"gradients on trainable modules only (NON-SCIENTIFIC)")

    # Trainable modules go to eval() for every readout check below, so the
    # trunk's dropout is off and the recorded gates are deterministic.
    models["B2"].eval()
    models["B3"].eval()

    # Tiny fixed-sample readout checks + G14 gates on 2 prefixes (B3).
    with torch.no_grad():
        b3 = models["B3"].to(device)
        prefixes = [b3.prefix_embeddings(
            lms["B3"], image_tokens[i:i + 1],
            question_tokens[i:i + 1], question_mask[i:i + 1]
        ).to(torch.bfloat16) for i in range(2)]
    results["g14_r1"] = readouts.g14_r1_gate(lms["B3"], prefixes, cache)
    results["g14_r2"] = readouts.g14_r2_gate(lms["B3"], prefixes, cache,
                                             trie)
    r3 = readouts.r3_generate(lms["B3"], prefixes[0], tokenizer)
    results["r3_sample"] = {k: r3[k] for k in (
        "n_generated", "terminated_by_eos", "empty", "overlong")}
    results["r3_sample"]["text_repr"] = repr(r3["text"])
    print(f"[G14] R1 argmax identity on {results['g14_r1']['examples']} "
          f"prefixes (max score delta "
          f"{results['g14_r1']['max_abs_score_delta']:.2e}); R2 identical")

    # E7b serial-extension smoke: every S8-S10 stage and integrity guard
    # executes once per readout, cached-feature regime always and the full
    # S1-S10 raw path when the pinned E7b row and its image are present.
    # NON-SCIENTIFIC; never aggregated with any timing record.
    from experiments.e8b_readout_generation import serial_efficiency
    rows_file = (config.RESULTS_DIR / "experiments"
                 / "e7b_serial_efficiency" / "benchmark_rows.json")
    full_row = None
    if rows_file.exists():
        candidate = json.loads(rows_file.read_text())["rows"][0]
        if (PROJECT_ROOT / candidate["image_path"]).exists():
            full_row = candidate
    results["e7b_serial_extension"] = serial_efficiency.nonscientific_smoke(
        lms["B3"], tokenizer, models["B3"], cache, trie, answers,
        image_tokens[:1], question_tokens[:1], question_mask[:1],
        full_path_row=full_row, device=device)
    print("[E7B-EXT] serial cached-feature smoke ran all three readouts"
          + ("; full S1-S10 path smoke ran" if full_row else
             "; full-path smoke skipped (no pinned row/image)")
          + " (NON-SCIENTIFIC)")

    if device.type == "cuda":
        results["memory"] = {
            "peak_allocated_mib": round(
                torch.cuda.max_memory_allocated() / 2 ** 20, 1),
            "peak_reserved_mib": round(
                torch.cuda.max_memory_reserved() / 2 ** 20, 1)}

    # Checkpoint save/load/resume-format validation on the live B3 modules
    # with a throwaway optimiser: constructed, never stepped.
    optimizer = torch.optim.AdamW(models["B3"].parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    generator = utils.make_generator(0)
    checkpoint_path = OUT_DIR / "preflight_resume_probe.pt"
    checkpoint_path.unlink(missing_ok=True)
    save_resume_checkpoint(
        checkpoint_path, model=models["B3"], optimizer=optimizer,
        scheduler=scheduler, epoch=0, global_step=0,
        best_model_state=None, best_metric=0.0, best_epoch=-1,
        loader_generator=generator, epoch_permutation_counter=0,
        recipe_sha256="preflight-probe", vocabulary_sha256=
        vocabulary["sha256"], store_sha256s={})
    state = verify_resume_checkpoint(checkpoint_path,
                                     recipe_sha256="preflight-probe")
    restored = restore_resume_state(state, model=models["B3"],
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    loader_generator=generator)
    if restored["epoch"] != 0 or restored["global_step"] != 0 \
            or restored["epoch_permutation_counter"] != 0:
        sys.exit("RESUME FORMAT FAILED: the restored counters do not match "
                 "the saved probe state")
    results["resume_format"] = {
        "fields_present": sorted(state.keys()),
        "complete": all(f in state for f in RESUME_FIELDS),
        "restore_round_trip": True,
        "optimizer_stepped": False}
    checkpoint_path.unlink()

    results["projection_demo_1s_per_epoch"] = project_resources(60.0)
    results["wall_seconds"] = round(time.time() - started, 1)
    results["optimizer_step_count"] = 0
    results["clean_test_accessed"] = False
    out = OUT_DIR / "preflight_nonscientific.json"
    out.unlink(missing_ok=True)
    atomic_write_json(out, {"metadata": utils.run_metadata(),
                            "e8b_preflight_nonscientific": results})
    print(f"preflight complete in {results['wall_seconds']}s; "
          f"NON-SCIENTIFIC record written")
    return 0


# --- G19 halt machinery (master protocol section 14) --------------------------

def pin_fp32_precision() -> dict:
    """Pin and record the FP32 matmul path (amendment OI4). Canonical
    FP32 must be true IEEE fp32: TF32 carries a 10-bit mantissa and
    would silently degrade the canonical path to roughly eight times
    BF16's resolution while every gate still called itself FP32."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {"cuda_matmul_allow_tf32":
                bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision":
                torch.get_float32_matmul_precision(),
            "rule": "canonical FP32 is true IEEE fp32; TF32 is pinned off"}


def promote_lm_to_fp32(lm) -> dict:
    """Amendment A1: canonical evaluation promotes the EXACT frozen state
    values to FP32. The pinned checkpoint stores BF16 on disk and BF16 is
    a truncated FP32, so the promotion is lossless and the model identity
    is unchanged -- only the arithmetic precision of the forward pass
    changes. Returns the identity record that every core result must
    carry, including both dtype hashes so the fp32 hash can never be
    misread as a different model."""
    before = {k: v.clone() for k, v in lm.state_dict().items()}
    bf16_sha = e8a.sha256_state_dict(before)
    lm = lm.to(torch.float32)
    after = lm.state_dict()
    round_trips = all(torch.equal(after[k].to(torch.bfloat16), before[k])
                      for k in before)
    if not round_trips:
        sys.exit("FP32 PROMOTION REFUSED: the promoted state does not "
                 "round-trip to the frozen bfloat16 values, so the "
                 "promotion would not be identity-preserving")
    for parameter in lm.parameters():
        parameter.requires_grad_(False)
    lm.eval()
    assert not any(p.requires_grad for p in lm.parameters())
    assert not lm.training
    return {"evaluation_precision": "fp32",
            "promotion": "the pinned frozen bfloat16 state values "
                         "promoted to fp32; identity-preserving",
            "round_trips_to_bf16_exactly": True,
            "bf16_state_dict_sha256": bf16_sha,
            "fp32_state_dict_sha256": e8a.sha256_state_dict(after),
            "sha_note": "the two hashes differ only because the digest "
                        "covers serialised bytes at different dtypes; "
                        "the VALUES are bitwise identical"}


def assert_core_family(state: dict, run_name: str, path: Path) -> None:
    """Amendment A8: a core run must never resume or reuse a checkpoint
    from the superseded BF16 search family. Recorded halt, not an
    uncaught assertion."""
    family = state.get("protocol_family")
    if family != PROTOCOL_FAMILY:
        gate_halt(run_name, "FAMILY",
                  f"checkpoint belongs to protocol family "
                  f"{family!r}, not {PROTOCOL_FAMILY!r}; the superseded "
                  f"BF16 search checkpoints are exploratory evidence "
                  f"and are never resumed or reused by a core run",
                  {"checkpoint": str(path), "found_family": family,
                   "required_family": PROTOCOL_FAMILY})


def halt_record_path(run_name: str, gate: str) -> Path:
    """One atomic JSON per halt, never overwritten. The gate name is part
    of the filename so a second, different failure cannot silently
    replace the first."""
    return OUT_DIR / f"HALT_{run_name}_{gate}.json"


def record_gate_halt(run_name: str, gate: str, reason: str,
                     detail: dict | None = None) -> Path:
    """Master protocol section 18: a gate failure is recorded verbatim.
    Every halting gate writes an atomic JSON artefact before exiting, so
    a halt is never evidenced only on stderr. Written with a temporary
    file and os.replace; an existing record for the same run and gate is
    never overwritten, and a suffixed record is written instead."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": utils.run_metadata(),
               "gate_halt": {"run": run_name, "gate": gate,
                             "reason": reason,
                             "detail": detail or {},
                             "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                  time.gmtime()),
                             "host": socket.gethostname(),
                             "status": "FAILED: halting gate fired; this "
                                       "run's result is not used and is "
                                       "never resumed automatically",
                             "clean_test_accessed": False}}
    path = halt_record_path(run_name, gate)
    if path.exists():
        index = 2
        while path.with_name(f"{path.stem}_{index}.json").exists():
            index += 1
        path = path.with_name(f"{path.stem}_{index}.json")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(temporary, path)
    return path


def gate_halt(run_name: str, gate: str, reason: str,
              detail: dict | None = None) -> None:
    """Record the halt atomically, then stop. Used by every halting gate
    so that no failure is evidenced only on the console."""
    path = record_gate_halt(run_name, gate, reason, detail)
    sys.exit(f"{gate} HALT: {reason} -- recorded at {path.name}; "
             f"execution stops and returns to the user with "
             f"pair-preserving alternatives; no arm is descoped "
             f"automatically (U2, P2)")


def g19_halt(fired: list, run_name: str = "e8b", detail: dict | None = None
             ) -> None:
    """Section 14: the halt is an explicit sys.exit with a recorded
    reason, in the style of the v3_01 GATE 4 wall-clock gate. Execution
    stops and returns to the user with pair-preserving alternatives;
    no arm is ever descoped automatically (U2, P2)."""
    if fired:
        gate_halt(run_name, "G19", "; ".join(fired), detail)


def memory_gate(allocated_bytes: int, total_bytes: int,
                reserved_bytes: int | None = None) -> dict:
    """Peak memory against 80 per cent of the memory the device actually
    reports. The user's decision of 2026-08-07 (P3) makes the RESERVED
    peak the hard gate: reserved is what the caching allocator has taken
    from the device and is the honest footprint, while allocated
    understates it (the E8B preflight measured 1333.2 MiB allocated
    against 1454.0 MiB reserved, a 9.1 per cent gap). Both are always
    recorded."""
    if reserved_bytes is None:
        reserved_bytes = allocated_bytes
    allocated_fraction = allocated_bytes / total_bytes
    reserved_fraction = reserved_bytes / total_bytes
    return {"peak_allocated_mib": round(allocated_bytes / 2 ** 20, 1),
            "peak_reserved_mib": round(reserved_bytes / 2 ** 20, 1),
            "device_total_mib": round(total_bytes / 2 ** 20, 1),
            "allocated_fraction": round(allocated_fraction, 4),
            "reserved_fraction": round(reserved_fraction, 4),
            "fraction_used": round(reserved_fraction, 4),
            "gate_basis": "peak reserved / device total (P3)",
            "ceiling_fraction": MEMORY_CEILING_FRACTION,
            "fires": reserved_fraction > MEMORY_CEILING_FRACTION}


def storage_gate(required_bytes: int, target_dir: Path) -> dict:
    """Projected additional storage against the space actually free on
    the target filesystem."""
    import shutil
    free = shutil.disk_usage(target_dir).free
    return {"required_gib": round(required_bytes / 2 ** 30, 3),
            "free_gib": round(free / 2 ** 30, 3),
            "fires": required_bytes > free}


def remaining_core_gate(expected_remaining_hours: float,
                        worst_case_remaining_hours: float) -> dict:
    """Section 14 as clarified by the user on 2026-08-07 (P3).

    The GOVERNING projection is the expected-epoch basis, 15 epochs at
    train_40k and 22 at train_250k. It is gated against the 180 GPU-hour
    core ceiling and HALTS if it exceeds it.

    The 100-epoch projection is a mandatory reported stress scenario and
    is explicitly NON-HALTING on its own. Its comparison against
    min(180, 3 x expected) is still computed and reported, under
    `stress_scenario`, so the previous rule's verdict remains visible;
    but `fires` -- the halting verdict -- is the governing comparison
    alone."""
    stress_threshold = min(CORE_CEILING_HOURS,
                           3.0 * expected_remaining_hours)
    return {"governing_basis": "expected epochs 15/22 (U3 as clarified "
                               "2026-08-07, P3)",
            "expected_remaining_hours": round(expected_remaining_hours, 3),
            "core_ceiling_hours": CORE_CEILING_HOURS,
            "rule": "governing expected remaining core > 180",
            "fires": expected_remaining_hours > CORE_CEILING_HOURS,
            "stress_scenario": {
                "worst_case_remaining_hours":
                    round(worst_case_remaining_hours, 3),
                "threshold_hours": round(stress_threshold, 3),
                "rule": "worst case > min(180, 3 x expected)",
                "exceeds_threshold":
                    worst_case_remaining_hours > stress_threshold,
                "halting": False,
                "note": "mandatory reported stress scenario; does not "
                        "halt by itself (P3)"}}


# --- Guarded training entry ---------------------------------------------------

def train(arm: str, scale: str, seed: int, grid_point: int | None = None
          ) -> int:
    """The only training entry.

    The eight-point search is permanently abandoned (A2), so no grid
    point runs under any circumstance. The successor is the 18-cell core
    matrix under the fixed pre-result recipe and FP32 canonical
    evaluation, and that matrix requires a SEPARATE explicit user
    approval which has not been given. Every path therefore refuses."""
    if grid_point is not None:
        sys.exit(f"E8B SEARCH REFUSED: the eight-point recipe search was "
                 f"permanently stopped on 2026-08-07. Grid points 4-8 "
                 f"are not authorised, grid points 1-3 are exploratory "
                 f"evidence only, and no hyperparameter winner is "
                 f"claimed (protocol_amendment_20260807_fp32.json A2)")
    if (arm, scale, seed) not in CORE_CELLS:
        sys.exit(f"E8B TRAINING REFUSED for {arm}/{scale}/seed{seed}: "
                 f"not one of the {len(CORE_CELLS)} core cells "
                 f"(B1/B2/B3 x train_40k/train_250k x seeds 0/1/2). "
                 f"E9, E10, F1, F2 and the clean test remain refused.")
    if TRAINING_AUTHORIZED != "core-matrix-approved":
        sys.exit(f"E8B CORE TRAINING IS NOT AUTHORISED: the amended "
                 f"protocol and the 18-cell matrix are frozen and under "
                 f"review, and the recorded authorisation state is "
                 f"{TRAINING_AUTHORIZED!r}. Core execution requires a "
                 f"separate explicit user approval.")
    # Reached only if a future approval flips TRAINING_AUTHORIZED. The
    # core trainer is deliberately NOT implemented yet: three amendment
    # open issues must be settled first (OI1 training determinism, OI2
    # the epoch-budget confound, OI3 per-row prediction dumps), and B1
    # needs its classifier training path. Refuse explicitly rather than
    # dispatch to a half-built runner.
    sys.exit("E8B CORE TRAINER NOT IMPLEMENTED: the amended protocol is "
             "frozen but the core training path is not yet built. "
             "Outstanding: the training-determinism probe (OI1), the "
             "epoch-budget decision (OI2), per-row prediction dumps "
             "(OI3), and the B1 classifier training path. Implement and "
             "re-review before flipping the authorisation state.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--pilot", action="store_true",
                        help="run the G19 pilot cell, search grid point 1")
    parser.add_argument("--search-point", type=int, default=None,
                        metavar="N", help="RETIRED; the search is stopped")
    parser.add_argument("--core-cell", nargs=3,
                        metavar=("ARM", "SCALE", "SEED"),
                        help="run one of the 18 core cells (requires a "
                             "separate explicit user approval)")
    args = parser.parse_args()
    utils.set_seed()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.preflight:
        return preflight(device)
    if args.pilot or args.search_point is not None:
        sys.exit("the eight-point recipe search was permanently stopped "
                 "on 2026-08-07; --pilot and --search-point are retired")
    if args.core_cell:
        arm, scale, seed = args.core_cell
        return train(arm, scale, int(seed))
    parser.error("this phase supports --preflight and "
                 "--core-cell ARM SCALE SEED only")


if __name__ == "__main__":
    raise SystemExit(main())
