"""OI1: NON-SCIENTIFIC strict-determinism probe.

Three epochs at train_40k under the amended execution design: BF16
autocast training, canonical FP32 evaluation, strict deterministic
algorithms (`warn_only=False`), TF32 pinned off, FP32 rotary buffers.

Two independently launched runs must agree BIT-FOR-BIT on every recorded
quantity: per-step losses, per-epoch canonical FP32 development R1
accuracy, the full development prediction vector, the model state, the
optimizer state and the global step counter.

PRECISE SCOPE. This probe DOES perform real training: three epochs and
939 optimizer steps on the B3/train_40k/seed0 configuration under the
CORE recipe. What it does NOT do is complete, write, or promote a
scientific final-core cell or result: it writes only
`determinism_probe_run{N}.json`, touches no core checkpoint path, and
every artefact it writes carries NON_SCIENTIFIC true so the promotion
guard refuses it.

Its standing authorisation was REVOKED on 2026-08-07 after it completed
its purpose; running it again requires a fresh explicit authorisation
granted BEFORE any GPU work.

    python -B experiments/e8b_readout_generation/determinism_probe.py --run 1
    python -B experiments/e8b_readout_generation/determinism_probe.py --run 2
    python -B experiments/e8b_readout_generation/determinism_probe.py --compare
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import g21_scorer as g21  # noqa: E402
from experiments.e8b_readout_generation import readouts  # noqa: E402
from experiments.e8b_readout_generation import run as e8b_run  # noqa: E402
from experiments.e8b_readout_generation import training as e8b_training  # noqa: E402

V2_DIR = config.DATA_DIR / "v2"
OUT_DIR = e8b_run.OUT_DIR
PROBE_EPOCHS = 3
PROBE_ARM, PROBE_SCALE, PROBE_SEED = "B3", "train_40k", 0


def tensor_digest(tensor) -> str:
    array = tensor.detach().cpu().contiguous()
    return hashlib.sha256(array.numpy().tobytes()).hexdigest()


def state_digest(state) -> str:
    parts = []
    for key in sorted(state):
        value = state[key]
        if torch.is_tensor(value):
            parts.append(f"{key}:{tensor_digest(value)}")
        else:
            parts.append(f"{key}:{json.dumps(value, sort_keys=True, default=str)}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def optimizer_digest(optimizer) -> str:
    state = optimizer.state_dict()
    parts = [json.dumps(state["param_groups"], sort_keys=True)]
    for key in sorted(state["state"]):
        entry = state["state"][key]
        for name in sorted(entry):
            value = entry[name]
            parts.append(f"{key}.{name}:"
                         + (tensor_digest(value) if torch.is_tensor(value)
                            else str(value)))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def probe(run_index: int) -> int:
    # The ONLY authorised non-scientific training (user authorisation of
    # 2026-08-07). Anything else refuses; the probe never writes to a
    # core or search path and is never promoted.
    e8b_run.authorize_optimizer_path("nonscientific-probe",
                                     f"determinism probe run {run_index}")
    from experiments.e8a_question_encoder import reinfer_g21
    reinfer_g21.assert_gpu_exclusive(f"e8b determinism probe {run_index}")
    started = time.time()
    # ORDER IS LOAD-BEARING: utils.set_seed re-enables warn_only, so it
    # must run BEFORE strict determinism is imposed, never after.
    utils.set_seed(PROBE_SEED)
    determinism = e8b_run.enable_strict_determinism()
    precision = e8b_run.pin_fp32_precision()
    determinism["verified_at_use"] = e8b_run.assert_strict_determinism()
    device = torch.device("cuda")

    lm, provenance = e8b_run.load_frozen_causal_lm(True, device=None)
    identity = e8b_run.promote_lm_to_fp32(lm)
    lm = lm.to(device)
    tokenizer = e8a.load_tokenizer()
    answers, vocabulary = g21.load_index_to_answer(
        V2_DIR / "answer_vocab_v2.json")
    cache = readouts.build_answer_cache(tokenizer, answers)

    recipe = e8b_training.build_core_recipe(PROBE_ARM, PROBE_SCALE,
                                            PROBE_SEED)
    model, scale_record = e8b_run.build_arm(PROBE_ARM, recipe["seed"],
                                            recipe["dropout"], lm)
    model = model.to(device)
    # build_arm re-seeds internally (G13 construction order), which
    # downgrades warn_only; re-impose before anything else runs.
    e8b_run.enable_strict_determinism()

    stores = tokens_data.TokenStores()
    train_loader, dev_loader = tokens_data.make_token_loaders(
        V2_DIR / "train_40k.csv", V2_DIR / "dev.csv", stores=stores,
        batch_size=recipe["batch_size"])
    train_loader.generator.manual_seed(recipe["seed"])

    optimizer = e8b_training.make_optimizer(model, recipe["lr"])
    scheduler = e8b_training.make_scheduler(
        optimizer, recipe["max_epochs"] * len(train_loader),
        recipe["warmup_frac"])

    e8b_run.assert_strict_determinism()   # re-read at the point of use
    torch.cuda.reset_peak_memory_stats()
    step_losses, epochs = [], []
    step_count = 0
    for epoch in range(1, PROBE_EPOCHS + 1):
        model.train()
        epoch_start = time.time()
        for images, questions, _, mask, labels in train_loader:
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
            step_losses.append(float(loss.detach()))
        train_seconds = time.time() - epoch_start
        eval_start = time.time()
        predictions, labels_np, _, margins = \
            e8b_training.canonical_dev_predictions(model, lm, dev_loader,
                                                   cache, device)
        accuracy = float((predictions == labels_np).mean())
        epochs.append({
            "epoch": epoch, "global_step": step_count,
            "canonical_fp32_dev_r1": accuracy,
            "predictions_sha256": hashlib.sha256(
                predictions.tobytes()).hexdigest(),
            "margins_sha256": hashlib.sha256(
                margins.tobytes()).hexdigest(),
            "model_state_digest": state_digest(model.state_dict()),
            "optimizer_digest": optimizer_digest(optimizer),
            "train_seconds": round(train_seconds, 1),
            "eval_seconds": round(time.time() - eval_start, 1)})
        print(f"  [probe {run_index}] epoch {epoch}: dev R1 {accuracy:.6f} "
              f"({train_seconds:.0f}s + {epochs[-1]['eval_seconds']:.0f}s)")

    record = {"metadata": utils.run_metadata(),
              "determinism_probe": {
        "NON_SCIENTIFIC": True,
        "scope": "this probe DOES perform real training (3 epochs, 939 "
                 "optimizer steps) on the B3/train_40k/seed0 "
                 "configuration under the core recipe. It does not "
                 "complete, write, or promote a scientific final-core "
                 "cell or result.",
        "never_promoted": "refused by assert_promotable at every final "
                          "aggregation, core-result loading, freeze and "
                          "promotion path, on the NON_SCIENTIFIC flag",
        "run_index": run_index, "epochs": PROBE_EPOCHS,
        "cell": [PROBE_ARM, PROBE_SCALE, PROBE_SEED],
        "recipe_sha256": e8b_training.recipe_sha256(recipe),
        "protocol_family": e8b_run.PROTOCOL_FAMILY,
        "determinism": determinism, "fp32_precision": precision,
        "model_identity": identity,
        "rotary_restoration": provenance["rotary_fp32_restoration"],
        "buffer_inventory": provenance["buffer_inventory"],
        "g13_projection_scale": scale_record,
        "vocabulary_sha256": vocabulary["sha256"],
        "step_losses_sha256": hashlib.sha256(
            np.asarray(step_losses, dtype=np.float64).tobytes()).hexdigest(),
        "step_losses_first10": step_losses[:10],
        "step_losses_last10": step_losses[-10:],
        "total_steps": step_count,
        "epochs_detail": epochs,
        "peak_allocated_mib": round(
            torch.cuda.max_memory_allocated() / 2 ** 20, 1),
        "peak_reserved_mib": round(
            torch.cuda.max_memory_reserved() / 2 ** 20, 1),
        "wall_seconds": round(time.time() - started, 1),
        "clean_test_accessed": False}}
    out = OUT_DIR / f"determinism_probe_run{run_index}.json"
    e8b_run.atomic_write_json(out, record)
    print(f"[probe {run_index}] written {out.name}")
    return 0


def compare() -> int:
    a = json.loads((OUT_DIR / "determinism_probe_run3.json").read_text()
                   )["determinism_probe"]
    b = json.loads((OUT_DIR / "determinism_probe_run4.json").read_text()
                   )["determinism_probe"]
    if not (len(a["epochs_detail"]) == len(b["epochs_detail"])
            == PROBE_EPOCHS):
        sys.exit(f"PROBE COMPARISON INVALID: epoch counts "
                 f"{len(a['epochs_detail'])} vs {len(b['epochs_detail'])} "
                 f"vs the required {PROBE_EPOCHS}; a truncated run must "
                 f"not silently pass")
    checks = {
        "determinism_blocks": a["determinism"] == b["determinism"],
        "fp32_precision_blocks": a["fp32_precision"] == b["fp32_precision"],
        "step_losses": a["step_losses_sha256"] == b["step_losses_sha256"],
        "total_steps": a["total_steps"] == b["total_steps"],
        "recipe_sha256": a["recipe_sha256"] == b["recipe_sha256"],
        "buffer_inventory": a["buffer_inventory"] == b["buffer_inventory"],
        "projection_scale": a["g13_projection_scale"]
                            == b["g13_projection_scale"],
    }
    per_epoch = []
    for ea, eb in zip(a["epochs_detail"], b["epochs_detail"]):
        per_epoch.append({
            "epoch": ea["epoch"],
            "dev_r1_identical": ea["canonical_fp32_dev_r1"]
                                == eb["canonical_fp32_dev_r1"],
            "predictions_identical": ea["predictions_sha256"]
                                     == eb["predictions_sha256"],
            "margins_identical": ea["margins_sha256"]
                                 == eb["margins_sha256"],
            "model_state_identical": ea["model_state_digest"]
                                     == eb["model_state_digest"],
            "optimizer_identical": ea["optimizer_digest"]
                                   == eb["optimizer_digest"],
            "global_step_identical": ea["global_step"] == eb["global_step"],
            "run1_dev_r1": ea["canonical_fp32_dev_r1"],
            "run2_dev_r1": eb["canonical_fp32_dev_r1"]})
    everything = (all(checks.values())
                  and all(all(v for k, v in e.items()
                              if k.endswith("identical"))
                          for e in per_epoch))
    verdict = {"metadata": utils.run_metadata(),
               "determinism_probe_comparison": {
        "NON_SCIENTIFIC": True,
        "strict_determinism": "torch.use_deterministic_algorithms("
                              "True, warn_only=False) with the flash and "
                              "mem-efficient SDPA backends disabled and "
                              "the deterministic math backend enabled",
        "run_level_checks": checks,
        "per_epoch_checks": per_epoch,
        "bitwise_identical": bool(everything),
        "verdict": ("STRICT DETERMINISM CONFIRMED: two independently "
                    "launched runs agree bit-for-bit on losses, canonical "
                    "FP32 development R1, the full prediction vector, "
                    "margins, model state, optimizer state and global "
                    "steps") if everything else
                   ("STRICT DETERMINISM NOT CONFIRMED: see the failing "
                    "checks; do not fall back to warn_only"),
        "clean_test_accessed": False}}
    out = OUT_DIR / "determinism_probe_comparison_strict.json"
    e8b_run.atomic_write_json(out, verdict)
    print(json.dumps(verdict["determinism_probe_comparison"]["run_level_checks"],
                     indent=2))
    for e in per_epoch:
        print(f"  epoch {e['epoch']}: " + "  ".join(
            f"{k.replace('_identical','')}={v}"
            for k, v in e.items() if k.endswith("identical")))
    print("\nBITWISE IDENTICAL:", everything)
    return 0 if everything else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.compare:
        return compare()
    if args.run:
        return probe(args.run)
    parser.error("use --run 1, --run 2 or --compare")


if __name__ == "__main__":
    raise SystemExit(main())
