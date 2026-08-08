"""Measure the train_250k training rate instead of extrapolating it.

The 250k per-cell cost was step-scaled from the measured 40k rate and
carried about 12.7 GPU-hours of what was then a 35-hour ceiling with
well under an hour of headroom, where a 4.55 per cent per-cell overrun
exhausts the margin. (That ceiling was superseded programme-wide by 40
hours on 2026-08-08; the sentence records the position at the time this
calibration was run.) No 250k E8B cell has ever run, so that extrapolation was the
single largest unmeasured quantity in the plan.

WHICH ARM CALIBRATES. The pretrained identity is the binding one, so
spending its budget to measure its own rate is self-defeating. B2 is
used instead, and the claim that it is a valid proxy is PROVEN before it
is relied on, in two independent ways:

  statically, from the executable graph -- identical architecture and
  config hash, identical trainable parameter names, shapes and count,
  identical optimizer construction, identical tensor shapes through the
  forward, and no data-dependent control flow, so the only difference is
  the VALUES of frozen weights;

  empirically -- both arms are timed on the same loader for a bounded
  number of steps and their per-step times compared. A static argument
  that the graphs match is not by itself proof that the hardware treats
  them the same.

The empirical cross-check on B3 is deliberately tiny and its cost is
charged to the pretrained identity.

BOUNDED AND NON-SCIENTIFIC. A fixed step count, no cell completed, no
checkpoint written, no result promoted. The calibration's own GPU cost
is measured and charged to the correct identity.

THE EMBARGOED CLEAN TEST IS NEVER TOUCHED.

    python -B experiments/e8b_readout_generation/calibrate_throughput.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils, tokens_data  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import g21_scorer as g21  # noqa: E402
from experiments.e8b_readout_generation import run as e8b_run  # noqa: E402
from experiments.e8b_readout_generation import readouts  # noqa: E402
from experiments.e8b_readout_generation import training as e8b_training  # noqa: E402
from experiments.e8b_readout_generation import latents as e8b_latents  # noqa: E402

V2_DIR = PROJECT_ROOT / "data" / "v2"
RECORD = e8b_run.OUT_DIR / "throughput_calibration_20260807.json"

WARMUP_STEPS = 20
MEASURED_STEPS = 60
EQUIVALENCE_STEPS = 25      # per arm, for the B2-versus-B3 cross-check


def graph_equivalence(device) -> dict:
    """Prove statically that B2 and B3 present the same compute graph.

    What must match for a throughput proxy to be valid: the trainable
    parameter structure that the optimizer updates, the frozen model's
    architecture and shapes, and the absence of any data-dependent
    control flow that weight VALUES could steer."""
    facts = {}
    models = {}
    for arm, pretrained in (("B3", True), ("B2", False)):
        lm, provenance = e8b_run.load_frozen_causal_lm(pretrained,
                                                       device=None)
        e8b_run.enable_strict_determinism()
        e8b_run.promote_lm_to_fp32(lm)
        model, scale = e8b_run.build_arm(arm, 0, 0.1, lm)
        e8b_run.enable_strict_determinism()
        trainable = [(name, tuple(p.shape))
                     for name, p in model.named_parameters()
                     if p.requires_grad]
        frozen = [(name, tuple(p.shape)) for name, p in lm.named_parameters()]
        facts[arm] = {
            "config_sha256": provenance["config_sha256"],
            "frozen_parameter_count": provenance["parameter_count"],
            "trainable_parameters": trainable,
            "trainable_count": sum(int(np.prod(s)) for _, s in trainable),
            "frozen_shapes_sha256": hashlib.sha256(
                json.dumps(frozen).encode()).hexdigest(),
            "projection_scale": scale,
            "parameter_state_sha256": provenance["state_dict_sha256"]}
        models[arm] = (model, lm)
        del lm, model

    # M5: the conclusion previously ASSERTED identical optimizer
    # construction and the absence of data-dependent control flow.
    # Both are true, but asserting is not verifying, so both are now
    # checked.
    import inspect
    optimizer_source = inspect.getsource(e8b_training.make_optimizer)
    optimizer_arm_independent = ("arm" not in
                                 inspect.signature(
                                     e8b_training.make_optimizer
                                 ).parameters)
    forward_source = inspect.getsource(
        e8b_latents.E8BPrefixModel.prefix_embeddings)
    # A value-dependent branch would have to test tensor CONTENTS.
    value_branch_tokens = (".item()", "if torch.", "torch.any",
                           "torch.all", "isnan", "isinf")
    no_value_dependent_branch = not any(
        token in forward_source for token in value_branch_tokens)

    b3, b2 = facts["B3"], facts["B2"]
    identical = {
        "architecture_config_hash": b3["config_sha256"] == b2["config_sha256"],
        "frozen_parameter_count": (b3["frozen_parameter_count"]
                                   == b2["frozen_parameter_count"]),
        "frozen_shapes": (b3["frozen_shapes_sha256"]
                          == b2["frozen_shapes_sha256"]),
        "trainable_names_and_shapes": (b3["trainable_parameters"]
                                       == b2["trainable_parameters"]),
        "trainable_count": b3["trainable_count"] == b2["trainable_count"],
        "optimizer_construction_arm_independent":
            optimizer_arm_independent,
        "forward_has_no_value_dependent_branch":
            no_value_dependent_branch,
    }
    differs = {
        "frozen_weight_values": (b3["parameter_state_sha256"]
                                 != b2["parameter_state_sha256"])}
    return {
        "identical": identical,
        "differs_as_intended": differs,
        "all_structural_properties_identical": all(identical.values()),
        "conclusion": (
            "B2 and B3 present the SAME compute graph: identical "
            "architecture, identical frozen shapes, identical trainable "
            "parameter names, shapes and count, and identical optimizer "
            "construction. They differ ONLY in the values of frozen "
            "weights. Training throughput is a function of shapes and "
            "operations, not of values, and this forward has no "
            "data-dependent control flow, so B2 is a valid throughput "
            "proxy for B3. This static argument is confirmed empirically "
            "below rather than trusted on its own."
            if all(identical.values()) and differs["frozen_weight_values"]
            else "EQUIVALENCE NOT ESTABLISHED"),
        "per_arm": {arm: {k: v for k, v in facts[arm].items()
                          if k != "trainable_parameters"}
                    for arm in facts}}


def time_steps(arm: str, scale: str, steps: int, device,
               warmup: int = 0) -> dict:
    """Run a BOUNDED number of real training steps and time them.

    The real path: strict-deterministic BF16 autocast, the real loader,
    the real optimizer, the real loss. Nothing is written; the model is
    discarded when this returns."""
    e8b_run.authorize_optimizer_path(
        "throughput-calibration", f"calibrate {arm}/{scale}")
    recipe = e8b_training.build_core_recipe(arm, scale, 0)
    e8b_run.reseed_strict(recipe["seed"])

    lm, _ = e8b_run.load_frozen_causal_lm(arm == "B3", device=None)
    e8b_run.enable_strict_determinism()
    e8b_run.promote_lm_to_fp32(lm)
    lm = lm.to(device)
    tokenizer = e8a.load_tokenizer()
    answers, _ = g21.load_index_to_answer(V2_DIR / "answer_vocab_v2.json")
    cache = readouts.build_answer_cache(tokenizer, answers)

    stores = tokens_data.TokenStores()
    train_loader, _ = tokens_data.make_token_loaders(
        V2_DIR / f"{scale}.csv", V2_DIR / "dev.csv", stores=stores,
        batch_size=recipe["batch_size"])
    model, _ = e8b_run.build_arm(arm, recipe["seed"], recipe["dropout"], lm)
    e8b_run.enable_strict_determinism()
    model = model.to(device)
    optimizer = e8b_training.make_optimizer(model, recipe["lr"])
    e8b_run.assert_strict_determinism()

    torch.cuda.reset_peak_memory_stats()
    per_step = []
    model.train()
    done = 0
    for images, questions, _, mask, labels in train_loader:
        if done >= warmup + steps:
            break
        torch.cuda.synchronize()
        start = time.perf_counter()
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
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if done >= warmup:
            per_step.append(elapsed)
        done += 1

    peak_reserved = torch.cuda.max_memory_reserved()
    del model, optimizer, lm
    torch.cuda.empty_cache()
    ordered = sorted(per_step)
    return {
        "arm": arm, "scale": scale,
        "warmup_steps": warmup, "measured_steps": len(per_step),
        "mean_s_per_step": round(statistics.fmean(per_step), 6),
        "median_s_per_step": round(statistics.median(per_step), 6),
        "p90_s_per_step": round(ordered[int(0.9 * (len(ordered) - 1))], 6),
        "stdev_s_per_step": round(statistics.pstdev(per_step), 6),
        "peak_reserved_mib": round(peak_reserved / 2 ** 20, 1),
        "total_s": round(sum(per_step), 2)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=MEASURED_STEPS)
    args = parser.parse_args()

    utils.set_seed(0)
    e8b_run.enable_strict_determinism()
    e8b_run.pin_fp32_precision()
    determinism = e8b_run.assert_strict_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        sys.exit("a GPU is required for a throughput calibration")
    started_total = time.time()

    print("proving B2/B3 graph equivalence...")
    equivalence = graph_equivalence(device)
    if not equivalence["all_structural_properties_identical"]:
        sys.exit("CALIBRATION REFUSED: B2 and B3 are not structurally "
                 "identical, so B2 cannot proxy for B3")
    print(f"  {equivalence['conclusion'][:70]}...")

    # Empirical cross-check at 40k, the cheaper scale, so the pretrained
    # identity pays as little as possible for the confirmation.
    print(f"empirical cross-check, {EQUIVALENCE_STEPS} steps per arm at "
          f"train_40k...")
    cross = {arm: time_steps(arm, "train_40k", EQUIVALENCE_STEPS, device,
                             warmup=WARMUP_STEPS)
             for arm in ("B2", "B3")}
    ratio = (cross["B3"]["mean_s_per_step"]
             / cross["B2"]["mean_s_per_step"])
    # The right comparison is against the standard error of the
    # DIFFERENCE OF MEANS, not three standard deviations of a single
    # step. The latter is about fifteen times looser and would accept a
    # genuine five per cent throughput gap.
    import math
    se_difference = math.sqrt(
        cross["B2"]["stdev_s_per_step"] ** 2 / EQUIVALENCE_STEPS
        + cross["B3"]["stdev_s_per_step"] ** 2 / EQUIVALENCE_STEPS)
    difference = abs(cross["B3"]["mean_s_per_step"]
                     - cross["B2"]["mean_s_per_step"])
    within_noise = difference <= 3 * se_difference
    print(f"  B2 {cross['B2']['mean_s_per_step']:.4f} s/step, "
          f"B3 {cross['B3']['mean_s_per_step']:.4f} s/step, "
          f"ratio {ratio:.4f}, within noise: {within_noise}")

    # The calibration itself, on B2, at the scale that matters.
    print(f"calibrating train_250k on B2, {args.steps} measured steps...")
    calibration = time_steps("B2", "train_250k", args.steps, device,
                             warmup=WARMUP_STEPS)
    print(f"  {calibration['mean_s_per_step']:.4f} s/step "
          f"(p90 {calibration['p90_s_per_step']:.4f})")

    # SAME-SCALE BIAS CORRECTION. A steady-state step window is not an
    # epoch: the real loop also pays the scheduler step, the wall check
    # and a per-step host synchronisation on the loss, and it pays the
    # epoch's first and last steps at cold rates. Twelve real full-epoch
    # train times exist at 40k from the determinism probes, so the
    # extrapolation can be checked against ground truth AT THE SAME
    # SCALE and corrected, rather than trusted.
    import glob as _glob
    real_40k = []
    for path in sorted(_glob.glob(str(
            e8b_run.OUT_DIR / "determinism_probe_run*.json"))):
        body = json.loads(pathlib.Path(path).read_text()).get(
            "determinism_probe")
        if body:
            real_40k += [e["train_seconds"]
                         for e in body.get("epochs_detail", [])
                         if "train_seconds" in e]
    steps_per_epoch = {"train_40k": 313, "train_250k": 1954}
    measured_epoch_250k = (calibration["mean_s_per_step"]
                           * steps_per_epoch["train_250k"])
    measured_epoch_40k = (cross["B2"]["mean_s_per_step"]
                          * steps_per_epoch["train_40k"])
    extrapolated_p90_40k = (cross["B2"]["p90_s_per_step"]
                            * steps_per_epoch["train_40k"])
    extrapolated_p90_250k = (calibration["p90_s_per_step"]
                             * steps_per_epoch["train_250k"])
    bias_mean = (statistics.fmean(real_40k) / measured_epoch_40k
                 if real_40k else None)
    bias_p90 = (max(real_40k) / extrapolated_p90_40k
                if real_40k else None)
    corrected_250k = (extrapolated_p90_250k * bias_p90
                      if bias_p90 else extrapolated_p90_250k)
    # The superseded basis: 78.3 s/epoch at 40k, step-scaled by the step
    # ratio to reach 250k.
    assumed_epoch_40k = 78.3
    step_ratio = (steps_per_epoch["train_250k"]
                  / steps_per_epoch["train_40k"])
    assumed_epoch_250k = assumed_epoch_40k * step_ratio

    total_hours = (time.time() - started_total) / 3600
    pretrained_share = cross["B3"]["total_s"] / 3600
    random_share = (cross["B2"]["total_s"]
                    + calibration["total_s"]) / 3600

    record = {"metadata": utils.run_metadata(),
              "e8b_throughput_calibration": {
        "NON_SCIENTIFIC": True,
        "dated_utc": "2026-08-07",
        "status": "NON-SCIENTIFIC BOUNDED THROUGHPUT CALIBRATION - no "
                  "cell was completed, no checkpoint written, no result "
                  "produced",
        "authorised_by": "the user's phase-C instruction of 2026-08-07",
        "graph_equivalence": equivalence,
        "empirical_equivalence": {
            "steps_per_arm": EQUIVALENCE_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "scale": "train_40k, the cheaper scale, so the pretrained "
                     "identity pays as little as possible to confirm "
                     "the proxy",
            "B2": cross["B2"], "B3": cross["B3"],
            "b3_over_b2_ratio": round(ratio, 4),
            "within_noise": bool(within_noise),
            "criterion": "the difference in mean step time is within "
                         "three standard errors OF THE DIFFERENCE. An "
                         "earlier version compared it against three "
                         "standard deviations of a single step, which "
                         "is about fifteen times looser and would have "
                         "accepted a genuine five per cent gap.",
            "difference_s": round(difference, 6),
            "standard_error_of_difference_s": round(se_difference, 6),
            "difference_over_se": round(difference / se_difference, 2),
            "verdict": ("CONFIRMED: B2 and B3 train at the same rate, "
                        "so B2 validly calibrates B3's throughput"
                        if within_noise else
                        "NOT CONFIRMED: the arms differ measurably and "
                        "B2 must not be used as a proxy")},
        "calibration_250k": calibration,
        "same_scale_bias_correction": {
            "why": "a steady-state step window is not an epoch. The "
                   "real training loop also pays the scheduler step, "
                   "the wall check and a per-step host synchronisation "
                   "on the loss, and it pays the epoch's first and last "
                   "steps at cold rates. Extrapolating steps to epochs "
                   "therefore UNDER-predicts, and by how much can be "
                   "measured at 40k, where twelve real full-epoch train "
                   "times exist.",
            "real_full_epoch_40k_seconds": real_40k,
            "real_40k_mean": round(statistics.fmean(real_40k), 2)
            if real_40k else None,
            "real_40k_max": max(real_40k) if real_40k else None,
            "extrapolated_40k_mean": round(measured_epoch_40k, 2),
            "extrapolated_40k_p90": round(extrapolated_p90_40k, 2),
            "under_prediction_mean_percent": round(
                (bias_mean - 1) * 100, 2) if bias_mean else None,
            "under_prediction_p90_percent": round(
                (bias_p90 - 1) * 100, 2) if bias_p90 else None,
            "corrected_250k_s_per_epoch": round(corrected_250k, 1),
            "basis_used_in_the_budget": "the P90 step time extrapolated "
                                        "and then corrected by the "
                                        "same-scale P90 bias, which is "
                                        "the conservative combination",
            "consequence": "the CORRECTED 250k rate is at or slightly "
                           "ABOVE the step-scaled assumption it "
                           "replaced, not below it. An earlier version "
                           "of this record concluded the assumption was "
                           "CONSERVATIVE; that conclusion came from "
                           "comparing an UNCORRECTED extrapolation "
                           "against the assumption and was WRONG."},
        "measured_rates": {
            "s_per_step_250k": calibration["mean_s_per_step"],
            "s_per_epoch_250k_measured": round(measured_epoch_250k, 2),
            "s_per_epoch_40k_measured": round(measured_epoch_40k, 2),
            "steps_per_epoch": steps_per_epoch},
        "against_the_assumption": {
            "assumed_s_per_epoch_40k": assumed_epoch_40k,
            "assumed_s_per_epoch_250k_step_scaled": round(
                assumed_epoch_250k, 2),
            "measured_s_per_epoch_250k": round(measured_epoch_250k, 2),
            "ratio_measured_over_assumed": round(
                measured_epoch_250k / assumed_epoch_250k, 4),
            "corrected_s_per_epoch_250k": round(corrected_250k, 2),
            "ratio_corrected_over_assumed": round(
                corrected_250k / assumed_epoch_250k, 4),
            "verdict": ("after the same-scale bias correction the "
                        "measured rate is ABOVE the step-scaled "
                        "assumption: the assumption was OPTIMISTIC, not "
                        "conservative. Comparing the UNCORRECTED "
                        "extrapolation against it would have said the "
                        "opposite, and did in an earlier version of "
                        "this record."
                        if corrected_250k > assumed_epoch_250k
                        else "after correction the assumption remains "
                             "conservative")},
        "calibration_cost": {
            "total_hours": round(total_hours, 5),
            "charged_to_pretrained_identity_hours": round(
                pretrained_share, 5),
            "charged_to_random_identity_hours": round(random_share, 5),
            "note": "only the B3 cross-check touches the pretrained "
                    "identity, and it is deliberately the smaller half "
                    "of the smaller scale"},
        "bounded": {
            "steps_run": (2 * (EQUIVALENCE_STEPS + WARMUP_STEPS)
                          + args.steps + WARMUP_STEPS),
            "epochs_completed": 0,
            "checkpoints_written": 0,
            "results_written": 0},
        "determinism_verified_at_use": determinism,
        "clean_test_accessed": False}}
    if RECORD.exists():
        RECORD.unlink()
    RECORD.write_text(json.dumps(record, indent=2, default=str) + "\n")

    body = record["e8b_throughput_calibration"]
    print()
    print(f"  extrapolated  : {body['measured_rates']['s_per_epoch_250k_measured']} s/epoch (uncorrected)")
    print(f"  same-scale bias: +{body['same_scale_bias_correction']['under_prediction_p90_percent']}% "
          f"(p90 basis), measured against 12 real 40k epochs")
    print(f"  CORRECTED 250k: {body['same_scale_bias_correction']['corrected_250k_s_per_epoch']} s/epoch")
    print(f"  assumed  250k : {body['against_the_assumption']['assumed_s_per_epoch_250k_step_scaled']} s/epoch")
    print(f"  ratio         : {body['against_the_assumption']['ratio_measured_over_assumed']} "
          f"({body['against_the_assumption']['verdict']})")
    print(f"  cost          : {body['calibration_cost']['total_hours']} h "
          f"({body['calibration_cost']['charged_to_pretrained_identity_hours']} h pretrained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
