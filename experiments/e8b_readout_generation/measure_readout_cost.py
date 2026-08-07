"""Measure what R1, R2 and R3 actually cost per query.

The projection carried 0.0291 s/row for R2 and 0.0437 s/row for R3.
Neither was measured: the E7b provenance once claimed for R2 was FALSE
(that figure is 0.0291 MILLISECONDS for a LayerNorm-plus-Linear stage,
a different operation in units a thousand apart), and no E8B readout had
ever been executed. The pipeline validation then took about 1.5 seconds
per row for all three readouts together, roughly twenty times the
assumption, so this needs a real measurement rather than another guess.

Measured here, separately and on a frozen exploratory checkpoint:

  * seconds per query for R1, R2 and R3 individually, and for the shared
    prefix that all three consume, so the readouts can be costed
    independently of each other;
  * the R3 generated-length distribution, which drives its cost;
  * peak allocated and reserved memory;
  * cold versus warm behaviour, since the first calls pay for CUDA
    context, autotuning and cache allocation that the remaining
    thousands of rows will not.

NON-SCIENTIFIC. No accuracy is reported and none may be. The cost of
this measurement is itself recorded and charged.

NO OPTIMIZER STEP. THE EMBARGOED CLEAN TEST IS NEVER TOUCHED.

    python -B experiments/e8b_readout_generation/measure_readout_cost.py \\
        --rows 200
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import g21_scorer as g21  # noqa: E402
from experiments.e8b_readout_generation import run as e8b_run  # noqa: E402
from experiments.e8b_readout_generation import readouts  # noqa: E402
from experiments.e8b_readout_generation import training as e8b_training  # noqa: E402
from experiments.e8b_readout_generation import validate_evaluation as ve  # noqa: E402

V2_DIR = PROJECT_ROOT / "data" / "v2"
RECORD = e8b_run.OUT_DIR / "readout_cost_measurement_20260807.json"
COLD_ROWS = 8      # excluded from the warm statistics, reported separately


def summarise(samples: list) -> dict:
    ordered = sorted(samples)
    return {"n": len(samples),
            "mean_s": round(statistics.fmean(samples), 6),
            "median_s": round(statistics.median(samples), 6),
            "p90_s": round(ordered[int(0.90 * (len(ordered) - 1))], 6),
            "max_s": round(ordered[-1], 6),
            "total_s": round(sum(samples), 3)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=200)
    args = parser.parse_args()

    utils.set_seed(0)
    e8b_run.enable_strict_determinism()
    e8b_run.pin_fp32_precision()
    determinism = e8b_run.assert_strict_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        sys.exit("a GPU is required for a cost measurement that will be "
                 "used to project GPU hours")
    started_total = time.time()

    raw = pd.read_csv(V2_DIR / "dev_raw.csv",
                      dtype={"questionId": str, "imageId": str},
                      keep_default_na=False, na_filter=False)
    raw["label"] = raw["label"].astype(int)
    raw["in_vocabulary"] = raw["in_vocabulary"].astype(bool)
    # Representative of the RAW denominator, which is what R2 and R3 are
    # charged over: sampled across it rather than taking a prefix, so
    # question length and answer difficulty are not systematically
    # unusual.
    subset = raw.sample(n=args.rows, random_state=0).reset_index(drop=True)

    stores = ve.ExtendedTokenStores()
    lm, provenance = e8b_run.load_frozen_causal_lm(True, device=None)
    e8b_run.enable_strict_determinism()
    e8b_run.promote_lm_to_fp32(lm)
    lm = lm.to(device)
    tokenizer = e8a.load_tokenizer()
    answers, _ = g21.load_index_to_answer(V2_DIR / "answer_vocab_v2.json")
    cache = readouts.build_answer_cache(tokenizer, answers)
    trie = readouts.build_trie(cache)

    model, _ = e8b_run.build_arm("B3", 0, 0.1, lm)
    e8b_run.enable_strict_determinism()
    checkpoint = torch.load(ve.EXPLORATORY, map_location="cpu",
                            weights_only=False)
    model.load_state_dict(checkpoint.get("state", checkpoint))
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    loader = torch.utils.data.DataLoader(
        ve.RawRowDataset(subset, stores), batch_size=16, shuffle=False,
        collate_fn=ve.collate)

    torch.cuda.reset_peak_memory_stats()
    # R1_batched is the CANONICAL path used by the per-epoch pass and by
    # the final evaluation. R1_cached is the per-row cross-check scorer
    # used only inside G14, where three implementations must agree.
    # Timing both is the point: they differ by two orders of magnitude,
    # and costing the evaluation with the wrong one was the mistake this
    # measurement caught.
    timings = {"prefix": [], "prefix_state": [], "R1_batched": [],
               "R1_cached": [], "R2": [], "R3": []}
    lengths, terminated = [], []
    with torch.no_grad():
        for images, questions, question_ids, mask, labels in loader:
            images = images.to(device)
            questions = questions.to(device)
            mask = mask.to(device)
            torch.cuda.synchronize()
            start = time.perf_counter()
            prefix = e8b_training.canonical_prefix(model, lm, images,
                                                   questions, mask)
            torch.cuda.synchronize()
            batch_prefix = time.perf_counter() - start
            per_row_prefix = batch_prefix / prefix.shape[0]

            torch.cuda.synchronize()
            start = time.perf_counter()
            scores = e8b_training.r1_scores_batched(lm, prefix, cache)
            torch.cuda.synchronize()
            batched = (time.perf_counter() - start) / prefix.shape[0]

            for index in range(prefix.shape[0]):
                timings["R1_batched"].append(batched)
                single = prefix[index:index + 1]
                timings["prefix"].append(per_row_prefix)

                torch.cuda.synchronize()
                start = time.perf_counter()
                state = readouts._prefix_cache(lm, single)
                torch.cuda.synchronize()
                timings["prefix_state"].append(time.perf_counter() - start)

                torch.cuda.synchronize()
                start = time.perf_counter()
                readouts.r1_cached(lm, single, cache, prefix_state=state)
                torch.cuda.synchronize()
                timings["R1_cached"].append(time.perf_counter() - start)

                torch.cuda.synchronize()
                start = time.perf_counter()
                readouts.r2_cached(lm, single, cache, trie,
                                   prefix_state=state)
                torch.cuda.synchronize()
                timings["R2"].append(time.perf_counter() - start)

                torch.cuda.synchronize()
                start = time.perf_counter()
                emitted = readouts.r3_generate(lm, single, tokenizer,
                                               prefix_state=state)
                torch.cuda.synchronize()
                timings["R3"].append(time.perf_counter() - start)
                lengths.append(int(emitted["n_generated"]))
                terminated.append(bool(emitted["terminated_by_eos"]))

    # --- the pass as it will ACTUALLY run -------------------------------
    # Composing per-part timings taken at batch 16 would misstate the
    # real cost: R1 is batched, so its per-row cost falls with batch
    # size, and the per-call synchronise above adds overhead the real
    # loop does not pay. This times the whole combined pass at the
    # batch size the evaluation uses.
    from experiments.e8b_readout_generation import final_evaluation as fe
    pass_loader = torch.utils.data.DataLoader(
        ve.RawRowDataset(subset, stores), batch_size=128, shuffle=False,
        collate_fn=ve.collate)
    neutral_image, neutral_question, neutral_mask = fe.neutral_inputs(
        pass_loader, device)
    context = {"neutral_image": neutral_image,
               "neutral_question": neutral_question,
               "neutral_mask": neutral_mask,
               "deranged_lookup": lambda qids: torch.stack([
                   torch.from_numpy(np.asarray(
                       stores.image_vector(subset["imageId"].iloc[0]),
                       dtype=np.float32)) for _ in qids]),
               "prefix_fn": e8b_training.canonical_prefix}
    torch.cuda.synchronize()
    start = time.perf_counter()
    fe.evaluate_condition(model, lm, pass_loader, cache, trie, tokenizer,
                          device, "normal", context)
    torch.cuda.synchronize()
    combined_seconds = time.perf_counter() - start
    combined_per_row = combined_seconds / len(subset)

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    total_device = torch.cuda.get_device_properties(0).total_memory

    warm = {name: summarise(values[COLD_ROWS:])
            for name, values in timings.items()}
    cold = {name: summarise(values[:COLD_ROWS])
            for name, values in timings.items()}
    measurement_hours = (time.time() - started_total) / 3600

    # What the projection needs: seconds per row on the RAW denominator.
    n_raw = 10_004
    per_row = {name: warm[name]["mean_s"] for name in warm}
    projected = {
        "r1_batched_hours_per_pass": round(
            n_raw * per_row["R1_batched"] / 3600, 4),
        "r1_cached_hours_per_pass_NOT_USED": round(
            n_raw * per_row["R1_cached"] / 3600, 4),
        "r2_hours_per_pass": round(n_raw * per_row["R2"] / 3600, 4),
        "r3_hours_per_pass": round(n_raw * per_row["R3"] / 3600, 4),
        "prefix_hours_per_pass": round(
            n_raw * (per_row["prefix"] + per_row["prefix_state"]) / 3600, 4),
        "r2_plus_r3_hours_per_pass": round(
            n_raw * (per_row["R2"] + per_row["R3"]) / 3600, 4),
        "combined_pass_measured_s_per_row": round(combined_per_row, 6),
        "combined_pass_hours": round(
            n_raw * combined_per_row / 3600, 4),
        "combined_pass_basis": "MEASURED end to end at batch 128 over "
                               f"{len(subset)} rows, which is how the "
                               "evaluation actually runs. Composing the "
                               "per-part timings above would overstate "
                               "it: those were taken at batch 16 with a "
                               "synchronise around every call.",
        "note": "the prefix and its cache are built ONCE per row and "
                "shared by all three readouts, so a combined pass costs "
                "prefix + R1 + R2 + R3, not three separate prefixes. R1 "
                "is costed at the CANONICAL BATCHED rate; the per-row "
                "cached scorer is roughly ninety times more expensive "
                "and is used only inside G14."}
    assumed_r2_r3 = round(n_raw * (0.0291 + 0.0437) / 3600, 4)

    record = {"metadata": utils.run_metadata(),
              "e8b_readout_cost_measurement": {
        "NON_SCIENTIFIC": True,
        "dated_utc": "2026-08-07",
        "status": "NON-SCIENTIFIC COST MEASUREMENT - no accuracy is "
                  "reported and none may be derived from it",
        "rows_measured": int(args.rows),
        "sampling": "sampled across the 10,004-row raw denominator with "
                    "a pinned seed, not a prefix of it, so question "
                    "length and answer difficulty are representative",
        "cold_rows_excluded_from_warm": COLD_ROWS,
        "warm_per_row_seconds": warm,
        "cold_per_row_seconds": cold,
        "cold_vs_warm": {
            name: {"cold_mean_s": cold[name]["mean_s"],
                   "warm_mean_s": warm[name]["mean_s"],
                   "ratio": round(cold[name]["mean_s"]
                                  / max(warm[name]["mean_s"], 1e-9), 2)}
            for name in warm},
        "r3_generated_length": {
            "mean": round(float(np.mean(lengths)), 3),
            "median": float(np.median(lengths)),
            "max": int(np.max(lengths)),
            "terminated_by_eos_rate": round(float(np.mean(terminated)), 4),
            "note": "R3 cost is driven by generated length: every extra "
                    "token is another sequential forward pass"},
        "memory": {
            "peak_allocated_mib": round(peak_allocated / 2 ** 20, 1),
            "peak_reserved_mib": round(peak_reserved / 2 ** 20, 1),
            "device_total_mib": round(total_device / 2 ** 20, 1),
            "reserved_fraction": round(peak_reserved / total_device, 4),
            "ceiling_fraction": e8b_run.MEMORY_CEILING_FRACTION,
            "fires": (peak_reserved / total_device
                      > e8b_run.MEMORY_CEILING_FRACTION)},
        "projected_over_raw_denominator": projected,
        "against_the_assumption": {
            "assumed_r2_plus_r3_hours_per_pass": assumed_r2_r3,
            "measured_r2_plus_r3_hours_per_pass":
                projected["r2_plus_r3_hours_per_pass"],
            "ratio": round(projected["r2_plus_r3_hours_per_pass"]
                           / assumed_r2_r3, 2),
            "verdict": "the assumption was UNSOURCED and is now "
                       "replaced by measurement"},
        "measurement_cost": {
            "hours": round(measurement_hours, 5),
            "charged_to": "the pretrained identity: this measurement "
                          "loads the frozen pretrained SmolLM2-135M"},
        "checkpoint": {
            "path": str(ve.EXPLORATORY.relative_to(PROJECT_ROOT)),
            "why_valid_for_timing": "cost depends on the compute graph, "
                                    "the shapes and the generated "
                                    "lengths, not on which weights the "
                                    "trunk holds. An exploratory "
                                    "checkpoint is therefore a valid "
                                    "timing subject even though it is "
                                    "never a valid scientific result."},
        "determinism_verified_at_use": determinism,
        "frozen_model": {"arm_model": provenance["arm_model"],
                         "pinned_revision": provenance["pinned_revision"]},
        "optimizer_steps": 0,
        "clean_test_accessed": False}}
    if RECORD.exists():
        RECORD.unlink()
    RECORD.write_text(json.dumps(record, indent=2, default=str) + "\n")

    body = record["e8b_readout_cost_measurement"]
    print(f"  rows            : {args.rows} "
          f"({COLD_ROWS} cold rows excluded from warm)")
    for name in ("prefix", "prefix_state", "R1_batched",
                 "R1_cached", "R2", "R3"):
        print(f"  {name:13s} : {warm[name]['mean_s']:.4f} s/row warm "
              f"(cold {cold[name]['mean_s']:.4f})")
    print(f"  R3 length       : mean {body['r3_generated_length']['mean']}, "
          f"max {body['r3_generated_length']['max']}, "
          f"eos {body['r3_generated_length']['terminated_by_eos_rate']}")
    print(f"  peak reserved   : {body['memory']['peak_reserved_mib']} MiB "
          f"({body['memory']['reserved_fraction']:.1%})")
    print(f"  R2+R3 per pass  : {projected['r2_plus_r3_hours_per_pass']} h "
          f"measured against {assumed_r2_r3} h assumed "
          f"({body['against_the_assumption']['ratio']}x)")
    print(f"  combined pass   : {projected['combined_pass_measured_s_per_row']:.4f} "
          f"s/row at batch 128 -> {projected['combined_pass_hours']} h "
          f"per condition over the raw denominator")
    print(f"  this measurement: {body['measurement_cost']['hours']} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
