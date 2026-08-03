"""E8A SmolLM2-135M core: the bounded efficiency table and the resource audit.

    python -B experiments/e8a_question_encoder/efficiency_resources.py

Two things, both from measured evidence already on disk and neither retraining
or re-timing anything:

  * the bounded E8A-135M accuracy-efficiency table and its Pareto front, over
    A0p, A1 and A1r at both scales;
  * the resource audit of the completed tranche against the operative gates.

Cost boundaries are labelled and never mixed. Three are reported:

  cached_head           the trained head on this arm's cached question states
                        and cached CLIP image tokens. No encoder forward pass.
  offline_extraction    the amortised cost of producing one arm's cached
                        question states, from the measured extraction rate.
  encoder_inclusive     not reported. E7a measured the CLIP and SigLIP towers
                        but never a SmolLM2 tower, and no online SmolLM2
                        forward pass has been timed in this project, so an
                        end-to-end per-query figure for A1 and A1r does not
                        exist and is not estimated here.

The E8A head latencies were measured under the 20 warm-up / 200 repeat mean
protocol inherited from v3_01. E7a explicitly supersedes that protocol and
states its numbers must not be mixed with it, so the front below is built
only from E8A's own internally consistent measurements and the E7a figures
appear as clearly labelled context, never as points on this front.

Nothing here reads the embargoed clean-test target.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import (  # noqa: E402
    analysis_integrity as integrity)

ARMS = ("A0p", "A1", "A1r")
SCALES = ("train_40k", "train_250k")
SEEDS = (0, 1, 2)

# The operative gates. The per-run wall and the memory fraction are the
# runner's own constants; the per-model aggregate and the core ceiling are the
# figures the user set on 2 August 2026.
PER_RUN_CEILING_HOURS = e8a.WALL_CLOCK_HALT_HOURS
PER_MODEL_CEILING_HOURS = 35.0
CORE_CEILING_HOURS = 180.0
MEMORY_CEILING_FRACTION = e8a.MEMORY_CEILING_FRACTION

# The three quantities the E8A record spells differently between Phase 1A and
# the core matrix. Reading either spelling keeps the two seed-0 pilots in the
# table instead of silently dropping them.
FROZEN_KEYS = ("frozen_question_encoder_parameters",
               "frozen_language_model_parameters")


def frozen_parameters(efficiency: dict) -> int:
    for key in FROZEN_KEYS:
        if key in efficiency:
            return efficiency[key]
    raise AssertionError(f"no frozen-encoder parameter count in {efficiency}")


def directory_bytes(path: Path, pattern: str = "*") -> int:
    return sum(p.stat().st_size for p in path.glob(pattern) if p.is_file())


def collect(manifest: dict, view: integrity.DevView) -> dict:
    """Every completed cell's cost and accuracy, integrity-checked first."""
    cells = {}
    for spec in manifest["runs"]:
        arm, scale, seed = spec["arm"], spec["scale"], spec["seed"]
        reuse, artefacts = spec["reuse_artefacts"], spec["expected_artefacts"]
        if reuse:
            path = e8a.OUT_DIR / reuse["record"]["path"]
            record = json.loads(path.read_text())
            block = (record["e8a_a0p_pilot"] if arm == "A0p"
                     else record["e8a_pilot"])
            run = block["run"] if arm == "A0p" else block["runs"][arm]
            evaluation = (block["evaluation"] if arm == "A0p"
                          else block["evaluations"][arm])
            correctness = e8a.OUT_DIR / reuse["correctness"]["path"]
        else:
            path = e8a.OUT_DIR / artefacts["record"]
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            block = record["e8a_core_run"]
            run, evaluation = block["run"], block["evaluation"]
            correctness = e8a.OUT_DIR / artefacts["correctness"]
        evidence = integrity.cell_evidence(
            f"{arm}/{scale}/seed{seed}", arm, scale, seed, path, record,
            block, run, evaluation, correctness, view)
        cells[(arm, scale, seed)] = {"run": run, "evaluation": evaluation,
                                     "integrity": evidence}
    return cells


def store_costs() -> dict:
    """Extraction time and store size, from the two extraction records.

    extraction_train_250k.json's storage_projection.train_40k_plus_dev is a
    copy of its own 250k figures, so the 40k storage numbers are taken from
    extraction.json, which is the only record that measured them.
    """
    small = json.loads((e8a.OUT_DIR / "extraction.json").read_text())[
        "e8a_extraction"]
    large = json.loads(
        (e8a.OUT_DIR / "extraction_train_250k.json").read_text())[
            "e8a_extraction"]
    out = {"train_40k": {}, "train_250k": {},
           "note": ("the 40k storage figures come from extraction.json; "
                    "extraction_train_250k.json's train_40k_plus_dev block "
                    "repeats its own 250k values and is not used")}
    for scale, block in (("train_40k", small), ("train_250k", large)):
        for arm, written in block["stores_written"].items():
            out[scale][arm] = {
                "extraction_gpu_hours": written.get("gpu_hours"),
                "store_bytes": written.get("bytes"),
                "store_gib": (round(written["bytes"] / 2 ** 30, 4)
                              if written.get("bytes") else None),
                "strings_per_second": written.get("strings_per_second")}
    out["A0p"] = {
        "extraction_gpu_hours": 0.0,
        "store_bytes": None,
        "note": ("A0p writes no store of its own; it reads the v3_00 CLIP "
                 "question tokens, which were extracted before E8A and are "
                 "shared with V3")}
    return out


def efficiency_table(cells: dict, stores: dict) -> dict:
    """One row per arm and scale, averaged over the three seeds."""
    rows = {}
    for arm in ARMS:
        for scale in SCALES:
            present = [cells[(arm, scale, s)] for s in SEEDS
                       if (arm, scale, s) in cells]
            if len(present) < len(SEEDS):
                continue
            accuracy = [c["evaluation"]["conditions"]["normal"]["accuracy"]
                        for c in present]
            latency = [c["evaluation"]["efficiency"]["head_latency_ms_mean"]
                       for c in present]
            hours = [c["run"]["wall_clock_hours"] for c in present]
            peak = [c["run"]["peak_allocated_mib"] for c in present]
            efficiency = present[0]["evaluation"]["efficiency"]
            store = stores[scale].get(arm, stores["A0p"])
            rows[f"{arm}/{scale}"] = {
                "arm": arm, "scale": scale,
                "dev_accuracy_mean": round(float(np.mean(accuracy)), 5),
                "dev_accuracy_std": round(float(np.std(accuracy, ddof=1)), 5),
                "dev_accuracy_per_seed": accuracy,
                "trainable_parameters": efficiency["trainable_parameters"],
                "trainable_projection": efficiency["trainable_projection"],
                "trainable_reasoner_trunk":
                    efficiency["trainable_reasoner_trunk"],
                "frozen_question_encoder_parameters":
                    frozen_parameters(efficiency),
                "total_loaded_parameters":
                    efficiency["total_loaded_parameters"],
                "checkpoint_mib": efficiency["checkpoint_mib"],
                "cached_head_latency_ms_mean":
                    round(float(np.mean(latency)), 4),
                "cached_head_latency_ms_spread":
                    round(float(max(latency) - min(latency)), 4),
                "cached_head_latency_per_seed": latency,
                "training_gpu_hours_mean": round(float(np.mean(hours)), 5),
                "training_gpu_hours_total": round(float(np.sum(hours)), 5),
                "peak_allocated_mib_max": max(peak),
                "question_store_gib": store.get("store_gib"),
                "question_store_extraction_gpu_hours":
                    store.get("extraction_gpu_hours"),
                "boundary": "cached_head",
            }
    return rows


def pareto(rows: dict, cost_key: str) -> list:
    """Rows not dominated on (higher accuracy, lower cost)."""
    usable = [(name, row) for name, row in rows.items()
              if row.get(cost_key) is not None]
    front = []
    for name, row in usable:
        dominated = any(
            other["dev_accuracy_mean"] >= row["dev_accuracy_mean"]
            and other[cost_key] <= row[cost_key]
            and (other["dev_accuracy_mean"] > row["dev_accuracy_mean"]
                 or other[cost_key] < row[cost_key])
            for other_name, other in usable if other_name != name)
        if not dominated:
            front.append(name)
    return sorted(front)


def resource_audit(cells: dict, stores: dict) -> dict:
    """Measured totals for the completed tranche and the gates they meet."""
    training = {arm: round(sum(
        cells[(arm, scale, seed)]["run"]["wall_clock_hours"]
        for scale in SCALES for seed in SEEDS
        if (arm, scale, seed) in cells), 5) for arm in ARMS}
    extraction = {arm: round(sum(
        stores[scale].get(arm, {}).get("extraction_gpu_hours") or 0.0
        for scale in SCALES), 6) for arm in ARMS}
    per_run = {f"{arm}/{scale}/seed{seed}":
               cells[(arm, scale, seed)]["run"]["wall_clock_hours"]
               for arm in ARMS for scale in SCALES for seed in SEEDS
               if (arm, scale, seed) in cells}
    longest = max(per_run.items(), key=lambda kv: kv[1])
    peak_mib = max(cells[k]["run"]["peak_allocated_mib"] for k in cells)
    ceiling_mib = max(cells[k]["run"]["memory_ceiling_mib"] for k in cells)

    checkpoints = directory_bytes(e8a.OUT_DIR / "checkpoints", "*.pt")
    predictions = directory_bytes(e8a.OUT_DIR, "correctness_*.npz")
    records = directory_bytes(e8a.OUT_DIR, "*.json")
    store_bytes = sum(
        (stores[scale].get(arm, {}).get("store_bytes") or 0)
        for scale in SCALES for arm in ARMS)

    training_total = round(sum(training.values()), 5)
    extraction_total = round(sum(extraction.values()), 6)
    total = round(training_total + extraction_total, 5)

    failed = sorted(p.name for p in e8a.OUT_DIR.glob("FAILED_*"))
    return {
        "scope": ("the completed E8A SmolLM2-135M core tranche: arms A0p, A1 "
                  "and A1r at train_40k and train_250k with seeds 0, 1 and 2"),
        "training_gpu_hours_by_arm": training,
        "training_gpu_hours_total": training_total,
        "extraction_gpu_hours_by_arm": extraction,
        "extraction_gpu_hours_total": extraction_total,
        "evaluation_gpu_hours": {
            "value": None,
            "note": ("evaluation runs inside each training invocation and is "
                     "included in that run's wall clock; it was never timed "
                     "separately, so it is not double-counted and cannot be "
                     "reported on its own")},
        "total_gpu_hours": total,
        "cells_completed": len(per_run),
        "cells_failed_or_interrupted": failed,
        "retried_cells": [],
        "storage_bytes": {
            "question_stores": store_bytes,
            "checkpoints": checkpoints,
            "predictions": predictions,
            "records": records,
            "total": store_bytes + checkpoints + predictions + records},
        "storage_gib": round((store_bytes + checkpoints + predictions
                              + records) / 2 ** 30, 4),
        "gates": {
            "per_run_8_gpu_hours": {
                "ceiling": PER_RUN_CEILING_HOURS,
                "largest_measured": round(longest[1], 5),
                "largest_measured_cell": longest[0],
                "fires": longest[1] >= PER_RUN_CEILING_HOURS,
                "margin_hours": round(PER_RUN_CEILING_HOURS - longest[1], 5)},
            "per_model_35_gpu_hours": {
                "ceiling": PER_MODEL_CEILING_HOURS,
                "measured_135m_aggregate": total,
                "fires": total >= PER_MODEL_CEILING_HOURS,
                "margin_hours": round(PER_MODEL_CEILING_HOURS - total, 5),
                "scope_caveat": (
                    "this aggregates only the E8A question-encoder arms of "
                    "SmolLM2-135M. The canonical per-model aggregate also "
                    "counts the E8B readout arms, which are neither "
                    "authorised nor run, so the model total is a lower bound "
                    "and the gate is not finally discharged")},
            "memory_80_per_cent": {
                "ceiling_mib": ceiling_mib,
                "fraction": MEMORY_CEILING_FRACTION,
                "peak_allocated_mib": peak_mib,
                "fires": peak_mib >= ceiling_mib,
                "headroom_mib": round(ceiling_mib - peak_mib, 1)},
            "storage_allocation": {
                "written_gib": round((store_bytes + checkpoints + predictions
                                      + records) / 2 ** 30, 4),
                "free_gib": round(
                    __import__("shutil").disk_usage(PROJECT_ROOT).free
                    / 2 ** 30, 1),
                "fires": False},
        },
        "unresolved": {
            "e8b_runtime_multiplier": (
                "unmeasured. E8B is not authorised and has not run, so the "
                "readout-branch cost per run is unknown"),
            "cross_e8_aggregate": (
                "not computable. Any complete SmolLM2-135M aggregate needs "
                "the E8B arms"),
            "core_programme_ceiling": (
                f"min({CORE_CEILING_HOURS}, three times the revised expected "
                f"projection) cannot be evaluated: the expected projection "
                f"spans branches that have not run"),
            "426_versus_420": (
                "presentational only. Canonical section 13.1 counts 426 "
                "evaluation passes and section 13.2 itemises 420; the six "
                "unprinted rows are the B1 classification evaluations and the "
                "stated totals already include them. No GPU-hour total is "
                "wrong and none is changed here"),
        },
    }


def main() -> int:
    utils.set_seed()
    manifest = json.loads(
        (e8a.OUT_DIR / "execution_manifest.json").read_text())[
            "e8a_execution_manifest"]
    view = integrity.canonical_dev_view()
    cells = collect(manifest, view)
    stores = store_costs()
    rows = efficiency_table(cells, stores)

    fronts = {
        "cached_head_latency_ms_mean": pareto(
            rows, "cached_head_latency_ms_mean"),
        "trainable_parameters": pareto(rows, "trainable_parameters"),
        "total_loaded_parameters": pareto(rows, "total_loaded_parameters"),
        "checkpoint_mib": pareto(rows, "checkpoint_mib"),
        "training_gpu_hours_mean": pareto(rows, "training_gpu_hours_mean"),
    }

    context = {}
    e7a_path = (config.RESULTS_DIR / "experiments" / "e7a_efficiency"
                / "results.json")
    if e7a_path.exists():
        reasoner = json.loads(e7a_path.read_text())["e7a_efficiency"][
            "cost"]["reasoner"]
        context["e7a_reasoner_arm_a0"] = {
            "head_only_ms_median": reasoner["single_ms"]["median"],
            "across_pass_spread_ms": reasoner["single_ms"]["spread"],
            "trainable_parameters": reasoner["trainable_parameters"],
            "label": (
                "arm A0, the same trunk WITHOUT the interface-matching "
                "projection, measured under the E7a protocol. E7a supersedes "
                "the 20/200 mean protocol the E8A latencies use, so this is "
                "context only and is NOT a point on the E8A front"),
        }

    record = {"metadata": utils.run_metadata(),
              "e8a_efficiency_and_resources": {
                  "boundaries": {
                      "cached_head": (
                          "the trained head on cached question states and "
                          "cached CLIP image tokens; no encoder forward pass"),
                      "offline_extraction": (
                          "the measured cost of writing one arm's cached "
                          "question states once, amortised over every run "
                          "that reads them"),
                      "encoder_inclusive": (
                          "NOT reported. No online SmolLM2-135M forward pass "
                          "has been timed in this project and E7a measured no "
                          "SmolLM2 tower, so no end-to-end per-query cost "
                          "exists for A1 or A1r and none is estimated"),
                      "protocol": (
                          "E8A head latency is 20 warm-up and 200 timed "
                          "repeats of a single example under torch.no_grad, "
                          "mean over repeats, one pass. E7a supersedes this "
                          "protocol and its numbers must not be mixed with "
                          "these"),
                  },
                  "table": rows,
                  "pareto_fronts": fronts,
                  "pareto_note": (
                      "accuracy is the three-seed mean development accuracy "
                      "and every cost is an E8A measurement, so the front is "
                      "bounded to E8A-135M and contains no cross-experiment "
                      "point"),
                  "context_not_on_the_front": context,
                  "resource_audit": resource_audit(cells, stores),
                  "clean_test_accessed": False}}
    utils.save_json(record, e8a.OUT_DIR / "efficiency_resources.json")

    print("=== bounded E8A-135M accuracy-efficiency table (cached head) ===")
    header = (f"{'row':22s} {'dev acc':>9s} {'params':>11s} "
              f"{'loaded':>12s} {'ms':>7s} {'ckpt MiB':>9s} {'GPU-h':>7s}")
    print(header)
    for name in sorted(rows):
        row = rows[name]
        print(f"{name:22s} {row['dev_accuracy_mean']:9.5f} "
              f"{row['trainable_parameters']:11,d} "
              f"{row['total_loaded_parameters']:12,d} "
              f"{row['cached_head_latency_ms_mean']:7.4f} "
              f"{row['checkpoint_mib']:9.2f} "
              f"{row['training_gpu_hours_mean']:7.4f}")
    print("\nPareto fronts:")
    for axis, front in fronts.items():
        print(f"  {axis:32s} {front}")
    audit = record["e8a_efficiency_and_resources"]["resource_audit"]
    print(f"\ntotal measured GPU-hours: {audit['total_gpu_hours']} "
          f"(training {audit['training_gpu_hours_total']}, extraction "
          f"{audit['extraction_gpu_hours_total']})")
    for gate, block in audit["gates"].items():
        print(f"  {gate:26s} fires={block['fires']}")
    print("\nwritten to results/experiments/e8a_question_encoder/"
          "efficiency_resources.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
