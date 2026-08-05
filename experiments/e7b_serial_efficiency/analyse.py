"""E7b analysis: aggregate the per-run records into the final artefacts.

    python -B experiments/e7b_serial_efficiency/analyse.py

Reads the immutable per-run records, the preflight references and the
cache-integrity hashes, and writes: e7b_results.json (per-system summary,
common-denominator Pareto frontier, E7a supersession block, provenance),
latency_memory.csv, stage_timings.csv and pareto_serial.png. Nothing is
re-timed and nothing is overwritten; the per-run records stay authoritative.
The embargoed clean-test target is never read.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import utils  # noqa: E402
from experiments.e7b_serial_efficiency import run as e7b  # noqa: E402

OUT_DIR = e7b.OUT_DIR

# Resident frozen-tower parameter counts, from measured sources: the CLIP
# and SigLIP dual-tower totals are E7a's stored encoder_total_parameters
# and the SmolLM2 count is the asserted e8a constant.
E7A_RESULTS = (PROJECT_ROOT / "results" / "experiments" / "e7a_efficiency"
               / "results.json")
SMOLLM2_PARAMETERS = 134_515_008

SUPERSEDED_E7A_KEYS = ("gpu_encoder_plus_head_ms", "full_pipeline_ms",
                       "amortised_ms",
                       "pareto_fronts.gpu_encoder_plus_head_ms",
                       "pareto_fronts.full_pipeline_ms",
                       "pareto_fronts.amortised_ms")


def tower_parameters() -> dict:
    """E7a's encoder_total_parameters is already the WHOLE dual-tower count
    (its note: "resident when either tower runs"), so it is taken once per
    encoder, never summed across the two tower entries."""
    cost = json.loads(E7A_RESULTS.read_text())["e7a_efficiency"]["cost"]
    clip = cost["clip_image_tower"]["encoder_total_parameters"]
    siglip = cost["siglip_image_tower"]["encoder_total_parameters"]
    assert clip == cost["clip_text_tower"]["encoder_total_parameters"]
    assert siglip == cost["siglip_text_tower"]["encoder_total_parameters"]
    return {"clip_dual_tower": int(clip), "siglip_dual_tower": int(siglip)}


def trainable_parameters(checkpoint: Path) -> int:
    import torch
    state = torch.load(checkpoint, map_location="cpu")
    return int(sum(v.numel() for v in state.values()))


def resident_parameters(name: str, trainable: int, towers: dict) -> dict:
    family = e7b.SYSTEMS[name]["family"]
    if family in ("clip_global", "clip_global_1000"):
        loaded, note = towers["clip_dual_tower"], "CLIP dual tower resident"
    elif family == "siglip_global":
        loaded, note = towers["siglip_dual_tower"], \
            "SigLIP dual tower resident"
    elif family == "clip_tokens":
        loaded, note = towers["clip_dual_tower"], \
            "CLIP dual tower resident (text side used for question tokens)"
    else:
        loaded, note = towers["clip_dual_tower"] + SMOLLM2_PARAMETERS, (
            "CLIP dual tower plus SmolLM2-135M resident; the CLIP text "
            "tower is loaded but unused by this pipeline")
    return {"trainable": trainable, "resident_frozen": loaded,
            "total_loaded": trainable + loaded, "convention": note}


def aggregate_system(name: str, runs: list) -> dict:
    warm = [run["warm_single_unsegmented"]["median_ms"] for run in runs]
    stage_names = list(runs[0]["stage_medians_ms"])
    stages = {s: round(float(np.median([run["stage_medians_ms"][s]
                                        for run in runs])), 4)
              for s in stage_names}
    entry = {
        "checkpoint": runs[0]["checkpoint"],
        "checkpoint_sha256": runs[0]["checkpoint_sha256"],
        "startup_load_seconds": [run["startup"][
            "process_and_model_load_seconds"] for run in runs],
        "cold_first_query_ms": [run["cold_first_query_ms"] for run in runs],
        "warm_median_ms_per_pass": warm,
        "warm_median_ms": round(float(np.median(warm)), 4),
        "warm_across_pass_spread_ms": round(max(warm) - min(warm), 4),
        "warm_p5_p95_ms_per_pass": [
            [run["warm_single_unsegmented"]["p5_ms"],
             run["warm_single_unsegmented"]["p95_ms"]] for run in runs],
        "warm_p5_p95_envelope_ms": [
            round(min(run["warm_single_unsegmented"]["p5_ms"]
                      for run in runs), 4),
            round(max(run["warm_single_unsegmented"]["p95_ms"]
                      for run in runs), 4)],
        "single_stream_qps": round(1000.0 / float(np.median(warm)), 1),
        "stage_medians_ms": stages,
        "additive_sum_ms": [run["additive_sum_ms"] for run in runs],
        "measured_minus_additive_ms": [run["measured_minus_additive_ms"]
                                       for run in runs],
        "cached_image_question_side_ms": (
            round(float(np.median([run["cached_image_question_side"]
                                   ["median_ms"] for run in runs])), 4)
            if runs[0]["cached_image_question_side"] else None),
        "cached_feature_head_only_ms": round(float(np.median(
            [run["cached_feature_head_only"]["median_ms"]
             for run in runs])), 4),
        "batched_image_branch_ms": (
            round(float(np.median([run["batched_image_branch"]["median_ms"]
                                   for run in runs])), 4)
            if runs[0]["batched_image_branch"] else None),
        "peak_allocated_mib": max(run["memory"]["peak_allocated_mib"]
                                  for run in runs),
        "peak_reserved_mib": max(run["memory"]["peak_reserved_mib"]
                                 for run in runs),
        "cpu_vmhwm_mib": max(run["memory"]["cpu"]["vmhwm_mib"]
                             for run in runs),
        "sync_floor_ms": runs[0]["sync_floor_ms"],
    }
    return entry


def pareto_front(points: dict) -> list:
    front = []
    for name, point in points.items():
        dominated = any(
            other["accuracy"] >= point["accuracy"]
            and other["latency"] <= point["latency"]
            and (other["accuracy"] > point["accuracy"]
                 or other["latency"] < point["latency"])
            for other_name, other in points.items() if other_name != name)
        if not dominated:
            front.append(name)
    return sorted(front)


def main() -> int:
    utils.set_seed()
    results = json.loads((OUT_DIR / "results.json").read_text())[
        "e7b_results"]
    preflight = json.loads((OUT_DIR / "preflight.json").read_text())[
        "e7b_preflight"]
    cache_before = (OUT_DIR / "cache_hashes_before.txt").read_text()
    cache_after = (OUT_DIR / "cache_hashes_after.txt").read_text()
    cache_unchanged = cache_before == cache_after
    if not cache_unchanged:
        sys.exit("CACHE INTEGRITY GATE FAILED: model cache changed during "
                 "the benchmark")
    towers = tower_parameters()

    systems = {}
    for name in e7b.SYSTEMS:
        runs = results["systems"][name]["runs"]
        entry = aggregate_system(name, runs)
        entry["reproduction"] = {
            k: results["systems"][name]["reproduction"][k]
            for k in ("n_rows", "agreements")}
        entry["reproduction"]["disagreements"] = \
            results["systems"][name]["reproduction"]["disagreements"]
        ref = preflight["systems"][name]
        entry["accuracy"] = {
            "timed_checkpoint_seed": 0,
            "frozen_seed0_dev_accuracy": ref["stored_seed0_accuracy"],
            "multiseed_context": preflight["multiseed_context"][name],
            "raw_distribution_accuracy_common_denominator":
                ref["raw_distribution_accuracy_common_denominator"],
            "vocabulary_supported_accuracy":
                ref["vocabulary_supported_accuracy"],
            "vocabulary_view_rows": ref["vocabulary_view_rows"],
            "denominator_note": (
                "the primary Pareto frontier uses only the common "
                f"{ref['common_denominator']}-row raw-distribution "
                "denominator; the 64-row timing sample supports no "
                "accuracy claim"),
        }
        entry["parameters"] = resident_parameters(
            name, trainable_parameters(PROJECT_ROOT / entry["checkpoint"]),
            towers)
        entry["context_only"] = e7b.SYSTEMS[name]["context_only"]
        systems[name] = entry

    frontier_points = {
        name: {"accuracy": entry["accuracy"][
                   "raw_distribution_accuracy_common_denominator"],
               "latency": entry["warm_median_ms"]}
        for name, entry in systems.items() if not entry["context_only"]}
    frontier = pareto_front(frontier_points)

    record = {"metadata": utils.run_metadata(),
              "e7b_analysis": {
                  "systems": systems,
                  "pass_orders": results["pass_orders"],
                  "primary_pareto_frontier_common_denominator": frontier,
                  "frontier_note": (
                      "multimodal systems only; question_only is a labelled "
                      "non-multimodal pipeline floor and is excluded"),
                  "fingerprint": preflight["fingerprint"],
                  "model_cache_integrity": {
                      "verified_files": len(cache_before.splitlines()),
                      "unchanged_across_benchmark": True,
                      "loading_mode": (
                          "online-capable Hugging Face configuration, as in "
                          "the qualified preflight: every model object "
                          "loaded from the staged cache; the byte-identical "
                          "before/after hash set proves no weight, "
                          "tokenizer or configuration file was downloaded, "
                          "updated or replaced. Network metadata probes "
                          "cannot alter the cache and are the only requests "
                          "this configuration can make for cached objects.")},
                  "e7a_supersession": {
                      "superseded_for_end_to_end_claims":
                          list(SUPERSEDED_E7A_KEYS),
                      "statement": (
                          "E7a's component measurements (tower and head "
                          "timings, CPU decode and tokenise terms, memory "
                          "deltas) remain valid as components. E7a's "
                          "additive gpu_encoder_plus_head_ms, "
                          "full_pipeline_ms and amortised_ms values and "
                          "their Pareto fronts are superseded for "
                          "end-to-end claims by E7b's measured serial "
                          "values, which are authoritative for online "
                          "inference claims.")},
                  "clean_test_accessed": False}}
    utils.save_json(record, OUT_DIR / "e7b_results.json")

    with open(OUT_DIR / "latency_memory.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["system", "context_only", "common_denominator_acc",
                         "vocab_supported_acc", "warm_median_ms",
                         "across_pass_spread_ms", "cold_ms_mean",
                         "load_s_mean", "qps", "cached_image_q_side_ms",
                         "cached_feature_ms", "peak_alloc_mib",
                         "peak_reserved_mib", "cpu_vmhwm_mib", "trainable",
                         "total_loaded"])
        for name, entry in systems.items():
            writer.writerow([
                name, entry["context_only"],
                entry["accuracy"][
                    "raw_distribution_accuracy_common_denominator"],
                entry["accuracy"]["vocabulary_supported_accuracy"],
                entry["warm_median_ms"],
                entry["warm_across_pass_spread_ms"],
                round(float(np.mean(entry["cold_first_query_ms"])), 1),
                round(float(np.mean(entry["startup_load_seconds"])), 2),
                entry["single_stream_qps"],
                entry["cached_image_question_side_ms"],
                entry["cached_feature_head_only_ms"],
                entry["peak_allocated_mib"], entry["peak_reserved_mib"],
                entry["cpu_vmhwm_mib"], entry["parameters"]["trainable"],
                entry["parameters"]["total_loaded"]])

    with open(OUT_DIR / "stage_timings.csv", "w", newline="") as handle:
        stage_names = ["S1_read", "S2_decode_preprocess", "S3_image_h2d",
                       "S4_vision_tower", "S5_tokenise", "S6_token_h2d",
                       "S7_text_tower", "S8_projection", "S9_reasoner",
                       "S9_head", "S10_decode"]
        writer = csv.writer(handle)
        writer.writerow(["system"] + stage_names + ["additive_sum",
                                                    "measured_median"])
        for name, entry in systems.items():
            writer.writerow([name] + [entry["stage_medians_ms"].get(s, "")
                                      for s in stage_names]
                            + [round(float(np.median(
                                entry["additive_sum_ms"])), 4),
                               entry["warm_median_ms"]])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 5))
    for name, entry in systems.items():
        x = entry["warm_median_ms"]
        y = entry["accuracy"]["raw_distribution_accuracy_common_denominator"]
        if entry["context_only"]:
            axis.scatter(x, y, marker="x", color="grey")
            axis.annotate(f"{name} (context)", (x, y), fontsize=7,
                          color="grey", xytext=(4, -8),
                          textcoords="offset points")
        else:
            on_front = name in frontier
            axis.scatter(x, y, marker="o" if on_front else "s",
                         color="tab:blue" if on_front else "tab:orange")
            axis.annotate(name, (x, y), fontsize=7, xytext=(4, 4),
                          textcoords="offset points")
    front_points = sorted(
        [(systems[n]["warm_median_ms"],
          systems[n]["accuracy"][
              "raw_distribution_accuracy_common_denominator"])
         for n in frontier])
    axis.plot([p[0] for p in front_points], [p[1] for p in front_points],
              "--", color="tab:blue", linewidth=1)
    axis.set_xlabel("measured serial end-to-end latency, warm median (ms)")
    axis.set_ylabel("raw-distribution accuracy (correct / 10,004)")
    axis.set_title("E7b: measured serial latency against "
                   "common-denominator accuracy")
    figure.tight_layout()
    figure.savefig(OUT_DIR / "pareto_serial.png", dpi=150)

    print("primary frontier (common denominator, multimodal):", frontier)
    for name, entry in systems.items():
        print(f"  {name:18s} warm {entry['warm_median_ms']:8.3f} ms "
              f"(spread {entry['warm_across_pass_spread_ms']:.3f}) "
              f"acc {entry['accuracy']['raw_distribution_accuracy_common_denominator']:.5f} "
              f"repro {entry['reproduction']['agreements']}/64 "
              f"{'CONTEXT' if entry['context_only'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
