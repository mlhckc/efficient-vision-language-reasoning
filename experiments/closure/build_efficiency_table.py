"""The canonical efficiency closure table.

    python -m experiments.closure.build_efficiency_table

Audits and freezes the efficiency evidence that already exists. It measures
nothing: every number is read from a frozen artefact, and the three
identified optional measurements are recorded as optional and left unrun.

The table's job is to make two distinctions impossible to lose.

TIMING CLASS. A cached-feature head latency and a raw-image-to-answer serial
latency differ by two orders of magnitude and answer different questions.
Every row therefore carries a required timing_kind, and any row that is not
END_TO_END is marked comparable_with_end_to_end false. E7a's additive
gpu_encoder_plus_head_ms, full_pipeline_ms and amortised_ms fields, which E7b
explicitly supersedes for end-to-end claims, enter only as
ADDITIVE_COMPONENT_SUM_SUPERSEDED and carry the superseding artefact's path.

MEASUREMENT NODE. E7b measured eight systems on otter155 and E9 measured
eleven on otter159. E9's own fusion bridge control failed its pre-registered
10 per cent tolerance against the historical otter155 figure, at -18.69 per
cent, so the two sets are two frontiers. The builder refuses to emit a
frontier that mixes nodes or timing classes; that refusal is a hard failure,
not a warning.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402
from experiments.closure import closure_common as cc  # noqa: E402

RESULTS_ROOT = config.RESULTS_DIR / "experiments"

TIMING_KINDS = {
    "END_TO_END_SERIAL": (
        "raw image plus raw question in, answer out, batch 1, warm median, "
        "measured serially on one node. The only class that supports an "
        "end-to-end latency claim"),
    "CACHED_IMAGE_QUESTION_SIDE": (
        "the image feature is already cached; only the question side and the "
        "head are timed. A partial pipeline, never an end-to-end figure"),
    "CACHED_FEATURE_HEAD_ONLY": (
        "both features are already cached; only the trainable head is timed. "
        "A head-only figure, never an end-to-end figure"),
    "ADDITIVE_COMPONENT_SUM_SUPERSEDED": (
        "a sum of separately measured components. Superseded for end-to-end "
        "claims by the measured serial values of E7b"),
    "COMPONENT": (
        "one measured stage of the pipeline in isolation; valid as a "
        "component, never as a system latency"),
    "CONTEXTUAL_EXTERNAL": (
        "a system measured under the same imported protocol but on a "
        "different node, or an external reference system. Quoted as context, "
        "never differenced against another node's number"),
}
END_TO_END = "END_TO_END_SERIAL"


def main() -> int:
    utils.set_seed()
    rows = []

    # ---------------- E7b, otter155, the authoritative serial set ----------
    e7b_path = RESULTS_ROOT / "e7b_serial_efficiency" / "e7b_results.json"
    e7b = json.loads(e7b_path.read_text())["e7b_analysis"]
    fingerprint = e7b["fingerprint"]
    with open(RESULTS_ROOT / "e7b_serial_efficiency"
              / "latency_memory.csv") as handle:
        latency = {r["system"]: r for r in csv.DictReader(handle)}

    def number(value):
        try:
            return float(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            return None

    for system, entry in sorted(e7b["systems"].items()):
        table = latency.get(system, {})
        accuracy = entry["accuracy"]
        rows.append({
            "system_id": system,
            "experiment_family": "E7b",
            "node": fingerprint["hostname"],
            "hardware": fingerprint["gpu"],
            "timing_kind": END_TO_END,
            "comparable_with_end_to_end": True,
            "comparability_class": "E7b/otter155 serial",
            "context_only": entry.get("context_only", False),
            "accuracy_raw_distribution_common_denominator":
                accuracy.get("raw_distribution_accuracy_common_denominator"),
            "accuracy_vocabulary_supported":
                accuracy.get("vocabulary_supported_accuracy"),
            "accuracy_denominator_note": accuracy.get("denominator_note"),
            "timed_checkpoint_seed": accuracy.get("timed_checkpoint_seed"),
            "trainable_parameters": entry["parameters"]["trainable"],
            "total_loaded_parameters": entry["parameters"]["total_loaded"],
            "resident_frozen_parameters":
                entry["parameters"].get("resident_frozen"),
            "checkpoint": entry.get("checkpoint"),
            "checkpoint_sha256": entry.get("checkpoint_sha256"),
            "warm_serial_median_ms": number(table.get("warm_median_ms")),
            "across_pass_spread_ms": number(table.get("across_pass_spread_ms")),
            "cold_first_query_ms": number(table.get("cold_ms_mean")),
            "load_seconds": number(table.get("load_s_mean")),
            "throughput_qps": number(table.get("qps")),
            "cached_image_question_side_ms":
                number(table.get("cached_image_q_side_ms")),
            "cached_feature_head_only_ms": number(table.get("cached_feature_ms")),
            "peak_allocated_mib": number(table.get("peak_alloc_mib")),
            "peak_reserved_mib": number(table.get("peak_reserved_mib")),
            "cpu_vmhwm_mib": number(table.get("cpu_vmhwm_mib")),
            "precision": "fp32 inference",
            "batch_size": 1,
            "timing_methodology": (
                "warm median over a fixed serial sample, three passes, "
                "per-pass medians and across-pass spread reported"),
            "training_gpu_hours": None,
            "evaluation_gpu_hours": None,
            "evidence_status": "VERIFIED",
            "source_artefact": str(e7b_path.relative_to(PROJECT_ROOT)),
        })

    # ---------------- E9, otter159, same protocol, different node ----------
    e9_path = RESULTS_ROOT / "e9_compact_vlm" / "e9_results.json"
    e9 = json.loads(e9_path.read_text())["e9_results"]
    efficiency = e9["efficiency"]
    bridge = efficiency["e7b_bridge"]
    for system, entry in sorted(efficiency["aggregate"].items()):
        rows.append({
            "system_id": system,
            "experiment_family": "E9",
            "node": entry["node"],
            "hardware": "NVIDIA RTX 4000 Ada Generation",
            "timing_kind": END_TO_END,
            "comparable_with_end_to_end": True,
            "comparability_class": "E9/otter159 serial",
            "context_only": False,
            "accuracy_raw_distribution_common_denominator": None,
            "accuracy_vocabulary_supported": None,
            "accuracy_denominator_note": (
                "accuracies for these systems live in the E9 scored_passes "
                "and E8B reference blocks on the 10,004-row raw denominator; "
                "they are not restated on a 64-row timing sample"),
            "timed_checkpoint_seed": 0,
            "trainable_parameters": entry.get("trainable_parameters"),
            "total_loaded_parameters": entry.get("total_parameters"),
            "resident_frozen_parameters": None,
            "checkpoint": None,
            "checkpoint_sha256": None,
            "warm_serial_median_ms": entry.get("warm_median_ms"),
            "across_pass_spread_ms": entry.get("spread_ms"),
            "cold_first_query_ms": entry.get("cold_first_query_ms"),
            "load_seconds": entry.get("load_seconds"),
            "throughput_qps": entry.get("queries_per_second"),
            "cached_image_question_side_ms": None,
            "cached_feature_head_only_ms": None,
            "peak_allocated_mib": None,
            "peak_reserved_mib": (
                round(entry["peak_memory_reserved_bytes"] / (1024 ** 2), 1)
                if entry.get("peak_memory_reserved_bytes") else None),
            "cpu_vmhwm_mib": None,
            "precision": "fp32 inference",
            "batch_size": efficiency["batch"],
            "timing_methodology": (
                f"E7b serial protocol imported read-only; warmup "
                f"{efficiency['warmup']}, iterations "
                f"{efficiency['iterations']}, passes {efficiency['passes']}, "
                f"pass-order seed {efficiency['pass_order_seed']}"),
            "training_gpu_hours": None,
            "evaluation_gpu_hours": None,
            "evidence_status": "VERIFIED",
            "source_artefact": str(e9_path.relative_to(PROJECT_ROOT)),
            "checkpoint_footprint_bytes": entry.get("checkpoint_bytes"),
            "mean_generated_tokens": entry.get("mean_generated_tokens"),
        })

    # ---------------- E7b cached and head-only views -----------------------
    # Same measurement session, different timing class. Split into their own
    # rows so no reader can mistake a cached figure for an end-to-end one.
    for system, entry in sorted(e7b["systems"].items()):
        table = latency.get(system, {})
        for kind, column in (("CACHED_IMAGE_QUESTION_SIDE",
                              "cached_image_q_side_ms"),
                             ("CACHED_FEATURE_HEAD_ONLY", "cached_feature_ms")):
            value = number(table.get(column))
            if value is None:
                continue
            rows.append({
                "system_id": f"{system}::{kind.lower()}",
                "experiment_family": "E7b",
                "node": fingerprint["hostname"],
                "hardware": fingerprint["gpu"],
                "timing_kind": kind,
                "comparable_with_end_to_end": False,
                "comparability_class": f"E7b/otter155 {kind.lower()}",
                "context_only": False,
                "accuracy_raw_distribution_common_denominator":
                    entry["accuracy"].get(
                        "raw_distribution_accuracy_common_denominator"),
                "accuracy_vocabulary_supported":
                    entry["accuracy"].get("vocabulary_supported_accuracy"),
                "accuracy_denominator_note": (
                    "the accuracy is the system's; the latency is a PARTIAL "
                    "pipeline and must never be reported as end-to-end"),
                "timed_checkpoint_seed":
                    entry["accuracy"].get("timed_checkpoint_seed"),
                "trainable_parameters": entry["parameters"]["trainable"],
                "total_loaded_parameters": entry["parameters"]["total_loaded"],
                "resident_frozen_parameters":
                    entry["parameters"].get("resident_frozen"),
                "checkpoint": entry.get("checkpoint"),
                "checkpoint_sha256": entry.get("checkpoint_sha256"),
                "warm_serial_median_ms": None,
                "cached_image_question_side_ms": (
                    value if kind == "CACHED_IMAGE_QUESTION_SIDE" else None),
                "cached_feature_head_only_ms": (
                    value if kind == "CACHED_FEATURE_HEAD_ONLY" else None),
                "precision": "fp32 inference",
                "batch_size": 1,
                "timing_methodology": "warm median, features pre-cached",
                "training_gpu_hours": None,
                "evaluation_gpu_hours": None,
                "evidence_status": "VERIFIED",
                "source_artefact": str(e7b_path.relative_to(PROJECT_ROOT)),
            })

    # ---------------- E7a: components and the superseded additive sums -----
    e7a_path = RESULTS_ROOT / "e7a_efficiency" / "results.json"
    superseded = e7b["e7a_supersession"]
    rows.append({
        "system_id": "e7a_additive_pipeline_sums",
        "experiment_family": "E7a",
        "node": "otter (E7a session)",
        "hardware": "NVIDIA RTX 4000 Ada Generation",
        "timing_kind": "ADDITIVE_COMPONENT_SUM_SUPERSEDED",
        "comparable_with_end_to_end": False,
        "comparability_class": "superseded additive sums",
        "context_only": True,
        "accuracy_denominator_note": None,
        "warm_serial_median_ms": None,
        "precision": "fp32 inference",
        "batch_size": 1,
        "timing_methodology": (
            "sum of separately measured components, not a measured serial "
            "query"),
        "training_gpu_hours": None,
        "evaluation_gpu_hours": None,
        "evidence_status": "SUPERSEDED_FOR_END_TO_END_CLAIMS",
        "superseded_by": str(e7b_path.relative_to(PROJECT_ROOT)),
        "superseded_fields": superseded["superseded_for_end_to_end_claims"],
        "supersession_statement": superseded["statement"],
        "source_artefact": str(e7a_path.relative_to(PROJECT_ROOT)),
    })
    rows.append({
        "system_id": "e7a_component_measurements",
        "experiment_family": "E7a",
        "node": "otter (E7a session)",
        "hardware": "NVIDIA RTX 4000 Ada Generation",
        "timing_kind": "COMPONENT",
        "comparable_with_end_to_end": False,
        "comparability_class": "component measurements",
        "context_only": False,
        "warm_serial_median_ms": None,
        "precision": "fp32 inference",
        "batch_size": 1,
        "timing_methodology": (
            "tower and head timings, CPU decode and tokenise terms and memory "
            "deltas, each measured in isolation"),
        "training_gpu_hours": None,
        "evaluation_gpu_hours": None,
        "evidence_status": "VERIFIED_AS_COMPONENTS",
        "note": ("E7a's component measurements remain valid AS COMPONENTS. "
                 "Only its additive sums and their Pareto fronts are "
                 "superseded"),
        "source_artefact": str(e7a_path.relative_to(PROJECT_ROOT)),
    })

    # system_id alone is NOT unique: E9 re-measured its own fusion and
    # vocab1000_product references on otter159, so those names exist under
    # two families on two nodes. Every row therefore carries a
    # family-qualified row_key, and frontiers are expressed in row_keys. A
    # frontier addressed by bare system_id would silently span both nodes,
    # which is exactly the merge this table exists to prevent.
    for entry in rows:
        entry["row_key"] = f"{entry['experiment_family']}::{entry['system_id']}"
        entry.setdefault("comparable_with_end_to_end",
                         entry["timing_kind"] == END_TO_END)
        if entry["timing_kind"] != END_TO_END:
            assert entry["comparable_with_end_to_end"] is False, (
                f"{entry['row_key']}: a non-end-to-end row may never be "
                f"marked comparable with end-to-end")
    assert len({r["row_key"] for r in rows}) == len(rows), \
        "row_key must uniquely identify an efficiency row"

    # ---------------- frontiers: one per node, never merged ----------------
    def frontier(family, node):
        members = [r for r in rows
                   if r["experiment_family"] == family
                   and r["node"] == node
                   and r["timing_kind"] == END_TO_END
                   and not r.get("context_only")]
        kinds = {r["timing_kind"] for r in members}
        nodes = {r["node"] for r in members}
        assert len(kinds) <= 1, "a frontier may not mix timing classes"
        assert len(nodes) <= 1, "a frontier may not mix measurement nodes"
        return sorted(r["row_key"] for r in members)

    frontiers = {
        "E7b/otter155": {
            "node": fingerprint["hostname"],
            "timing_kind": END_TO_END,
            "members": frontier("E7b", fingerprint["hostname"]),
            "recorded_primary_frontier":
                e7b["primary_pareto_frontier_common_denominator"],
            "frontier_note": e7b["frontier_note"],
        },
        "E9/otter159": {
            "node": "otter159.eps.surrey.ac.uk",
            "timing_kind": END_TO_END,
            "members": frontier("E9", "otter159.eps.surrey.ac.uk"),
            "recorded_primary_frontier": None,
            "frontier_note": ("same-node systems only; the two CLIP-global "
                              "references are E9's own re-measurements, not "
                              "E7b's otter155 numbers"),
        },
    }

    record = {
        "closure_output": "efficiency closure table",
        "title": "canonical efficiency evidence, audited and frozen",
        "provenance": cc.provenance(
            "experiments/closure/build_efficiency_table.py",
            [cc.record_source(e7b_path, "E7b canonical analysis"),
             cc.record_source(RESULTS_ROOT / "e7b_serial_efficiency"
                              / "latency_memory.csv", "E7b latency table"),
             cc.record_source(e9_path, "E9 canonical analysis"),
             cc.record_source(e7a_path, "E7a component measurements")]),
        "measured_anything": False,
        "measurement_statement": (
            "this closure measured no latency, memory or throughput. Every "
            "value is read from a frozen artefact"),
        "timing_kind_definitions": TIMING_KINDS,
        "hard_rule": (
            "any row whose timing_kind is not END_TO_END_SERIAL carries "
            "comparable_with_end_to_end false, and the builder refuses to "
            "emit a frontier that mixes timing classes or measurement nodes"),
        "two_nodes": {
            "statement": (
                "E7b measured on otter155 and E9 measured on otter159 under "
                "the same imported protocol. They are two frontiers and are "
                "NEVER merged"),
            "bridge_control": bridge,
            "consequence": (
                "the fusion bridge control failed its pre-registered 10 per "
                "cent tolerance at -18.69 per cent, so no adjustment factor "
                "is applied and no cross-node frontier is constructed. A "
                "cross-node latency difference is not evidence about the "
                "systems"),
        },
        "r1_exclusion": efficiency["r1_exclusion"],
        "e10_timing": {
            "status": "OPTIONAL_EVALUATION_ONLY",
            "statement": (
                "E10's B4 and B4r systems have no timing evidence of any "
                "kind. This is recorded, not estimated, and is NOT a blocker: "
                "the dissertation can rest project-wide efficiency on E7b and "
                "E9 and use E10 for capacity and pretraining science"),
            "not_measured_here": True,
        },
        "frontiers": frontiers,
        "row_count": len(rows),
        "rows": sorted(rows, key=lambda r: (r["experiment_family"],
                                            r["system_id"])),
        "clean_test_accessed": False,
    }
    digest = cc.write_json(record,
                           cc.CLOSURE_DIR / "efficiency_closure_table.json")
    cc.write_csv(record["rows"],
                 ["row_key", "system_id", "experiment_family", "node",
                  "hardware",
                  "timing_kind", "comparable_with_end_to_end",
                  "comparability_class",
                  "accuracy_raw_distribution_common_denominator",
                  "accuracy_vocabulary_supported", "trainable_parameters",
                  "total_loaded_parameters", "checkpoint_footprint_bytes",
                  "warm_serial_median_ms", "across_pass_spread_ms",
                  "cached_image_question_side_ms",
                  "cached_feature_head_only_ms", "throughput_qps",
                  "peak_allocated_mib", "peak_reserved_mib",
                  "training_gpu_hours", "evaluation_gpu_hours", "precision",
                  "batch_size", "timing_methodology", "evidence_status",
                  "source_artefact"],
                 cc.CLOSURE_DIR / "efficiency_closure_table.csv")

    kinds = {}
    for entry in rows:
        kinds[entry["timing_kind"]] = kinds.get(entry["timing_kind"], 0) + 1
    print(f"efficiency rows {len(rows)}: {kinds}")
    print(f"frontiers: " + ", ".join(
        f"{k} ({len(v['members'])})" for k, v in frontiers.items()))
    print(f"efficiency_closure_table.json sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
