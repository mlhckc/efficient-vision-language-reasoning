"""Tests for the canonical E7b evidence supersession overlay.

CPU-only and read-only. Nothing is trained, no model is loaded, no timing is
taken and no checkpoint is opened. The embargoed clean-test target is never
read or named; the token is assembled at run time so this source does not
contain it.

WHAT THESE TESTS ARE FOR. The overlay's whole job is to stop a reader
mistaking a withdrawn E7b number for current evidence. That job fails
silently: the historical artefacts still parse, still look complete and
still carry a row-level status of VERIFIED. So the invariants worth pinning
are the ones a future edit could quietly break — the exact three-field
withdrawal, the two separate causes, the retention of warm serial latency,
the absence of any substitute value, the task-scoped clean-test wording, and
the fact that no closed packet moved.

The checks are semantic and field-level, over parsed JSON and the AST of the
builder, not over line numbers or rendered text, so ordinary reformatting
does not break them. Load-bearing guards are exercised as known-negatives
too, because a check that cannot fail is not evidence.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RECORD = PROJECT_ROOT / "results" / "closure" / "e7b_evidence_supersession.json"
BUILDER = (PROJECT_ROOT / "experiments" / "closure"
           / "build_e7b_supersession.py")

EMBARGOED = "test_" + "clean_targets"

# The three withdrawn fields and their bound statuses and causes. Written
# out here rather than read from the record, so the test disagrees with the
# record if either one changes.
WITHDRAWN = {
    "peak_allocated_mib": ("INVALID_WITHDRAWN", "MEASUREMENT_BOUNDARY_DEFECT"),
    "peak_reserved_mib": ("INVALID_WITHDRAWN", "MEASUREMENT_BOUNDARY_DEFECT"),
    "cold_first_query_ms": ("SUPERSEDED_WITHDRAWN", "GRAD_MODE_STANDARDISATION"),
}

RETAINED = {
    "warm_serial_median_ms",
    "across_pass_spread_ms",
    "accuracy_raw_distribution_common_denominator",
    "trainable_parameters",
    "total_loaded_parameters",
    "resident_frozen_parameters",
    "primary_pareto_frontier_common_denominator",
}

# Words that would mean the cold-query withdrawal had been blamed on the
# peak-memory measurement-boundary defect, or the reverse.
GRAD_WORDS = ("grad", "no_grad", "autograd")
BOUNDARY_WORDS = ("boundary", "measurement window", "envelope")

# Closed packets. Byte identity here is the proof that publishing the
# overlay reopened nothing.
CLOSED_PACKETS = {
    "results/ve0/VE0_MANIFEST.json":
        "294e11d48ade7316908f8cc6dc37ab9b8f8cc2cc6d454b3df4524fa8ecf6c7f3",
    "results/ve1/VE1_MANIFEST.json":
        "021592025cbfcdeb145441799d7b5c6ff341a0945289f8927230a89e16eb986a",
    "results/ve2/VE2_MANIFEST.json":
        "f54218e9666b8d2574ba87ccd7868874d4ef35de7560b401e1dce91f3210c386",
    "results/ve0/supersession_map.json":
        "e1163023509920dddf182989341a2517057941bffc813cc645e6b65928e66f5c",
    "results/ve0/canonical_evidence_inventory.json":
        "a1f49a0f99ddf4cffac591fcf24b3cf7ba9638de349898fd4a525aed96ae66d3",
    "results/ve0/figure_specifications.json":
        "7d1c1d4588e97ded196a208e32021b9f21cbfa18aa9a9d3a031e11572eb2486c",
    "results/ve0/table_specifications.json":
        "98f0e756d7817ba2a78444e2ecadfd021ffda129752604e6f4a7bed7c3b4a3f8",
    "docs/experiments/ve0_evidence_contract.md":
        "326bbc7c259aea808478e38358b3dc0eab12e1b61e03501927b9c2d4fc591188",
    "results/closure/efficiency_closure_table.json":
        "be1781efeece841cc75df126fab43c9e9946437957c1ea6401ee1f86c8b47d5e",
    "results/closure/efficiency_closure_table.csv":
        "2d929d5d42d82d71bdab5e4f42d41f325117610a7a542947f549a298a1da339e",
    "results/closure/A_evidence_registry.json":
        "b8b47e024c49a599a425894f34c24b4375958998401a311605694027257de122",
    "results/closure/A_evidence_registry.csv":
        "da77d32c2567d85831669d701f6390aaefc9053ac584fef741759a554e82d3ce",
    "results/closure/manifest_e7b_serial_efficiency.json":
        "a7a1d692bad137418673ad20c2fe6fb2823ba6874bb2441db39b80b5baab28db",
    "results/experiments/e7b_serial_efficiency/e7b_results.json":
        "c55d61600124ed76a6383f87d6f0a8744f4d51eec24b2f53ef8aa9edeb76bae4",
    "results/experiments/e7b_serial_efficiency/latency_memory.csv":
        "deb4a67e793612c822e645d78d314081b9413bc83579d8694d6a83ff647942cc",
    "results/experiments/e7b_serial_efficiency/stage_timings.csv":
        "aeb9f7ef15230278faa1f1a5e5c24430f87506b0e1f1135b8b61eb5ac11cd266",
    "results/experiments/e7b_serial_efficiency/results.json":
        "fddf880fc6a023ffe8255e6a8001f225dc4a253d6f87d0d1db2136c81bd672b9",
}

DISCLOSURE = (
    "The clean-test contents were never inspected or used for development, "
    "model selection, or reporting decisions. Mechanical byte access occurred "
    "in two documented governance incidents, on 13 and 14 August 2026.")

# The wording written when only the first incident was known. Accurate then,
# incomplete as a history now. It is permitted ONLY inside a closed packet
# that cannot be reopened, and only where it is labelled as superseded.
SUPERSEDED_DISCLOSURE_FRAGMENT = "mechanically read once"

# Frozen surfaces that may still carry the superseded fragment. VE2_MANIFEST
# is immutable; run_ve2.py is the source string the manifest is rebuilt from,
# and tests/test_ve2.py compares that rebuild, so editing it reopens VE-2.
PERMITTED_RESIDUALS = {
    "results/ve2/VE2_MANIFEST.json",
    "experiments/ve2/run_ve2.py",
}

# Human-facing documents a fresh reviewer would actually open.
HUMAN_FACING_DOCS = (
    "README.md",
    "CLAUDE.md",
    "docs/REPRODUCIBILITY.md",
    "docs/experiments/ve2_qualitative_evidence.md",
    "docs/experiments/e7b_serial_efficiency.md",
)

CANONICAL_E9 = (
    "SmolVLM-500M was approximately 24.7x slower in warm serial latency "
    "under the same-node contextual protocol.")

FORBIDDEN_E9 = ("cheaper", "25x cheaper", "~25x cheaper", "more efficient",
                "lower cost")

# Terms no timing result alone can license.
UNBOUNDED_COST = ("energy", "power", "carbon", "monetary cost",
                  "general computational cost")

VERBOSE = False
_CHECKS: list[tuple[str, bool]] = []
_INSIDE_KNOWN_NEGATIVE = False


def check(name: str, condition: bool, detail: str = "") -> None:
    if not _INSIDE_KNOWN_NEGATIVE:
        _CHECKS.append((name, bool(condition)))
        if VERBOSE:
            print(f"  [{'PASS' if condition else 'FAIL'}] {name}"
                  + (f" — {detail}" if detail else ""))
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def must_fail(name: str, thunk) -> None:
    """A known-negative: the guard must reject the bad input."""
    global _INSIDE_KNOWN_NEGATIVE
    _INSIDE_KNOWN_NEGATIVE = True
    try:
        thunk()
        raised = False
    except Exception:
        raised = True
    finally:
        _INSIDE_KNOWN_NEGATIVE = False
    check(f"known-negative: {name}", raised,
          "the guard accepted input it must reject")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record() -> dict:
    return json.loads(RECORD.read_text())


def _flat(value) -> str:
    return json.dumps(value, ensure_ascii=False).lower()


def _find_key(node, wanted: str, path: str = "") -> list:
    """Every dotted path at which `wanted` appears as a mapping key."""
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            if key == wanted:
                found.append(here)
            found.extend(_find_key(value, wanted, here))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_find_key(value, wanted, f"{path}[{index}]"))
    return found


# --------------------------------------------------------------------------
# A. schema and content hash
# --------------------------------------------------------------------------

def test_schema_and_content_hash() -> None:
    check("the overlay exists", RECORD.exists(), str(RECORD))
    record = _record()
    for key in ("title", "schema_version", "overlay_kind", "experiment_family",
                "precedence", "withdrawn_fields", "retained_fields",
                "no_substitute_value", "remeasurement", "e9_wording",
                "clean_test_governance", "conflicting_historical_artefacts",
                "residual_discoverability_risk", "frozen_pins", "status",
                "content_sha256", "provenance"):
        check(f"the overlay carries {key}", key in record)
    check("the overlay is field-level",
          record["overlay_kind"] == "FIELD_LEVEL", record["overlay_kind"])
    check("the overlay is E7b-scoped",
          record["experiment_family"] == "E7b")

    # The content digest is self-verifying: recomputing it from the written
    # bytes must reproduce the value stored inside them. Provenance is
    # excluded because it carries the repository HEAD and worktree state,
    # which move for reasons that are not scientific.
    stripped = {k: v for k, v in record.items()
                if k not in ("provenance", "content_sha256")}
    text = json.dumps(stripped, indent=2, sort_keys=True,
                      ensure_ascii=False) + "\n"
    recomputed = hashlib.sha256(text.encode("utf-8")).hexdigest()
    check("content_sha256 re-derives from the record's own bytes",
          recomputed == record["content_sha256"],
          f"{recomputed} against {record['content_sha256']}")

    # Known-negative: a changed scientific field must move the digest.
    def _tamper():
        bad = dict(stripped)
        bad["withdrawn_field_count"] = 99
        moved = hashlib.sha256(
            (json.dumps(bad, indent=2, sort_keys=True,
                        ensure_ascii=False) + "\n").encode()).hexdigest()
        assert moved == record["content_sha256"]
    must_fail("a changed scientific field leaves the content digest alone",
              _tamper)

    check("the record declares zero GPU hours",
          record["provenance"]["gpu_hours_charged"] == 0.0)
    check("the record declares nothing measured",
          record["provenance"]["measured_anything"] is False
          and record["provenance"]["gpu_used"] is False
          and record["provenance"]["trained_anything"] is False
          and record["provenance"]["selected_any_checkpoint"] is False)


# --------------------------------------------------------------------------
# B. frozen pin verification
# --------------------------------------------------------------------------

def test_frozen_pins() -> None:
    record = _record()
    pins = record["frozen_pins"]
    check("the overlay pins the historical artefacts",
          pins["pin_count"] >= 50, str(pins["pin_count"]))
    check("every pin is recorded as unchanged", pins["all_unchanged"] is True)
    check("the pin records match the declared count",
          len(pins["records"]) == pins["pin_count"])

    moved, missing = [], []
    for row in pins["records"]:
        path = PROJECT_ROOT / row["path"]
        if not path.exists():
            missing.append(row["path"])
            continue
        if _sha256(path) != row["pinned_sha256"]:
            moved.append(row["path"])
        if row["actual_sha256"] != row["pinned_sha256"]:
            moved.append(row["path"] + " (recorded mismatch)")
    check("no pinned historical artefact is missing", missing == [],
          str(missing[:5]))
    check(f"every pinned artefact still has its frozen hash "
          f"({len(pins['records'])} pins rehashed)", moved == [],
          str(moved[:5]))

    # The raw child records are the measurement evidence and must all be
    # pinned, not merely most of them.
    children = [row["path"] for row in pins["records"]
                if "/e7b_run_" in row["path"]]
    check("all 27 raw E7b child records are pinned",
          len(children) == 27, str(len(children)))

    # Known-negative: the builder's own guard must refuse a moved pin.
    from experiments.closure import build_e7b_supersession as builder
    must_fail("the builder writes when a pinned artefact has moved",
              lambda: _with_broken_pin(builder))


def _with_broken_pin(builder) -> None:
    original = dict(builder.PINS)
    victim = "results/experiments/e7b_serial_efficiency/e7b_results.json"
    try:
        builder.PINS[victim] = "0" * 64
        builder.build()
    finally:
        builder.PINS.clear()
        builder.PINS.update(original)


# --------------------------------------------------------------------------
# C. VE-0, VE-1 and VE-2 are not reopened
# D. closed packet byte identity
# --------------------------------------------------------------------------

def test_closed_packets_untouched() -> None:
    for relative, expected in sorted(CLOSED_PACKETS.items()):
        path = PROJECT_ROOT / relative
        check(f"closed packet unchanged: {relative}",
              path.exists() and _sha256(path) == expected)

    # Nothing new was added under the E7b results directory.
    e7b_dir = PROJECT_ROOT / "results" / "experiments" / "e7b_serial_efficiency"
    names = sorted(p.name for p in e7b_dir.iterdir() if p.is_file())
    check("the E7b results directory still holds its 35 top-level files",
          len(names) == 35, str(len(names)))
    check("the overlay was not written into the E7b results directory",
          not (e7b_dir / RECORD.name).exists())
    check("the overlay lives under results/closure/",
          RECORD.parent.name == "closure")

    # The VE manifests still describe what is on disk.
    for manifest_name, tree in (("VE0_MANIFEST.json", "ve0"),
                                ("VE1_MANIFEST.json", "ve1"),
                                ("VE2_MANIFEST.json", "ve2")):
        manifest = json.loads(
            (PROJECT_ROOT / "results" / tree / manifest_name).read_text())
        entries = _manifest_entries(manifest)
        bad = [row["path"] for row in entries
               if (PROJECT_ROOT / row["path"]).exists()
               and _sha256(PROJECT_ROOT / row["path"]) != row["sha256"]]
        check(f"{manifest_name} entries still rehash ({len(entries)} entries)",
              bad == [], str(bad[:5]))
        check(f"{manifest_name} is not empty", len(entries) > 0)

    record = _record()
    check("the overlay records VE-0, VE-1 and VE-2 as closed and unchanged",
          all(record["status"][tree] == "CLOSED, unchanged"
              for tree in ("ve0", "ve1", "ve2")))


def _manifest_entries(manifest: dict) -> list:
    """Every {path, sha256} pair a VE manifest carries, at any depth."""
    found, stack = [], [manifest]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if isinstance(node.get("path"), str) and \
                    isinstance(node.get("sha256"), str):
                found.append({"path": node["path"], "sha256": node["sha256"]})
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return found


# --------------------------------------------------------------------------
# E. exactly three withdrawn fields
# I. two separate cause classes
# J. the cold reason is grad-mode, not boundary
# K. the peak reasons are boundary, not grad-mode
# --------------------------------------------------------------------------

def test_withdrawal_is_exact() -> None:
    record = _record()
    entries = record["withdrawn_fields"]
    names = sorted(entry["field"] for entry in entries)
    check("exactly three fields are withdrawn", len(entries) == 3, str(names))
    check("the withdrawn fields are exactly the three named ones",
          names == sorted(WITHDRAWN), str(names))
    check("the declared count agrees with the entries",
          record["withdrawn_field_count"] == 3)
    check("the declared names agree with the entries",
          sorted(record["withdrawn_field_names"]) == names)

    for entry in entries:
        status, cause = WITHDRAWN[entry["field"]]
        check(f"{entry['field']} carries final_status {status}",
              entry["final_status"] == status, entry["final_status"])
        check(f"{entry['field']} carries cause_class {cause}",
              entry["cause_class"] == cause, entry["cause_class"])
        check(f"{entry['field']} states a reason",
              isinstance(entry["reason"], str) and len(entry["reason"]) > 80)
        check(f"{entry['field']} applies to all eight timed systems",
              len(entry["applies_to_systems"]) == 8)

    # I. the two causes are separate sets, and neither is empty.
    classes = record["cause_classes"]
    check("the peak fields carry the measurement-boundary cause",
          classes["MEASUREMENT_BOUNDARY_DEFECT"]
          == ["peak_allocated_mib", "peak_reserved_mib"])
    check("the cold field carries the grad-mode cause",
          classes["GRAD_MODE_STANDARDISATION"] == ["cold_first_query_ms"])
    check("the two cause classes are disjoint",
          not (set(classes["MEASUREMENT_BOUNDARY_DEFECT"])
               & set(classes["GRAD_MODE_STANDARDISATION"])))
    check("the record states the causes must not be conflated",
          "must not be conflated" in record["cause_separation"])

    by_field = {entry["field"]: entry for entry in entries}

    # J. the cold reason names torch.no_grad() and does not borrow the
    # peak-memory cause.
    cold = by_field["cold_first_query_ms"]["reason"]
    check("the cold reason names torch.no_grad()", "torch.no_grad()" in cold)
    check("the cold reason names the grad-enabled historical path",
          "grad-enabled" in cold)
    check("the cold reason does not use the measurement-boundary cause",
          not any(word in cold.lower() for word in BOUNDARY_WORDS),
          cold)

    # K. the peak reasons state the boundary defect and never mention
    # grad mode.
    for field in ("peak_allocated_mib", "peak_reserved_mib"):
        reason = by_field[field]["reason"]
        check(f"the {field} reason states the measurement window did not "
              f"isolate the batch-1 serial envelope",
              "measurement window did not isolate" in reason
              and "batch-1 serial inference envelope" in reason)
        check(f"the {field} reason does not use the grad-mode cause",
              not any(word in reason.lower() for word in GRAD_WORDS),
              reason)


# --------------------------------------------------------------------------
# F. warm serial latency and the rest of the retained evidence
# --------------------------------------------------------------------------

def test_retention_is_explicit() -> None:
    record = _record()
    retained = {entry["field"]: entry for entry in record["retained_fields"]}
    check("every retained field is named", set(retained) == RETAINED,
          str(sorted(set(retained) ^ RETAINED)))
    for name, entry in sorted(retained.items()):
        check(f"{name} is RETAINED_VALID",
              entry["final_status"] == "RETAINED_VALID",
              entry["final_status"])
    check("warm serial latency is retained",
          retained["warm_serial_median_ms"]["final_status"]
          == "RETAINED_VALID")
    check("no field is both withdrawn and retained",
          not (set(retained) & set(WITHDRAWN)))

    # The frontier is unchanged and neither withdrawn field is an axis.
    check("the Pareto frontier is the recorded one",
          record["primary_pareto_frontier_common_denominator"]
          == ["concat", "fusion", "vocab1000_product"])
    check("Pareto membership is declared unchanged",
          record["pareto_membership_unchanged"] is True)
    check("the Pareto axes are accuracy and warm serial latency",
          record["pareto_axes"]
          == ["accuracy_raw_distribution_common_denominator",
              "warm_serial_median_ms"])
    check("no withdrawn field is a Pareto axis",
          not (set(record["pareto_axes"]) & set(WITHDRAWN)))

    # The frontier the overlay reports is the one E7b actually stored.
    stored = json.loads(
        (PROJECT_ROOT / "results" / "experiments" / "e7b_serial_efficiency"
         / "e7b_results.json").read_text())
    check("the overlay's frontier matches the stored E7b frontier",
          stored["e7b_analysis"]["primary_pareto_frontier_common_denominator"]
          == record["primary_pareto_frontier_common_denominator"])

    # Warm serial medians on disk are untouched by the overlay.
    systems = stored["e7b_analysis"]["systems"]
    check("the stored warm serial medians are unchanged",
          [round(systems[name]["warm_median_ms"], 4) for name in
           ("concat", "fusion", "vocab1000_product")]
          == [7.6104, 7.6351, 7.6708])


# --------------------------------------------------------------------------
# G. precedence completeness
# --------------------------------------------------------------------------

def test_precedence_is_complete() -> None:
    record = _record()
    precedence = record["precedence"]
    order = precedence["order"]
    check("precedence has exactly three ranks", len(order) == 3)
    check("the ranks are 1, 2, 3",
          [row["rank"] for row in order] == [1, 2, 3])
    check("rank 1 is this overlay",
          order[0]["authority"]
          == "results/closure/e7b_evidence_supersession.json")
    check("rank 2 is the VE-0 supersession map",
          order[1]["authority"] == "results/ve0/supersession_map.json")
    check("rank 3 is the historical stored evidence_status",
          order[2]["authority"] == "historical stored evidence_status")
    check("rank 1 controls named E7b field validity",
          order[0]["controls"] == "named E7b field validity")
    check("rank 2 controls unnamed artefact-group status",
          order[1]["controls"] == "unnamed artefact-group status")
    check("historical numbers are provenance only",
          precedence["historical_numbers"]
          == "Historical numbers remain readable only as provenance.")

    # Every artefact that still shows a conflicting status is named, and the
    # named set covers every surface that actually carries a withdrawn field.
    named = {row["path"] for row in record["conflicting_historical_artefacts"]}
    for required in (
            "results/closure/efficiency_closure_table.json",
            "results/closure/efficiency_closure_table.csv",
            "results/experiments/e7b_serial_efficiency/e7b_results.json",
            "results/experiments/e7b_serial_efficiency/latency_memory.csv"):
        check(f"the overlay names the conflicting artefact {required}",
              required in named)
    for row in record["conflicting_historical_artefacts"]:
        check(f"{row['path']} is resolved by precedence",
              "rank 3" in row["resolution"])
        check(f"{row['path']} is recorded as immutable",
              row["immutable"] is True)

    # The conflict the overlay claims is real: those rows do carry a
    # withdrawn field next to a VERIFIED status.
    table = json.loads(
        (PROJECT_ROOT / "results" / "closure"
         / "efficiency_closure_table.json").read_text())
    e7b_rows = [row for row in table["rows"]
                if row["experiment_family"] == "E7b"]
    check("the closure table really carries 23 E7b rows", len(e7b_rows) == 23)
    # Only the eight end-to-end serial rows carry the withdrawn fields; the
    # fifteen cached rows are a partial pipeline and never recorded memory or
    # a cold query. Both groups are stored as VERIFIED.
    serial = [row for row in e7b_rows
              if row["timing_kind"] == "END_TO_END_SERIAL"]
    check("eight of them are end-to-end serial rows", len(serial) == 8,
          str(len(serial)))
    check("those eight really carry all three withdrawn fields next to a "
          "VERIFIED status",
          all(row["evidence_status"] == "VERIFIED"
              and all(field in row for field in WITHDRAWN)
              for row in serial))
    check("every E7b row in the closure table is stored as VERIFIED",
          {row["evidence_status"] for row in e7b_rows} == {"VERIFIED"})

    # The overlay does not silently widen its scope to other families.
    check("the overlay declares the out-of-scope peak-memory artefacts",
          len(record["out_of_scope"]["paths"]) >= 3)
    check("the out-of-scope statement refuses to withdraw them",
          "NOT withdrawn" in record["out_of_scope"]["statement"])


# --------------------------------------------------------------------------
# H. no substitute value
# --------------------------------------------------------------------------

def test_no_substitute_value() -> None:
    record = _record()
    substitute = record["no_substitute_value"]
    check("no replacement peak-memory value exists",
          substitute["replacement_peak_memory_value_exists"] is False)
    check("no replacement cold-query value exists",
          substitute["replacement_cold_query_value_exists"] is False)
    for key in ("estimation_permitted", "inference_permitted",
                "back_calculation_permitted",
                "cross_node_substitution_permitted",
                "adjustment_factor_permitted"):
        check(f"{key} is false", substitute[key] is False)

    for entry in record["withdrawn_fields"]:
        check(f"{entry['field']} carries no replacement value",
              entry["replacement_value"] is None)
        check(f"{entry['field']} permits no substitute",
              entry["substitute_permitted"] is False)
        check(f"{entry['field']} was not remeasured",
              entry["remeasured"] is False)

    remeasurement = record["remeasurement"]
    check("nothing was remeasured", remeasurement["remeasured"] is False)
    check("remeasurement is not authorised",
          remeasurement["remeasurement_authorised"] is False)
    check("zero GPU hours are charged",
          remeasurement["gpu_hours_charged"] == 0.0)
    check("the historical node is otter155",
          remeasurement["historical_node"] == "otter155")
    check("the current project copy is otter159",
          remeasurement["current_project_copy_node"] == "otter159")
    check("the bridge control difference is recorded",
          remeasurement["bridge_control_relative_difference"] == -0.1869)
    check("the pre-registered tolerance is recorded",
          remeasurement["bridge_control_pre_registered_tolerance"] == 0.10)
    check("the bridge outcome is FAILED",
          remeasurement["bridge_outcome"] == "FAILED")
    check("the bridge really failed its tolerance",
          abs(remeasurement["bridge_control_relative_difference"])
          > remeasurement["bridge_control_pre_registered_tolerance"])

    # No number anywhere in the record can be read as a replacement: the
    # withdrawn entries hold no numeric field beyond the booleans above.
    for entry in record["withdrawn_fields"]:
        numbers = [key for key, value in entry.items()
                   if isinstance(value, (int, float))
                   and not isinstance(value, bool)]
        check(f"{entry['field']} carries no numeric payload",
              numbers == [], str(numbers))


# --------------------------------------------------------------------------
# L. task-scoped clean-test governance
# M. the exact disclosure
# N. the builder names no embargoed target
# --------------------------------------------------------------------------

def test_task_scoped_governance() -> None:
    record = _record()
    governance = record["clean_test_governance"]
    check("the governance block is task-scoped",
          governance["scope"] == "THIS_TASK_ONLY")
    # The two access fields are read literally, by the user decision of
    # 14 August 2026: true, because a mechanical read of the target's bytes
    # did occur during this task. The third field is the one that carries
    # the scientific guarantee, and it is false.
    for key in ("clean_test_accessed_by_this_task",
                "clean_test_read_stat_hashed_opened_searched_or_parsed_"
                "by_this_task"):
        check(f"{key} is present and reports the access literally",
              governance.get(key) is True, str(governance.get(key)))
    check("clean_test_used_for_development_model_selection_or_reporting "
          "is false",
          governance.get(
              "clean_test_used_for_development_model_selection_or_reporting")
          is False)
    check("the two access fields are read literally, not by the flag "
          "convention used elsewhere",
          "read LITERALLY" in governance["field_semantics"]
          and "must not be read as one" in governance["field_semantics"])

    # L. no project-global claim, at any depth. A bare clean_test_accessed
    # anywhere in the record — including inside the provenance block, where
    # the shared closure helper stamps one by default — would assert more
    # than this task can attest, because the clean-test bytes were in fact
    # read once by a reviewer's integrity-hash command.
    globals_found = _find_key(record, "clean_test_accessed")
    check("the record asserts no project-global clean_test_accessed field "
          "at any depth", globals_found == [], str(globals_found[:5]))
    check("the provenance block carries a pointer instead",
          "this record deliberately makes no project-global clean-test claim"
          in record["provenance"]["clean_test_governance_pointer"])
    # Known-negative: the deep search really finds such a field when present.
    check("the deep search can find a project-global field",
          _find_key({"a": {"b": {"clean_test_accessed": False}}},
                    "clean_test_accessed") == ["a.b.clean_test_accessed"])
    check("the record says why it makes no project-global claim",
          "would be false" in governance["no_project_global_claim"])

    # The scoped fields are only honest if their meaning is stated and the
    # one mechanical access this task did make is disclosed rather than
    # hidden behind them.
    check("the scoped fields state what they do not mean",
          "do NOT mean that experiment, builder, test or reporting code "
          "opened the target" in governance["field_semantics"])
    incident = governance["executor_mechanical_access_during_this_task"]
    check("the executor's mechanical access is disclosed, not omitted",
          incident["occurred"] is True)
    check("it is classified as a mechanical access incident",
          incident["classification"].startswith("B"))
    check("the disclosure states the contents were not inspected",
          incident["contents_inspected"] is False)
    check("the disclosure states nothing reached a model or a number",
          incident["reached_any_model_or_number"] is False
          and incident["used_for_development_model_selection_or_reporting"]
          is False)
    check("the disclosure states the target was not hashed",
          incident["hashed"] is False)
    check("the disclosure states the target is named in no written file",
          incident["target_named_in_any_written_file"] is False)
    check("the disclosure records a corrective rule",
          "exclude data/" in incident["corrective_rule"])

    # M. the disclosure is the two-incident wording, verbatim, whitespace-
    # normalised so reflowing the JSON does not break the check.
    stored = re.sub(r"\s+", " ", governance["disclosure"]).strip()
    check("the governance disclosure is the two-incident wording, verbatim",
          stored == re.sub(r"\s+", " ", DISCLOSURE).strip(), stored)
    check("the disclosure does not claim the clean test was never accessed",
          "never accessed" not in stored)
    check("the disclosure does not claim the bytes were read only once",
          SUPERSEDED_DISCLOSURE_FRAGMENT not in stored
          and "read only once" not in stored)
    check("the disclosure names both dates",
          "13 and 14 August 2026" in stored)
    check("the disclosure keeps the contents and use guarantees",
          "never inspected or used for development, model selection, or "
          "reporting decisions" in stored)

    # N. neither the builder nor the record names the embargoed target.
    check("the builder source does not name the embargoed target",
          EMBARGOED not in BUILDER.read_text())
    check("this test source does not name the embargoed target",
          EMBARGOED not in Path(__file__).read_text())
    check("the overlay does not name the embargoed target",
          EMBARGOED not in RECORD.read_text())

    # Known-negative: the builder's payload guard must refuse the token.
    from experiments.closure import build_e7b_supersession as builder
    must_fail("the payload guard accepts the embargoed token",
              lambda: builder.cc.write_json(
                  {"x": f"data/v2/{EMBARGOED}.csv"},
                  PROJECT_ROOT / "results" / "closure" / "_never_written.json"))
    check("the known-negative wrote no file",
          not (PROJECT_ROOT / "results" / "closure"
               / "_never_written.json").exists())


# --------------------------------------------------------------------------
# Two-incident governance: both incidents recorded, the superseded wording
# confined to closed packets, and the global entry points accurate.
# --------------------------------------------------------------------------

def test_two_incident_governance() -> None:
    record = _record()
    governance = record["clean_test_governance"]

    check("two incidents are recorded", governance["incident_count"] == 2)
    incidents = {row["incident_id"]: row for row in governance["incidents"]}
    check("the incidents are A and B", sorted(incidents) == ["A", "B"])
    check("incident A is the 13 August reviewer integrity hash",
          incidents["A"]["date"] == "2026-08-13"
          and "independent reviewer" in incidents["A"]["actor"]
          and "integrity-hash" in incidents["A"]["mechanism"])
    check("incident B is the 14 August dependency-mapping scan",
          incidents["B"]["date"] == "2026-08-14"
          and "dependency-mapping scan" in incidents["B"]["mechanism"]
          and "data/" in incidents["B"]["mechanism"])
    for key, row in sorted(incidents.items()):
        check(f"incident {key} is classification B",
              row["classification"].startswith("B"))
        check(f"incident {key} is mechanical byte access",
              row["nature"] == "mechanical byte access")
        for guarantee in ("contents_inspected",
                          "rows_labels_distributions_or_predictions_examined",
                          "informed_development",
                          "informed_model_selection",
                          "informed_reporting_decisions"):
            check(f"incident {key}: {guarantee} is false",
                  row[guarantee] is False)

    common = governance["common_to_both_incidents"]
    check("both incidents share the classification-B guarantees",
          common["contents_inspected"] is False
          and common["informed_development"] is False
          and common["informed_model_selection"] is False
          and common["informed_reporting_decisions"] is False)
    check("the record separates governance from scientific selection",
          "not test-informed scientific selection" in common["statement"])

    # The superseded wording is kept, labelled, and never presented as the
    # whole history.
    superseded = governance["superseded_single_incident_wording"]
    check("the superseded wording is retained for lineage",
          SUPERSEDED_DISCLOSURE_FRAGMENT in superseded["text"])
    check("it is dated as superseded", superseded["superseded_on"]
          == "2026-08-14")
    check("the record says why it is now incomplete",
          "incomplete as a history" in superseded["why"]
          and "never be presented as the complete history" in superseded["why"])

    # The permitted residuals are exactly the two frozen VE-2 surfaces, and
    # each carries a reason that names the reopening risk.
    residuals = {row["path"]: row["why"]
                 for row in superseded["permitted_residual_locations"]}
    check("the permitted residuals are the two frozen VE-2 surfaces",
          set(residuals) == PERMITTED_RESIDUALS, str(sorted(residuals)))
    check("the immutable manifest residual is labelled historical and "
          "superseded",
          "immutable" in residuals["results/ve2/VE2_MANIFEST.json"]
          and "predates the second incident"
          in residuals["results/ve2/VE2_MANIFEST.json"])
    check("the run_ve2.py residual explains the rebuild coupling",
          "rebuilds VE-2" in residuals["experiments/ve2/run_ve2.py"]
          and "reopen" in residuals["experiments/ve2/run_ve2.py"])

    # The frozen residuals really are frozen, and really do still carry it.
    for relative in sorted(PERMITTED_RESIDUALS):
        text = (PROJECT_ROOT / relative).read_text()
        check(f"{relative} still carries the superseded wording, unedited",
              SUPERSEDED_DISCLOSURE_FRAGMENT in text)
    check("VE2_MANIFEST.json is byte-identical",
          _sha256(PROJECT_ROOT / "results" / "ve2" / "VE2_MANIFEST.json")
          == CLOSED_PACKETS["results/ve2/VE2_MANIFEST.json"])
    check("experiments/ve2/run_ve2.py is byte-identical",
          _sha256(PROJECT_ROOT / "experiments" / "ve2" / "run_ve2.py")
          == "7fab3289488e05d89bd258aa04394df6bbdc88b8c82abe7071196b79dccd4e89")

    # No editable human-facing surface presents the one-incident sentence as
    # the complete history. Where the fragment appears at all, it must be
    # accompanied by the correction in the same document.
    for relative in HUMAN_FACING_DOCS:
        text = (PROJECT_ROOT / relative).read_text()
        flat = re.sub(r"\s+", " ", text)
        if SUPERSEDED_DISCLOSURE_FRAGMENT in flat:
            check(f"{relative} labels the one-incident sentence as "
                  f"superseded rather than current",
                  "SUPERSEDED" in text.upper()
                  and "two documented governance incidents" in flat)
        # The prohibited claim is "the clean test was never accessed". A
        # sentence saying that claim WOULD BE FALSE is the opposite, and is
        # exactly what the disclosure has to say, so each occurrence is
        # judged in context rather than banned outright.
        for hit in re.finditer("never accessed", flat):
            window = flat[max(0, hit.start() - 160): hit.end() + 160]
            check(f"{relative} refutes rather than makes the "
                  f"never-accessed claim",
                  "would be false" in window or "must not" in window,
                  window[:200])

    # The two global entry points carry the accurate disclosure.
    for relative in ("README.md", "docs/REPRODUCIBILITY.md"):
        flat = re.sub(r"\s+", " ",
                      (PROJECT_ROOT / relative).read_text())
        check(f"{relative} discloses two mechanical-access incidents",
              "two documented governance incidents, on 13 and 14 August 2026"
              in flat)
        check(f"{relative} states neither inspected contents",
              "never inspected or used for development, model selection, or "
              "reporting decisions" in flat)
        check(f"{relative} separates governance from scientific selection",
              "not test-informed scientific selection" in flat)
    reproducibility = (PROJECT_ROOT / "docs" / "REPRODUCIBILITY.md").read_text()
    check("REPRODUCIBILITY identifies both incidents separately",
          "13 August 2026 — an independent reviewer" in reproducibility
          and "14 August 2026 — during the E7b canonical-supersession"
          in reproducibility)
    check("REPRODUCIBILITY names the frozen residuals",
          "results/ve2/VE2_MANIFEST.json" in reproducibility
          and "superseded residuals" in reproducibility)
    check("README points a reader to the disclosure",
          "docs/REPRODUCIBILITY.md" in (PROJECT_ROOT / "README.md").read_text())

    # The VE-2 report carries the corrected wording next to the residual note.
    # Markdown is hard-wrapped, so every prose assertion is made against the
    # whitespace-normalised text rather than the raw bytes.
    ve2 = re.sub(r"\s+", " ", (PROJECT_ROOT / "docs" / "experiments"
                               / "ve2_qualitative_evidence.md").read_text())
    check("the VE-2 report carries the two-incident disclosure",
          "two documented governance incidents, on 13 and 14 August 2026"
          in ve2)
    check("the VE-2 report marks the manifest residual as superseded",
          "HISTORICAL, SUPERSEDED RESIDUALS" in ve2)
    check("the VE-2 report explains why the residual is not corrected",
          "reopen a closed packet" in ve2)
    check("the VE-2 report does not blame VE-2 for either incident",
          "Neither incident was caused by VE-2 code" in ve2)


# --------------------------------------------------------------------------
# Conflict coverage: every artefact carrying a withdrawn field is named.
# --------------------------------------------------------------------------

def test_every_carrier_is_named() -> None:
    record = _record()
    coverage = record["conflict_coverage"]
    check("the coverage set is computed, not hand-listed",
          "not hand-listed" in coverage["method"])
    check("every carrier is named", coverage["all_carriers_named"] is True)

    # Re-derive the carrier set here rather than trusting the record.
    spellings = set()
    for entry in record["withdrawn_fields"]:
        spellings.add(entry["field"])
        spellings.update(entry["also_written_as"])
    carriers = []
    for row in record["frozen_pins"]["records"]:
        path = PROJECT_ROOT / row["path"]
        if path.suffix not in (".json", ".csv", ".txt"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(name in text for name in spellings):
            carriers.append(row["path"])
    check("the record's carrier count matches a fresh derivation",
          coverage["carrier_count"] == len(carriers),
          f"{coverage['carrier_count']} against {len(carriers)}")
    check("33 pinned artefacts carry a withdrawn field",
          len(carriers) == 33, str(len(carriers)))
    missing = [path for path in carriers if path not in coverage["resolution"]]
    check("every derived carrier resolves to a named entry", missing == [],
          str(missing[:5]))

    # The artefact the independent review found missing is now named in its
    # own right, not merely swept up by a glob.
    raw = "results/experiments/e7b_serial_efficiency/results.json"
    check("results.json is a carrier", raw in carriers)
    check("results.json resolves to itself, as its own named entry",
          coverage["resolution"].get(raw) == raw)
    named = {row["path"] for row in record["conflicting_historical_artefacts"]}
    check("results.json is named in the precedence controls", raw in named)
    check("results.json is named in the residual surfaces",
          raw in record["residual_discoverability_risk"]["residual_surfaces"])
    entry = next(row for row in record["conflicting_historical_artefacts"]
                 if row["path"] == raw)
    check("the results.json entry is resolved by field-level precedence",
          "rank 3" in entry["resolution"])
    check("the results.json entry records it as immutable",
          entry["immutable"] is True)
    check("the results.json entry says what it carries",
          all(field in entry["conflict"] for field in WITHDRAWN))
    check("results.json is byte-identical",
          _sha256(PROJECT_ROOT / raw)
          == CLOSED_PACKETS[
              "results/experiments/e7b_serial_efficiency/results.json"])

    # Known-negative: dropping a named entry must make the builder refuse.
    from experiments.closure import build_e7b_supersession as builder
    must_fail("the builder writes when a carrier is left unnamed",
              lambda: builder._assert_conflict_coverage(
                  [row for row
                   in builder.build()["conflicting_historical_artefacts"]
                   if row["path"] != raw],
                  builder._carriers()))


# --------------------------------------------------------------------------
# O. human-facing discoverability
# --------------------------------------------------------------------------

def test_human_facing_discoverability() -> None:
    record = _record()
    relative = "results/closure/e7b_evidence_supersession.json"

    report = (PROJECT_ROOT / "docs" / "experiments"
              / "e7b_serial_efficiency.md").read_text()
    first_screen = "\n".join(report.splitlines()[:15])
    check("the E7b report shows the withdrawal within its first 15 lines",
          "WITHDRAWN" in first_screen.upper())
    check("the first screen marks peak memory INVALID",
          "INVALID" in first_screen.upper()
          and "peak_allocated_mib" in first_screen)
    check("the first screen marks the cold first query SUPERSEDED",
          "SUPERSEDED" in first_screen.upper()
          and "cold_first_query_ms" in first_screen)
    check("the report states warm serial latency remains valid",
          re.search(r"[Ww]arm serial latency.{0,200}VALID", report,
                    re.S) is not None)
    check("the report points to the canonical record", relative in report)
    check("the report marks the withdrawn table columns",
          re.search(r"\|\s*cold \(ms\) WITHDRAWN\s*\|", report) is not None
          and re.search(r"peak MiB alloc/res WITHDRAWN", report) is not None)
    check("the report does not declare E7b closed",
          "E7b remains OPEN" in report)

    readme = (PROJECT_ROOT / "README.md").read_text()
    check("README points to the canonical record", relative in readme)
    check("README names the withdrawal", "WITHDRAWN" in readme.upper())
    check("README does not alter the valid warm-latency numbers",
          "7.6 ms per query" in readme and "9.2-20.1 ms" in readme)

    reproducibility = (PROJECT_ROOT / "docs"
                       / "REPRODUCIBILITY.md").read_text()
    check("REPRODUCIBILITY documents the precedence rule",
          relative in reproducibility
          and "results/ve0/supersession_map.json" in reproducibility
          and "lowest precedence" in reproducibility)
    check("REPRODUCIBILITY orders the three ranks",
          reproducibility.index(relative)
          < reproducibility.index("results/ve0/supersession_map.json")
          < reproducibility.index("lowest precedence"))

    claude = (PROJECT_ROOT / "CLAUDE.md").read_text()
    check("CLAUDE.md records the current E7b status",
          "E7b status" in claude and relative in claude)
    check("CLAUDE.md names both cause classes",
          "MEASUREMENT_BOUNDARY_DEFECT" in claude
          and "GRAD_MODE_STANDARDISATION" in claude)
    check("CLAUDE.md records E7b as open",
          "E7b\n  remains OPEN" in claude or "E7b remains OPEN" in claude)

    # The record's own pointer list agrees with the files that really point.
    pointed = {row["path"] for row in record["human_facing_pointers"]}
    check("the record lists its four human-facing pointers",
          pointed == {"docs/experiments/e7b_serial_efficiency.md",
                      "README.md", "docs/REPRODUCIBILITY.md", "CLAUDE.md"},
          str(sorted(pointed)))
    for path in sorted(pointed):
        check(f"{path} really names the canonical record",
              relative in (PROJECT_ROOT / path).read_text())

    # A machine resolver can answer for all three fields.
    check("the record resolves all three withdrawn fields",
          record["machine_resolution"]["resolves_all_three_withdrawn_fields"]
          is True)
    check("retention is explicit, not merely implied by omission",
          record["machine_resolution"]["retention_is_explicit"] is True)

    # The residual risk is recorded rather than claimed away.
    residual = record["residual_discoverability_risk"]
    check("the residual discoverability risk is not claimed to be zero",
          residual["risk_is_zero"] is False)
    check("the residual surfaces are named", len(residual["residual_surfaces"])
          >= 3)
    check("the residual risk is recorded for the F1 review",
          "F1" in residual["recorded_for"])

    # No dissertation figure or table can resolve to a withdrawn value,
    # because VE-0 registers no evidence id for one.
    inventory = json.loads(
        (PROJECT_ROOT / "results" / "ve0"
         / "canonical_evidence_inventory.json").read_text())
    e7b_ids = [row for row in inventory["rows"]
               if str(row.get("evidence_id", "")).startswith("EV-EFF-E7b")]
    check("VE-0 registers 23 E7b efficiency evidence ids",
          len(e7b_ids) == 23, str(len(e7b_ids)))
    offenders = [row["evidence_id"] for row in e7b_ids
                 if any(field in _flat(row) for field in WITHDRAWN)]
    check("no canonical E7b evidence id exposes a withdrawn field",
          offenders == [], str(offenders[:5]))


# --------------------------------------------------------------------------
# P. E9 latency-specific wording
# Q. no unbounded efficiency or cost claim from timing alone
# --------------------------------------------------------------------------

def test_e9_wording() -> None:
    record = _record()
    wording = record["e9_wording"]
    check("the canonical E9 statement is exact",
          wording["canonical_statement"] == CANONICAL_E9,
          wording["canonical_statement"])
    check("the canonical statement is about latency, not cost",
          "slower in warm serial latency" in CANONICAL_E9)
    check("the measured values are recorded",
          wording["measured_values_ms"]["top_1000_global_head"] == 6.199
          and wording["measured_values_ms"]["smolvlm_500m"] == 152.854)
    ratio = (wording["measured_values_ms"]["smolvlm_500m"]
             / wording["measured_values_ms"]["top_1000_global_head"])
    check("24.7x is what the two measured values give",
          abs(ratio - 24.7) < 0.05, f"{ratio:.4f}")
    check("the comparison is labelled contextual positioning",
          wording["comparison_kind"]
          == "contextual positioning, not a matched causal comparison")

    check("every forbidden latency-alone conclusion is listed",
          sorted(wording["forbidden_conclusions_from_latency_alone"])
          == sorted(FORBIDDEN_E9))
    check("every unbounded cost term is refused",
          sorted(wording["not_supported_by_the_timing_result_alone"])
          == sorted(UNBOUNDED_COST))

    # Q. the docs this task updated must not draw a cost or efficiency
    # conclusion from the E9 timing. Scoped to sentences that carry one of
    # the three measured E9 figures, so it is a semantic check on the claim
    # rather than a lexical sweep of unrelated prose.
    for name in ("CLAUDE.md", "README.md",
                 "docs/experiments/e7b_serial_efficiency.md",
                 "docs/REPRODUCIBILITY.md"):
        text = re.sub(r"\s+", " ", (PROJECT_ROOT / name).read_text())
        for sentence in re.split(r"(?<=[.;]) ", text):
            if not any(value in sentence
                       for value in ("152.854", "6.199", "24.7x")):
                continue
            hits = [term for term in FORBIDDEN_E9 if term in sentence.lower()]
            check(f"{name}: the E9 timing sentence draws no cost or "
                  f"efficiency conclusion", hits == [],
                  f"{hits} in {sentence[:160]}")
            hits = [term for term in UNBOUNDED_COST
                    if term in sentence.lower() and "no " + term
                    not in sentence.lower()]
            check(f"{name}: the E9 timing sentence claims no energy, power, "
                  f"carbon or monetary cost", hits == [],
                  f"{hits} in {sentence[:160]}")

    # CLAUDE.md carries the canonical statement, not the withdrawn phrasing.
    claude = re.sub(r"\s+", " ", (PROJECT_ROOT / "CLAUDE.md").read_text())
    check("CLAUDE.md carries the canonical E9 wording",
          "approximately 24.7x slower in warm serial latency" in claude)
    check("CLAUDE.md no longer calls the global head 25 times cheaper",
          "25 times cheaper" not in claude)


# --------------------------------------------------------------------------
# R. the builder is mechanical: no GPU, no model, no measurement
# --------------------------------------------------------------------------

def test_builder_is_mechanical() -> None:
    source = BUILDER.read_text()
    tree = ast.parse(source)

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for banned in ("torch", "open_clip", "transformers", "timm"):
        check(f"the builder does not import {banned}", banned not in imported,
              str(sorted(imported)))

    called = {node.func.attr for node in ast.walk(tree)
              if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute)}
    for banned in ("cuda", "to", "eval", "forward", "load_state_dict",
                   "from_pretrained", "synchronize",
                   "reset_peak_memory_stats", "max_memory_allocated"):
        check(f"the builder never calls {banned}", banned not in called,
              str(sorted(called)))

    # It writes exactly one file, and that file is the overlay.
    writes = [node for node in ast.walk(tree)
              if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute)
              and node.func.attr in ("write_json", "write_csv", "write_bytes",
                                     "write_text", "savefig", "save")]
    check("the builder performs exactly one write", len(writes) == 1,
          str(len(writes)))
    check("the single write is a deterministic JSON write",
          writes[0].func.attr == "write_json")
    check("the builder's output path is the overlay",
          "results/closure/e7b_evidence_supersession.json" in source)

    # The pin assertion runs before the write, not after it.
    check("the builder asserts the pins inside build()",
          "_assert_pins()" in source)
    build_fn = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "build")
    first = ast.dump(build_fn.body[0])
    check("the pin assertion is the first statement of build()",
          "_assert_pins" in first, first[:120])

    from experiments.closure import build_e7b_supersession as builder
    check("the builder declares a zero-GPU provenance block",
          builder.build()["provenance"]["gpu_used"] is False)


# --------------------------------------------------------------------------
# S. the surrounding suites are unchanged
# --------------------------------------------------------------------------

def test_no_regression_in_neighbouring_suites() -> None:
    # The overlay adds one file under results/closure/ and nothing else, so
    # the artefact sets the other suites walk must be the sets they expect.
    closure_dir = PROJECT_ROOT / "results" / "closure"
    names = sorted(p.name for p in closure_dir.iterdir() if p.is_file())
    check("the closure directory gained exactly the overlay",
          RECORD.name in names and len(names) == 20, str(len(names)))

    # run_all.py is inside E10's frozen SOURCE_PATHS, so this task must not
    # have edited it. That is why these tests ship their own runner.
    run_all = (PROJECT_ROOT / "tests" / "run_all.py").read_text()
    check("run_all.py was not edited to import this module",
          "test_e7b_supersession" not in run_all)
    check("run_all.py still globs tests/*.py for the embargo scan",
          'glob("*.py")' in run_all)
    runner = PROJECT_ROOT / "tests" / "run_e7b_supersession.py"
    check("this module ships its own runner", runner.exists())
    check("the runner imports this module",
          "test_e7b_supersession" in runner.read_text())

    # The E7b regression guard from the accepted repair is untouched.
    check("tests/test_e7b.py is unchanged by this task",
          _sha256(PROJECT_ROOT / "tests" / "test_e7b.py")
          == "ac95a923a13241ef5a6e2d828f89be8ca8b8a41e472722662e08306027231595")
    check("the repaired run.py is unchanged by this task",
          _sha256(PROJECT_ROOT / "experiments" / "e7b_serial_efficiency"
                  / "run.py")
          == "fd62ccb2cd7532be154d19cd7c0ab598215940340f7295b29ca8d5ca54495455")


def run() -> None:
    _CHECKS.clear()
    test_schema_and_content_hash()
    test_frozen_pins()
    test_closed_packets_untouched()
    test_withdrawal_is_exact()
    test_retention_is_explicit()
    test_precedence_is_complete()
    test_no_substitute_value()
    test_task_scoped_governance()
    test_two_incident_governance()
    test_every_carrier_is_named()
    test_human_facing_discoverability()
    test_e9_wording()
    test_builder_is_mechanical()
    test_no_regression_in_neighbouring_suites()
    failed = [name for name, ok in _CHECKS if not ok]
    if failed:
        raise AssertionError(
            f"{len(failed)} E7b supersession checks failed: {failed}")
    print(f"  {len(_CHECKS)} E7b supersession checks passed")


if __name__ == "__main__":
    VERBOSE = True
    run()
