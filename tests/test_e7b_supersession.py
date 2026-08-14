"""Tests for the E7b evidence supersession overlay and the pre-F1 lifecycle
record that supersedes its status block.

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

POST-CLOSURE EXTENSION, 14 August 2026. This suite was accepted by the
independent E7b review at E7B_CANONICAL_SUPERSESSION_REVIEW_PASS. It is
extended here for LIFECYCLE and DISCOVERABILITY only: the five packets are
now closed, the overlay's own status block still reads OPEN because it is
byte-pinned, and a consolidated record supersedes that lifecycle text. The
extension asserts no scientific quantity, reinstates no withdrawn field and
reopens no E7b science — but a suite an independent review accepted should
not grow without one, so this modification REQUIRES INDEPENDENT PRE-F1
REVIEW.

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

# The consolidated pre-F1 lifecycle and wording record. It supersedes the
# status block above and nothing else.
STATUS_RECORD_NAME = "pre_f1_status_supersession_20260814.json"
STATUS_RECORD = PROJECT_ROOT / "results" / "closure" / STATUS_RECORD_NAME
STATUS_BUILDER = (PROJECT_ROOT / "experiments" / "closure"
                  / "build_pre_f1_status_supersession.py")
STATUS_RELATIVE = f"results/closure/{STATUS_RECORD_NAME}"

# The E7b overlay is byte-pinned by this task: the lifecycle supersession is
# published beside it, never into it.
E7B_OVERLAY_SHA256 = (
    "2fb6561b3916b83603631c0feb35d716dc9bbf3e1ed1ef3d4e7286516a0a0c75")

# The five packets the consolidated record covers, with the verdict each was
# accepted at. Written here rather than read from the record, so the test
# disagrees with the record if either moves.
CLOSED_PACKETS_LIFECYCLE = {
    "e7b": "E7B_CANONICAL_SUPERSESSION_REVIEW_PASS",
    "ve0": "VE0_PASS",
    "ve1": "VE1_PASS",
    "ve2": "VE2_PASS",
    "e10": "E10_PASS",
}

# Statements VE-0's supersession map records as removed and forbidden to
# return, plus the additive comparison this task demoted. None may appear as
# a live claim on a current-facing surface.
E7A_FORBIDDEN_STATEMENTS = (
    "cuts a query from 6.35 to 2.25 ms",
    "on both end-to-end latency Pareto fronts",
)
E7A_INVALID_COMPARISON = ("6.35 against 7.71", "6.35 ms to 2.25 ms")

# Documents a reader treats as current. The E7a report is included because
# its banner is the first thing on the page; the forbidden sentences may
# appear there only as quoted prohibitions. collab/PROJECT_CONTEXT.md is here
# because the independent pre-F1 review found the two forbidden sentences
# alive in its "Current truth" section, where the previous coverage set could
# not see them: it is the file the collaboration protocol tells an agent to
# read first, so a superseded claim there is as current-facing as one in
# README.
E7A_CURRENT_FACING = (
    "README.md",
    "CLAUDE.md",
    "docs/REPRODUCIBILITY.md",
    "collab/PROJECT_CONTEXT.md",
    "docs/experiments/e7a_efficiency.md",
    "docs/experiments/e7b_serial_efficiency.md",
)

# Surfaces where the forbidden sentences may not appear at all, not even as a
# quoted prohibition, because none of them is the report that publishes the
# supersession. Keeping this separate from E7A_CURRENT_FACING is what makes
# the guard non-vacuous on the E7a report itself.
E7A_NO_QUOTED_EXEMPTION = (
    "README.md",
    "CLAUDE.md",
    "collab/PROJECT_CONTEXT.md",
)

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
    "collab/PROJECT_CONTEXT.md",
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


def _status_record() -> dict:
    return json.loads(STATUS_RECORD.read_text())


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
    # E7b closed on 14 August 2026. The report must say so, and must send the
    # reader to the lifecycle record rather than to the overlay's own status
    # block, which still reads OPEN because it is byte-pinned.
    check("the report no longer says E7b remains open",
          "E7b remains OPEN" not in report)
    check("the report declares E7b CLOSED", "E7b is CLOSED" in report)
    check("the report names the accepted verdict",
          "E7B_CANONICAL_SUPERSESSION_REVIEW_PASS" in report)
    check("the report points to the lifecycle record",
          STATUS_RELATIVE in report)

    readme = (PROJECT_ROOT / "README.md").read_text()
    check("README points to the canonical record", relative in readme)
    check("README points to the lifecycle record", STATUS_RELATIVE in readme)
    check("README names the withdrawal", "WITHDRAWN" in readme.upper())
    check("README does not alter the valid warm-latency numbers",
          "7.6 ms per query" in readme and "9.2-20.1 ms" in readme)
    check("README no longer says E7b remains open",
          "E7b remains open" not in readme)
    check("README declares E7b CLOSED", "E7b is CLOSED" in readme)
    check("README is current to 14 August 2026",
          "current to 14 August 2026" in readme)
    for command in ("tests/run_closure.py", "tests/run_ve0.py",
                    "tests/run_ve1.py", "tests/run_ve2.py",
                    "tests/run_e7b_supersession.py"):
        check(f"README documents the {command} runner", command in readme)

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
    # Lifecycle precedence is stated, and stated BEFORE field validity, so a
    # reader settles "is the packet open" before "is the number current".
    check("REPRODUCIBILITY documents the lifecycle record",
          STATUS_RELATIVE in reproducibility)
    check("REPRODUCIBILITY puts lifecycle precedence above field validity",
          reproducibility.index(STATUS_RELATIVE)
          < reproducibility.index("lowest precedence"))
    check("REPRODUCIBILITY says the lifecycle record changes no result",
          "changes no accuracy, latency," in reproducibility)
    for command in ("tests/run_closure.py", "tests/run_ve0.py",
                    "tests/run_ve1.py", "tests/run_ve2.py",
                    "tests/run_e7b_supersession.py"):
        check(f"REPRODUCIBILITY documents the {command} runner",
              command in reproducibility)

    claude = (PROJECT_ROOT / "CLAUDE.md").read_text()
    check("CLAUDE.md records the current E7b status",
          "E7b status" in claude and relative in claude)
    check("CLAUDE.md names both cause classes",
          "MEASUREMENT_BOUNDARY_DEFECT" in claude
          and "GRAD_MODE_STANDARDISATION" in claude)
    check("CLAUDE.md no longer records E7b as open",
          "E7b\n  remains OPEN" not in claude
          and "E7b remains OPEN" not in claude)
    check("CLAUDE.md records E7b as CLOSED", "E7b is\n  CLOSED" in claude
          or "E7b is CLOSED" in claude)
    check("CLAUDE.md points to the lifecycle record",
          STATUS_RELATIVE in claude)
    flat_claude = re.sub(r"\s+", " ", claude)
    check("CLAUDE.md records E10 as closed at E10_PASS",
          "E10 is CLOSED at E10_PASS" in flat_claude)
    check("CLAUDE.md no longer says no final E10 PASS is declared",
          "no final E10 PASS is declared. The clean test" not in flat_claude)
    for packet, verdict in sorted(CLOSED_PACKETS_LIFECYCLE.items()):
        check(f"CLAUDE.md names the accepted {packet} verdict",
              verdict in flat_claude, verdict)
    check("CLAUDE.md carries the required clean-test wording",
          "Mechanical byte access occurred in two documented governance "
          "incidents, on 13 and 14 August 2026." in flat_claude)
    check("CLAUDE.md identifies both incidents separately",
          "13 August 2026 — an independent reviewer" in flat_claude
          and "14 August 2026 — during the E7b canonical-supersession"
          in flat_claude)
    check("CLAUDE.md records the corrective scan rule",
          "must exclude data/ before reading or traversing" in flat_claude)
    check("CLAUDE.md states F1 and F2 remain unstarted",
          "F1 is UNSTARTED and unauthorised" in flat_claude)

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
    # The overlay adds one file under results/closure/, the pre-F1
    # lifecycle repair adds exactly one more, and the pre-F1
    # evidence-metadata repair, which supersedes the packet that wrote this
    # guard, adds exactly one more at the location its accepted plan review
    # specified. The artefact set the other suites walk must be exactly
    # these plus the closure tables.
    closure_dir = PROJECT_ROOT / "results" / "closure"
    names = sorted(p.name for p in closure_dir.iterdir() if p.is_file())
    check("the closure directory holds the overlay, the lifecycle record "
          "and the evidence-metadata successor and nothing else new",
          RECORD.name in names and STATUS_RECORD_NAME in names
          and "pre_f1_evidence_metadata_repair_20260814.json" in names
          and len(names) == 22, str(len(names)))

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
    check("run_all.py is byte-identical",
          _sha256(PROJECT_ROOT / "tests" / "run_all.py")
          == "0a53250ac3c10edd788ea756a64dff54e83d2dec824ad557a6f8ea629e741786")
    check(".gitignore is byte-identical",
          _sha256(PROJECT_ROOT / ".gitignore")
          == "9ec096d242e61dd67650f619680d7a0d34115872eca06475dd701ce0e87ed7d1")


# --------------------------------------------------------------------------
# T. the consolidated pre-F1 lifecycle record: schema and digest
# --------------------------------------------------------------------------

def test_status_record_schema_and_digest() -> None:
    check("the lifecycle record exists by its exact name",
          STATUS_RECORD.exists(), STATUS_RELATIVE)
    check("its builder exists", STATUS_BUILDER.exists())
    record = _status_record()

    for key in ("title", "record_type", "scope", "changes_no_scientific_result",
                "purpose", "what_this_record_does_not_do", "precedence",
                "lifecycle_status", "e7b_lifecycle_supersession",
                "ve0_pinned_status_treatment", "ve1_ve2_status_edits",
                "e10_status_correction", "e7a_wording_repair",
                "frozen_rq_matrix_wording_erratum", "clean_test_governance",
                "f1_f2", "task_footprint", "verification_runners",
                "frozen_pins", "content_sha256", "provenance"):
        check(f"the lifecycle record carries {key}", key in record)

    check("the record type is PRE_F1_STATUS_SUPERSESSION",
          record["record_type"] == "PRE_F1_STATUS_SUPERSESSION",
          record["record_type"])
    check("the scope is LIFECYCLE_AND_WORDING_ONLY",
          record["scope"] == "LIFECYCLE_AND_WORDING_ONLY", record["scope"])
    check("it declares that it changes no scientific result",
          record["changes_no_scientific_result"] is True)

    # The content digest is self-verifying, on the same two-tier rule the
    # overlay uses: provenance carries the repository HEAD and is excluded.
    stripped = {k: v for k, v in record.items()
                if k not in ("provenance", "content_sha256")}
    text = json.dumps(stripped, indent=2, sort_keys=True,
                      ensure_ascii=False) + "\n"
    recomputed = hashlib.sha256(text.encode("utf-8")).hexdigest()
    check("its content_sha256 re-derives from its own bytes",
          recomputed == record["content_sha256"],
          f"{recomputed} against {record['content_sha256']}")

    def _tamper():
        bad = dict(stripped)
        bad["scope"] = "EVERYTHING"
        moved = hashlib.sha256(
            (json.dumps(bad, indent=2, sort_keys=True,
                        ensure_ascii=False) + "\n").encode()).hexdigest()
        assert moved == record["content_sha256"]
    must_fail("a changed scope leaves the lifecycle digest alone", _tamper)

    check("the lifecycle record declares zero GPU hours",
          record["provenance"]["gpu_hours_charged"] == 0.0)
    check("the lifecycle record measured, trained and selected nothing",
          record["provenance"]["gpu_used"] is False
          and record["provenance"]["trained_anything"] is False
          and record["provenance"]["measured_anything"] is False
          and record["provenance"]["selected_any_checkpoint"] is False)
    check("the lifecycle record reopened no closed packet",
          record["provenance"]["reopened_any_closed_packet"] is False)

    # It must not smuggle in a project-global clean-test claim either.
    globals_found = _find_key(record, "clean_test_accessed")
    check("the lifecycle record asserts no project-global "
          "clean_test_accessed field", globals_found == [],
          str(globals_found[:5]))
    check("neither the lifecycle record nor its builder names the embargoed "
          "target",
          EMBARGOED not in STATUS_RECORD.read_text()
          and EMBARGOED not in STATUS_BUILDER.read_text())


# --------------------------------------------------------------------------
# U. lifecycle: five closed packets, F1 and F2 unstarted
# --------------------------------------------------------------------------

def test_status_record_lifecycle() -> None:
    record = _status_record()
    lifecycle = record["lifecycle_status"]
    check("exactly the five closed packets are covered",
          sorted(lifecycle) == sorted(CLOSED_PACKETS_LIFECYCLE),
          str(sorted(lifecycle)))
    check("the declared count agrees",
          record["closed_packet_count"] == len(CLOSED_PACKETS_LIFECYCLE))
    for packet, verdict in sorted(CLOSED_PACKETS_LIFECYCLE.items()):
        entry = lifecycle[packet]
        check(f"{packet} is CLOSED", entry["status"] == "CLOSED",
              entry["status"])
        check(f"{packet} records its accepted verdict {verdict}",
              entry["accepted_verdict"] == verdict, entry["accepted_verdict"])
        check(f"{packet} names the stale statement it supersedes",
              len(entry["superseded_statements"]) >= 1)
    check("VE-0's amendment verdict is recorded",
          lifecycle["ve0"]["amendment_verdict"] == "VE0_AMENDMENT_PASS")
    check("E10's closure is attributed to the frozen supersession map",
          lifecycle["e10"]["recorded_in"]
          == "results/ve0/supersession_map.json")

    # F1 is not declared ready by the record that removes E7b's blocker.
    f1f2 = record["f1_f2"]
    check("F1 is unstarted", f1f2["f1"] == "UNSTARTED")
    check("F2 is unstarted and unauthorised",
          f1f2["f2"] == "UNSTARTED_AND_UNAUTHORISED")
    check("F1 is not declared ready by this record",
          f1f2["f1_declared_ready_by_this_record"] is False)
    check("F1 is not authorised", f1f2["f1_authorised"] is False)
    check("the clean-test embargo is unchanged",
          f1f2["clean_test_embargo"] == "UNCHANGED")

    # The scope limit is asserted field by field, not merely in prose.
    limits = record["what_this_record_does_not_do"]
    for key in ("changes_no_accuracy", "changes_no_latency",
                "changes_no_parameter_count", "changes_no_interval",
                "changes_no_pareto_membership", "changes_no_figure_or_table",
                "changes_no_raw_artefact", "reinstates_no_withdrawn_field",
                "introduces_no_replacement_value", "remeasures_nothing",
                "reopens_no_closed_packet", "authorises_nothing"):
        check(f"the record declares {key}", limits[key] is True)
    check("the record does not declare F1 ready",
          limits["declares_f1_ready"] is False)

    # Precedence: lifecycle here, field validity elsewhere.
    precedence = record["precedence"]
    check("this record controls no field validity",
          precedence["this_record_controls_no_field_validity"] is True)
    check("lifecycle rank 1 is this record",
          precedence["lifecycle_order"][0]["authority"] == STATUS_RELATIVE)
    field_order = precedence["field_validity_order"]
    check("field rank 1 is still the E7b overlay",
          field_order[0]["authority"]
          == "results/closure/e7b_evidence_supersession.json")
    check("field rank 2 is still the VE-0 supersession map",
          field_order[1]["authority"] == "results/ve0/supersession_map.json")
    check("field rank 3 is still the historical evidence_status",
          field_order[2]["authority"] == "historical stored evidence_status")

    # The task is recorded as a lifecycle extension needing its own review.
    footprint = record["task_footprint"]
    check("the task charges zero GPU hours",
          footprint["gpu_hours_charged"] == 0.0)
    check("the task trained, evaluated and measured nothing",
          footprint["trained_anything"] is False
          and footprint["evaluated_anything"] is False
          and footprint["measured_anything"] is False)
    check("exactly one test file was modified",
          footprint["only_test_file_modified"]
          == "tests/test_e7b_supersession.py")
    check("the test modification is a lifecycle and discoverability "
          "extension",
          footprint["test_modification_kind"]
          == "POST_CLOSURE_LIFECYCLE_DISCOVERABILITY_EXTENSION")
    check("the test modification reopens no E7b science",
          footprint["test_modification_reopens_e7b_science"] is False)
    check("the test modification requires independent pre-F1 review",
          footprint["test_modification_requires_independent_pre_f1_review"]
          is True)
    check("this test source records that it needs that review",
          "REQUIRES INDEPENDENT PRE-F1" in Path(__file__).read_text().upper())


# --------------------------------------------------------------------------
# V. the E7b lifecycle supersession leaves field validity alone
# --------------------------------------------------------------------------

def test_e7b_lifecycle_supersession() -> None:
    record = _status_record()
    block = record["e7b_lifecycle_supersession"]
    check("it supersedes the E7b overlay",
          block["supersedes"]
          == "results/closure/e7b_evidence_supersession.json")
    check("it supersedes the status block only",
          block["supersedes_scope"] == "the status block only")
    check("status.e7b OPEN is marked historical lifecycle state only",
          block["status_e7b_open_is_historical_lifecycle_state_only"] is True)
    check("status.f1 BLOCKED is marked historical lifecycle state only",
          block["status_f1_blocked_is_historical_lifecycle_state_only"]
          is True)
    check("the historical values are recorded verbatim",
          block["historical_values"]["e7b"] == "OPEN"
          and block["historical_values"]["f1"] == "BLOCKED")
    check("it records that E7b was independently accepted and closed",
          "accepted and formally CLOSED" in block["what_happened_since"])

    for field in ("withdrawn_fields", "retained_fields",
                  "field-level precedence", "metric validity"):
        check(f"the older record stays authoritative for {field}",
              field in block["older_record_remains_authoritative_for"])
    check("the older record was not edited",
          block["older_record_edited"] is False)
    check("the older record was not regenerated",
          block["older_record_regenerated"] is False)
    check("the withdrawn fields are unchanged",
          block["withdrawn_fields_unchanged"] is True)
    check("the retained fields are unchanged",
          block["retained_fields_unchanged"] is True)
    check("Pareto membership is unchanged",
          block["pareto_membership_unchanged"] is True)

    # The proof, not the claim: the overlay's bytes.
    actual = _sha256(RECORD)
    check("the E7b overlay is byte-identical",
          actual == E7B_OVERLAY_SHA256, actual)
    check("the record pins the same overlay hash",
          block["older_record_sha256"] == E7B_OVERLAY_SHA256)
    overlay = _record()
    check("the overlay still carries the lifecycle text being superseded",
          overlay["status"]["e7b"] == "OPEN"
          and overlay["status"]["f1"] == "BLOCKED")
    check("the overlay's field-level content is untouched",
          overlay["withdrawn_field_count"] == 3
          and overlay["retained_field_count"] == 7
          and overlay["primary_pareto_frontier_common_denominator"]
          == ["concat", "fusion", "vocab1000_product"])

    # VE-0's report is pinned and was not edited to carry its own correction.
    ve0 = record["ve0_pinned_status_treatment"]
    check("the VE-0 report is recorded as not edited",
          ve0["report_edited"] is False)
    check("the VE-0 report is byte-identical",
          _sha256(PROJECT_ROOT / ve0["report"]) == ve0["report_sha256"])
    check("the VE-0 report is byte-pinned by its manifest",
          ve0["report_byte_pinned_by"] == "results/ve0/VE0_MANIFEST.json")
    check("the still-true parts of the stale section are named",
          any("F1 and F2" in line
              for line in ve0["still_true_in_the_stale_section"]))

    # VE-1 and VE-2 reports were corrected; their result trees were not.
    edits = record["ve1_ve2_status_edits"]
    check("results/ve1 was not modified",
          edits["results_ve1_modified"] is False)
    check("results/ve2 was not modified",
          edits["results_ve2_modified"] is False)
    check("no figure or table was modified",
          edits["figures_or_tables_modified"] is False)
    for relative, verdict in (("docs/experiments/ve1_figures_and_tables.md",
                               "VE1_PASS"),
                              ("docs/experiments/ve2_qualitative_evidence.md",
                               "VE2_PASS")):
        text = (PROJECT_ROOT / relative).read_text()
        flat = re.sub(r"\s+", " ", text)
        name = "VE-1" if "ve1" in relative else "VE-2"
        check(f"{relative} declares {name} CLOSED",
              f"{name} is CLOSED" in flat, relative)
        check(f"{relative} names {verdict}", verdict in flat)
        check(f"{relative} points to the lifecycle record",
              STATUS_RELATIVE in flat)
    ve1_doc = re.sub(r"\s+", " ", (PROJECT_ROOT / "docs" / "experiments"
                                   / "ve1_figures_and_tables.md").read_text())
    check("the VE-1 report no longer says VE-2 has not been started",
          "VE-2, F1 and F2 have not been started" not in ve1_doc)

    # E10's correction cites the frozen map and touched no E10 surface.
    e10 = record["e10_status_correction"]
    check("E10 is recorded as CLOSED", e10["current_status"] == "CLOSED")
    check("E10's verdict is E10_PASS",
          e10["accepted_verdict"] == "E10_PASS")
    check("the E10 status is established by the frozen supersession map",
          e10["established_by"] == "results/ve0/supersession_map.json"
          and _sha256(PROJECT_ROOT / e10["established_by"])
          == e10["established_by_sha256"])
    for key in ("e10_source_modified", "e10_result_modified",
                "e10_test_modified", "e10_config_modified"):
        check(f"E10 {key} is false", e10[key] is False)
    check("the twelve-cell matrix is unchanged",
          e10["twelve_cell_matrix_unchanged"] is True)


# --------------------------------------------------------------------------
# W. the frozen surfaces really did not move
# --------------------------------------------------------------------------

def test_frozen_surfaces_untouched() -> None:
    record = _status_record()
    pins = record["frozen_pins"]
    check("the lifecycle record pins the frozen surfaces",
          pins["pin_count"] >= 15, str(pins["pin_count"]))
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
    check("no pinned surface is missing", missing == [], str(missing[:5]))
    check(f"every pinned surface still has its frozen hash "
          f"({len(pins['records'])} pins rehashed)", moved == [],
          str(moved[:5]))

    # The three frozen VE trees gained and lost nothing, and every manifest
    # entry still rehashes.
    for tree, expected_files in (("ve0", 15), ("ve1", 83), ("ve2", 19)):
        root = PROJECT_ROOT / "results" / tree
        found = sum(1 for p in root.rglob("*") if p.is_file())
        check(f"results/{tree}/ still holds {expected_files} files",
              found == expected_files, str(found))
    for tree, name in (("ve0", "VE0_MANIFEST.json"),
                       ("ve1", "VE1_MANIFEST.json"),
                       ("ve2", "VE2_MANIFEST.json")):
        manifest = json.loads(
            (PROJECT_ROOT / "results" / tree / name).read_text())
        entries = _manifest_entries(manifest)
        bad = [row["path"] for row in entries
               if (PROJECT_ROOT / row["path"]).exists()
               and _sha256(PROJECT_ROOT / row["path"]) != row["sha256"]]
        check(f"{name} entries still rehash after the repair "
              f"({len(entries)} entries)", bad == [], str(bad[:5]))

    # Nothing was written into the frozen experiment trees.
    check("the lifecycle record lives under results/closure/",
          STATUS_RECORD.parent.name == "closure")
    for tree in ("ve0", "ve1", "ve2"):
        check(f"the lifecycle record was not written into results/{tree}/",
              not (PROJECT_ROOT / "results" / tree
                   / STATUS_RECORD_NAME).exists())
    check("the lifecycle record was not written into the E7b results "
          "directory",
          not (PROJECT_ROOT / "results" / "experiments"
               / "e7b_serial_efficiency" / STATUS_RECORD_NAME).exists())

    # Known-negative: the builder's own pin guard must refuse a moved pin.
    from experiments.closure import build_pre_f1_status_supersession as builder
    must_fail("the lifecycle builder writes when a pinned surface has moved",
              lambda: _with_broken_status_pin(builder))


def _with_broken_status_pin(builder) -> None:
    original = dict(builder.PINS)
    try:
        builder.PINS[builder.E7B_OVERLAY] = "0" * 64
        builder.build()
    finally:
        builder.PINS.clear()
        builder.PINS.update(original)


# --------------------------------------------------------------------------
# X. the frozen RQ matrix: wording superseded, nothing else
# --------------------------------------------------------------------------

def test_rq_matrix_wording_erratum() -> None:
    record = _status_record()
    erratum = record["frozen_rq_matrix_wording_erratum"]
    check("the erratum names the frozen RQ matrix",
          erratum["artefact"] == "results/ve0/rq_evidence_matrix.json")
    check("the RQ matrix was not edited",
          erratum["artefact_edited"] is False)
    check("the RQ matrix is recorded as immutable",
          erratum["artefact_immutable"] is True)
    check("the offending wording is 'far cheaper'",
          erratum["offending_wording"] == "far cheaper")
    check("its final status is WORDING_SUPERSEDED",
          erratum["final_status"] == "WORDING_SUPERSEDED",
          erratum["final_status"])

    # The supersession is by WORDING AUTHORITY only: no answer status, no
    # evidence id and no number moves.
    check("no RQ answer status changed",
          erratum["rq_answer_status_changed"] is False)
    check("no evidence id changed", erratum["evidence_ids_changed"] is False)
    check("no numeric result changed",
          erratum["numeric_results_changed"] is False)
    check("the superseding authority is the claim ledger",
          erratum["superseding_authority"] == "results/ve0/claim_ledger.json")
    check("the superseding claims are C13 and C18",
          erratum["superseding_claims"] == ["C13", "C18"])
    check("the superseding wording is the measured-latency form",
          erratum["superseding_wording"]
          == "at a small fraction of the measured end-to-end serial latency")
    check("the claim ledger was already correct",
          erratum["already_correct_in_the_claim_ledger"] is True)

    # The frozen artefacts really say what the erratum says they say.
    matrix = json.loads((PROJECT_ROOT / "results" / "ve0"
                         / "rq_evidence_matrix.json").read_text())
    offending = {row["rq_id"] for row in matrix["questions"]
                 if "far cheaper" in row.get("supported_answer", "")}
    check("the frozen matrix still carries the superseded wording, unedited",
          offending == set(erratum["affected_questions"]),
          str(sorted(offending)))
    for rq_id in sorted(offending):
        row = next(r for r in matrix["questions"] if r["rq_id"] == rq_id)
        check(f"{rq_id} keeps its answer status",
              row["answer_status"] == "ANSWERED", row["answer_status"])

    ledger = json.loads((PROJECT_ROOT / "results" / "ve0"
                         / "claim_ledger.json").read_text())
    texts = {row["claim_id"]: row["claim_text"] for row in ledger["claims"]}
    # C13 and C18 state the same bound in different words, so the record
    # carries both phrasings rather than collapsing them into one string.
    by_claim = erratum["superseding_wording_by_claim"]
    check("C13's phrasing is the headline superseding wording",
          by_claim["C13"] == erratum["superseding_wording"])
    for claim_id in ("C13", "C18"):
        check(f"{claim_id} already states the measured-latency form",
              by_claim[claim_id] in texts[claim_id], claim_id)
        check(f"{claim_id} bounds the claim to measured latency",
              "measured" in by_claim[claim_id]
              and "latency" in by_claim[claim_id])
        check(f"{claim_id} makes no cheapness claim",
              "cheaper" not in texts[claim_id])
    check("the record states that neither claim says cheaper",
          erratum["neither_claim_says_cheaper"] is True)


# --------------------------------------------------------------------------
# Y. no superseded E7a claim survives on a current-facing surface
# --------------------------------------------------------------------------

def _sentences(text: str) -> list:
    return re.split(r"(?<=[.;]) ", re.sub(r"\s+", " ", text))


# Words that mark an occurrence as a prohibition rather than a claim.
_PROHIBITION_MARKERS = ("must not", "superseded", "forbidden", "prohibit")


def _superseded_e7a_violations(relative: str, text: str) -> list:
    """Every place `text` presents a superseded E7a timing conclusion as fact.

    Two rules, both semantic rather than lexical.

    A banned sentence or the additive 6.35/7.71 comparison is exonerated only
    where the surrounding window marks it as superseded or forbidden, which is
    how the E7a report is able to quote the two banned sentences in the act of
    banning them. Surfaces in E7A_NO_QUOTED_EXEMPTION get no such exemption:
    none of them publishes the supersession, so a quoted occurrence there
    reads to a reader exactly like a live claim.

    A cheapness word is a violation unless its own sentence marks it as a
    prohibition. That is what the independent pre-F1 review found alive in
    collab/PROJECT_CONTEXT.md, where the previous coverage set could not see
    it.
    """
    flat = re.sub(r"\s+", " ", text)
    violations = []
    quoting_allowed = relative not in E7A_NO_QUOTED_EXEMPTION
    for phrase in E7A_FORBIDDEN_STATEMENTS + E7A_INVALID_COMPARISON:
        for hit in re.finditer(re.escape(phrase), flat):
            window = flat[max(0, hit.start() - 240): hit.end() + 240]
            if quoting_allowed and ("must not return" in window
                                    or "SUPERSEDED" in window):
                continue
            violations.append((relative, phrase, window[:200]))
    for sentence in _sentences(flat):
        lowered = sentence.lower()
        if not any(term in lowered
                   for term in ("cheaper", "more efficient", "lower cost")):
            continue
        if any(marker in lowered for marker in _PROHIBITION_MARKERS):
            continue
        violations.append((relative, "latency-only cheapness claim",
                           sentence[:200]))
    return violations


def test_e7a_superseded_claims_are_gone() -> None:
    record = _status_record()
    repair = record["e7a_wording_repair"]
    check("the two forbidden statements are named",
          sorted(repair["forbidden_statements_that_must_not_return"])
          == sorted(E7A_FORBIDDEN_STATEMENTS))
    check("the invalid comparison is recorded with both values",
          repair["invalid_current_comparison"]["values_ms"] == [6.35, 7.71])
    check("the invalid comparison is removed from current-facing surfaces",
          repair["invalid_current_comparison"]
          ["removed_from_current_facing_surfaces"] is True)
    check("no E7b value was substituted",
          repair["replacement_policy"]["replaced_with_e7b_values"] is False)
    check("no new cross-experiment comparison was manufactured",
          repair["replacement_policy"]
          ["new_cross_experiment_comparison_manufactured"] is False)
    check("no underlying number was altered",
          repair["replacement_policy"]["underlying_numbers_altered"] is False
          and repair["e7a_numbers_altered"] is False)
    check("the E7a artefact itself was not edited",
          repair["e7a_artefact_edited"] is False)
    check("the bounded wording keeps only accuracy and parameters",
          "more accurate (0.4904 against 0.4594)" in repair["bounded_wording"]
          and "16 times fewer parameters" in repair["bounded_wording"]
          and "superseded" in repair["bounded_wording"])

    # The bounded wording is what the documents actually carry: the accuracy
    # pair and the parameter ratio, and no latency half.
    for relative in ("docs/experiments/e7a_efficiency.md", "CLAUDE.md",
                     "README.md"):
        flat = re.sub(r"\s+", " ", (PROJECT_ROOT / relative).read_text())
        check(f"{relative} carries the bounded accuracy-and-parameter "
              f"statement",
              "more accurate (0.4904 against 0.4594" in flat
              and "16 times fewer parameters" in flat, relative)

    # The forbidden sentences, the additive comparison and any latency-only
    # cheapness claim survive nowhere current-facing. collab/PROJECT_CONTEXT.md
    # is in this set because the independent pre-F1 review found both banned
    # sentences alive in its "Current truth" section.
    check("the guard covers collab/PROJECT_CONTEXT.md",
          "collab/PROJECT_CONTEXT.md" in E7A_CURRENT_FACING
          and "collab/PROJECT_CONTEXT.md" in HUMAN_FACING_DOCS)
    for relative in E7A_CURRENT_FACING:
        violations = _superseded_e7a_violations(
            relative, (PROJECT_ROOT / relative).read_text())
        check(f"{relative} presents no superseded E7a timing conclusion",
              violations == [], str(violations[:2]))

    # PROJECT_CONTEXT's repaired paragraph keeps the valid facts and says
    # plainly that the additive conclusions are superseded.
    context = re.sub(r"\s+", " ", (PROJECT_ROOT / "collab"
                                   / "PROJECT_CONTEXT.md").read_text())
    check("PROJECT_CONTEXT marks E7a as partly superseded",
          "PARTLY SUPERSEDED" in context)
    check("PROJECT_CONTEXT names the three superseded additive fields",
          all(field in context for field in ("gpu_encoder_plus_head_ms",
                                             "full_pipeline_ms",
                                             "amortised_ms")))
    check("PROJECT_CONTEXT sends the reader to E7b for measured latency",
          "docs/experiments/e7b_serial_efficiency.md" in context)
    check("PROJECT_CONTEXT refuses to substitute E7b values",
          "NOT replaced by E7b values" in context)
    check("PROJECT_CONTEXT keeps the valid component measurement",
          "2.2510 ms image plus 1.7309 ms text" in context)
    check("PROJECT_CONTEXT carries the bounded accuracy-and-parameter "
          "statement",
          "more accurate (0.4904 against 0.4594" in context
          and "16 times fewer parameters" in context)
    check("PROJECT_CONTEXT points to the status authorities",
          "results/ve0/supersession_map.json" in context
          and STATUS_RELATIVE in context)

    # The E7a report leads with the supersession, not with the numbers.
    e7a = (PROJECT_ROOT / "docs" / "experiments"
           / "e7a_efficiency.md").read_text()
    banner = "\n".join(e7a.splitlines()[:30])
    check("the E7a report shows the supersession within its first 30 lines",
          "SUPERSEDED" in banner.upper())
    check("the banner names the three additive fields",
          all(field in banner for field in ("gpu_encoder_plus_head_ms",
                                            "full_pipeline_ms",
                                            "amortised_ms")))
    check("the banner states what remains valid",
          "VALID" in banner.upper() and "parameter counts" in banner)
    check("the banner sends the reader to E7b for measured latency",
          "e7b_serial_efficiency.md" in banner)
    check("the banner refuses to substitute E7b values",
          "not** replaced by" in banner or "not replaced by" in banner)
    check("the E7a report points to the lifecycle record",
          STATUS_RELATIVE in e7a)
    check("the superseded columns are marked in the results table",
          "full pipeline SUPERSEDED" in e7a
          and "amortised SUPERSEDED" in e7a)


# --------------------------------------------------------------------------
# Y2. the guard is not vacuous: reintroducing the claim is caught
# --------------------------------------------------------------------------

# The exact paragraph the independent pre-F1 review found alive in
# PROJECT_CONTEXT's "Current truth" section, reassembled from parts so this
# source does not itself carry the banned sentences as prose.
_REGRESSION_CASES = (
    ("the removed caching claim",
     "The encoder dominates; caching image features across about ten "
     "questions per image " + "cuts a query from 6.35 to 2.25 ms" + "."),
    ("the removed front claim",
     "Measured under one protocol, " + "on both end-to-end latency Pareto "
     "fronts" + " only top-1000 global heads are optimal."),
    ("the additive full-pipeline comparison",
     "The top-1000 product head is cheaper (" + "6.35 against 7.71"
     + " ms) than the reasoner."),
    ("a bare latency-only cheapness claim",
     "The small global head is therefore cheaper than the reasoner at "
     "comparable accuracy."),
)


def test_e7a_guard_is_not_vacuous() -> None:
    """Prove the guard fires, on every surface it now covers.

    A regression guard that has never been shown to fail is not evidence that
    the claim is gone; it is evidence that nothing looked. Each case is
    injected into an in-memory copy of a live document. No file is written and
    no tracked document is modified.
    """
    for relative in E7A_CURRENT_FACING:
        live = (PROJECT_ROOT / relative).read_text()
        check(f"{relative} is clean before injection",
              _superseded_e7a_violations(relative, live) == [])
        for label, injected in _REGRESSION_CASES:
            mutated = live + "\n\n## Current truth\n\n" + injected + "\n"
            found = _superseded_e7a_violations(relative, mutated)
            check(f"{relative}: the guard catches {label}",
                  len(found) > len(_superseded_e7a_violations(relative, live)),
                  f"{label} survived in {relative}")

    # The quoted-prohibition exemption is real where it is meant to be, and
    # absent where it is not: the same banned sentence, presented as a
    # prohibition, is tolerated in the E7a report and refused in
    # PROJECT_CONTEXT, which publishes no supersession of its own.
    quoted = ('The sentence "' + E7A_FORBIDDEN_STATEMENTS[0]
              + '" was removed and must not return.')
    check("a quoted prohibition is tolerated in the E7a report",
          _superseded_e7a_violations(
              "docs/experiments/e7a_efficiency.md", quoted) == [])
    check("the same quotation is refused in PROJECT_CONTEXT",
          _superseded_e7a_violations(
              "collab/PROJECT_CONTEXT.md", quoted) != [])
    check("PROJECT_CONTEXT is in the no-quoted-exemption set",
          "collab/PROJECT_CONTEXT.md" in E7A_NO_QUOTED_EXEMPTION)

    # And the cheapness rule does not fire on a sentence that forbids the
    # claim, or the repaired documents themselves could never pass.
    check("a prohibition sentence is not treated as a claim",
          _superseded_e7a_violations(
              "CLAUDE.md",
              'They must not be used in any "cheaper" claim.') == [])


# --------------------------------------------------------------------------
# Z. the lifecycle builder is mechanical too
# --------------------------------------------------------------------------

def test_status_builder_is_mechanical() -> None:
    source = STATUS_BUILDER.read_text()
    tree = ast.parse(source)

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for banned in ("torch", "open_clip", "transformers", "timm"):
        check(f"the lifecycle builder does not import {banned}",
              banned not in imported, str(sorted(imported)))

    called = {node.func.attr for node in ast.walk(tree)
              if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute)}
    for banned in ("cuda", "to", "eval", "forward", "load_state_dict",
                   "from_pretrained", "synchronize"):
        check(f"the lifecycle builder never calls {banned}",
              banned not in called, str(sorted(called)))

    writes = [node for node in ast.walk(tree)
              if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute)
              and node.func.attr in ("write_json", "write_csv", "write_bytes",
                                     "write_text", "savefig", "save")]
    check("the lifecycle builder performs exactly one write",
          len(writes) == 1, str(len(writes)))
    check("that write is a deterministic JSON write",
          writes[0].func.attr == "write_json")
    check("its output path is the lifecycle record", STATUS_RELATIVE in source)

    build_fn = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "build")
    first = ast.dump(build_fn.body[0])
    check("the pin assertion is the first statement of build()",
          "_assert_pins" in first, first[:120])

    from experiments.closure import build_pre_f1_status_supersession as builder
    rebuilt = builder.build()
    check("a rebuild declares a zero-GPU provenance block",
          rebuilt["provenance"]["gpu_used"] is False)
    check("a rebuild reproduces the stored content digest",
          rebuilt["content_sha256"] == _status_record()["content_sha256"],
          rebuilt["content_sha256"])

    # Known-negative: the builder refuses to publish a supersession of a
    # statement that is no longer there.
    must_fail("the builder publishes when the superseded status has gone",
              lambda: _with_missing_superseded_status(builder))


def _with_missing_superseded_status(builder) -> None:
    original = builder.E7B_OVERLAY
    try:
        builder.E7B_OVERLAY = "results/closure/A_evidence_registry.json"
        builder._assert_superseded_status_still_present()
    finally:
        builder.E7B_OVERLAY = original


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
    test_status_record_schema_and_digest()
    test_status_record_lifecycle()
    test_e7b_lifecycle_supersession()
    test_frozen_surfaces_untouched()
    test_rq_matrix_wording_erratum()
    test_e7a_superseded_claims_are_gone()
    test_e7a_guard_is_not_vacuous()
    test_status_builder_is_mechanical()
    failed = [name for name, ok in _CHECKS if not ok]
    if failed:
        raise AssertionError(
            f"{len(failed)} E7b supersession checks failed: {failed}")
    print(f"  {len(_CHECKS)} E7b supersession checks passed")


if __name__ == "__main__":
    VERBOSE = True
    run()
