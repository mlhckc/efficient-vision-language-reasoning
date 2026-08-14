"""Build the consolidated pre-F1 lifecycle status and wording supersession.

    python -m experiments.closure.build_pre_f1_status_supersession

WHAT THIS IS. Five review packets — E7b, VE-0, VE-1, VE-2 and E10 — are
closed, but the documents and records that state their status were written
before their reviews returned. A reader arriving at the repository today is
told E7b "remains OPEN", VE-0 and VE-1 are "submitted for review", and E10
has "no final PASS", none of which is still true. This record states the
current lifecycle of all five in one machine-resolvable place, and supersedes
the stale status text without editing any closed artefact.

It also resolves three wording defects that survive on current-facing
surfaces: the E7a additive full_pipeline_ms comparison, which VE-0 already
recorded as superseded and whose two named sentences it forbids from
returning; and the phrase "far cheaper" inside the frozen RQ evidence matrix,
which the claim ledger already states correctly at C13 and C18.

WHAT THIS IS NOT. It is not a remeasurement, not a re-review and not an
authorisation. No accuracy, latency, interval, parameter count, Pareto
membership, figure, table or raw artefact changes. No withdrawn field is
reinstated and no withdrawn field is given a replacement value. F1 is not
declared ready here: this record states that F1 is unstarted, and readiness
is a judgement for the independent pre-F1 review and for the user, not for
the executor that wrote this file.

PRECEDENCE. This record controls LIFECYCLE and WORDING only. It does not
touch field validity: `results/closure/e7b_evidence_supersession.json`
remains the sole authority on which E7b fields are current evidence, and
`results/ve0/supersession_map.json` remains the authority on unnamed
artefact-group status. Those two records are unchanged and byte-pinned here.

Determinism follows the project's two-tier rule. content_sha256 is the digest
of the scientific content with the provenance block excluded, and it is what
a later build must reproduce; provenance carries the repository HEAD and the
worktree state, which move for reasons that are not scientific. The record
deliberately carries no hash of the documents this task edits, so rebuilding
it before or after those edits gives the same content digest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.closure import closure_common as cc  # noqa: E402

SCRIPT = "experiments/closure/build_pre_f1_status_supersession.py"
OUTPUT = "results/closure/pre_f1_status_supersession_20260814.json"

# The embargoed clean-test target. Assembled from parts and never written
# literally, so this source carries no reference to it while the guard below
# still refuses any path that names it. closure_common.write_json refuses a
# payload carrying the token as well, so the output cannot contain it either.
EMBARGOED_TOKEN = "test_" + "clean_" + "targets"

E7B_OVERLAY = "results/closure/e7b_evidence_supersession.json"
RQ_MATRIX = "results/ve0/rq_evidence_matrix.json"
CLAIM_LEDGER = "results/ve0/claim_ledger.json"
SUPERSESSION_MAP = "results/ve0/supersession_map.json"
VE0_REPORT = "docs/experiments/ve0_evidence_contract.md"

# --------------------------------------------------------------------------
# frozen pins
#
# Everything this task must not disturb: the field-level overlay it defers
# to, the closed VE trees, the frozen RQ matrix and claim ledger it cites as
# an erratum, the E10 and E7b evidence, and the three source surfaces the
# task forbids editing. The builder refuses to write if any one has moved.
# --------------------------------------------------------------------------

PINS = {
    E7B_OVERLAY:
        "2fb6561b3916b83603631c0feb35d716dc9bbf3e1ed1ef3d4e7286516a0a0c75",
    SUPERSESSION_MAP:
        "e1163023509920dddf182989341a2517057941bffc813cc645e6b65928e66f5c",
    RQ_MATRIX:
        "edad2d62a3db4b4ebae5fae9108889c7156dd0cb4ab585b64f325b37654d532d",
    CLAIM_LEDGER:
        "ab1a1a52f246832ef48e266cbac6db82453b8f334035f5a4234fba4ae073a5b7",
    "results/ve0/canonical_evidence_inventory.json":
        "a1f49a0f99ddf4cffac591fcf24b3cf7ba9638de349898fd4a525aed96ae66d3",
    "results/ve0/figure_specifications.json":
        "7d1c1d4588e97ded196a208e32021b9f21cbfa18aa9a9d3a031e11572eb2486c",
    "results/ve0/table_specifications.json":
        "98f0e756d7817ba2a78444e2ecadfd021ffda129752604e6f4a7bed7c3b4a3f8",
    "results/ve0/VE0_MANIFEST.json":
        "294e11d48ade7316908f8cc6dc37ab9b8f8cc2cc6d454b3df4524fa8ecf6c7f3",
    "results/ve1/VE1_MANIFEST.json":
        "021592025cbfcdeb145441799d7b5c6ff341a0945289f8927230a89e16eb986a",
    "results/ve2/VE2_MANIFEST.json":
        "f54218e9666b8d2574ba87ccd7868874d4ef35de7560b401e1dce91f3210c386",
    VE0_REPORT:
        "326bbc7c259aea808478e38358b3dc0eab12e1b61e03501927b9c2d4fc591188",
    "results/closure/efficiency_closure_table.json":
        "be1781efeece841cc75df126fab43c9e9946437957c1ea6401ee1f86c8b47d5e",
    "results/closure/A_evidence_registry.json":
        "b8b47e024c49a599a425894f34c24b4375958998401a311605694027257de122",
    "results/experiments/e7b_serial_efficiency/e7b_results.json":
        "c55d61600124ed76a6383f87d6f0a8744f4d51eec24b2f53ef8aa9edeb76bae4",
    "results/experiments/e10_capacity_360m_core/e10_core_analysis.json":
        "37ffdcd09aa2bcd306ca9b54229a1d0d17ff203b6b0900e09e5f876ad1ed0748",
    "tests/run_all.py":
        "0a53250ac3c10edd788ea756a64dff54e83d2dec824ad557a6f8ea629e741786",
    "tests/test_e7b.py":
        "ac95a923a13241ef5a6e2d828f89be8ca8b8a41e472722662e08306027231595",
    "experiments/e7b_serial_efficiency/run.py":
        "fd62ccb2cd7532be154d19cd7c0ab598215940340f7295b29ca8d5ca54495455",
    ".gitignore":
        "9ec096d242e61dd67650f619680d7a0d34115872eca06475dd701ce0e87ed7d1",
}

# --------------------------------------------------------------------------
# lifecycle
#
# One entry per closed packet: the accepted verdict, where the acceptance is
# recorded, and which stale statement this record supersedes.
# --------------------------------------------------------------------------

LIFECYCLE = {
    "e7b": {
        "packet": "E7b canonical evidence supersession",
        "status": "CLOSED",
        "accepted_verdict": "E7B_CANONICAL_SUPERSESSION_REVIEW_PASS",
        "review": "independent, two rounds, approved",
        "closed_on": "2026-08-14",
        "field_validity_authority": E7B_OVERLAY,
        "superseded_statements": [
            "docs/experiments/e7b_serial_efficiency.md: 'E7b remains OPEN "
            "pending independent review'",
            "README.md: 'E7b remains open pending independent review'",
            "CLAUDE.md: 'E7b remains OPEN pending independent review'",
            f"{E7B_OVERLAY}: status.e7b = OPEN and status.f1 = BLOCKED",
        ],
    },
    "ve0": {
        "packet": "VE-0 canonical evidence contract",
        "status": "CLOSED",
        "accepted_verdict": "VE0_PASS",
        "amendment_verdict": "VE0_AMENDMENT_PASS",
        "review": ("independent; the contract closed at VE0_PASS and the "
                   "13 August 2026 amendment closed at VE0_AMENDMENT_PASS in "
                   "the same packet that closed VE-1"),
        "closed_on": "2026-08-13",
        "superseded_statements": [
            f"{VE0_REPORT}: 'submitted for independent re-review at packet "
            "state review_requested. It authorises nothing. VE-1, VE-2, F1 "
            "and F2 have not been started'",
        ],
        "report_is_byte_pinned": True,
    },
    "ve1": {
        "packet": "VE-1 final quantitative figures and tables",
        "status": "CLOSED",
        "accepted_verdict": "VE1_PASS",
        "review": ("independent; the first round returned "
                   "VE1_CHANGES_REQUIRED with seven blocking findings, and "
                   "the authorised amendment and repair closed at VE1_PASS"),
        "closed_on": "2026-08-13",
        "superseded_statements": [
            "docs/experiments/ve1_figures_and_tables.md: 'submitted for "
            "independent review at packet state review_requested ... VE-2, "
            "F1 and F2 have not been started'",
        ],
        "report_is_byte_pinned": False,
    },
    "ve2": {
        "packet": "VE-2 deterministic qualitative development evidence",
        "status": "CLOSED",
        "accepted_verdict": "VE2_PASS",
        "review": "independent, two rounds, approved",
        "closed_on": "2026-08-14",
        "superseded_statements": [
            "docs/experiments/ve2_qualitative_evidence.md carried no status "
            "section at all, so a reader could not tell the packet was "
            "closed",
        ],
        "report_is_byte_pinned": False,
    },
    "e10": {
        "packet": "E10 SmolLM2-360M answer-readout capacity matrix",
        "status": "CLOSED",
        "accepted_verdict": "E10_PASS",
        "review": ("independent; recorded as closed by the frozen VE-0 "
                   "supersession map, which states that E10 is CLOSED at "
                   "E10_PASS and must not be reopened"),
        "recorded_in": SUPERSESSION_MAP,
        "superseded_statements": [
            "CLAUDE.md: 'R3.1 is review_requested and no final E10 PASS is "
            "declared'",
        ],
        "grant_state": "operationally SPENT",
        "cells_immutable": True,
    },
}

# --------------------------------------------------------------------------
# the E7a wording repair
#
# VE-0 already recorded these as superseded. What this task repairs is that
# the superseded numbers were still being used as current end-to-end evidence
# on documents a reader opens first.
# --------------------------------------------------------------------------

E7A_FORBIDDEN = (
    "cuts a query from 6.35 to 2.25 ms",
    "on both end-to-end latency Pareto fronts",
)

E7A_BOUNDED_WORDING = (
    "vocab1000_product@250k is more accurate (0.4904 against 0.4594) than "
    "the 21.1M-parameter reasoner, using 16 times fewer parameters. Its "
    "additive full-pipeline cost figures are superseded; for measured "
    "end-to-end latency see E7b.")

# --------------------------------------------------------------------------
# clean-test governance
#
# The same two documented classification-B incidents the E7b overlay
# records. Restated here because CLAUDE.md, which had no governance section,
# now carries the summary and a reader must be able to resolve it.
# --------------------------------------------------------------------------

_NEITHER = {
    "classification": "B — MECHANICAL ACCESS INCIDENT",
    "contents_inspected": False,
    "rows_labels_distributions_or_predictions_examined": False,
    "informed_development": False,
    "informed_model_selection": False,
    "informed_reporting_decisions": False,
}

INCIDENTS = (
    dict(_NEITHER,
         incident_id="A",
         date="2026-08-13",
         actor="an independent reviewer",
         mechanism=("an integrity-hash command over the target file, which "
                    "read its bytes and moved its atime"),
         nature="mechanical byte access"),
    dict(_NEITHER,
         incident_id="B",
         date="2026-08-14",
         actor=("the executor of the E7b canonical-supersession "
                "implementation"),
         mechanism=("an overly broad dependency-mapping scan that walked the "
                    "whole project root and read every .json, .py, .md, .csv "
                    "and .txt file, searching for four documentation "
                    "digests; the traversal was not scoped away from data/"),
         nature="mechanical byte access"),
)

CORRECTIVE_RULE = (
    "A repository-wide scan must exclude data/ before reading or traversing "
    "candidate files, not filter it afterward. Scoping the traversal is the "
    "control; filtering the results is not, because by then the bytes have "
    "already been read.")

# --------------------------------------------------------------------------
# the verification runners
#
# tests/run_all.py is inside E10's frozen SOURCE_PATHS, so no closed packet's
# suite may be imported into it. Each final-evidence layer therefore ships
# its own runner, and a reviewer needs the list.
# --------------------------------------------------------------------------

RUNNERS = (
    {"command": "python -B tests/run_closure.py",
     "covers": "the statistical and efficiency evidence closure"},
    {"command": "python -B tests/run_ve0.py",
     "covers": "the VE-0 canonical evidence contract"},
    {"command": "python -B tests/run_ve1.py",
     "covers": "the VE-1 figures and tables"},
    {"command": "python -B tests/run_ve2.py",
     "covers": "the VE-2 qualitative evidence"},
    {"command": "python -B tests/run_e7b_supersession.py",
     "covers": ("the E7b field-level withdrawal overlay and this "
                "consolidated lifecycle record")},
)

CHANGED_PATHS = (
    "CLAUDE.md",
    "README.md",
    "docs/REPRODUCIBILITY.md",
    "docs/experiments/e7a_efficiency.md",
    "docs/experiments/e7b_serial_efficiency.md",
    "docs/experiments/ve1_figures_and_tables.md",
    "docs/experiments/ve2_qualitative_evidence.md",
    "tests/test_e7b_supersession.py",
)

NEW_PATHS = (OUTPUT, SCRIPT)


def _assert_pins() -> list:
    """Fail closed unless every pinned artefact still has its frozen hash."""
    records, moved, missing = [], [], []
    for relative in sorted(PINS):
        path = PROJECT_ROOT / cc.assert_not_embargoed(relative)
        if not path.exists():
            missing.append(relative)
            continue
        actual = cc.sha256_file(path)
        records.append({"path": relative, "pinned_sha256": PINS[relative],
                        "actual_sha256": actual,
                        "unchanged": actual == PINS[relative]})
        if actual != PINS[relative]:
            moved.append(relative)
    if missing:
        raise AssertionError(f"pinned artefact missing: {missing}")
    if moved:
        raise AssertionError(
            "pinned artefact moved; refusing to publish a lifecycle record "
            f"that names artefacts by hashes they no longer have: {moved}")
    return records


def _assert_superseded_status_still_present() -> dict:
    """Fail closed unless the stale status this record supersedes is real.

    A supersession that names a statement no longer present is not evidence
    of anything, and would quietly become a lie the moment somebody edited
    the frozen artefact. Both are read from the pinned bytes.
    """
    overlay = json.loads((PROJECT_ROOT / E7B_OVERLAY).read_text())
    status = overlay["status"]
    if status.get("e7b") != "OPEN" or status.get("f1") != "BLOCKED":
        raise AssertionError(
            "the E7b overlay no longer carries the lifecycle status this "
            f"record supersedes: {status}")
    report = (PROJECT_ROOT / VE0_REPORT).read_text()
    if "review_requested" not in report:
        raise AssertionError(
            "the pinned VE-0 report no longer carries the stale status "
            "section this record supersedes")
    return {
        "e7b_overlay_status_block": dict(status),
        "ve0_report_states_review_requested": True,
        "method": ("read from the pinned bytes at build time; the builder "
                   "refuses to publish if either statement has gone"),
    }


def _rq_wording_erratum() -> dict:
    """Locate the frozen 'far cheaper' wording and the correct form."""
    matrix = json.loads((PROJECT_ROOT / RQ_MATRIX).read_text())
    offending = [row["rq_id"] for row in matrix["questions"]
                 if "far cheaper" in row.get("supported_answer", "")]
    if not offending:
        raise AssertionError(
            "the frozen RQ matrix no longer carries the wording this "
            "erratum supersedes")
    ledger = json.loads((PROJECT_ROOT / CLAIM_LEDGER).read_text())
    correct = {row["claim_id"]: row["claim_text"] for row in ledger["claims"]
               if row["claim_id"] in ("C13", "C18")}
    phrase = "at a small fraction of the measured end-to-end serial latency"
    if phrase not in correct.get("C13", ""):
        raise AssertionError(
            "the claim ledger no longer carries the C13 wording this erratum "
            "defers to")
    # C18 states the same bound in the thesis-level form. Both are latency
    # statements and neither says "cheaper"; the two phrasings are recorded
    # separately rather than collapsed, because they are not the same string.
    c18_phrase = "at a small fraction of the measured per-query latency"
    if c18_phrase not in correct.get("C18", ""):
        raise AssertionError(
            "the claim ledger no longer carries the C18 wording this erratum "
            "defers to")
    for claim_id, text in correct.items():
        if "cheaper" in text:
            raise AssertionError(
                f"the claim ledger's {claim_id} now carries a cheapness "
                "claim, which this erratum defers to as the correct form")
    return {
        "artefact": RQ_MATRIX,
        "artefact_edited": False,
        "artefact_immutable": True,
        "why_immutable": "byte-pinned by the closed VE-0 packet.",
        "affected_questions": sorted(offending),
        "offending_wording": "far cheaper",
        "final_status": "WORDING_SUPERSEDED",
        "why": (
            "'far cheaper' reads as a cost claim. What was measured is warm "
            "serial batch-1 latency on one node. No energy, power, carbon, "
            "monetary or general computational-cost quantity was measured "
            "anywhere in this project, so the only defensible form of the "
            "statement is a latency statement."),
        "superseding_wording": phrase,
        "superseding_wording_by_claim": {"C13": phrase, "C18": c18_phrase},
        "superseding_authority": CLAIM_LEDGER,
        "superseding_claims": sorted(correct),
        "already_correct_in_the_claim_ledger": True,
        "neither_claim_says_cheaper": True,
        "rq_answer_status_changed": False,
        "evidence_ids_changed": False,
        "numeric_results_changed": False,
        "scope": (
            "wording only. RQ9 keeps its answer status of ANSWERED, its "
            "evidence set and every number it rests on."),
    }


def _content_digest(payload: dict) -> str:
    """Digest of the scientific content, provenance excluded."""
    stripped = {k: v for k, v in payload.items()
                if k not in ("provenance", "content_sha256")}
    text = json.dumps(stripped, indent=2, sort_keys=True,
                      ensure_ascii=False) + "\n"
    return cc.sha256_bytes(text.encode("utf-8"))


def build() -> dict:
    pin_records = _assert_pins()
    superseded = _assert_superseded_status_still_present()

    payload = {
        "title": "Consolidated pre-F1 lifecycle status and wording "
                 "supersession",
        "record_type": "PRE_F1_STATUS_SUPERSESSION",
        "schema_version": 1,
        "scope": "LIFECYCLE_AND_WORDING_ONLY",
        "changes_no_scientific_result": True,
        "record_date": "2026-08-14",

        "purpose": (
            "State, in one machine-resolvable place, the current lifecycle "
            "of the five closed review packets — E7b, VE-0, VE-1, VE-2 and "
            "E10 — and supersede the stale status text that survives on "
            "documents a reader opens first. It also records two wording "
            "errata that no longer state what was measured: E7a's additive "
            "full-pipeline comparison, and the phrase 'far cheaper' inside "
            "the frozen RQ evidence matrix."),

        "what_this_record_does_not_do": {
            "changes_no_accuracy": True,
            "changes_no_latency": True,
            "changes_no_parameter_count": True,
            "changes_no_interval": True,
            "changes_no_pareto_membership": True,
            "changes_no_figure_or_table": True,
            "changes_no_raw_artefact": True,
            "reinstates_no_withdrawn_field": True,
            "introduces_no_replacement_value": True,
            "remeasures_nothing": True,
            "reopens_no_closed_packet": True,
            "authorises_nothing": True,
            "declares_f1_ready": False,
            "statement": (
                "This record is lifecycle and wording only. It resolves who "
                "says what about status, and it removes superseded numbers "
                "from current-facing prose. It creates no evidence, and it "
                "does not judge whether F1 may begin: that is the "
                "independent pre-F1 review's finding and the user's "
                "decision."),
        },

        # ------------------------------------------------------------------
        # precedence
        # ------------------------------------------------------------------
        "precedence": {
            "rule": (
                "Lifecycle status and field validity are separate questions "
                "and are resolved by separate records. For LIFECYCLE and "
                "STATUS, this record is highest precedence. For FIELD "
                "VALIDITY, it defers entirely: rank 1 is "
                "results/closure/e7b_evidence_supersession.json for named "
                "E7b fields, rank 2 is results/ve0/supersession_map.json for "
                "unnamed artefact-group status, and historical stored "
                "evidence_status is lowest."),
            "lifecycle_order": [
                {"rank": 1,
                 "controls": "packet lifecycle and review status",
                 "authority": OUTPUT},
                {"rank": 2,
                 "controls": "packet lifecycle recorded inside a closed "
                             "packet's own artefacts",
                 "authority": "the closed packet's stored status block"},
            ],
            "field_validity_order": [
                {"rank": 1,
                 "controls": "named E7b field validity",
                 "authority": E7B_OVERLAY},
                {"rank": 2,
                 "controls": "unnamed artefact-group status",
                 "authority": SUPERSESSION_MAP},
                {"rank": 3,
                 "controls": "nothing that ranks 1 or 2 already controls",
                 "authority": "historical stored evidence_status"},
            ],
            "this_record_controls_no_field_validity": True,
            "historical_numbers": (
                "Historical numbers remain readable only as provenance."),
        },

        # ------------------------------------------------------------------
        # lifecycle
        # ------------------------------------------------------------------
        "lifecycle_status": LIFECYCLE,
        "closed_packet_count": len(LIFECYCLE),
        "closed_packets": sorted(LIFECYCLE),
        "all_named_packets_closed": True,

        # ------------------------------------------------------------------
        # E7b: lifecycle superseded, field validity untouched
        # ------------------------------------------------------------------
        "e7b_lifecycle_supersession": {
            "supersedes": E7B_OVERLAY,
            "supersedes_scope": "the status block only",
            "superseded_fields": ["status.e7b", "status.f1",
                                  "status.e7b_closed_by_this_record",
                                  "status.note"],
            "historical_values": superseded["e7b_overlay_status_block"],
            "historical_values_are_lifecycle_state_only": True,
            "status_e7b_open_is_historical_lifecycle_state_only": True,
            "status_f1_blocked_is_historical_lifecycle_state_only": True,
            "what_happened_since": (
                "E7b was subsequently reviewed independently, accepted and "
                "formally CLOSED at E7B_CANONICAL_SUPERSESSION_REVIEW_PASS. "
                "The overlay's status block was written before that review "
                "returned and was accurate when written."),
            "older_record_remains_authoritative_for": [
                "withdrawn_fields",
                "retained_fields",
                "field-level precedence",
                "metric validity",
            ],
            "older_record_edited": False,
            "older_record_regenerated": False,
            "older_record_sha256": PINS[E7B_OVERLAY],
            "older_record_byte_identical": True,
            "why_not_edited": (
                "Editing it would move a record the independent review "
                "accepted by its bytes, and would reopen a closed packet to "
                "correct a sentence that was true when it was written. The "
                "supersession is published as a separate record for the same "
                "reason the field-level withdrawal was."),
            "withdrawn_fields_unchanged": True,
            "retained_fields_unchanged": True,
            "pareto_membership_unchanged": True,
        },

        # ------------------------------------------------------------------
        # VE-0: the report is pinned and cannot carry its own correction
        # ------------------------------------------------------------------
        "ve0_pinned_status_treatment": {
            "report": VE0_REPORT,
            "report_edited": False,
            "report_byte_pinned_by": "results/ve0/VE0_MANIFEST.json",
            "report_sha256": PINS[VE0_REPORT],
            "stale_section": "## Status",
            "stale_text_summary": (
                "It says VE-0 is 'submitted for independent re-review at "
                "packet state review_requested' and that 'VE-1, VE-2, F1 and "
                "F2 have not been started'. VE-0 has since closed at "
                "VE0_PASS with its amendment accepted at "
                "VE0_AMENDMENT_PASS, and VE-1 and VE-2 have both run and "
                "closed."),
            "resolution": (
                "This record supersedes that section's lifecycle claims. "
                "VE-0 is CLOSED. The pinned report is not edited and remains "
                "readable as the state of the packet on the day it was "
                "submitted."),
            "still_true_in_the_stale_section": [
                "VE-0 authorises nothing",
                "F1 and F2 have not been started",
                "F1 and F2 remain unauthorised",
            ],
        },

        # ------------------------------------------------------------------
        # VE-1 and VE-2: unpinned reports, corrected in place
        # ------------------------------------------------------------------
        "ve1_ve2_status_edits": {
            "ve1_report": "docs/experiments/ve1_figures_and_tables.md",
            "ve1_report_edited": True,
            "ve1_change": (
                "the Status section now states VE-1 is CLOSED at VE1_PASS "
                "and no longer says VE-2 has not been started; it points "
                "here."),
            "ve2_report": "docs/experiments/ve2_qualitative_evidence.md",
            "ve2_report_edited": True,
            "ve2_change": (
                "a Status section was added, stating VE-2 is CLOSED at "
                "VE2_PASS and pointing here. The report had no status "
                "section at all."),
            "results_ve1_modified": False,
            "results_ve2_modified": False,
            "figures_or_tables_modified": False,
            "scientific_content_modified": False,
        },

        # ------------------------------------------------------------------
        # E10
        # ------------------------------------------------------------------
        "e10_status_correction": {
            "stale_text": ("CLAUDE.md: 'R3.1 is review_requested and no "
                           "final E10 PASS is declared'"),
            "current_status": "CLOSED",
            "accepted_verdict": "E10_PASS",
            "established_by": SUPERSESSION_MAP,
            "established_by_sha256": PINS[SUPERSESSION_MAP],
            "quoted_reason": (
                "E10 is CLOSED at E10_PASS and must not be reopened. Its "
                "grant is operationally SPENT, its completed cells are "
                "immutable and its live source digest is unchanged."),
            "e10_source_modified": False,
            "e10_result_modified": False,
            "e10_test_modified": False,
            "e10_config_modified": False,
            "twelve_cell_matrix_unchanged": True,
        },

        # ------------------------------------------------------------------
        # E7a
        # ------------------------------------------------------------------
        "e7a_wording_repair": {
            "artefact_group": "E7a component efficiency measurements",
            "artefact_group_status": "PARTLY_SUPERSEDED",
            "status_authority": SUPERSESSION_MAP,
            "invalid_fields": ["gpu_encoder_plus_head_ms", "full_pipeline_ms",
                               "amortised_ms",
                               "every Pareto front derived from those "
                               "additive sums"],
            "valid_fields": ["isolated encoder component latencies",
                             "isolated head component latencies",
                             "peak memory components",
                             "parameter counts"],
            "forbidden_statements_that_must_not_return":
                list(E7A_FORBIDDEN),
            "invalid_current_comparison": {
                "values_ms": [6.35, 7.71],
                "field": "full_pipeline_ms",
                "why": (
                    "Both are additive component estimates: sums of stage "
                    "medians each measured in isolation. No serially "
                    "executed decode-encode-head pass was timed in E7a, so "
                    "the difference between them is a difference between two "
                    "estimates, not a measured latency difference. E7b "
                    "measured the serial pass and found the CLIP-global "
                    "medians about 1.3 ms above the additive figures, which "
                    "is exactly the error an additive estimate cannot see."),
                "removed_from_current_facing_surfaces": True,
            },
            "replacement_policy": {
                "replaced_with_e7b_values": False,
                "new_cross_experiment_comparison_manufactured": False,
                "underlying_numbers_altered": False,
                "statement": (
                    "The superseded figures are not replaced by E7b values. "
                    "E7a and E7b are different measurements and substituting "
                    "one into the other's sentence would manufacture a "
                    "comparison neither experiment made. The latency half of "
                    "the claim is dropped and the reader is sent to E7b for "
                    "measured end-to-end latency."),
            },
            "bounded_wording": E7A_BOUNDED_WORDING,
            "bounded_wording_retains": [
                "raw-distribution accuracy 0.4904 against 0.4594",
                "the 16-times parameter ratio",
            ],
            "bounded_wording_drops": [
                "the additive full-pipeline latency comparison",
                "any latency-only claim of being 'cheaper'",
            ],
            "applies_to": [
                "docs/experiments/e7a_efficiency.md",
                "CLAUDE.md",
            ],
            "e7a_artefact_edited": False,
            "e7a_numbers_altered": False,
        },

        # ------------------------------------------------------------------
        # the frozen RQ matrix erratum
        # ------------------------------------------------------------------
        "frozen_rq_matrix_wording_erratum": _rq_wording_erratum(),

        # ------------------------------------------------------------------
        # clean-test governance
        # ------------------------------------------------------------------
        "clean_test_governance": {
            "canonical_record": E7B_OVERLAY,
            "restated_here_because": (
                "CLAUDE.md carried no clean-test governance section, so a "
                "reader of the project's own instruction file could not "
                "resolve the two incidents at all. The summary added there "
                "is the same two-incident record, and this block is what it "
                "resolves against."),
            "incident_count": 2,
            "incidents": list(INCIDENTS),
            "common_to_both_incidents": {
                "classification": "B — MECHANICAL ACCESS INCIDENT",
                "contents_inspected": False,
                "rows_labels_distributions_or_predictions_examined": False,
                "informed_development": False,
                "informed_model_selection": False,
                "informed_reporting_decisions": False,
                "statement": (
                    "These are governance incidents, not test-informed "
                    "scientific selection. In both cases a process read the "
                    "file's bytes and nothing else: no row, label, "
                    "distribution or prediction was examined, and no "
                    "information from the target entered development, model "
                    "selection or any reporting decision."),
            },
            "corrective_rule": CORRECTIVE_RULE,
            "never_accessed_claim_permitted": False,
            "why_the_never_accessed_claim_is_refused": (
                "It would be false. The bytes were read twice. The record "
                "states what is true — that nothing from the target reached "
                "any model, checkpoint, selection, figure, table or reported "
                "number — rather than a stronger claim that is not."),
            "this_task_touched_the_target": False,
            "this_task_scan_policy": (
                "Repository scans in this task operate on tracked Git "
                "surfaces or exclude data/ before traversal. The embargoed "
                "target was not read, opened, hashed, stat-ed, globbed, "
                "searched, parsed or traversed, and the firewall was not "
                "'verified' by touching it."),
            "embargo_intact": True,
        },

        # ------------------------------------------------------------------
        # F1 and F2
        # ------------------------------------------------------------------
        "f1_f2": {
            "f1": "UNSTARTED",
            "f2": "UNSTARTED_AND_UNAUTHORISED",
            "f1_declared_ready_by_this_record": False,
            "f1_authorised": False,
            "clean_test_embargo": "UNCHANGED",
            "statement": (
                "E7b's closure removes the specific blocker its own status "
                "block recorded. It does not start F1, does not authorise "
                "F1 and does not assert that F1 is ready. The model-list "
                "freeze requires an explicit user decision that has not been "
                "given, and this record is itself submitted for independent "
                "pre-F1 review."),
        },

        # ------------------------------------------------------------------
        # what this task changed
        # ------------------------------------------------------------------
        "task_footprint": {
            "changed_tracked_paths": list(CHANGED_PATHS),
            "new_tracked_paths": list(NEW_PATHS),
            "frozen_trees_untouched": ["results/ve0/", "results/ve1/",
                                       "results/ve2/",
                                       "results/experiments/"],
            "only_test_file_modified": "tests/test_e7b_supersession.py",
            "test_modification_kind": (
                "POST_CLOSURE_LIFECYCLE_DISCOVERABILITY_EXTENSION"),
            "test_modification_reopens_e7b_science": False,
            "test_modification_requires_independent_pre_f1_review": True,
            "why_the_test_review_is_required": (
                "The accepted E7b supersession suite is being extended after "
                "E7b closed. The extension is lifecycle and discoverability "
                "only and asserts no scientific quantity, but a suite an "
                "independent review accepted should not grow without one."),
            "gpu_hours_charged": 0.0,
            "trained_anything": False,
            "evaluated_anything": False,
            "measured_anything": False,
            "selected_any_checkpoint": False,
        },

        "verification_runners": {
            "run_all_may_not_be_used_for_this": (
                "tests/run_all.py imports tests/test_reproduction.py, which "
                "performs a CUDA forward pass. This task is zero-GPU, so it "
                "was not run. run_all.py is also inside E10's frozen "
                "SOURCE_PATHS and is byte-identical."),
            "runners": list(RUNNERS),
            "runner_count": len(RUNNERS),
            "why_separate": (
                "tests/run_all.py is listed in E10's frozen SOURCE_PATHS, so "
                "importing a new suite into it moves the live source digest "
                "and the E10 phase verifiers correctly refuse the stale "
                "state. Each final-evidence layer therefore ships its own "
                "runner. New files under tests/ are safe, because "
                "SOURCE_PATHS names individual files."),
        },

        "frozen_pins": {
            "pin_count": len(PINS),
            "all_unchanged": True,
            "policy": ("fail closed: the builder refuses to write if any "
                       "pinned artefact has moved"),
            "records": sorted(pin_records, key=lambda row: row["path"]),
        },

        "superseded_statement_verification": superseded,

        "determinism": {
            "content_sha256": ("the digest of this record with the "
                               "provenance block and the digest field itself "
                               "excluded; it is what a later build must "
                               "reproduce"),
            "file_bytes": ("may differ across commits, because provenance "
                           "carries the repository HEAD and worktree state"),
            "no_hash_of_an_edited_document_is_stored": (
                "The record carries no digest of the documents this task "
                "edits, so rebuilding it before or after those edits gives "
                "the same content digest."),
        },
    }

    prov = cc.provenance(
        SCRIPT,
        inputs=[cc.record_source(PROJECT_ROOT / relative)
                for relative in (E7B_OVERLAY, SUPERSESSION_MAP, RQ_MATRIX,
                                 CLAIM_LEDGER, VE0_REPORT)],
        extra={
            "measured_anything": False,
            "loaded_any_model": False,
            "evaluated_anything": False,
            "remeasured_anything": False,
            "reopened_any_closed_packet": False,
            "embargoed_token_guard": (
                "assembled from parts and asserted absent from the payload"),
        })
    # The shared closure provenance helper stamps a project-global
    # clean_test_accessed flag. This record must not carry one: at project
    # scope the claim would be false, because the target's bytes were read in
    # the two documented governance incidents above. What is true of THIS
    # task is stated in clean_test_governance and is scoped to it.
    prov.pop("clean_test_accessed", None)
    prov["clean_test_governance_pointer"] = (
        "see clean_test_governance; this record makes no project-global "
        "clean-test claim, and this task did not touch the target")
    payload["provenance"] = prov
    payload["content_sha256"] = _content_digest(payload)
    return payload


def main() -> int:
    payload = build()
    if EMBARGOED_TOKEN in json.dumps(payload):
        raise AssertionError(
            "the record payload names the embargoed clean-test target")
    digest = cc.write_json(payload, PROJECT_ROOT / OUTPUT)
    print(f"written {OUTPUT}")
    print(f"  pins verified  : {len(PINS)}/{len(PINS)} unchanged")
    print(f"  packets closed : {payload['closed_packet_count']} "
          f"{payload['closed_packets']}")
    print(f"  scope          : {payload['scope']}")
    print(f"  content_sha256 : {payload['content_sha256']}")
    print(f"  file sha256    : {digest}")
    print("  gpu hours      : 0.0, nothing measured, nothing loaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
