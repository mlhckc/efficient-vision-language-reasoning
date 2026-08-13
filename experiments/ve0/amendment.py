"""Output N: the VE-0 reporting and evidence-binding amendment record.

VE-0 was closed at verdict VE0_PASS on 2026-08-13. An independent review of
VE-1, which renders from that contract, then found seven blocking defects. Four
of them are defects in VE-0 itself rather than in the rendering: two research
questions whose supported answers over-generalise what the evidence carries,
and three reporting specifications that require per-seed values the inventory
did not bind, although those values already existed fully computed inside
VE-0's own frozen closure inputs.

This record exists so the amendment can never be mistaken for part of the
original freeze. It states what VE0_PASS covered, what the independent review
found, who authorised the amendment, exactly which fields moved, exactly which
frozen artefact each newly bound vector was read from, and that no scientific
value was recomputed.

It is a registered VE-0 output, so it is rebuilt deterministically with
everything else and carries the same two-tier determinism contract.
"""

from __future__ import annotations

from . import ve0_common as vc

ORIGINAL_FREEZE = {
    "verdict": "VE0_PASS",
    "approved_head": "d58cd66b0b3e7e64fd49b10b0a0342588df367fd",
    "approved_patch_sha256":
        "fbab37b5162147dc3468985dfa415583d7ffecaa13fa3f03fc05c0041e63fe01",
    "closure_record": ".agent-bridge/tasks/ve0-canonical-evidence-inventory/"
                      "60-final.json",
    "closed_at_utc": "2026-08-13",
    "review_rounds": ["VE0_CHANGES_REQUIRED", "VE0_PASS"],
    "note": "the original freeze stands as the historical record. It is not "
            "rewritten, reinterpreted or backdated by this amendment, and no "
            "historical review artefact was modified.",
}

DISCOVERY = {
    "found_by": "the independent VE-1 review, which rendered the dissertation "
                "figures and tables from this contract and reported verdict "
                "VE1_CHANGES_REQUIRED",
    "reviewed_ve1_head": "f91d0770b6733a0b3eb4f161dd2ca36d60050bbb",
    "numerical_verification_reported_by_that_review": {
        "figure_drawn_rows_re_derived": 203,
        "figure_field_comparisons": 2436,
        "strictly_numeric_table_cells": 898,
        "scientific_numerical_mismatches": 0,
    },
    "blocking_findings_that_are_ve0_defects": {
        "B1": "RQ6's supported answer over-generalised the train_40k "
              "conclusion to every tested scale. The reasoner's mean overall "
              "accuracy is higher at 100k and 250k; only the "
              "four-or-more-step deficit conclusion holds at every scale.",
        "B2": "RQ9's supported answer implied that every frozen-SLM readout "
              "has accuracy-paired serial cost evidence. On otter159 every "
              "lightweight and E8B row is latency-only, because VE-0 records "
              "its accuracy pairing as unresolved.",
        "B5": "VE0-TAB-04 requires a per-seed column and calls it mandatory, "
              "but no inventory row bound per-seed values.",
        "B6": "VE0-TAB-05's footnote states that the per-seed effects are "
              "printed so a reader can see where seeds disagree in sign, "
              "while no inventory row bound them.",
        "B7": "VE0-TAB-A2 exists to answer what the per-seed value behind "
              "every reported mean is, and could not.",
    },
    "blocking_findings_repaired_in_ve1_rather_than_here": {
        "B3": "mislabelled trainable-parameter scalar keys in the VE-1 "
              "provenance registry",
        "B4": "VE-1 source validation did not resolve canonical sources in a "
              "clean checkout",
    },
}

AUTHORISATION = {
    "authorised_by": "the user",
    "date_utc": "2026-08-13",
    "scope": "one narrow governed amendment and repair, existing only to "
             "close the seven blockers the independent VE-1 review raised and "
             "to apply the directly associated dissertation-readability "
             "improvements that review already accepted.",
    "explicitly_not_authorised": [
        "training", "evaluation", "new bootstrap or statistics", "timing",
        "checkpoint selection", "any change to a historical scientific value, "
        "confidence interval, training-seed standard deviation or sample "
        "count", "any v2_07 repair", "clean-test access", "VE-2", "F1", "F2",
        "A5 or A8c", "reopening E10", "reopening the statistical and "
        "efficiency closure",
    ],
    "permitted_reading": "the frozen source artefacts may be READ only to "
                         "bind values that were already computed before VE-0 "
                         "and whose hashes are already part of the accepted "
                         "evidence provenance.",
}

FIELDS_CHANGED = {
    "canonical_evidence_inventory": {
        "rows_added": 0,
        "rows_removed": 0,
        "row_count_before_and_after": 428,
        "new_fields": ["per_seed_values", "per_seed_seed_ids",
                       "per_seed_range", "per_seed_effects",
                       "per_seed_effect_range", "per_seed_status",
                       "per_seed_provenance"],
        "existing_fields_changed": "none. Every point estimate, interval, "
                                   "across-training-seed standard deviation, "
                                   "sample count, seed set, metric identity, "
                                   "comparison class, scorer, "
                                   "checkpoint-selection class, claim "
                                   "boundary and limitation is byte-identical "
                                   "to the VE0_PASS state.",
    },
    "rq_evidence_matrix": {
        "questions_changed": ["RQ6", "RQ9"],
        "fields_changed": ["supported_answer", "limitation"],
        "answer_status_changed": "none. Both remain ANSWERED.",
        "claim_bindings_changed": "none. RQ6 still rests on C07 and RQ9 on "
                                  "C13 and C14.",
    },
    "table_specifications": {
        "specifications_changed": ["VE0-TAB-07"],
        "fields_changed": ["placement", "dissertation_section", "footnotes"],
        "reason": "twenty-one rows of claim text, mandatory caveat and "
                  "forbidden stronger version is reference apparatus rather "
                  "than a Results table. No claim, status, caveat or "
                  "prohibition changed, and no replacement main-text table "
                  "was invented to fill the slot.",
    },
    "results_section_mapping": {
        "sections_added": ["R_APPENDIX_CLAIM_LEDGER"],
        "reason": "the claim ledger needed an appendix home once it left the "
                  "main text. The Limitations section keeps its claims and "
                  "states the boundaries in prose.",
    },
    "claim_ledger": {
        "changed": "nothing. 21 claims, identical texts, statuses, caveats "
                   "and forbidden stronger versions. C02 remains TENTATIVE "
                   "and C19 remains SUPPORTED.",
    },
    "figure_specifications": {"changed": "nothing"},
    "supersession_map": {"changed": "nothing"},
    "qualitative_protocol": {"changed": "nothing; the frozen selection salt "
                                        "is untouched"},
}


def build(ctx, rows: list) -> dict:
    """The amendment record, with the per-seed sources counted from the rows."""
    seed_bound = [r for r in rows
                  if r["evidence_class"] == "SEED"
                  and r["per_seed_status"] == "BOUND"]
    contrast_bound = [r for r in rows
                      if r["evidence_class"] == "CON"
                      and r["per_seed_status"] == "BOUND"]
    unavailable = [r for r in rows
                   if str(r["per_seed_status"]).startswith("NOT_AVAILABLE")]

    sources = {}
    for row in seed_bound + contrast_bound:
        provenance = row["per_seed_provenance"]
        sources.setdefault(provenance["read_from"], {
            "path": provenance["read_from"],
            "sha256": provenance["read_from_sha256"],
            "rows_bound_from_it": 0,
            "role": "a frozen closure output that VE-0 already declares as "
                    "an input and already hashes",
        })["rows_bound_from_it"] += 1

    return {
        "title": "VE-0 reporting and evidence-binding amendment, 2026-08-13",
        "ve0_output": "N",
        "rule": "this amendment is NOT part of the original freeze. It binds "
                "values that already existed in VE-0's own frozen inputs and "
                "repairs two over-general research-question answers. It "
                "computes no scientific value, changes no historical result "
                "and reopens nothing.",
        "original_freeze": ORIGINAL_FREEZE,
        "discovery": DISCOVERY,
        "authorisation": AUTHORISATION,
        "fields_changed": FIELDS_CHANGED,
        "per_seed_binding": {
            "seed_rows_bound": len(seed_bound),
            "seed_rows_total": sum(1 for r in rows
                                   if r["evidence_class"] == "SEED"),
            "contrast_rows_bound": len(contrast_bound),
            "contrast_rows_total": sum(1 for r in rows
                                       if r["evidence_class"] == "CON"),
            "rows_without_a_per_seed_vector": sorted(
                r["evidence_id"] for r in unavailable),
            "why_those_rows_have_none":
                "the closure's contrast artefact stores no per-seed effect "
                "for the three V3 reasoner-minus-fusion deficit differences. "
                "No rendered specification requires a per-seed effect for "
                "them, so nothing is blocked; the absence is recorded on the "
                "rows rather than filled.",
            "sources": [sources[k] for k in sorted(sources)],
            "consistency_guard": "every bound vector must reproduce the mean, "
                                 "the sample standard deviation (ddof=1) and, "
                                 "for accuracies, the range stored beside it "
                                 "in the same frozen row, and its seed "
                                 "identifiers must equal the row's own seed "
                                 "set. A vector that does not is a build "
                                 "failure, not a warning.",
            "computation": "READ_ONLY. No value was recomputed, no model was "
                           "loaded, no checkpoint or embedding store was "
                           "opened, no evaluation or timing was run and no "
                           "GPU hour was charged.",
        },
        "dependency_into_ve1": {
            "unblocks": ["VE0-FIG-A2"],
            "populates": ["VE0-TAB-04 per-seed values",
                          "VE0-TAB-05 per-seed effects",
                          "VE0-TAB-A2 per-seed accuracies and range"],
            "note": "VE-1 must be rebuilt against this amended contract. Its "
                    "validation suite refuses to render from a contract whose "
                    "content digests it cannot verify.",
        },
        "scope_confirmations": {
            "trained_anything": False,
            "evaluated_anything": False,
            "measured_anything": False,
            "recomputed_any_scientific_value": False,
            "selected_any_checkpoint": False,
            "clean_test_accessed": False,
            "gpu_used": False,
            "gpu_hours_charged": 0.0,
            "e10_reopened": False,
            "closure_reopened": False,
            "v2_07_repaired": False,
            "historical_review_artefact_rewritten": False,
        },
    }
