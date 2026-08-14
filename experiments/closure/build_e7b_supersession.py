"""Build the canonical E7b evidence supersession overlay.

    python -m experiments.closure.build_e7b_supersession

WHAT THIS IS. E7b's warm serial latency, its accuracy column and its
parameter counts are sound. Three of its stored fields are not, and the
artefacts carrying them are closed: e7b_results.json is byte-pinned by the
VE-0 and VE-1 packets, and efficiency_closure_table.json and
A_evidence_registry.json are byte-pinned by VE-0. Amending any of them in
place reopens a closed packet, which was reproduced by experiment and
refused. So the withdrawal is published as a separate FIELD-LEVEL overlay
that takes precedence over the historical row-level evidence_status,
instead of as an edit to the rows themselves.

WHAT THIS IS NOT. It is not a remeasurement and it does not repair a
number. The historical E7b node was otter155 and the current project copy
runs on otter159; the E9 fusion bridge control between those nodes failed
its pre-registered 10 per cent tolerance at -18.69 per cent, so no
cross-node substitution and no adjustment factor is permitted. No
replacement peak-memory or cold-query value exists, and none is estimated,
inferred or back-calculated here.

WHAT IT DOES. It asserts the frozen SHA-256 identity of every pinned
historical artefact, fails closed if any one differs, and then writes
exactly one new file. It performs no GPU work, loads no model, runs no
evaluation, takes no measurement and selects no checkpoint. The three
withdrawn fields have two DIFFERENT causes and the record keeps them
apart: the peak-memory withdrawal is a measurement-boundary defect, the
cold-query withdrawal is the grad-mode standardisation the accepted source
repair introduced. Conflating them would misstate both.

Determinism follows the project's two-tier rule. content_sha256 is the
digest of the scientific content with the provenance block excluded, and
it is what a later build must reproduce; provenance carries the repository
HEAD and the worktree state, which move for reasons that are not
scientific, so the file bytes may legitimately differ across commits while
the content digest does not.
"""

from __future__ import annotations

import json
import sys
from fnmatch import fnmatch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.closure import closure_common as cc  # noqa: E402

SCRIPT = "experiments/closure/build_e7b_supersession.py"
OUTPUT = "results/closure/e7b_evidence_supersession.json"

# The embargoed clean-test target. Assembled from parts and never written
# literally, so this source carries no reference to it while the guard below
# still refuses any path that names it. closure_common.write_json refuses a
# payload carrying the token as well, so the output cannot contain it either.
EMBARGOED_TOKEN = "test_" + "clean_" + "targets"

# --------------------------------------------------------------------------
# clean-test governance
#
# TWO classification-B mechanical-access incidents are documented. The older
# single-incident sentence was accurate when written and is now incomplete as
# a history; it survives only where a closed packet froze it, and those
# residuals are named as such in the record below.
# --------------------------------------------------------------------------

DISCLOSURE = (
    "The clean-test contents were never inspected or used for development, "
    "model selection, or reporting decisions. Mechanical byte access occurred "
    "in two documented governance incidents, on 13 and 14 August 2026.")

SUPERSEDED_DISCLOSURE = (
    "The clean-test contents were never inspected or used for development, "
    "model selection, or reporting decisions. Its bytes were mechanically "
    "read once by an independent reviewer integrity-hash command on "
    "13 August 2026.")

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

# The eight timed E7b systems. Every withdrawn field applies to all eight.
SYSTEMS = ("concat", "e8a_a0p", "e8a_a1", "fusion", "question_only",
           "reasoner", "siglip_fusion", "vocab1000_product")

# --------------------------------------------------------------------------
# frozen pins
#
# Every artefact that carries a withdrawn field, governs its status, or is a
# closed packet this overlay must not disturb. The builder refuses to write
# if any one of them has moved: an overlay that names an artefact by a hash
# it no longer has is not evidence of anything.
# --------------------------------------------------------------------------

PINS = {
    "results/experiments/e7b_serial_efficiency/audit_verdicts.json":
        "d8df78208a4a542d9446c266139062cfee365452d056d6c05ea06c54ec9c2572",
    "results/experiments/e7b_serial_efficiency/benchmark_rows.json":
        "ca370a7dc0885fa69df2b1fa6076ca2f647d8a7047e4c1d9fbcb30d40e5621a5",
    "results/experiments/e7b_serial_efficiency/cache_hashes_after.txt":
        "da3ce51b6f5cc45e18a0991834d68ec201fea91890e6b580aa7234bcec2e0750",
    "results/experiments/e7b_serial_efficiency/cache_hashes_before.txt":
        "da3ce51b6f5cc45e18a0991834d68ec201fea91890e6b580aa7234bcec2e0750",
    "results/experiments/e7b_serial_efficiency/e7b_results.json":
        "c55d61600124ed76a6383f87d6f0a8744f4d51eec24b2f53ef8aa9edeb76bae4",
    "results/experiments/e7b_serial_efficiency/e7b_run_concat_pass1.json":
        "d10d7ccf094718e3f9946b74e9340ce28ea66c55683a0d616cfbee99a8a18f69",
    "results/experiments/e7b_serial_efficiency/e7b_run_concat_pass2.json":
        "1def4b8eaf67cff56a411ddd4bff9d910dc1d6e12cb202cffac89fe3feb19d6d",
    "results/experiments/e7b_serial_efficiency/e7b_run_concat_pass3.json":
        "36bf4f211cfe8bb7f46cd7cca8b471bc17bad4565cc52c5210f293844162a1fb",
    "results/experiments/e7b_serial_efficiency/e7b_run_e8a_a0p_pass1.json":
        "d1e3ebee30561e00e7e56d2546b960d90266064ee27b4dffca2165a6f249acc5",
    "results/experiments/e7b_serial_efficiency/e7b_run_e8a_a0p_pass2.json":
        "eaaad976be7f7854480fd06e045e60f6fff34bec019c2001c6b502362357a2d1",
    "results/experiments/e7b_serial_efficiency/e7b_run_e8a_a0p_pass3.json":
        "e6256d49b161ffb80db921ccded7f5da3e6e20d4f8796605e2ac3a84b9f5bfd1",
    "results/experiments/e7b_serial_efficiency/e7b_run_e8a_a1_pass1.json":
        "5d5dd4fe02b313441d16490d43911b6ab082a6fe44669ded238fadf064164b2a",
    "results/experiments/e7b_serial_efficiency/e7b_run_e8a_a1_pass2.json":
        "32cebe7829e5ef7dcb560775f9cbe9dd8bf8162598e9a54e7c663bbafd01a4b5",
    "results/experiments/e7b_serial_efficiency/e7b_run_e8a_a1_pass3.json":
        "371b10347ac5cbed0ce9cd4090592dcfabdc325d7c33428fc6ddb70bc17d97c6",
    "results/experiments/e7b_serial_efficiency/e7b_run_fusion_pass1.json":
        "a26c1f2a2fb07c3109c86b2ec7bd33f048b00d46d438cbac7969ca17415a9f57",
    "results/experiments/e7b_serial_efficiency/e7b_run_fusion_pass2.json":
        "1ef3e498045a063962e2f84ae54ee3bebe37d0c30451aeabd357b290d3ad2a09",
    "results/experiments/e7b_serial_efficiency/e7b_run_fusion_pass3.json":
        "63b9801d26c431cf1f9d800a7d9d8572d4950e8c83296473f6fd9f1255efb98e",
    "results/experiments/e7b_serial_efficiency/e7b_run_question_only_pass1.json":
        "6c8e901bb557afa2be8ae8c2f6f36cdee6615ea5f349e9168ad4a05c3316bb30",
    "results/experiments/e7b_serial_efficiency/e7b_run_question_only_pass2.json":
        "3173c45c7af7844ad4b9f88664bb4cdc76347c465b5c441d93fc93a03d40987d",
    "results/experiments/e7b_serial_efficiency/e7b_run_question_only_pass3.json":
        "f441af579cbe0ac493f82261956d84f7e0f4870e3f5bc2ecda531b8f7aee4f25",
    "results/experiments/e7b_serial_efficiency/e7b_run_reasoner_pass1.json":
        "0158b21b14bb596fcc23c78454931066b5efe27984b28f9e820fe2d6f114ce36",
    "results/experiments/e7b_serial_efficiency/e7b_run_reasoner_pass2.json":
        "451d11cf59f003b5e6b73ac8afb9ee661d16f22d8d864f7509d9823ea0bd5f7c",
    "results/experiments/e7b_serial_efficiency/e7b_run_reasoner_pass3.json":
        "8443ec097e25763bade3eee7564015795c27a8f84dfa53d83f779e340ef4f79e",
    "results/experiments/e7b_serial_efficiency/e7b_run_siglip_fusion_pass1.json":
        "d62703395b4fc7c319fe5cfcb32ce84c3778fab954b6ff63e96327cc65edc1ee",
    "results/experiments/e7b_serial_efficiency/e7b_run_siglip_fusion_pass2.json":
        "475f35c524fb55ac93a194e1d5d52df5146565a6ff7ce34be994d49096254a66",
    "results/experiments/e7b_serial_efficiency/e7b_run_siglip_fusion_pass3.json":
        "1d9d3205dd0c5013054b49b3966c06f52977e2f955f5383c9b3fffd7d80cca4e",
    "results/experiments/e7b_serial_efficiency/e7b_run_vocab1000_product_pass1.json":
        "0e82d6da90b03189afca3330b775ca617fee60cb2e98205d5b0ca2840ad12c25",
    "results/experiments/e7b_serial_efficiency/e7b_run_vocab1000_product_pass2.json":
        "8be8e0525f04fa533a7e4c5557ed4dcc76f7d3f6d6823a98334668fc57fa0589",
    "results/experiments/e7b_serial_efficiency/e7b_run_vocab1000_product_pass3.json":
        "4bc431d642d69fbe7247c5fce32b8b91bca318c4fb480f07387b44e9684cfff5",
    "results/experiments/e7b_serial_efficiency/latency_memory.csv":
        "deb4a67e793612c822e645d78d314081b9413bc83579d8694d6a83ff647942cc",
    "results/experiments/e7b_serial_efficiency/pareto_serial.png":
        "c65631ea318ff712ef26380edda7c91de2dfbc562f2d37bef059655e65c9ffb5",
    "results/experiments/e7b_serial_efficiency/pilot_e8a_a1.json":
        "792cef85bb5b0f4ff410b130ffe6631b87ddaa47a2d736f27d41a4fd62c3180c",
    "results/experiments/e7b_serial_efficiency/pilot_evidence/e7b_run_e8a_a1_pass1.json":
        "570990f5e129c60545e9ec505931bbf428af221b738b22833e872fdf70859adc",
    "results/experiments/e7b_serial_efficiency/pilot_evidence/e7b_run_e8a_a1_pass2.json":
        "494064aeec7ba524d4c04b8f0df32b1517246c7d463def72b477c22aa221b697",
    "results/experiments/e7b_serial_efficiency/pilot_evidence/e7b_run_e8a_a1_pass3.json":
        "938bef44f39609fe1fe4bb8c73bcbeb0714fbc90b7d08b9f9f754b9c7ec81733",
    "results/experiments/e7b_serial_efficiency/preflight.json":
        "9c96a04d1602117fe0c7b3f27b0368355c6f2c6b5f487e8455e4ec53e6df00d7",
    "results/experiments/e7b_serial_efficiency/results.json":
        "fddf880fc6a023ffe8255e6a8001f225dc4a253d6f87d0d1db2136c81bd672b9",
    "results/experiments/e7b_serial_efficiency/stage_timings.csv":
        "aeb9f7ef15230278faa1f1a5e5c24430f87506b0e1f1135b8b61eb5ac11cd266",
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
    "results/ve0/supersession_map.json":
        "e1163023509920dddf182989341a2517057941bffc813cc645e6b65928e66f5c",
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
    "docs/experiments/ve0_evidence_contract.md":
        "326bbc7c259aea808478e38358b3dc0eab12e1b61e03501927b9c2d4fc591188",
}

# --------------------------------------------------------------------------
# the withdrawal, stated once
# --------------------------------------------------------------------------

PEAK_REASON = (
    "The historical measurement window did not isolate the intended batch-1 "
    "serial inference envelope. The CUDA peak counters were read after the "
    "whole benchmark rather than at the close of the primary serial timing, "
    "so a single reported peak spans the serial, segmented, answer-sweep, "
    "cached-image, cached-feature and batch-64 regimes together. The "
    "contamination is also asymmetric across systems, because question_only "
    "never enters the batch-64 image branch. The published value therefore "
    "does not measure what it is labelled as measuring, at any system, and "
    "no arithmetic on the stored artefacts can recover the intended "
    "quantity."
)

COLD_REASON = (
    "Historical cold-query measurements were executed on a grad-enabled "
    "path. The accepted source repair standardised the cold query to "
    "torch.no_grad(). The historical values therefore do not measure the "
    "current cold-query path and are not current cold-start evidence."
)

WITHDRAWN = (
    {
        "field": "peak_allocated_mib",
        "final_status": "INVALID_WITHDRAWN",
        "cause_class": "MEASUREMENT_BOUNDARY_DEFECT",
        "reason": PEAK_REASON,
        "also_written_as": ["peak_alloc_mib"],
    },
    {
        "field": "peak_reserved_mib",
        "final_status": "INVALID_WITHDRAWN",
        "cause_class": "MEASUREMENT_BOUNDARY_DEFECT",
        "reason": PEAK_REASON,
        "also_written_as": ["peak_reserved_mib"],
    },
    {
        "field": "cold_first_query_ms",
        "final_status": "SUPERSEDED_WITHDRAWN",
        "cause_class": "GRAD_MODE_STANDARDISATION",
        "reason": COLD_REASON,
        "also_written_as": ["cold_ms_mean", "cold_ms"],
    },
)

RETAINED = (
    {
        "field": "warm_serial_median_ms",
        "final_status": "RETAINED_VALID",
        "also_written_as": ["warm_median_ms", "warm_median_ms_per_pass"],
        "why": "the warm serial timing loop was already inside a hoisted "
               "no_grad context and the accepted source repair did not "
               "change it, so the headline latency measures the same path "
               "before and after the repair.",
    },
    {
        "field": "across_pass_spread_ms",
        "final_status": "RETAINED_VALID",
        "also_written_as": ["warm_across_pass_spread_ms"],
        "why": "derived from the same unchanged warm serial medians.",
    },
    {
        "field": "accuracy_raw_distribution_common_denominator",
        "final_status": "RETAINED_VALID",
        "also_written_as": ["common_denominator_acc"],
        "why": "a stored development accuracy over the common 10,004-row "
               "raw denominator, reproduced by the preflight gate before any "
               "timing ran; it is independent of the timing harness.",
    },
    {
        "field": "trainable_parameters",
        "final_status": "RETAINED_VALID",
        "also_written_as": ["trainable"],
        "why": "a structural count of the loaded modules, independent of "
               "the timing and memory instrumentation.",
    },
    {
        "field": "total_loaded_parameters",
        "final_status": "RETAINED_VALID",
        "also_written_as": ["total_loaded"],
        "why": "a structural count of the loaded modules, independent of "
               "the timing and memory instrumentation.",
    },
    {
        "field": "resident_frozen_parameters",
        "final_status": "RETAINED_VALID",
        "also_written_as": ["resident_frozen"],
        "why": "a structural count of the loaded modules, independent of "
               "the timing and memory instrumentation.",
    },
    {
        "field": "primary_pareto_frontier_common_denominator",
        "final_status": "RETAINED_VALID",
        "also_written_as": [],
        "why": "its two axes are raw-distribution accuracy and warm serial "
               "latency. Neither withdrawn field is a frontier axis, so "
               "frontier membership is untouched: concat, fusion and "
               "vocab1000_product, exactly as recorded.",
    },
)

# Fields that carry a system-level cached-regime timing. They are measured,
# not derived, and the repair did not touch them, but they are labelled in
# the closed artefacts as not comparable with end-to-end and that labelling
# stands.
RETAINED_LABELLED = (
    "cached_image_question_side_ms",
    "cached_feature_head_only_ms",
)


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
        raise AssertionError(
            f"pinned historical artefact missing: {missing}")
    if moved:
        raise AssertionError(
            "pinned historical artefact moved; refusing to write an overlay "
            f"that names artefacts by hashes they no longer have: {moved}")
    return records


def _withdrawn_spellings() -> tuple:
    """Every spelling a withdrawn field is written under, in any artefact."""
    spellings = set()
    for entry in WITHDRAWN:
        spellings.add(entry["field"])
        spellings.update(entry["also_written_as"])
    return tuple(sorted(spellings))


def _carriers() -> list:
    """Pinned artefacts whose text actually contains a withdrawn field.

    Computed, not hand-listed. The first version of this record named the
    conflicting artefacts by hand and missed
    results/experiments/e7b_serial_efficiency/results.json, which carries all
    three. A list a human maintains is a list that goes stale; this derives
    the set from the bytes each time.
    """
    spellings = _withdrawn_spellings()
    carriers = []
    for relative in sorted(PINS):
        path = PROJECT_ROOT / cc.assert_not_embargoed(relative)
        if path.suffix not in (".json", ".csv", ".txt"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        found = sorted({name for name in spellings if name in text})
        if found:
            carriers.append({"path": relative, "fields_present": found})
    return carriers


def _assert_conflict_coverage(named: list, carriers: list) -> dict:
    """Fail closed unless every carrier is named in the precedence controls.

    A carrier may be named exactly, or matched by an entry's match_globs, so
    the 27 per-pass child records stay one entry instead of 27.
    """
    exact = {entry["path"] for entry in named}
    globs = [(pattern, entry["path"]) for entry in named
             for pattern in entry.get("match_globs", ())]
    resolved, uncovered = {}, []
    for carrier in carriers:
        relative = carrier["path"]
        if relative in exact:
            resolved[relative] = relative
            continue
        hit = next((owner for pattern, owner in globs
                    if fnmatch(relative, pattern)), None)
        if hit is None:
            uncovered.append(relative)
        else:
            resolved[relative] = hit
    if uncovered:
        raise AssertionError(
            "a pinned artefact carries a withdrawn field but is named in no "
            f"precedence control: {uncovered}")
    return {
        "carrier_count": len(carriers),
        "named_entry_count": len(named),
        "all_carriers_named": True,
        "method": "computed from the pinned artefacts' bytes, not "
                  "hand-listed; the builder refuses to write if any carrier "
                  "is unnamed",
        "resolution": dict(sorted(resolved.items())),
    }


def _content_digest(payload: dict) -> str:
    """Digest of the scientific content, provenance excluded.

    Same two-tier rule and same excluded keys as the VE-0 contract, computed
    over this file's own serialisation so it is self-verifying from the
    written bytes.
    """
    stripped = {k: v for k, v in payload.items()
                if k not in ("provenance", "content_sha256")}
    text = json.dumps(stripped, indent=2, sort_keys=True,
                      ensure_ascii=False) + "\n"
    return cc.sha256_bytes(text.encode("utf-8"))


def build() -> dict:
    pin_records = _assert_pins()

    payload = {
        "title": "E7b canonical evidence supersession overlay",
        "schema_version": 1,
        "overlay_kind": "FIELD_LEVEL",
        "experiment_family": "E7b",
        "applies_to_systems": list(SYSTEMS),
        "applies_to_system_count": len(SYSTEMS),

        "purpose": (
            "Declare, in one machine-resolvable place, which E7b fields are "
            "current canonical dissertation evidence and which are "
            "withdrawn. The artefacts carrying the withdrawn fields are "
            "closed and byte-pinned by VE-0 and VE-1, so they cannot be "
            "amended in place without reopening a closed packet. This "
            "overlay supersedes their field-level validity instead."),

        # ------------------------------------------------------------------
        # precedence
        # ------------------------------------------------------------------
        "precedence": {
            "rule": (
                "1. results/closure/e7b_evidence_supersession.json controls "
                "named E7b field validity; "
                "2. results/ve0/supersession_map.json controls unnamed "
                "artefact-group status; "
                "3. historical stored evidence_status is lowest precedence."),
            "order": [
                {"rank": 1,
                 "controls": "named E7b field validity",
                 "authority": "results/closure/e7b_evidence_supersession.json"},
                {"rank": 2,
                 "controls": "unnamed artefact-group status",
                 "authority": "results/ve0/supersession_map.json"},
                {"rank": 3,
                 "controls": "nothing that ranks 1 or 2 already controls",
                 "authority": "historical stored evidence_status"},
            ],
            "historical_numbers": (
                "Historical numbers remain readable only as provenance."),
            "scope_limit": (
                "This overlay is E7b-scoped. It says nothing about any field "
                "of any other experiment family, and it does not change the "
                "artefact-group status VE-0 assigns to E7b as a whole."),
        },

        # ------------------------------------------------------------------
        # the withdrawal
        # ------------------------------------------------------------------
        "withdrawn_fields": [
            dict(entry,
                 applies_to_systems=list(SYSTEMS),
                 replacement_value=None,
                 substitute_permitted=False,
                 remeasured=False,
                 permitted_use="readable as historical provenance only",
                 forbidden_use=[
                     "any dissertation number",
                     "any figure",
                     "any table",
                     "any comparison between systems",
                     "any Pareto axis",
                     "any GPU-memory footprint claim",
                 ])
            for entry in WITHDRAWN
        ],
        "withdrawn_field_count": len(WITHDRAWN),
        "withdrawn_field_names": sorted(e["field"] for e in WITHDRAWN),
        "cause_classes": {
            "MEASUREMENT_BOUNDARY_DEFECT": sorted(
                e["field"] for e in WITHDRAWN
                if e["cause_class"] == "MEASUREMENT_BOUNDARY_DEFECT"),
            "GRAD_MODE_STANDARDISATION": sorted(
                e["field"] for e in WITHDRAWN
                if e["cause_class"] == "GRAD_MODE_STANDARDISATION"),
        },
        "cause_separation": (
            "The two causes are distinct and must not be conflated. The peak "
            "memory withdrawal is NOT attributed to the grad-mode "
            "standardisation, and the cold-query withdrawal is NOT "
            "attributed to the measurement-boundary defect."),

        # ------------------------------------------------------------------
        # what survives
        # ------------------------------------------------------------------
        "retained_fields": [
            dict(entry, applies_to_systems=list(SYSTEMS))
            for entry in RETAINED
        ],
        "retained_field_count": len(RETAINED),
        "retained_field_names": sorted(e["field"] for e in RETAINED),
        "retained_but_labelled_not_comparable_with_end_to_end":
            list(RETAINED_LABELLED),
        "primary_pareto_frontier_common_denominator":
            ["concat", "fusion", "vocab1000_product"],
        "pareto_membership_unchanged": True,
        "pareto_axes": ["accuracy_raw_distribution_common_denominator",
                        "warm_serial_median_ms"],

        # ------------------------------------------------------------------
        # no substitute
        # ------------------------------------------------------------------
        "no_substitute_value": {
            "replacement_peak_memory_value_exists": False,
            "replacement_cold_query_value_exists": False,
            "estimation_permitted": False,
            "inference_permitted": False,
            "back_calculation_permitted": False,
            "cross_node_substitution_permitted": False,
            "adjustment_factor_permitted": False,
            "statement": (
                "No replacement memory or cold-query value exists. None is "
                "estimated, inferred, back-calculated or substituted from "
                "another node, and this overlay introduces no number that "
                "was not already measured."),
        },

        # ------------------------------------------------------------------
        # no remeasurement
        # ------------------------------------------------------------------
        "remeasurement": {
            "remeasured": False,
            "remeasurement_authorised": False,
            "gpu_hours_charged": 0.0,
            "historical_node": "otter155",
            "current_project_copy_node": "otter159",
            "bridge_control": "fusion, the same system measured on both nodes",
            "bridge_control_relative_difference": -0.1869,
            "bridge_control_pre_registered_tolerance": 0.10,
            "bridge_outcome": "FAILED",
            "reason": (
                "The historical E7b measurements were taken on otter155 and "
                "the current project copy runs on otter159. The fusion "
                "bridge control differs by -0.1869 against a pre-registered "
                "tolerance of 0.10, so the bridge FAILED. Cross-node "
                "substitution is forbidden and no adjustment factor is "
                "permitted, so remeasuring on the current node would not "
                "produce a value comparable with the historical record."),
        },

        # ------------------------------------------------------------------
        # E9 wording
        # ------------------------------------------------------------------
        "e9_wording": {
            "canonical_statement": (
                "SmolVLM-500M was approximately 24.7x slower in warm serial "
                "latency under the same-node contextual protocol."),
            "measured_values_ms": {
                "top_1000_global_head": 6.199,
                "smolvlm_500m": 152.854,
            },
            "comparison_kind": (
                "contextual positioning, not a matched causal comparison"),
            "forbidden_conclusions_from_latency_alone": [
                "cheaper",
                "25x cheaper",
                "~25x cheaper",
                "more efficient",
                "lower cost",
            ],
            "not_supported_by_the_timing_result_alone": [
                "energy",
                "power",
                "carbon",
                "monetary cost",
                "general computational cost",
            ],
            "why": (
                "A latency ratio measured on one node states how long each "
                "system takes, and nothing else. Training history, "
                "multimodal pretraining, answer support and output format "
                "all differ between the two systems, so no efficiency, cost "
                "or resource conclusion follows from the timing alone."),
        },

        # ------------------------------------------------------------------
        # clean-test governance, task-scoped
        # ------------------------------------------------------------------
        "clean_test_governance": {
            "scope": "THIS_TASK_ONLY",
            # LITERAL reading, set by explicit user decision of 14 August
            # 2026. A mechanical read of the target's bytes did occur during
            # this task, so both access fields are true. This deliberately
            # diverges from the semantics the other VE-0 and VE-1 artefacts
            # give their clean_test_accessed flags, where false means "no
            # experiment or reporting code opened the target". Those records
            # remain correct under their own semantics; this one is stricter.
            "clean_test_accessed_by_this_task": True,
            "clean_test_read_stat_hashed_opened_searched_or_parsed_by_this_"
            "task": True,
            "clean_test_used_for_development_model_selection_or_reporting":
                False,
            "disclosure": DISCLOSURE,
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
            "superseded_single_incident_wording": {
                "text": SUPERSEDED_DISCLOSURE,
                "superseded_on": "2026-08-14",
                "why": (
                    "It was accurate when written and is now incomplete as a "
                    "history: it predates the second incident and, read as "
                    "the whole record, understates the access to one "
                    "occasion. It must never be presented as the complete "
                    "history."),
                "permitted_residual_locations": [
                    {"path": "results/ve2/VE2_MANIFEST.json",
                     "why": "an immutable VE-2 artefact. VE-2 is CLOSED and "
                            "may not reopen, so the sentence stays as a "
                            "historical, superseded residual whose wording "
                            "predates the second incident."},
                    {"path": "experiments/ve2/run_ve2.py",
                     "why": "the source string the immutable manifest is "
                            "rebuilt from. tests/test_ve2.py rebuilds VE-2 "
                            "and compares scientific content, so editing it "
                            "would change the manifest and reopen a closed "
                            "packet."},
                ],
                "corrected_in": [
                    "results/closure/e7b_evidence_supersession.json",
                    "docs/experiments/ve2_qualitative_evidence.md",
                    "docs/REPRODUCIBILITY.md",
                    "README.md",
                ],
            },
            "no_project_global_claim": (
                "This record deliberately asserts no project-global "
                "clean_test_accessed field. The claim that the clean test "
                "was never accessed at all would be false, and the scoped "
                "fields above are what this task can truthfully attest."),
            "field_semantics": (
                "The two access fields above are read LITERALLY, by explicit "
                "user decision of 14 August 2026: true because some process "
                "acting for this task read the target's bytes, whatever its "
                "purpose. They do NOT mean that experiment, builder, test or "
                "reporting code opened the target, and none did. The third "
                "field is what carries the scientific guarantee: no byte of "
                "the target reached any model, checkpoint, selection, "
                "figure, table or reported number. Elsewhere in the project "
                "a clean_test_accessed flag of false means 'no experiment or "
                "reporting code opened the target'; this record is stricter "
                "and the two conventions must not be read as one."),
            "executor_mechanical_access_during_this_task": {
                "occurred": True,
                "classification": "B — MECHANICAL ACCESS INCIDENT",
                "what_happened": (
                    "While mapping which artefacts carry the withdrawn "
                    "fields, the executor ran one ad-hoc repository-wide "
                    "scan that walked the whole project root and called "
                    "read_text() on every .json, .py, .md, .csv and .txt "
                    "file, searching for four documentation digests. The "
                    "traversal was not scoped away from data/, so the "
                    "embargoed target was among the files mechanically "
                    "read."),
                "contents_inspected": False,
                "reached_any_model_or_number": False,
                "used_for_development_model_selection_or_reporting": False,
                "target_named_in_any_written_file": False,
                "hashed": False,
                "same_class_as": (
                    "the independent reviewer's sha256sum integrity check of "
                    "13 August 2026, recorded as classification B"),
                "consequence": (
                    "No scientific invalidation and no F2 remediation. It is "
                    "recorded here rather than omitted, because the project "
                    "record must not overstate the embargo."),
                "avoidable": True,
                "corrective_rule": (
                    "A repository-wide scan must exclude data/ or filter the "
                    "embargoed name before reading, not after."),
            },
            "embargo_intact": True,
        },

        # ------------------------------------------------------------------
        # what conflicts, and why this record wins
        # ------------------------------------------------------------------
        "conflicting_historical_artefacts": [
            {
                "path": "results/closure/efficiency_closure_table.json",
                "conflict": "23 E7b rows carry evidence_status VERIFIED "
                            "alongside peak_allocated_mib, peak_reserved_mib "
                            "and cold_first_query_ms.",
                "stored_row_level_status": "VERIFIED",
                "resolution": "rank 3, superseded field-by-field by this "
                              "overlay for the three withdrawn fields only; "
                              "every other field in those rows stands.",
                "immutable": True,
                "why_immutable": "byte-pinned by the closed VE-0 packet.",
            },
            {
                "path": "results/closure/efficiency_closure_table.csv",
                "conflict": "the flat view of the same 23 rows, carrying the "
                            "same three withdrawn columns.",
                "stored_row_level_status": "VERIFIED",
                "resolution": "rank 3, superseded field-by-field by this "
                              "overlay.",
                "immutable": True,
                "why_immutable": "a closed closure artefact; the task "
                                 "forbids modifying it.",
            },
            {
                "path": "results/experiments/e7b_serial_efficiency/"
                        "e7b_results.json",
                "conflict": "the aggregate carries peak_allocated_mib, "
                            "peak_reserved_mib and cold_first_query_ms for "
                            "all eight systems with no withdrawal marker.",
                "stored_row_level_status": "none recorded",
                "resolution": "rank 3, superseded field-by-field by this "
                              "overlay; retained as raw provenance.",
                "immutable": True,
                "why_immutable": "byte-pinned by the closed VE-0 and VE-1 "
                                 "packets.",
            },
            {
                "path": "results/experiments/e7b_serial_efficiency/"
                        "latency_memory.csv",
                "conflict": "columns peak_alloc_mib, peak_reserved_mib and "
                            "cold_ms_mean carry withdrawn values.",
                "stored_row_level_status": "none recorded",
                "resolution": "rank 3, superseded field-by-field by this "
                              "overlay.",
                "immutable": True,
                "why_immutable": "a closed E7b artefact; the task forbids "
                                 "modifying it.",
            },
            {
                "path": "results/experiments/e7b_serial_efficiency/"
                        "results.json",
                "conflict": "the full raw aggregate carries "
                            "peak_allocated_mib, peak_reserved_mib and "
                            "cold_first_query_ms for every system and pass, "
                            "with no withdrawal marker. It is the artefact "
                            "e7b_results.json is derived from.",
                "stored_row_level_status": "none recorded",
                "resolution": "rank 3, superseded field-by-field by this "
                              "overlay; retained as raw provenance.",
                "immutable": True,
                "why_immutable": "raw evidence is never rewritten.",
            },
            {
                "path": "results/experiments/e7b_serial_efficiency/"
                        "e7b_run_<system>_pass<n>.json",
                "match_globs": [
                    "results/experiments/e7b_serial_efficiency/"
                    "e7b_run_*.json",
                    "results/experiments/e7b_serial_efficiency/"
                    "pilot_evidence/e7b_run_*.json",
                ],
                "conflict": "the 24 full-run child records and the 3 pilot "
                            "child records carry the raw per-pass peak and "
                            "cold values.",
                "stored_row_level_status": "none recorded",
                "resolution": "rank 3. Raw measurement evidence, preserved "
                              "byte-for-byte and readable as provenance "
                              "only.",
                "immutable": True,
                "why_immutable": "raw evidence is never rewritten.",
            },
            {
                "path": "results/experiments/e7b_serial_efficiency/"
                        "pilot_e8a_a1.json",
                "conflict": "embeds its three pilot child records verbatim, "
                            "including their peak and cold values.",
                "stored_row_level_status": "none recorded",
                "resolution": "rank 3, raw provenance only.",
                "immutable": True,
                "why_immutable": "raw evidence is never rewritten.",
            },
        ],

        "out_of_scope": {
            "statement": (
                "Peak-memory fields recorded by other experiments are NOT "
                "withdrawn by this overlay. They were measured by different "
                "harnesses under different protocols and this record makes "
                "no claim about them."),
            "paths": [
                "artifacts/results_export/v3_01_reasoner/results.json",
                "artifacts/results_export/v3_03_scaling/results.json",
                "artifacts/results_export/v3_03_scaling/preflight.json",
                "results/experiments/e7a_efficiency/results.json",
            ],
        },

        # ------------------------------------------------------------------
        # discoverability
        # ------------------------------------------------------------------
        "human_facing_pointers": [
            {"path": "docs/experiments/e7b_serial_efficiency.md",
             "role": "status banner on the first screen, withdrawn columns "
                     "marked in the results table"},
            {"path": "README.md",
             "role": "pointer to this record from the project overview"},
            {"path": "docs/REPRODUCIBILITY.md",
             "role": "the precedence rule"},
            {"path": "CLAUDE.md",
             "role": "current E7b status"},
        ],
        "machine_resolution": {
            "resolves_all_three_withdrawn_fields": True,
            "lookup": "withdrawn_fields[].field, with the artefact-local "
                      "spellings in withdrawn_fields[].also_written_as",
            "retention_is_explicit": True,
        },
        "residual_discoverability_risk": {
            "risk_is_zero": False,
            "statement": (
                "A reviewer who opens only an immutable historical closure "
                "file may still see a row-level evidence_status of VERIFIED "
                "next to a withdrawn value, with no marker in that file "
                "pointing here. Those files are byte-pinned by closed "
                "packets and cannot carry a marker without reopening them."),
            "residual_surfaces": [
                "results/closure/efficiency_closure_table.json",
                "results/closure/efficiency_closure_table.csv",
                "results/experiments/e7b_serial_efficiency/e7b_results.json",
                "results/experiments/e7b_serial_efficiency/results.json",
                "results/experiments/e7b_serial_efficiency/"
                "latency_memory.csv",
                "results/experiments/e7b_serial_efficiency/"
                "e7b_run_<system>_pass<n>.json, 27 raw child records",
                "results/experiments/e7b_serial_efficiency/pilot_e8a_a1.json",
            ],
            "residual_surfaces_are_the_computed_carrier_set": True,
            "mitigation": (
                "Every human entry point that is not itself frozen names "
                "this record: the E7b report banner, README and "
                "REPRODUCIBILITY. No dissertation figure or table draws a "
                "withdrawn value, because the VE-0 canonical evidence "
                "inventory registers no evidence id for peak memory or cold "
                "first query at all."),
            "recorded_for": "the later F1 discoverability review",
        },
        "no_canonical_evidence_id_for_withdrawn_fields": {
            "checked_against":
                "results/ve0/canonical_evidence_inventory.json",
            "e7b_evidence_id_count": 23,
            "evidence_ids_covering_a_withdrawn_field": 0,
            "statement": (
                "The 23 canonical EV-EFF-E7b evidence ids cover warm serial "
                "latency, the cached-image question side and the "
                "cached-feature head only. None registers peak memory or "
                "cold first query, so no VE-1 figure or table can resolve to "
                "a withdrawn value."),
        },

        # ------------------------------------------------------------------
        # status
        # ------------------------------------------------------------------
        "status": {
            "e7b": "OPEN",
            "e7b_closed_by_this_record": False,
            "ve0": "CLOSED, unchanged",
            "ve1": "CLOSED, unchanged",
            "ve2": "CLOSED, unchanged",
            "f1": "BLOCKED",
            "f2": "unstarted and unauthorised",
            "note": "This record is a withdrawal declaration. It does not "
                    "close E7b, which needs an independent review.",
        },

        "frozen_pins": {
            "pin_count": len(PINS),
            "all_unchanged": True,
            "policy": "fail closed: the builder refuses to write if any "
                      "pinned artefact has moved",
            "records": sorted(pin_records, key=lambda row: row["path"]),
        },

        "determinism": {
            "content_sha256": "the digest of this record with the provenance "
                              "block and the digest field itself excluded; "
                              "it is what a later build must reproduce",
            "file_bytes": "may differ across commits, because provenance "
                          "carries the repository HEAD and worktree state",
        },
    }

    # The complete pin set, with both the pinned and the recomputed hash,
    # is in frozen_pins.records above and is therefore inside the content
    # digest. Provenance names only the four artefacts this overlay directly
    # governs, so the two blocks do not restate the same 51 hashes.
    prov = cc.provenance(
        SCRIPT,
        inputs=[cc.record_source(PROJECT_ROOT / relative)
                for relative in (
                    "results/experiments/e7b_serial_efficiency/"
                    "e7b_results.json",
                    "results/experiments/e7b_serial_efficiency/"
                    "latency_memory.csv",
                    "results/closure/efficiency_closure_table.json",
                    "results/ve0/supersession_map.json")],
        extra={
            "measured_anything": False,
            "loaded_any_model": False,
            "evaluated_anything": False,
            "remeasured_anything": False,
            "embargoed_token_guard": (
                "assembled from parts and asserted absent from the payload"),
        })
    # The shared closure provenance helper stamps a project-global
    # clean_test_accessed flag. This record must not carry one. At project
    # scope the claim is not true: the clean-test bytes were read once by an
    # independent reviewer's integrity-hash command on 13 August 2026. What
    # this task can truthfully attest is task-scoped, and that is what
    # clean_test_governance states. The field is removed rather than set,
    # because a false value here would be a project-global claim either way.
    # Every pinned artefact that actually carries a withdrawn field must be
    # named in the precedence controls. Computed and asserted, so the
    # omission the independent review found cannot recur.
    payload["conflict_coverage"] = _assert_conflict_coverage(
        payload["conflicting_historical_artefacts"], _carriers())

    prov.pop("clean_test_accessed", None)
    prov["clean_test_governance_pointer"] = (
        "see clean_test_governance; this record deliberately makes no "
        "project-global clean-test claim")
    payload["provenance"] = prov
    payload["content_sha256"] = _content_digest(payload)
    return payload


def main() -> int:
    payload = build()
    if EMBARGOED_TOKEN in json.dumps(payload):
        raise AssertionError(
            "the overlay payload names the embargoed clean-test target")
    digest = cc.write_json(payload, PROJECT_ROOT / OUTPUT)
    print(f"written {OUTPUT}")
    print(f"  pins verified   : {len(PINS)}/{len(PINS)} unchanged")
    print(f"  withdrawn fields: {payload['withdrawn_field_count']} "
          f"{payload['withdrawn_field_names']}")
    print(f"  retained fields : {payload['retained_field_count']}")
    print(f"  content_sha256  : {payload['content_sha256']}")
    print(f"  file sha256     : {digest}")
    print("  gpu hours       : 0.0, nothing measured, nothing loaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
