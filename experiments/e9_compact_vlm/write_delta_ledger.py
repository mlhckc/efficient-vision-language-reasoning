"""Write the final D1-D9 delta ledger, before any E9 score is produced.

Amendment 13. D2, D4, D5, D6 and D9 were approved subject to amendments and
are recorded here as amended. D1, D3, D7 and D8 are retained only if each is
exactly the already-reviewed canonical-plan delta and introduces no new model,
dataset, tuning search, scoring rule or scientific claim beyond that reviewed
scope; each is evaluated against those five tests explicitly below, and any
delta failing one would halt before scientific execution.

The ledger is a COMPANION frozen artefact rather than an edit to
`frozen_protocol.json`: that file is already hashed, every stage asserts
against it, and mutating a hashed artefact after freezing is exactly the
stale-output ambiguity the process hardening forbids. The ledger pins the
frozen protocol's SHA-256, so the two are bound and the pair is the frozen
protocol record set.
"""

from __future__ import annotations

import sys

import e9_common as e9

SUBSTANTIVE_TESTS = ("new model", "new dataset", "new tuning search",
                     "new scoring rule",
                     "new scientific claim beyond the reviewed scope")

LEDGER = {
    "D1": {
        "title": "SmolVLM-500M demoted to a SECONDARY, normal-condition-only "
                 "capacity point",
        "status": "RETAINED",
        "approval": "Amendment 2 approves SmolVLM-256M-Instruct as the one "
                    "primary baseline and keeps 500M secondary and "
                    "normal-condition only.",
        "direction": "narrowing of the reviewed canonical plan",
        "substantive_change": {test: False for test in SUBSTANTIVE_TESTS},
        "constraint": "500M results are never presented as a clean causal "
                      "135M-to-360M language-backbone effect. E10 remains "
                      "the project's controlled SLM size-sensitivity "
                      "experiment and is not started here.",
    },
    "D2": {
        "title": "Trie-constrained closed readout",
        "status": "APPROVED AS AMENDED",
        "approval": "Amendment 4.",
        "amendments_applied": [
            "Open generation remains the canonical PRIMARY E9 readout.",
            "The constrained readout is a SECONDARY DIAGNOSTIC whose purpose "
            "is to separate task and content failure from output-format "
            "failure.",
            "It is never promoted to primary even if it scores higher.",
            "The answer support is frozen at TOP-1000 before any result, "
            "because the reviewed canonical plan pins the top-1000 "
            "vocabulary and reports a top-1000 view.",
            "Metrics are also reported on the common 7,714 "
            "top-100-supported rows, with the candidate-support difference "
            "disclosed; a top-1000 constrained task is never silently "
            "equated with E8B's top-100 classifier task.",
        ],
    },
    "D3": {
        "title": "Paired row-level contrasts against the stored B1/B2/B3 "
                 "per-row vectors",
        "status": "RETAINED",
        "approval": "Amendment 10 requires that paired E9-versus-existing "
                    "comparisons preserve row and image pairing.",
        "direction": "uses the already-frozen project clustered procedure",
        "substantive_change": {test: False for test in SUBSTANTIVE_TESTS},
        "constraint": "The deterministic E9 checkpoint and the three "
                      "independently trained E8B seeds are never treated as "
                      "equivalent sources of uncertainty; the distinction is "
                      "recorded on every comparison.",
    },
    "D4": {
        "title": "Derangement-hash equality with E8B's frozen mapping",
        "status": "APPROVED AS AMENDED",
        "approval": "Amendment 5.",
        "amendments_applied": [
            "Described as an identical IMAGE-PARTNER intervention.",
            "NOT described as an identical feature or representation "
            "intervention, because model preprocessing differs.",
        ],
        "verified_sha256": e9.EXPECTED_DERANGEMENT_SHA256,
    },
    "D5": {
        "title": "Contemporaneous fusion re-timing on the E9 node",
        "status": "APPROVED AS AMENDED",
        "approval": "Amendment 8.",
        "amendments_applied": [
            "otter159-against-otter159 measurements are the PRIMARY "
            "numerical comparison for every E9 efficiency conclusion.",
            "E7b's otter155 timings are retained as HISTORICAL / REFERENCE "
            "evidence only.",
            "The pre-registered plus or minus 10 per cent tolerance decides "
            "only whether the new fusion timing is CONSISTENT with the "
            "earlier E7b measurement.",
            "Different-node latency measurements are never numerically "
            "merged into one frontier and no adjustment factor is applied.",
        ],
    },
    "D6": {
        "title": "E8B timing bridge over the frozen 250k seed-0 checkpoints",
        "status": "APPROVED AS AMENDED",
        "approval": "Amendment 7.",
        "amendments_applied": [
            "Read-only. This does NOT reopen E8B.",
            "All new timing artefacts are written under the E9 experiment "
            "namespace and reference the frozen E8B checkpoint hashes.",
            "No E8B scientific artefact is modified.",
            "Pairings obtained where the existing frozen readouts support "
            "them: E9 open against E8B R3, E9 constrained against E8B R2, "
            "and B1 as the language-model-free reference.",
            "Identical warm-up, synchronisation, batch, timing and reporting "
            "semantics across every timed system.",
        ],
    },
    "D7": {
        "title": "Pre-registered 6.0 GPU-hour E9 branch halt",
        "status": "RETAINED",
        "approval": "Amendment 11 keeps the 6.0 GPU-hour branch halt and "
                    "forbids changing it after seeing results.",
        "direction": "tightening; stricter than every binding project gate",
        "substantive_change": {test: False for test in SUBSTANTIVE_TESTS},
    },
    "D8": {
        "title": "Top-1000 view computed as a row subset of the same open "
                 "generation pass",
        "status": "RETAINED",
        "approval": "The reviewed canonical plan already promises a "
                    "top-1000 view and pins the top-1000 vocabulary hash for "
                    "it; this only fixes how the view is computed.",
        "direction": "clarification; no additional decoding",
        "substantive_change": {test: False for test in SUBSTANTIVE_TESTS},
    },
    "D9": {
        "title": "Blank-image condition",
        "status": "APPROVED AS AMENDED",
        "approval": "Amendment 6.",
        "amendments_applied": [
            "Labelled AUXILIARY / UNMATCHED ABLATION everywhere.",
            "Its drop is never directly equated with E8B fixed_image, which "
            "is a feature-space mean intervention.",
            "The matched visual-reliance comparison across E8B and E9 is the "
            "deranged-image condition.",
        ],
    },
}


def main() -> int:
    path = e9.RESULTS_DIR / "delta_ledger.json"
    if path.exists():
        print(f"E9 REFUSED: {path.name} already exists and is frozen.",
              file=sys.stderr)
        return 4
    e9.require(e9.FROZEN_PROTOCOL_PATH.exists(), "G-PROMPT",
               "the frozen protocol must exist before the ledger is written")
    for key in ("D1", "D3", "D7", "D8"):
        entry = LEDGER[key]
        offending = [test for test, flag in entry["substantive_change"].items()
                     if flag]
        e9.require(not offending, "G-DELTA",
                   f"{key} introduces {offending}; amendment 13 requires "
                   f"stopping before scientific execution and reporting only "
                   f"that delta")
    record = {"e9_delta_ledger": {
        "frozen_protocol_sha256": e9.sha256_file(e9.FROZEN_PROTOCOL_PATH),
        "written_before_any_score": True,
        "relationship_to_frozen_protocol":
            "Companion frozen artefact. frozen_protocol.json is already "
            "hashed and asserted against by every stage, so it is not "
            "edited after freezing; this ledger pins its SHA-256 instead, "
            "and the two together are the frozen protocol record set.",
        "substantive_change_tests": list(SUBSTANTIVE_TESTS),
        "retained_without_substantive_change": ["D1", "D3", "D7", "D8"],
        "approved_as_amended": ["D2", "D4", "D5", "D6", "D9"],
        "deltas": LEDGER,
    }, "metadata": e9.run_metadata()}
    digest = e9.atomic_write_json(path, record)
    print(f"delta_ledger.json written, sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
