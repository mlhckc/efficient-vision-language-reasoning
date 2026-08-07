"""Derive the E8B MEDIUM/LOW findings ledger summary MECHANICALLY.

The audit found the earlier summary counts declared by hand, so a
finding could be recorded as fixed in the summary while its own entry
said otherwise. This module removes the possibility: the summary is a
pure function of `entries[].disposition` under a published state map,
and nothing in it is written by hand.

The ledger is ADDITIVE. Entries accumulate; a later finding is appended
with its own identifier and is never merged into an earlier one, and no
historical entry is rewritten. Re-running is idempotent: an entry whose
identifier is already present is left exactly as it stands.

NO OPTIMIZER STEP. This module reads and writes one JSON record.

    python -B experiments/e8b_readout_generation/medium_ledger.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import utils  # noqa: E402
from experiments.e8b_readout_generation import run as e8b_run  # noqa: E402

LEDGER = e8b_run.OUT_DIR / "medium_findings_ledger_20260807.json"

STATE_MAP = {
    "FIXED": "FIXED_CLOSED",
    "FIXED AT SOURCE": "FIXED_CLOSED",
    "FIXED BY DELETION": "FIXED_CLOSED",
    "FIXED BY DELETION + AUTHORISATION": "FIXED_CLOSED",
    "FIXED BY DISCLOSURE": "FIXED_CLOSED",
    "ACCEPTED AND DISCLOSED": "ACCEPTED_LIMITATION",
    "PARTIALLY FIXED": "PARTIALLY_FIXED",
    "OPEN": "OPEN",
}

# The three execution-critical MEDIUMs raised by review A of 2026-08-07,
# appended additively to the entries already on record.
REVIEW_A_ENTRIES = [
    {"id": "REVIEW-A-M1",
     "lens": "A (scientific and provenance honesty)",
     "issue": "CORE_FIXED_JUSTIFICATION stated only that the recipe was "
              "the pre-result default and did not disclose that the same "
              "configuration also showed the highest bf16 development "
              "maximum of the three grid points that completed. A reader "
              "of the code alone could not see the coincidence and judge "
              "it.",
     "classification": "scientific-validity",
     "disposition": "FIXED BY DISCLOSURE",
     "evidence": "training.CORE_FIXED_JUSTIFICATION now carries the "
                 "disclosure explicitly, framed exactly as CLAUDE.md "
                 "gate item 2 requires: exploratory, NON-CANONICAL, "
                 "SUPERSEDED, invalidated for ranking because the pools "
                 "differ in length, and offered as support for nothing. "
                 "The stated reason for the freeze remains SOLELY that "
                 "it was the pre-result preregistered default, and the "
                 "freeze is outcome-independent by construction.",
     "closed_by": "the user's item-7 instruction of 2026-08-07, which "
                  "governs over the reviewer's plainer wording"},
    {"id": "REVIEW-A-M2",
     "lens": "A (scientific and provenance honesty)",
     "issue": "The G8 overfit gate trains to a convergence threshold and "
              "so consumes a different number of RNG draws in B2 than in "
              "B3. The paired arms therefore entered the training loop on "
              "divergent global RNG states, so their dropout masks and "
              "shuffle orders differed for a reason unrelated to the "
              "pretrained-versus-random contrast that B3 minus B2 exists "
              "to isolate. The loader generator was re-seeded after the "
              "gate; the global RNG was not.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "training.py re-seeds with e8b_run.reseed_strict("
                 "recipe['seed']) after the G8/G14 gates and before the "
                 "resume block, so both arms enter training on a common "
                 "RNG state; the placement before the resume block "
                 "preserves resume fidelity, since a resume legitimately "
                 "overwrites RNG from the saved state. Ordering is "
                 "asserted in tests (gate < reseed < loop, and reseed < "
                 "resume).",
     "closed_by": "test_remediation_known_negatives, M-2 ordering checks"},
    {"id": "REVIEW-A-M5",
     "lens": "B (executable safety)",
     "issue": "history, train_times and eval_times were not resume "
              "fields. The fixed-endpoint rule reads history[-1] to "
              "identify the epoch-22 canonical checkpoint, so a cell "
              "resumed at or near the final epoch would die on an "
              "IndexError inside a loop that would then retry and die "
              "again, or would select a canonical epoch from a truncated "
              "record.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "run.RESUME_FIELDS now carries history, train_times and "
                 "eval_times; save_resume_checkpoint requires them and "
                 "restore_resume_state returns them; training.py "
                 "restores them on resume. The empty-history path halts "
                 "through gate_halt with the reason stated instead of "
                 "raising IndexError. Superseded pre-amendment "
                 "checkpoints stay READABLE FOR AUDIT via "
                 "protocol_family=None but cannot be restored, which is "
                 "asserted as a known negative.",
     "closed_by": "test_checkpoint_resume and "
                  "test_remediation_known_negatives, M-5 checks"},
]


# The findings raised by the three fresh-context audits of 2026-08-07,
# appended additively.
AUDIT_20260807_ENTRIES = [
    {"id": "AUDIT-B-BLOCKER-1",
     "lens": "B (executable safety)",
     "issue": "B1Classifier.forward returned AttentionPoolReadout's "
              "(logits, attention_weights) pair unchanged, while every "
              "B1 consumer -- the loss, the dtype gate, the shape gate "
              "and the canonical evaluation -- expects a tensor. All six "
              "B1 core cells would have acquired the NFS lock, hashed "
              "the token stores and then died with an uncaught "
              "AttributeError, with no HALT and no FAILED record.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "B1Classifier.forward now unpacks and returns logits "
                 "only; forward_with_weights keeps the diagnostic pair. "
                 "Reproduced before the fix and asserted after it by a "
                 "REAL forward and backward pass, not a source-text "
                 "check -- which is why the defect survived: every prior "
                 "B1 test was a hasattr or substring assertion.",
     "closed_by": "test_b1_forward_pass"},
    {"id": "AUDIT-B-HIGH-1",
     "lens": "B (executable safety)",
     "issue": "Strict determinism was downgraded for the whole of both "
              "overfit gates. utils.set_seed sets warn_only=True and the "
              "builders re-seed again, so optimizer.step ran for up to "
              "200 epochs with enforcement downgraded, inside a HALTING "
              "gate, while the hashed recipe recorded strict enforcement "
              "throughout. Confirmed empirically by reading the live "
              "flag.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "both gates now call reseed_strict at entry, re-impose "
                 "after the internal builder re-seed, and assert at the "
                 "point of use.",
     "closed_by": "test_audit_known_negatives, gate determinism checks"},
    {"id": "AUDIT-B-MEDIUM-1",
     "lens": "B (executable safety)",
     "issue": "Resume verified the recipe hash and protocol family but "
              "never the implementation, though code_head was stored "
              "from the beginning. A cell could crash at epoch 14, have "
              "a defect fixed, resume, and run epochs 15-22 under "
              "different code, producing an epoch-22 canonical "
              "checkpoint spliced from two implementations.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "verify_resume_checkpoint now compares code_head AND a "
                 "new code_digest over the E8B sources, so a dirty "
                 "-worktree edit that leaves the commit unchanged is "
                 "caught too.",
     "closed_by": "test_audit_known_negatives, code-identity negatives"},
    {"id": "AUDIT-C-HIGH-1",
     "lens": "C (operational and resource integrity)",
     "issue": "The 35 GPU-hour per-model-identity ceiling and the "
              "storage gate, both re-ratified as hard halts, had no "
              "call site in the execution path. Nothing summed GPU "
              "hours across cells, so the ceiling could only ever have "
              "been checked by hand.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "run.per_identity_gate reads an append-only ledger and "
                 "is checked before every cell against its projected "
                 "cost and after it against measured hours, which are "
                 "charged; run.storage_gate is checked before every "
                 "cell; both halt through gate_halt. The gate "
                 "reproduces the published 32.960 h and fires between a "
                 "16 and 17 per cent overrun on the 250k rate.",
     "closed_by": "test_audit_known_negatives, ceiling checks"},
    {"id": "AUDIT-C-HIGH-2",
     "lens": "C (operational and resource integrity)",
     "issue": "G14_ROW_S 5.301 and R2_ROW_S 0.0291 were emitted under "
              "keys naming them MEASURED, but neither is traceable to "
              "any E8B artefact: the recorded G14 per-row costs are "
              "5.149/5.137/5.135 and no E8B R2 readout has ever run.",
     "classification": "provenance/reproducibility",
     "disposition": "FIXED BY DISCLOSURE",
     "evidence": "both are relabelled as ASSUMPTIONS with their real "
                 "basis stated (pooled 5.139 plus a 3 per cent margin; "
                 "R2 carried over from the E7b S8 per-row cost), and "
                 "their sensitivity is quantified.",
     "closed_by": "test_audit_known_negatives, rate-labelling checks"},
    {"id": "AUDIT-C-HIGH-3",
     "lens": "C (operational and resource integrity)",
     "issue": "The record claimed to cover the COMPLETE planned "
              "pretrained-identity programme while omitting the "
              "mandatory section-19 serial efficiency measurement, "
              "which loads the pretrained language model and is costed "
              "at 0.500-1.000 h.",
     "classification": "scientific-validity",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the pass is charged at its upper bound; the "
                 "pretrained identity rises from 31.960 to 32.960 h and "
                 "the headroom falls from 3.04 to 2.04 h. A correction "
                 "UPWARD against the ceiling.",
     "closed_by": "identity_reconciliation_20260807.json"},
    {"id": "AUDIT-A-HIGH-1",
     "lens": "A (scientific and provenance honesty)",
     "issue": "The anchor caveat claimed 'both files are tracked and "
              "the worktree was clean'. The checkpoint is git-ignored "
              "and has never been in git, so its hash is a first-and"
              "-only self-attestation, not a corroborated anchor.",
     "classification": "provenance/reproducibility",
     "disposition": "FIXED BY DISCLOSURE",
     "evidence": "the caveat now quotes its own erroneous sentence and "
                 "records the verified tracking status of each artefact "
                 "separately, stating what each anchor is and is not "
                 "worth. The tests now HASH both files instead of "
                 "string-matching the digests inside the record.",
     "closed_by": "test_audit_known_negatives, anchor verification"},
    {"id": "AUDIT-A-HIGH-2",
     "lens": "A (scientific and provenance honesty)",
     "issue": "determinism_probe_run1/run2/comparison record "
              "'warn_only': false and 'STRICT DETERMINISM CONFIRMED' "
              "for runs the project has itself established ran under "
              "warn_only=True, with no in-file marker; the correction "
              "existed only in an external map.",
     "classification": "provenance/reproducibility",
     "disposition": "FIXED BY DISCLOSURE",
     "evidence": "each file now carries a leading SUPERSEDED block "
                 "naming the withdrawn claim, the replacement "
                 "artefacts, the only reliable discriminator "
                 "(verified_at_use) and the fact that the hours remain "
                 "charged. Original content is unmodified below it.",
     "closed_by": "test_audit_known_negatives, supersession markers"},
]


def derive_summary(entries: list) -> dict:
    """Every count here is computed from entries[].disposition. Nothing
    is declared by hand, and no state is collapsed into another."""
    by_state: dict = {}
    buckets: dict = {}
    by_classification: dict = {}
    for entry in entries:
        state = STATE_MAP[entry["disposition"]]
        by_state[state] = by_state.get(state, 0) + 1
        buckets.setdefault(state, []).append(entry["id"])
        cls = entry["classification"]
        by_classification[cls] = by_classification.get(cls, 0) + 1
    not_closed = [e for e in entries
                  if STATE_MAP[e["disposition"]] != "FIXED_CLOSED"]
    return {
        "total": len(entries),
        "derivation": "computed mechanically from entries[].disposition "
                      "via the state map below; never declared by hand. "
                      "ACCEPTED LIMITATION is NOT counted as FIXED, and "
                      "PARTIALLY FIXED is NOT counted as CLOSED.",
        "state_map": STATE_MAP,
        "by_state": by_state,
        "by_classification": by_classification,
        "fixed_closed": buckets.get("FIXED_CLOSED", []),
        "accepted_limitation": buckets.get("ACCEPTED_LIMITATION", []),
        "partially_fixed": buckets.get("PARTIALLY_FIXED", []),
        "open": buckets.get("OPEN", []),
        "execution_critical_open": [
            e["id"] for e in entries
            if e["classification"] == "execution-critical"
            and STATE_MAP[e["disposition"]] in ("OPEN", "PARTIALLY_FIXED")],
        "not_fully_closed_any_class": [e["id"] for e in not_closed],
    }


def main() -> int:
    body = json.loads(LEDGER.read_text())["e8b_medium_ledger"]
    entries = list(body["entries"])
    known = {e["id"] for e in entries}
    pending = REVIEW_A_ENTRIES + AUDIT_20260807_ENTRIES
    appended = [e for e in pending if e["id"] not in known]
    for entry in appended:
        entry = dict(entry)
        entry["state"] = STATE_MAP[entry["disposition"]]
        entries.append(entry)
    for entry in entries:
        entry["state"] = STATE_MAP[entry["disposition"]]

    body["entries"] = entries
    body["summary"] = derive_summary(entries)
    body["additive_history"] = (
        "entries accumulate and are never rewritten or merged; the "
        "summary is recomputed from them on every run, so a hand-edited "
        "count cannot survive")
    record = {"metadata": utils.run_metadata(),
              "e8b_medium_ledger": body}
    if LEDGER.exists():
        LEDGER.unlink()
    LEDGER.write_text(json.dumps(record, indent=2, default=str) + "\n")

    summary = body["summary"]
    print(f"  entries          : {summary['total']} "
          f"({len(appended)} appended this run)")
    print(f"  by state         : {summary['by_state']}")
    print(f"  open             : {summary['open']}")
    print(f"  exec-critical open: {summary['execution_critical_open']}")
    return 0 if not summary["execution_critical_open"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
