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
    appended = [e for e in REVIEW_A_ENTRIES if e["id"] not in known]
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
