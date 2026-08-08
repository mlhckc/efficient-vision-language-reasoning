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


# Findings from the RE-AUDIT of 94d8792, including one blocker that the
# ceiling fix itself introduced.
REAUDIT_20260807_ENTRIES = [
    {"id": "REAUDIT-B-BLOCKER-1",
     "lens": "B (executable safety, re-audit)",
     "issue": "The per-identity and storage gates were computed in "
              "train_core_cell but read in _train_core_locked, which "
              "does not receive them, so identity_gate and storage were "
              "unbound globals. Every core cell would have trained to "
              "completion, written its canonical checkpoint and per-row "
              "dump, then died with an uncaught NameError while "
              "assembling the result -- before the ledger charge and "
              "before the result was written. No result, no charge, no "
              "HALT and no FAILED record, and the re-run would then "
              "refuse at the existing npz. INTRODUCED BY THE FIX FOR "
              "AUDIT-C-HIGH-1.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the gate results are passed as parameters; asserted "
                 "by disassembling _train_core_locked and checking that "
                 "no gate name is read as LOAD_GLOBAL, and by asserting "
                 "none of them is a module global.",
     "closed_by": "test_audit_known_negatives, unbound-name checks"},
    {"id": "REAUDIT-B-HIGH-1",
     "lens": "B (executable safety, re-audit)",
     "issue": "charge_identity_hours was reachable only after every "
              "gate passed, and every gate_halt is a sys.exit, so the "
              "hours burned by a halted or crashed cell were never "
              "charged. A cell that ran 7.9 hours and tripped the wall "
              "would vanish from the ledger and the 35-hour ceiling "
              "would green-light the next cell on an identity that had "
              "already spent the headroom.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the charge moved into train_core_cell's finally "
                 "block, so it runs on every exit path, and the ledger "
                 "now records ONE ENTRY PER PROCESS rather than a "
                 "single overwritten total, which is what makes "
                 "charging on every path safe without double-counting a "
                 "resume.",
     "closed_by": "test_audit_known_negatives, per-process charging"},
    {"id": "REAUDIT-B-MEDIUM-1",
     "lens": "B (executable safety, re-audit)",
     "issue": "e8b_code_digest omitted src/reasoner.py, which defines "
              "the trunk architecture -- precisely the dirty-worktree "
              "splice the digest exists to catch, and the realistic "
              "case, since every 2026-08-07 record has git_dirty true.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the digest now covers all of src/, config.py and the "
                 "E8A modules the core path imports, not only the E8B "
                 "directory.",
     "closed_by": "test_audit_known_negatives, digest scope checks"},
    {"id": "REAUDIT-B-MEDIUM-2",
     "lens": "B (executable safety, re-audit)",
     "issue": "The resumed-cell wall carried only per-epoch train and "
              "eval time, so store hashing, the LM load and promotion, "
              "G8 and G14 pre-selection were bought back free on every "
              "crash-resume cycle. Ledger corruption raised uncaught "
              "exceptions with no halt artefact, and the ledger lived "
              "on node-local scratch while the duplicate-run lock "
              "deliberately does not.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the wall now reads every prior process's full wall "
                 "from the ledger; an unreadable or malformed ledger "
                 "REFUSES rather than assuming zero hours spent; and "
                 "the ledger moved to the shared filesystem beside the "
                 "execution locks.",
     "closed_by": "test_audit_known_negatives, ledger checks"},
    {"id": "REAUDIT-B-MEDIUM-3",
     "lens": "B (executable safety, re-audit)",
     "issue": "The 180 GPU-hour core ceiling was still not executable: "
              "remaining_core_gate had no call site outside tests, the "
              "identical defect that had just been closed for the "
              "35-hour ceiling.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "core_remaining_hours derives the remaining hours from "
                 "the governing projection minus everything charged to "
                 "the ledger, and train_core_cell halts on it before "
                 "any GPU work.",
     "closed_by": "test_audit_known_negatives, core ceiling checks"},
    {"id": "REAUDIT-B-LOW-1",
     "lens": "B (executable safety, re-audit)",
     "issue": "restore_resume_state mutated the model, optimizer, "
              "scheduler and all four RNG streams before raising on a "
              "legacy checkpoint, leaving a half-restored process; and "
              "the overfit gates verified determinism into a variable "
              "that was then discarded, so the verified state never "
              "reached any record.",
     "classification": "provenance/reproducibility",
     "disposition": "FIXED AT SOURCE",
     "evidence": "restore validates the full field set before touching "
                 "anything, and both gates record their verified "
                 "determinism block in the gate record.",
     "closed_by": "test_audit_known_negatives"},
]


# Findings from the honesty-and-resources re-audit of 94d8792.
REAUDIT_AC_20260807_ENTRIES = [
    {"id": "REAUDIT-AC-HIGH-1",
     "lens": "A/C (honesty and resources, re-audit)",
     "issue": "The fix for AUDIT-C-HIGH-2 replaced a mislabel with a "
              "FALSE PROVENANCE. It claimed R2 0.0291 s/row was carried "
              "over from the E7b S8-stage per-row cost. The E7b figure "
              "is S8_projection = 0.0291 MILLISECONDS, a "
              "LayerNorm-plus-Linear projection stage: a different "
              "operation in units 1000x apart. The matching digits are "
              "a coincidence, and R3 was derived from it.",
     "classification": "scientific-validity",
     "disposition": "FIXED BY DISCLOSURE",
     "evidence": "both rates are declared UNSOURCED, the false claim is "
                 "quoted and corrected in the generator and in the "
                 "reconciliation, and NO replacement provenance was "
                 "invented. The exposure is quantified and listed as an "
                 "OPEN RISK for the user to accept or close with a "
                 "measurement.",
     "closed_by": "test_audit_known_negatives, rate-provenance checks"},
    {"id": "REAUDIT-AC-HIGH-2",
     "lens": "A/C (honesty and resources, re-audit)",
     "issue": "The 'complete planned programme' claim was still false. "
              "Canonical plan section 6 requires every readout "
              "evaluation to run once over the 10,004-row RAW "
              "denominator and section 6.1 requires every final "
              "checkpoint to receive raw-distribution evaluation; the "
              "projection budgeted 7,714 in-vocabulary rows and HB3b "
              "was neither implemented nor budgeted. Section 7 applies "
              "three matched intervention conditions to EVERY trained "
              "checkpoint, B1 included, and B1's 18 intervention passes "
              "were omitted.",
     "classification": "scientific-validity",
     "disposition": "PARTIALLY FIXED",
     "evidence": "BUDGET ONLY. N_RAW = 10004 is charged for R2, R3 and "
                 "a raw-distribution R1 pass per cell, and B1 is "
                 "charged its three intervention conditions, so the "
                 "COSTING half is closed and the pretrained identity "
                 "rose accordingly. The IMPLEMENTATION half is NOT "
                 "done: N_RAW appears in no execution module, "
                 "intervention_inputs has no caller outside the tests, "
                 "and no code path performs a raw-denominator "
                 "evaluation, an R2/R3 final-checkpoint readout or an "
                 "intervention pass. An earlier disposition of FIXED AT "
                 "SOURCE was WRONG: the finding said 'neither "
                 "implemented nor budgeted' and only the budget was "
                 "addressed. This does not block the 18 training cells; "
                 "it blocks the branch from producing its primary "
                 "readout comparisons.",
     "closed_by": "NOT CLOSED - implementation work remains, and is "
                  "recorded as an open risk in "
                  "identity_reconciliation_20260807.json"},
    {"id": "REAUDIT-AC-MEDIUM-1",
     "lens": "A/C (honesty and resources, re-audit)",
     "issue": "The G14 per-row input was presented as conservative, but "
              "its measurement covers R1 brute force plus R1 cached "
              "only, while the live gate also runs r2_brute_force and "
              "r2_cached per row under clause C3. The 3.15 per cent "
              "margin does not cover two extra constrained walks over "
              "224 rows and 12 cells, so the input is optimistic.",
     "classification": "provenance/reproducibility",
     "disposition": "FIXED BY DISCLOSURE",
     "evidence": "the shortfall is stated in the generator and listed "
                 "as an OPEN, UNQUANTIFIED risk in the reconciliation "
                 "rather than described as a conservative margin.",
     "closed_by": "identity_reconciliation_20260807.json open_risks"},
    {"id": "REAUDIT-AC-MEDIUM-2",
     "lens": "A/C (honesty and resources, re-audit)",
     "issue": "A live record still certified the superseded 31.96 h / "
              "3.04 h headroom figure, and four further records the "
              "project's own map treats as superseded carried no "
              "in-file marker: u4_decision.json still reading PROMOTE, "
              "protocol_clarification still authorising grid points 2 "
              "to 8, g14_v2_preregistration, and preregistration.json "
              "carrying U4_OPEN.",
     "classification": "provenance/reproducibility",
     "disposition": "FIXED BY DISCLOSURE",
     "evidence": "all five carry in-place additive markers naming what "
                 "is superseded, what still stands and what replaced "
                 "it; original content is unmodified below each marker.",
     "closed_by": "test_audit_known_negatives, marker checks"},
    {"id": "REAUDIT-AC-MEDIUM-3",
     "lens": "A/C (honesty and resources, re-audit)",
     "issue": "The FP32 amendment's withdrawal block said 'the text "
              "below states', but both texts it corrects appear ABOVE "
              "it, so the correction misdirected the reader. Several "
              "quantified sensitivities were also internally "
              "inconsistent with their own carried-hours figures, and "
              "the container key measured_inputs held four entries that "
              "are not measurements.",
     "classification": "reporting/documentation",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the block states ABOVE and carries a placement note; "
                 "the sensitivities are computed from the same "
                 "constants they describe rather than hand-written; and "
                 "the container carries an explicit note that not every "
                 "entry is a measurement.",
     "closed_by": "the regenerated projection and reconciliation"},
]


# Findings from the executable-safety re-audit of 8ad573b.
REAUDIT_B2_20260807_ENTRIES = [
    {"id": "REAUDIT-B2-BLOCKER-1",
     "lens": "B (executable safety, second re-audit)",
     "issue": "A THIRD instance of the unbound-name defect, again "
              "introduced by the previous fix: moving the ledger charge "
              "into the caller's finally block left the trailing "
              "[LEDGER] print and the post-charge halt behind, still "
              "referencing a name that now lived in the caller. All 18 "
              "cells would have trained to completion, written every "
              "artefact, then died with NameError before returning.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the leftover block was removed and a bytecode binding "
                 "sweep over every function, method, nested function and "
                 "comprehension on the executable path now runs as a "
                 "test. It found this instance; reading the code had "
                 "missed the same class three times.",
     "closed_by": "test_no_unbound_names"},
    {"id": "REAUDIT-B2-HIGH-1",
     "lens": "B (executable safety, second re-audit)",
     "issue": "preflight() called save_resume_checkpoint without the "
              "four arguments that had become required two commits "
              "earlier, so --preflight would have died with a TypeError "
              "after minutes of GPU work. The same block would then "
              "have failed verification and restore for the same "
              "reason. No test executed preflight.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the call site passes protocol_family, history, "
                 "train_times and eval_times, and a static arity sweep "
                 "now binds EVERY internal call against its callee's "
                 "live signature.",
     "closed_by": "test_call_arity_everywhere"},
    {"id": "REAUDIT-B2-MEDIUM-1",
     "lens": "B (executable safety, second re-audit)",
     "issue": "The finally-block charge could destroy the failure that "
              "brought execution there: read_spend_ledger is "
              "fail-closed and raises `from None`, so an unreadable "
              "ledger would replace a CUDA OOM with a JSONDecodeError, "
              "leave the cell lock held, and write no artefact.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the lock is released FIRST, and every accounting step "
                 "is contained: an accounting failure writes a "
                 "G19_LEDGER halt record and is reported, never raised, "
                 "so the original failure survives.",
     "closed_by": "test_audit_known_negatives"},
    {"id": "REAUDIT-B2-MEDIUM-2",
     "lens": "B (executable safety, second re-audit)",
     "issue": "The code digest hashed every file in src/, the E8A "
              "directory and the E8B directory, most of which cannot "
              "affect a trajectory, and code_head equality was ALSO "
              "required. Any commit during a run -- including one to a "
              "report -- would have invalidated every outstanding "
              "resume, and since the ledger charges every process and "
              "never refunds, one crash plus one commit on a 250k cell "
              "could push the identity past its ceiling.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the digest covers a FIXED list of eleven trajectory "
                 "sources, including src/reasoner.py, and refuses if "
                 "any is absent rather than silently covering less than "
                 "it claims. The digest is the authority; a differing "
                 "code_head with byte-identical sources is reported and "
                 "recorded, not treated as a prohibition.",
     "closed_by": "test_audit_known_negatives, resume-contract checks"},
    {"id": "REAUDIT-B2-MEDIUM-3",
     "lens": "B (executable safety, second re-audit)",
     "issue": "charge_identity_hours was an unsynchronised "
              "read-modify-write to a shared ledger through a FIXED "
              "temporary filename, so two concurrent cells could drop "
              "each other's charge -- the exact under-count the ceiling "
              "exists to prevent.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the whole read-modify-write runs under an exclusive "
                 "ledger lock with a per-process staged filename, and "
                 "refuses rather than writing unsynchronised if the "
                 "lock is held too long. Verified with 24 concurrent "
                 "charges: none lost.",
     "closed_by": "test_audit_known_negatives"},
    {"id": "REAUDIT-B2-LOW-1",
     "lens": "B (executable safety, second re-audit)",
     "issue": "Two halting paths raised bare AssertionErrors without "
              "the halt artefact the protocol requires, and a resumed "
              "cell did not check its wall budget until the first "
              "training step, so an already-exhausted cell would re-run "
              "the model load, the parity load, G8 and G14 "
              "pre-selection before it could halt.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the core gate writes a halt record before raising, "
                 "and the wall is checked before the loop on a resumed "
                 "cell. CORRECTED 2026-08-07: this entry previously "
                 "claimed the check runs 'before the LM load, the "
                 "parity load, G8 and G14 pre-selection have run "
                 "again'. THAT WAS FALSE -- all of those run earlier in "
                 "the function and the check cannot precede them. It "
                 "saves one batch, and it is now SKIPPED entirely when "
                 "the loop will not run, so a completed 22-epoch "
                 "trajectory resumed for finalisation is not destroyed.",
     "closed_by": "the training and run modules"},
]


# Findings from the confirmation audit of d404891.
CONFIRM_20260807_ENTRIES = [
    {"id": "CONFIRM-HIGH-1",
     "lens": "A/C (confirmation audit)",
     "issue": "The false E7b provenance survived VERBATIM in "
              "identity_reconciliation's residual_assumptions, fifteen "
              "lines below the open_risks entry declaring it false, "
              "with no marker. Its exposure figures were also stale "
              "(computed on the superseded 7,714-row basis and the "
              "superseded headroom), understating the true exposure and "
              "inverting its sign: the correct increase at 3x is "
              "+2.428 h, which EXHAUSTS the headroom rather than "
              "leaving 0.17 h.",
     "classification": "scientific-validity",
     "disposition": "FIXED BY DISCLOSURE",
     "evidence": "residual_assumptions is rewritten: its first entry "
                 "withdraws both the E7b claim and the stale figures by "
                 "quoting them, and every remaining entry is derived "
                 "from the current basis. The correction commit had "
                 "fixed open_risks and missed this list entirely.",
     "closed_by": "test_audit_known_negatives, provenance-survival "
                  "checks"},
    {"id": "CONFIRM-HIGH-2",
     "lens": "A/C (confirmation audit)",
     "issue": "The 'complete planned programme' claim was false for a "
              "THIRD time. The identity substituted the MEASURED A1 "
              "core run (2.292 h) for protocol 13.2b's full A1 row "
              "(2.987 h), silently dropping 0.723 h of retained "
              "pretrained-135M work: the 7.1b secondary optimisation "
              "study, the bounded representation ablation and the "
              "middle-layer extraction. All three are retained in the "
              "operative protocol and none is descoped.",
     "classification": "scientific-validity",
     "disposition": "FIXED AT SOURCE",
     "evidence": "all three are charged to the pretrained identity and "
                 "the two that appear in the A1r row to the random one. "
                 "With the per-epoch rates also moved to the MAXIMUM of "
                 "their recorded sets, the pretrained identity rises "
                 "from 33.472 to 34.423 h and the headroom falls to "
                 "0.577 h.",
     "closed_by": "identity_reconciliation_20260807.json"},
    {"id": "CONFIRM-HIGH-3",
     "lens": "A/C (confirmation audit)",
     "issue": "REAUDIT-AC-HIGH-2 was disposed FIXED AT SOURCE on the "
              "BUDGET alone, though the finding read 'neither "
              "implemented nor budgeted'. No code path performs a "
              "raw-denominator evaluation, an R2/R3 final-checkpoint "
              "readout or an intervention pass, so the mechanically "
              "derived summary reported zero open findings while a "
              "mandatory evaluation remained unbuilt.",
     "classification": "scientific-validity",
     "disposition": "PARTIALLY FIXED",
     "evidence": "the entry is re-dispositioned to PARTIALLY FIXED, "
                 "which the ledger's own state map excludes from "
                 "CLOSED, and the implementation gap is recorded as an "
                 "open risk. The summary now reports one partially "
                 "fixed finding rather than none.",
     "closed_by": "NOT CLOSED - this entry records the correction of "
                  "the disposition, not the completion of the work"},
    {"id": "CONFIRM-MEDIUM-1",
     "lens": "A/C (confirmation audit)",
     "issue": "The 180-hour gate's scope note cited 'about 17.6-21.6 h "
              "expected per section 13.2' for the remaining E8A arms. "
              "That range appears nowhere in the protocol. An Aug-6 "
              "implementation-audit record also lacked a marker though "
              "several of its verdicts no longer describe HEAD, and "
              "CLAUDE.md still stated the core ceiling as the min() "
              "form that user decision P3 made non-halting.",
     "classification": "provenance/reproducibility",
     "disposition": "FIXED BY DISCLOSURE",
     "evidence": "the citation is corrected to the protocol's actual "
                 "figures with the spent hours deducted; the Aug-6 "
                 "record carries a marker distinguishing what is stale "
                 "from what still stands; and CLAUDE.md records P3's "
                 "clarification that the expected-epoch comparison "
                 "alone halts. RE-OPENED AND RE-CLOSED 2026-08-07: the "
                 "generator was corrected but the published artefact "
                 "was not regenerated, so the fabricated citation "
                 "stayed live in the governing record while this entry "
                 "reported it closed. The artefact is regenerated and "
                 "a test now asserts it reproduces from its own "
                 "generator.",
     "closed_by": "the corrected records and CLAUDE.md"},
]


# Findings from the executable-safety confirmation audit of d404891.
CONFIRM_B_20260807_ENTRIES = [
    {"id": "CONFIRM-B-MEDIUM-1",
     "lens": "B (executable safety, confirmation audit)",
     "issue": "The M-1 fix traded exception masking for resume "
              "blocking. An accounting failure wrote "
              "HALT_{run}_G19_LEDGER.json, and train_core_cell refuses "
              "any cell with a HALT record, so a transient shared "
              "-filesystem error during the finally block turned a "
              "resumable trajectory into one needing manual repair -- "
              "at 4.2 h a restart against 0.6 h of headroom.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "an accounting failure now writes LEDGER_FAILURE_*, "
                 "not HALT_*: an operational warning about the ledger, "
                 "not a scientific halt about the cell. It is still "
                 "recorded durably and still says the ceiling is "
                 "under-counted until repaired, but it no longer blocks "
                 "resume.",
     "closed_by": "test_audit_known_negatives, halt-semantics checks"},
    {"id": "CONFIRM-B-MEDIUM-2",
     "lens": "B (executable safety, confirmation audit)",
     "issue": "The pre-loop wall check destroyed a completed cell. A "
              "cell resumed at epoch 22 skips the loop and needs only "
              "finalising, but if its accumulated hours exceeded the "
              "wall the check wrote both a FAILED record and a halt "
              "record, so a complete 22-epoch trajectory was killed and "
              "could never be re-entered, with no canonical checkpoint "
              "ever written. The comment justifying the check was also "
              "factually wrong about what it precedes.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the check is skipped when start_epoch exceeds "
                 "max_epochs, and the false comment is corrected in "
                 "place by quoting it.",
     "closed_by": "test_audit_known_negatives, resume-at-final checks"},
    {"id": "CONFIRM-B-MEDIUM-3",
     "lens": "B (executable safety, confirmation audit)",
     "issue": "A fired 35-hour ceiling on the completion path stamped a "
              "COMPLETED, valid cell with a halt record reading "
              "'FAILED: halting gate fired; this run's result is not "
              "used', contradicting the cell's own valid result, and "
              "returned exit code 0 so a driver would read success. "
              "Replacing the sys.exit was correct in a finally block "
              "but was done unconditionally and undocumented.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the ceiling now writes IDENTITY_EXHAUSTED_{identity}, "
                 "which records that the IDENTITY is exhausted without "
                 "asserting anything about this cell, and exits "
                 "non-zero only when sys.exc_info() shows no exception "
                 "already propagating.",
     "closed_by": "test_audit_known_negatives, ceiling-semantics checks"},
    {"id": "CONFIRM-B-MEDIUM-4",
     "lens": "B (executable safety, confirmation audit)",
     "issue": "Nothing sequenced B2 and B3 for a given scale and seed. "
              "With under an hour of headroom, running the six B2 cells "
              "first could exhaust the pretrained ceiling partway "
              "through B3 and leave unpaired B2 results -- breaking a "
              "pairing CLAUDE.md declares binding, by accident rather "
              "than by decision. The halt message promised "
              "'pair-preserving alternatives'; nothing computed any.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "run.pair_preserving_order interleaves each pair so "
                 "that if the ceiling fires it fires BETWEEN pairs and "
                 "every completed pair is whole. B1, charged to neither "
                 "identity ceiling, runs last. CORRECTED 2026-08-07: "
                 "this entry originally said 'B2 immediately before its "
                 "B3'. That ordering was BACKWARDS -- the binding "
                 "ceiling is the pretrained one B3 charges, so a firing "
                 "at B3's pre-gate would have stranded the B2 that had "
                 "just completed. B3 now runs first.",
     "closed_by": "test_audit_known_negatives, pair-order checks"},
    {"id": "CONFIRM-B-MEDIUM-5",
     "lens": "B (executable safety, confirmation audit)",
     "issue": "assert_promotable, the guard against promoting a "
              "non-scientific probe artefact or a superseded search "
              "record, had NO production call site -- the same "
              "'constant with no caller' defect closed earlier for the "
              "35-hour and 180-hour ceilings. Peak memory was also "
              "measured only across the training loop, excluding the "
              "gates and so excluding the one combination never "
              "measured anywhere: the fp32-promoted model with a "
              "non-autocast G7 backward.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "run.load_core_result is now THE sanctioned reader and "
                 "calls the guard; it has no caller today only because "
                 "no aggregation script exists (F1 is unauthorised), "
                 "and that is stated in its docstring. Peak memory is "
                 "reset once at the start of the cell, so the gates are "
                 "inside the measurement.",
     "closed_by": "test_audit_known_negatives"},
    {"id": "CONFIRM-B-LOW-1",
     "lens": "B (executable safety, confirmation audit)",
     "issue": "src/models.py was in TRAJECTORY_SOURCES but is imported "
              "nowhere on the E8B path, so an unrelated V1/V2 edit "
              "would have invalidated every outstanding E8B resume -- "
              "the exact harm the fixed list exists to prevent. "
              "reinfer_g21.py, which DOES execute on the core path and "
              "enforces GPU exclusivity, was absent.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "models.py removed with the reason recorded in place, "
                 "reinfer_g21.py added.",
     "closed_by": "test_audit_known_negatives, digest-scope checks"},
]


# Findings from the final audits of 28afec5.
FINAL_20260807_ENTRIES = [
    {"id": "FINAL-B-HIGH-1",
     "lens": "B (executable safety, final audit)",
     "issue": "A fix reached the generator but never the artefact. The "
              "fabricated '17.6-21.6 h' citation was corrected in "
              "core_resource_projection.py, but the published JSON was "
              "not regenerated, so the governing 180-hour record still "
              "carried the fabricated citation as a live statement "
              "while the ledger reported it FIXED. The artefact no "
              "longer reproduced from the script its own docstring "
              "promises it reproduces from.",
     "classification": "provenance/reproducibility",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the artefact is regenerated, and a test asserts that "
                 "the projection AND the findings ledger each reproduce "
                 "from their committed generators, field for field. "
                 "SCOPE, stated exactly: two records -- "
                 "identity_reconciliation_20260807.json and "
                 "blocker_high_reverification_20260807.json -- have NO "
                 "generator at all and are maintained by hand, so no "
                 "reproduction test can cover them. That is a real "
                 "residual gap, not a closed one, and it is recorded as "
                 "such rather than papered over by a broader claim than "
                 "the guard supports.",
     "closed_by": "test_records_reproduce_from_generators"},
    {"id": "FINAL-AC-HIGH-1",
     "lens": "A/C (honesty, final audit)",
     "issue": "The largest residual assumption's break-even was stale "
              "and understated the risk by about 2.6 times: it quoted "
              "12 per cent per cell and 17 per cent on the training "
              "rate, both computed verbatim on the superseded 4.189 h "
              "basis, while the per-cell figure at the time was 4.226 h (superseded again by measurement on 2026-08-07). The "
              "whole sensitivity block was hardcoded inside a generator "
              "whose docstring claims it recomputes every figure.",
     "classification": "scientific-validity",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the entire block is now DERIVED from the same "
                 "constants as the projection. The true break-evens are "
                 "4.55 per cent per cell and 5.24 per cent on the "
                 "training rate.",
     "closed_by": "the regenerated projection"},
    {"id": "FINAL-AC-HIGH-2",
     "lens": "A/C (honesty, final audit)",
     "issue": "The claim 'A1 came in 23 per cent BELOW its projection', "
              "offered as the sole reassurance for the 3.668 h A4/A7c "
              "residual, was the EXACT mis-comparison the same file "
              "charges 0.723 h to correct: it divided a partial "
              "measurement by the full protocol row, treating unrun "
              "components as having cost nothing.",
     "classification": "scientific-validity",
     "disposition": "FIXED AT SOURCE",
     "evidence": "like for like, A1 came in 1.25 per cent ABOVE "
                 "projection and its extraction component overran by "
                 "82.4 per cent. The precedent is recorded as mildly "
                 "UNFAVOURABLE, and the error is quoted where it was "
                 "made.",
     "closed_by": "the regenerated projection"},
    {"id": "FINAL-AC-HIGH-3",
     "lens": "A/C (honesty, final audit)",
     "issue": "The published projection called the G14 margin "
              "'conservative' while its own generator and the risk "
              "register both call it OPTIMISTIC. A withdrawn "
              "characterisation surviving in the governing artefact.",
     "classification": "scientific-validity",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the artefact now carries an explicit direction field "
                 "reading OPTIMISTIC, NOT CONSERVATIVE, with the reason "
                 "(the measurement covers the two R1 scorers only, "
                 "while the live gate also runs both R2 walks).",
     "closed_by": "the regenerated projection"},
    {"id": "FINAL-AC-HIGH-4",
     "lens": "A/C (honesty, final audit)",
     "issue": "Superseded identity figures survived in the very markers "
              "added to fix superseded figures: 33.472 h / 1.528 h in "
              "the reverification record's 'current' line, a gate "
              "verification citing 33.4747 against 33.472, "
              "net_change_hours 0.786 from the previous round, and "
              "coverage lines still quoting 108.0 s and 7,714 rows.",
     "classification": "provenance/reproducibility",
     "disposition": "FIXED AT SOURCE",
     "evidence": "every one is recomputed from the governing "
                 "projection, and the reverification line now records "
                 "that it has itself been corrected twice, always "
                 "upward.",
     "closed_by": "the regenerated records"},
    {"id": "FINAL-AC-HIGH-5",
     "lens": "A/C (honesty, final audit)",
     "issue": "The selection-sensitivity study was REMOVED, on stated "
              "and defensible grounds, while the certifying record said "
              "'NOTHING was descoped - every correction ADDED work'. "
              "Reinstating it costs several times the remaining "
              "headroom, and that number appeared nowhere.",
     "classification": "scientific-validity",
     "disposition": "FIXED BY DISCLOSURE",
     "evidence": "both records now state plainly that this one "
                 "component was removed, cost it at 4.451 h if "
                 "reinstated, and explain that 'nothing descoped' "
                 "refers to core arms and cells. The two statements can "
                 "no longer be read as contradicting each other.",
     "closed_by": "identity_reconciliation_20260807.json"},
    {"id": "FINAL-B-MEDIUM-1",
     "lens": "B (executable safety, final audit)",
     "issue": "Three residual ways a bad day became unrecoverable: an "
              "uncontained ledger read AFTER the per-row dump was "
              "written but before the result was, which would strand a "
              "completed cell behind its own npz guard; the lock unlink "
              "sitting OUTSIDE the containment, where one stale NFS "
              "handle would leak the lock, skip the charge and mask the "
              "failure at once; and nothing anywhere READING a "
              "LEDGER_FAILURE record, so the under-count it warns about "
              "was never enforced.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the final ledger read is contained and degrades to a "
                 "recorded 'unavailable' field; the unlink moved inside "
                 "the containment; and train_core_cell now refuses to "
                 "start while any LEDGER_FAILURE or matching "
                 "IDENTITY_EXHAUSTED record exists.",
     "closed_by": "test_audit_known_negatives"},
    {"id": "FINAL-B-MEDIUM-2",
     "lens": "B (executable safety, final audit)",
     "issue": "pair_preserving_order ran B2 before B3, but the BINDING "
              "ceiling is the pretrained one that B3 charges. A firing "
              "at B3's pre-gate would therefore strand the B2 that had "
              "just completed - the exact outcome the function claims "
              "to prevent. The 8-hour wall was also removed entirely "
              "from the finalisation path rather than relaxed.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the pair runs B3 FIRST, so a firing leaves nothing "
                 "stranded: the pair either both run or neither does. "
                 "Finalisation gets a bounded one-hour allowance on top "
                 "of the wall rather than no wall at all, and "
                 "exceeding it still halts with a recorded failure "
                 "status.",
     "closed_by": "test_audit_known_negatives, pair-order checks"},
    {"id": "FINAL-AC-MEDIUM-1",
     "lens": "A/C (honesty, final audit)",
     "issue": "The per-identity gate summed only spent, committed and "
              "THIS cell, so a rate overrun in cell 1 of 6 was not "
              "detected until cell 6 - after most of the arm's hours "
              "were already burned and unrecoverable.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the gate now RESERVES the cells that have not run "
                 "yet, so detection scales with severity: a sustained "
                 "20 per cent overrun stops at B3 cell 3 having burned "
                 "4.1 h of a planned 17.8 h, where before it would have "
                 "run to cell 6 regardless.",
     "closed_by": "test_audit_known_negatives, reservation checks"},
]


# Findings from the verification audit of b27696e.
VERIFY_20260807_ENTRIES = [
    {"id": "VERIFY-HIGH-1",
     "lens": "verification audit",
     "issue": "The closure evidence for FINAL-B-HIGH-1 claimed a test "
              "asserts EVERY published record reproduces from its "
              "generator. The test covered one record. Two governing "
              "records have no generator at all and are hand-edited, so "
              "the claim was broader than any guard could be -- the "
              "same over-claiming this project has repeatedly had to "
              "correct.",
     "classification": "provenance/reproducibility",
     "disposition": "PARTIALLY FIXED",
     "evidence": "the test now covers the projection AND the findings "
                 "ledger, and the evidence states the scope exactly, "
                 "naming the two hand-maintained records it cannot "
                 "cover. Those two remain a real residual gap.",
     "closed_by": "NOT CLOSED - two records remain unguarded by "
                  "construction, and that is now stated rather than "
                  "hidden behind a broader claim"},
    {"id": "VERIFY-HIGH-2",
     "lens": "verification audit",
     "issue": "The commit that fixed the counting claims broke two of "
              "its own: the reconciliation said 'Five separate "
              "omissions' above a list of four, and 'Five open risks "
              "... ANY ONE of them exhausts the remaining margin' above "
              "a list of six, two of which are not hours risks at all.",
     "classification": "reporting/documentation",
     "disposition": "FIXED AT SOURCE",
     "evidence": "both counts are now DERIVED from the lists they "
                 "describe rather than written beside them, and the "
                 "verdict distinguishes the risks that can exhaust the "
                 "margin from those that cannot.",
     "closed_by": "the regenerated reconciliation"},
    {"id": "VERIFY-MEDIUM-1",
     "lens": "verification audit",
     "issue": "The reservation skipped any cell with ANY ledger entry, "
              "so a cell that crashed after six minutes had its whole "
              "projected cost dropped from the reservation. A 250k cell "
              "that OOMed early would leave every later pre-gate "
              "reading 4.2 h of headroom that does not exist -- "
              "restoring the 'not detected until cell 6' behaviour the "
              "reservation was added to fix.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the gate reserves max(0, projected minus hours "
                 "already charged to that cell), so a crashed cell is "
                 "still reserved for the work it has left to do. Also "
                 "removed a module-level 'current cell' global that "
                 "could have leaked between calls in one process.",
     "closed_by": "test_audit_known_negatives, reservation checks"},
    {"id": "VERIFY-MEDIUM-2",
     "lens": "verification audit",
     "issue": "The A1 precedent was corrected once and was STILL not "
              "like for like: '1.25 per cent ABOVE' credited an "
              "interventions component the measurement excludes. The "
              "measured 2.29222 h decomposes exactly as training "
              "1.56262 plus extraction 0.729599, and the source record "
              "states interventions are reported separately and never "
              "added. Two successive framings each picked a more "
              "favourable comparator than the evidence supports.",
     "classification": "scientific-validity",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the comparator is now core 1.774 plus extraction "
                 "0.400, the components the measurement actually "
                 "covers, giving 5.44 per cent ABOVE. Both earlier "
                 "framings are quoted and withdrawn in place, and the "
                 "precedent is stated as UNFAVOURABLE.",
     "closed_by": "the regenerated projection"},
    {"id": "VERIFY-MEDIUM-3",
     "lens": "verification audit",
     "issue": "One mandatory component was still uncosted after the "
              "list was declared complete: A1's own matched "
              "interventions, 0.090 h in protocol 13.2b's A1 row. BASE "
              "covers core and extraction only, and the retained-row "
              "constant covered the other three components but not "
              "this one.",
     "classification": "scientific-validity",
     "disposition": "FIXED AT SOURCE",
     "evidence": "charged for both A1 and A1r. The pretrained identity "
                 "rises from 34.423 to 34.513 h and the headroom falls "
                 "to 0.487 h. This is the FOURTH omission found and the "
                 "fourth correction upward.",
     "closed_by": "the regenerated projection"},
    {"id": "VERIFY-MEDIUM-4",
     "lens": "verification audit",
     "issue": "Two operator-facing claims were falsified by code added "
              "in the same commit: the ledger-failure message said "
              "'this cell can still resume' while a new pre-check "
              "stopped every cell including that one, and the "
              "finalisation allowance was tested once on entry while "
              "its comment claimed exceeding it is recorded -- the only "
              "wall check lives inside a loop that does not run on that "
              "path.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the message now says plainly that no cell starts "
                 "while the record exists and how to clear it, and "
                 "finalisation carries a real deadline that records a "
                 "G19_FINALISATION overrun after the result is safely "
                 "written.",
     "closed_by": "test_audit_known_negatives"},
    {"id": "VERIFY-MEDIUM-5",
     "lens": "verification audit",
     "issue": "pair_preserving_order still had no production call site, "
              "so the documented ordering was whatever the operator "
              "typed, and an entry closing it described the superseded "
              "B2-first ordering.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "run.py --core-order prints the sequence, and "
                 "--core-cell warns when a cell is run ahead of "
                 "earlier ones in the order. The stale entry text is "
                 "corrected in place by quotation.",
     "closed_by": "test_audit_known_negatives, pair-order checks"},

    # --- Final operational closure, 2026-08-08. Two fresh-context
    # reviews of the corrected state; the first REJECTED it. Recorded
    # here so this ledger's "0 open" summary stays truthful.
    {"id": "CLOSURE-BLOCKER-1",
     "lens": "C (fresh-context narrow review, 2026-08-08)",
     "issue": "One governed quantity was published as two numbers, in "
              "three places at once. The enforceable recovery margin "
              "was 4.195 in run.py and 4.197 in "
              "identity_reconciliation.py, which re-derived it from the "
              "projection's headroom on a different baseline; the "
              "reconciliation record carried 4.197 in its verdict and "
              "all six judged_against fields while carrying 4.195 three "
              "lines away. The baseline was 34.803 in the projection "
              "and 34.805 in the gate, and the rounding tolerance was "
              "0.005 in code against 0.01 in the amendment record. A "
              "comment in run.py described this defect as already "
              "fixed; it had been fixed on one side only.",
     "classification": "execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the two baselines are real and are now named apart -- "
                 "MEASURED_PROGRAMME_HOURS (34.803, component sum) and "
                 "BASELINE_BUDGET_HOURS (34.805, what the gate sums "
                 "from per-cell constants rounded to 3dp). The gate "
                 "basis governs and every consumer derives from it. No "
                 "authorised value moved and the B3 recipe hash is "
                 "byte-identical.",
     "closed_by": "test_final_operational_closure, the record audit "
                  "over 15 governed fields in 37 records"},
    {"id": "CLOSURE-HIGH-1",
     "lens": "C (fresh-context narrow review, 2026-08-08)",
     "issue": "The guard against the withdrawn retry guarantee was "
              "near-vacuous. Its exemption asked whether 'withdraw' "
              "appeared ANYWHERE in the file, so 5 of its 7 targets "
              "were wholly exempt and a fresh guarantee could be "
              "reintroduced in them and still pass. It scanned a "
              "hand-listed four sources and three records, and not the "
              "test file, where the withdrawn wording was in fact still "
              "standing as current fact.",
     "classification": "not execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the exemption is per-occurrence within 300 "
                 "characters; the surface is every e8b source, every "
                 "record, this test file, docs/, collab/, CLAUDE.md and "
                 "AGENTS.md; both self-scan markers are asserted "
                 "present; and two synthetic cases prove the guard can "
                 "fail and can still exempt a genuine local withdrawal.",
     "closed_by": "test_governance_amendment, synthetic offend/exempt "
                  "pair"},
    {"id": "CLOSURE-MEDIUM-1",
     "lens": "C (fresh-context narrow review, 2026-08-08)",
     "issue": "training.py stored the previous SIGTERM/SIGINT/SIGHUP "
              "handlers in a dict that nothing ever read, beside a "
              "comment promising they were restored at the end.",
     "classification": "not execution-critical",
     "disposition": "FIXED BY DISCLOSURE",
     "evidence": "not restoring is correct and measured: restoring "
                 "before the charge leaves a window in which a second "
                 "SIGTERM kills the process with nothing written, "
                 "because the charge can wait up to 30 s on the ledger "
                 "lock. The dead state is removed and the residual -- "
                 "the handler outlives the call -- is disclosed.",
     "closed_by": "test_final_operational_closure"},
    {"id": "CLOSURE-MEDIUM-2",
     "lens": "D (confirmation review of the remediation, 2026-08-08)",
     "issue": "The BLOCKER fix MOVED the mislabel rather than removing "
              "it. The reconciliation's concluding verdict came to read "
              "'gated against the 34.805 h measured baseline' -- the "
              "right number under the wrong label, where before it had "
              "been the wrong number under the right one. In the same "
              "record the note 'the margin is the headroom less the "
              "floor' no longer closed, because headroom_hours is "
              "measured-basis: 5.197 - 1.0 = 4.197, not the published "
              "4.195.",
     "classification": "not execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "the verdict now names the BASELINE BUDGET and "
                 "contrasts it with the measured component sum; both "
                 "headrooms are published and named "
                 "(headroom_gate_basis_hours 5.195), and the note "
                 "states the rule on the gate basis so it closes "
                 "exactly, while explaining why the measured basis "
                 "gives 4.197.",
     "closed_by": "test_final_operational_closure"},
    {"id": "CLOSURE-MEDIUM-3",
     "lens": "D (confirmation review of the remediation, 2026-08-08)",
     "issue": "Nine '35 GPU-hour / 35-hour ceiling' statements survived "
              "in LIVE source after the ceiling became 40 h, five of "
              "them present tense beside the code enforcing 40 -- "
              "including per_identity_gate's own docstring and the "
              "operator-facing message shown when the ledger is "
              "unreadable and execution refuses. The earlier sweep "
              "claimed to have covered this and had scanned records "
              "only.",
     "classification": "not execution-critical",
     "disposition": "FIXED AT SOURCE",
     "evidence": "every present-tense mention now names the constant or "
                 "the ceiling generically; the gate docstring names no "
                 "figure at all; the refusal message interpolates "
                 "PER_IDENTITY_CEILING_HOURS and is verified by "
                 "RENDERING it, not by grepping. An exact per-file "
                 "allowlist pins the surviving historical mentions so "
                 "any new one fails.",
     "closed_by": "test_final_operational_closure, stale-ceiling "
                  "allowlist and rendered-refusal check"},
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
    pending = (REVIEW_A_ENTRIES + AUDIT_20260807_ENTRIES
               + REAUDIT_20260807_ENTRIES
               + REAUDIT_AC_20260807_ENTRIES
               + REAUDIT_B2_20260807_ENTRIES
               + CONFIRM_20260807_ENTRIES
               + CONFIRM_B_20260807_ENTRIES
               + FINAL_20260807_ENTRIES
               + VERIFY_20260807_ENTRIES)
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
    # M5. The metadata stamp names the tree this record was GENERATED
    # from, which is necessarily the tree before the commit that carries
    # it -- a record cannot know its own commit. Chasing the stamp by
    # regenerating after each commit only moves the lag by one. Stated
    # rather than chased: this is a NARRATIVE governance record, not an
    # attestation that a particular code state passed a check. Records
    # that DO attest to a code state (final_preflight) are re-stamped on
    # a clean tree so the only later change is the record itself.
    body["provenance_note"] = (
        "narrative governance record. Its metadata stamps the tree it "
        "was generated from, not the commit that carries it. It is not "
        "an attestation that a code state passed a check.")
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
