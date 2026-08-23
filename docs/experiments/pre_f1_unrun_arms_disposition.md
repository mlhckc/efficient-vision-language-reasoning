# Pre-F1 disposition of the unexecuted E8A arms

## Purpose

Record the explicit user decision of 23 August 2026 that the seven
unexecuted E8A arms — A2, A2r, A4, A5, A7c, A8c and AF (the
FLAN-T5-small sensitivity point) — are retained as authorised-but-unrun
and are deferred from the current MSc execution scope, before F1 and
before F2. The decision supersedes only the operational requirement to
execute these arms within the current MSc pre-F1 scope. It withdraws
nothing, cancels nothing, authorises no training, reopens no arm,
declares no arm failed or scientifically disproven, and changes no
scientific claim, number, status or model selection.

## Method

The decision rests on a read-only reconstruction of the complete
authorization chain, performed on 23 August 2026 at commit c151118.
Sources: the verbatim user-decision records under
.agent-bridge/tasks/_shared/ (the 2 August 2026 programme authorization,
the 19-decision message, and the round-05 through round-08 decision
records), the canonical E8A plan
(.agent-bridge/tasks/e8a-question-encoder/10-canonical-plan.md, section
2), the canonical master protocol
(.agent-bridge/tasks/_shared/e8-master-protocol-canonical.md), the
tracked mirrors in CLAUDE.md and collab/PROJECT_CONTEXT.md, the E10
Phase-0 record (docs/experiments/e10_phase0_calibration.md), the
canonical claim and evidence layer (results/closure/,
results/ve0/rq_evidence_matrix.json) and a scoped execution/result audit
of the results tree. Repository scans were scoped to named non-data
directories; the clean-test area was not read, listed or traversed.

## Outputs

- docs/experiments/pre_f1_unrun_arms_disposition.md (this record);
- a dated deferral note in CLAUDE.md (gate section and locked-scope
  section) and in collab/PROJECT_CONTEXT.md, each appended after the
  historical sentences it qualifies, which are preserved verbatim;
- an updated bullet in docs/PROJECT_STATUS.md replacing the former
  "open user decision" wording.

No results/closure/ record is created: this disposition controls
execution scope, a lifecycle decision; it carries no field override,
digest or numerical value that a machine resolver would query, and field
validity continues to resolve through the existing authorities
unchanged. No historical or verbatim authorization record is edited.

## Results

### Authorization history, preserved

1. All seven arms were authorised. The user's 2 August 2026 message
   authorised the staged execution of the broader P2607 SLM-alignment
   programme; the user's own A0-A8 arm list is recorded verbatim, and
   the canonical E8A plan carries the full run matrix (A0p, A1, A1r,
   A2, A2r, A4, A5, A7c, A8c, AF; six cells per run arm) with the
   sentence "No result-dependent trigger governs any arm". AF was a
   planner-added cross-family sensitivity point, bound by the user's
   decision to the semantic-encoder sensitivity role only, and is by
   its own protocol wording a multi-factor sensitivity point, not a
   causal comparator.
2. A5 and A8c were later explicitly made unconditional core. The
   original 2 August wording described the 360M controls as
   "conditionally authorised"; the user's round-05 Decision A, recorded
   verbatim, states "I explicitly authorise both A5 and A8c as
   unconditional core controls", and round-06 reaffirms "A5 and A8c
   remain unconditional core". The tracked mirrors carry this as the
   2 August 2026 decision.
3. The 10 August 2026 E10 Phase-0 record re-deferred A5/A8c behind E10
   and explicitly did not cancel or supersede them: "A5 and A8c remain
   scientifically owed. They are outside E10, re-deferred behind it,
   and are neither cancelled nor superseded."
4. A2/A2r remained authorised but insufficiently defined on tracked
   binding surfaces. The same Phase-0 record states "A2/A2r remain
   unresolved because the binding text does not define them
   sufficiently; no definition was invented, they are not cancelled."
   Their full definition exists in the gitignored canonical plan and
   master protocol; the tracked binding surfaces named them only in the
   pair-preservation clause. Section "A2/A2r definition" below closes
   that gap by transcription.
5. None of the seven arms was executed. No canonical or accepted
   experimental result exists for any of them, and the scoped
   execution/result audit found no execution artefact for them: the
   E8A results tree (results/experiments/e8a_question_encoder/) holds
   checkpoints, predictions, per-row correctness vectors and analysis
   records for A0p, A1 and A1r only, and the canonical analysis records
   name only those three arms.

Nothing in this record implies that any of the seven arms was never
planned, never authorised, obsolete, failed, or scientifically
disproven.

### The decision of 23 August 2026

By explicit user decision, each of the seven arms takes the status:

    authorised-but-unrun; deferred from the current MSc execution scope

| Arm | Role (canonical plan) | Status |
|---|---|---|
| A2 | pretrained SmolLM2-360M question-token reasoner encoder | authorised-but-unrun; deferred from the current MSc execution scope |
| A2r | random-initialised 360M twin; inseparable pair with A2 | authorised-but-unrun; deferred from the current MSc execution scope |
| A4 | SmolLM2-135M question-only head (v2_02 recipe) | authorised-but-unrun; deferred from the current MSc execution scope |
| A5 | SmolLM2-360M question-only head (v2_02 recipe) | authorised-but-unrun; deferred from the current MSc execution scope |
| A7c | CLIP image ++ projected pooled 135M, concat (v2_02 recipe) | authorised-but-unrun; deferred from the current MSc execution scope |
| A8c | CLIP image ++ projected pooled 360M, concat (v2_02 recipe) | authorised-but-unrun; deferred from the current MSc execution scope |
| AF | FLAN-T5-small cross-family sensitivity point | authorised-but-unrun; deferred from the current MSc execution scope |

For A5 and A8c this is a new explicit user decision that supersedes
only the operational execution obligation of their unconditional-core
status, for the current MSc scope. The earlier authorization was not
erroneous and its record is preserved verbatim where it stands. The
canonical protocol's own rules — that removal of a core pair requires a
new user decision and that no core arm is ever descoped automatically —
are satisfied, not bypassed: this record is that new user decision,
taken after the read-only analysis was returned to the user.

### Not outcome-selected

No canonical or accepted experimental result exists for any of these
seven arms, and the scoped execution/result audit found no execution
artefact for them. This decision is made before F1 and before F2, the
blinded held-out evaluation. The pre-F1 deferral decision is therefore
not based on observing their experimental outcomes.

### Scientific justification

The primary justification is scientific:

- Every current question-side RQ3 claim is explicitly bounded to the
  executed evidence. Claim C08's recorded limitation states that
  A1 minus A1r "is the only clean within-size question-side pretraining
  contrast", and the findings summary bounds the question-side
  pretraining result to pretrained-versus-random at fixed size.
- A0p, A1 and A1r provide the executed, matched 135M question-side
  comparison; B2/B3 (E8B) and B4/B4r (E10) provide the executed
  answer-side evidence, including the 360M capacity sensitivity.
- The canonical evidence layer confirms no dependency: the VE-0
  research-question matrix marks the frozen-SLM interface question
  ANSWERED on the executed evidence, and none of its partially answered
  questions cites an unexecuted arm.
- The seven arms therefore answer extension and completeness questions;
  none is an unresolved validity requirement for a claim actually made.

Secondarily, executing them now would reopen closed evidence, review
and reporting layers (VE-0 and VE-1 are closed and byte-pinned) and
expand late MSc scope, at material schedule risk to F1, F2 and the
dissertation.

No current RQ3 scientific claim is broadened or changed by this
disposition.

### A2/A2r definition, tracked transcription

This is a tracked transcription of the pre-existing canonical
definition for governance and reproducibility; it is not a new
experimental design, new authorization, or new execution instruction.
The definition below is verified in the gitignored canonical E8A plan
(section 2) and canonical master protocol; it was previously missing
from tracked binding surfaces, which is the gap the 10 August 2026
Phase-0 record named. Nothing is invented or modified.

- A2: frozen pretrained SmolLM2-360M base (pinned revision
  f8027fd0eaeea54caa13c31d31b9fdc459c38b49) as the E8A question-token
  encoder, feeding the latent-query reasoner through one trainable
  Linear(960->512) projection (492,032 parameters), under the inherited
  section-7.1 (v3_01) recipe; six cells: train_40k and train_250k at
  seeds 0/1/2. Planned primary contrasts: A2 - A2r (causal), A2 - A1
  (size), A2 - A0p (system).
- A2r: the architecture- and tokenizer-identical random-initialised
  SmolLM2-360M control, constructed from the pretrained model's own
  configuration with from_config, weights initialised with
  utils.set_seed(20260802) immediately before construction, the
  pretrained tokenizer reused unchanged, every parameter frozen and no
  pretrained weights loaded; the same projection, recipe and six cells.
- A2 and A2r are an inseparable pair: neither runs at a scale where the
  other does not. Pair integrity binds any future reopening.

### Reopening

Before F1, an explicit user authorization — including authorization
made in response to supervisor feedback — may reopen an arm or required
pair. Supervisor feedback may motivate reopening but is not itself
execution authorization. Any reopening preserves pair integrity,
in particular A2 with A2r. If F1 has already occurred, reopening
development work would additionally require the applicable freeze and
supersession governance before any later F2 use. Nothing is reopened or
authorised by this record.

### Publication extension

A2/A2r remain a natural high-value post-MSc publication extension
because they would test whether the question-side pretraining effect
persists at 360M; this does not create a current MSc execution
obligation.

## Decisions and problems

- Location: docs/experiments/ was chosen; no results/closure/ sibling
  was created, for the reasons in Outputs.
- The verbatim authorization records under .agent-bridge/tasks/_shared/
  declare themselves never-to-be-edited and are untouched; the
  gitignored canonical plan and master protocol were read-only sources.
- Validation: after these edits the six CPU-only layer runners were
  re-run (run_closure, run_ve0, run_ve1, run_ve2, run_e7b_supersession,
  run_pre_f1_evidence_metadata_repair); all six exited 0 on the first
  run, so every sentence guard on CLAUDE.md and
  collab/PROJECT_CONTEXT.md holds after the appended notes.
- Invariance: no training and no experiment execution; no scientific
  result, status, claim, number, interval, figure, table or model
  selection changed; no frozen, hash-pinned or historical artefact
  modified; the clean-test area was not read, listed or traversed;
  F1 and F2 remain unstarted and unauthorised; GPU work was zero.
