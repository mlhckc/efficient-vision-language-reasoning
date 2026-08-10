# E10 Phase 2: the scientific pipeline

## Purpose

Phase 0 established the B4/B4r model identity, provenance, the frozen recipe
contract and one bounded non-scientific calibration. Phase 1 set the two
mandatory resource constants and made them enforceable through a signed,
process-bound cell permit and a whole-cell guard. Neither phase implemented the
science: `run core-cell` opened the guard and raised inside the first stage.

Phase 2 implements the missing pipeline behind those reviewed guardrails:
labelled train and development loading, B4/B4r construction, the frozen
22-epoch training loop, the pinned development-evaluation cadence, canonical
scoring, the G14 scorer-identity diagnostics, the final R1 evaluation,
checkpoint and result publication with cell-level reconciliation, and the
frozen post-matrix statistical analysis.

Phase 2 authorises nothing. `E10_TRAINING_AUTHORIZED` is still `None`,
`run core-cell B4 train_40k 0` still refuses, and no scientific cell has run.

## Method

The base state was the independently reviewed Phase-1 HEAD
`231ad3bdfa3efc7e3caee3137987739b43d3193a`, verified before any change:
branch `v2-protocol`, origin equal, divergence 0/0, clean worktree,
`phase1-verify` PASS, `tests/test_e10.py` 21 checks, `tests/test_e10_phase1.py`
15 checks, and the scientific core refusing.

The scientific behaviour is the frozen E8B B2/B3 behaviour at `d_lm` 960. The
reuse boundary was drawn deliberately. Pure, parameterised E8B and E8A helpers
are imported and used unchanged: the latent trunk, the projection and its
preregistered scale rule, the teacher-forced answer-token loss, the answer cache
and prefix trie, the R1 and R2 reference and cached scorers, the batched R1
scorer, the G14 row construction and strata record, the G10 collapse
diagnostic, the labels-path reference loss, the G21 normaliser and its two
metrics, and the image-clustered bootstrap. The E8B core runner itself is hard
bound to E8B's authorisation state, gate-halt records, run locks, result tree
and identity ledger, so the parts of it E10 needs were ported into
`experiments/e10_capacity_360m/science.py` against E10's own guard, ledger and
artefact tree, preserving the arithmetic and the tolerances exactly.

Every scientifically relevant stage runs inside `ScientificCellGuard.stage()`.
The training loop enters `training` once per epoch and `development_evaluation`
once per evaluated epoch, so the 12-hour whole-cell deadline is re-checked at
every epoch boundary as well as at every optimizer step. `guarded_optimizer_step`
is the only door to `optimizer.step()`; a source-level check proves no bare
optimizer step exists in E10.

The recipe is verified against the immutable Phase-0 contract before a cell may
train, not merely built from it: the executable recipe must hash to the frozen
digest for that cell, the cadence must match the contract, and the twelve
digests are unchanged from Phase 0.

Commands run from the project root with the environment sourced:

    python -B -m experiments.e10_capacity_360m.run phase2-record
    python -B -m experiments.e10_capacity_360m.run phase2-verify
    python -B -m experiments.e10_capacity_360m.run phase1-verify
    python -B -m experiments.e10_capacity_360m.run core-order
    python -B -m experiments.e10_capacity_360m.run analysis
    python -B -m experiments.e10_capacity_360m.run core-cell B4 train_40k 0
    python -B tests/run_all.py

## Outputs

New sources under `experiments/e10_capacity_360m/`: `science.py` (the cell
pipeline), `analysis.py` (the frozen post-matrix analysis) and `phase2.py` (the
Phase-2 records and the executable pipeline-contract checks). `run.py` gains
`phase2-record`, `phase2-verify` and `analysis`, and its `core-cell` stub is
replaced by the real runner behind the unchanged refusal. `e10_common.py` gains
the scientific data contract, the scientific artefact paths, the Phase-2 binding
amendment and a no-scientific-cells proof. `tests/test_e10_phase2.py` is new and
registered in `tests/run_all.py`.

Two immutable records were written to `results/experiments/e10_capacity_360m/`:
`phase2_binding_amendment_20260810.json` and
`phase2_scientific_pipeline_20260810.json`.

Scientific artefacts will be published to
`results/experiments/e10_capacity_360m_core/`, deliberately beside rather than
inside the Phase-0/Phase-1 governance tree, so the existing
no-scientific-outputs proof over that tree keeps its meaning. The directory does
not exist yet, because no cell has run.

## Results

No scientific result exists and none was produced. The measurable outcomes are
contract and regression outcomes.

`tests/test_e10_phase2.py` passes 34 checks. `tests/test_e10.py` passes its 21
Phase-0 checks and `tests/test_e10_phase1.py` its 15 Phase-1 checks, both
unchanged. The full repository runner `tests/run_all.py` passes every module,
including E8A (1,035 E8B checks, 25 E9 checks and the rest unchanged).
`phase1-verify` and `phase2-verify` both pass.

The frozen twelve-cell order is unchanged and pair preserving. All twelve recipe
digests are identical to the Phase-0 recipe contract. The pinned cadence
resolves, from the contract rather than from prose, to epochs
1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 19 and 22, and epoch 22 is always evaluated so
the primary result is always measured. The scheduler horizon is 100 epochs and
the budget is exactly 22, so training covers only the first part of the
inherited trajectory. The frozen E8A, E8B and E9 result trees hash to their
recorded values: 111, 201 and 62 files respectively.

The known-negative still refuses. `run core-cell B4 train_40k 0` exits non-zero
with `E10 CORE REFUSED: scientific authorization is None`, before model, data,
CUDA, optimizer or output-directory initialisation; a test patches those six
entry points with exploding stubs and proves none is reached. `run analysis`
refuses with `E10 ANALYSIS REFUSED`, because no cell exists and a missing cell
is never imputed.

Zero optimizer steps ran on real E10 data, zero GPU-hours were charged, zero
checkpoints exist and the spend ledger records zero core cells. The only E10
ledger entry remains the 0.032714 GPU-hour Phase-0 calibration charge.

## Decisions and problems

Three deliberate differences from the frozen E8B behaviour are recorded in the
pipeline record rather than left to a diff.

The development cadence is the pinned E10 one: twelve of twenty-two epochs,
where E8B evaluated every epoch. This is the frozen E10 contract, not a choice
made here.

The G8 tiny-subset overfit gate is not run. The frozen Gate-1 per-cell budget
enumerates training, twelve development evaluations, the inherited G14 schedule,
a small fixed overhead and one final R1 pass. E8B's G8 can train for up to 200
epochs on 1,000 examples and is not funded by that budget; running it unbudgeted
would put the 12-hour wall at risk on the 250k cells, whose stress projection is
already 7.88 hours. The cheap implementation gates G4, G5, G6, G7 and G12, and
the closing G9, G10 and G11 checks, are all retained. This exclusion is flagged
for the reviewer as an explicit decision rather than an omission.

E10 never resumes. A leftover training state at cell start is a hard refusal,
because continuing an interrupted trajectory is a retry and no retry exists
without a fresh explicit authorization record. A durability checkpoint is still
written each evaluated epoch, labelled for forensics only.

One further point is flagged rather than resolved. The frozen Gate-1 per-cell
budget carries a single line for one full-development pass beyond the twelve
evaluations. The published Phase-0 table names that line "final R1", and the
guarded stage is named `final_r1_evaluation`, but the field in the projection
source is named `final_r1_intervention_seconds`. The pipeline implements it as
one canonical R1 pass on the epoch-22 checkpoint under the normal condition,
which is what the stage name and the published table state and what the primary
result requires. No visual-reliance intervention pass is implemented, because no
E10 record registers an executable intervention set or a target-cell subset, and
inventing one would both broaden the frozen design and add roughly three
unbudgeted full-development passes per cell. The analysis module's
visual-reliance summary therefore reports the conditions the published cells
actually contain and refuses to estimate anything else. If the intended reading
was an intervention pass, that is a design question for the user, not a gap to
fill silently.

A defect was introduced and repaired during implementation. Making the Phase-1
repair record consume the amendment chain, which adding a newer Phase-2 link
requires, created mutual recursion between the repair and the amendment
validators. The fix is substantive rather than cosmetic: a chain link may now
only be carried forward by a strictly newer link, which both terminates the walk
and states the real rule, since a record cannot be carried forward by evidence
that already existed when it was written.

The Phase-2 records bind the live source digest, so they were regenerated
whenever a source changed during this open, unapproved revision, exactly as the
Phase-1 repair record was. Nothing approved was rewritten: the amendment carries
all four Phase-1 records forward byte for byte and re-checks their hashes on
every read.
