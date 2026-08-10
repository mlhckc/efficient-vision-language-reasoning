# E10 Phase 3: the scientific authorization binding

## Purpose

The independent Phase-2 review returned PASS and left one gate: explicit
scientific authorization. The only mechanism implemented for that gate was the
source constant `e10_common.E10_TRAINING_AUTHORIZED`, and that mechanism is
self-defeating.

`e10_common.py` is inside `SOURCE_PATHS`, so the constant is part of the live
source digest that every immutable E10 record binds. Setting it to the approval
token therefore moves the digest and invalidates the reviewed provenance chain
the authorization is supposed to open. There was no way to grant scientific
authorization without destroying the evidence that the granted source had been
reviewed.

Phase 3 changes only how an authorization is expressed. It changes no
scientific behaviour, and it authorises nothing: no grant is created here,
`run core-cell B4 train_40k 0` still refuses before any model, dataset, CUDA
context, optimizer or output directory is touched, and no scientific cell has
run.

## Method

The base state was the approved Phase-2 closure HEAD
`5d79bf5fb435ce365335593aff6912e37071f65c` (reviewed scientific source
`45986fc49a8117ed951339457099f5d3debda3a3`), verified before any change:
branch `v2-protocol`, origin equal, divergence 0/0, clean worktree,
`phase1-verify` and `phase2-verify` PASS, the four E10 test modules passing,
`tests/run_all.py` passing, the scientific authorization unset, the known
negative refusing, and zero scientific cells, checkpoints, results or core
GPU-hours.

The defect was reproduced against that state before any Phase-3 source existed,
by changing the single constant and nothing else. The result is recorded in
`phase3.DEFECT` and in Results below.

The repair moves the runtime grant out of the source and into an immutable
external record:

- `CORE_AUTHORIZATION_DIR` is a directory in shared state, outside the
  repository and outside both `SOURCE_PATHS` and `CONFIG_PATHS`, so creating a
  grant cannot move a digest that a reviewed record binds. It holds at most one
  file; a second record refuses rather than being ranked.
- `e10_common.core_authorization_grant_expectations()` states, from frozen
  contracts and pinned constants alone, exactly what the one valid grant must
  say. It can never be shaped by an observed result.
- `e10_common.validate_core_authorization_grant()` is the single canonical
  validator. It grants exactly the frozen, pair-preserved twelve-cell matrix or
  it refuses; there is no partial authorisation.
- `E10_TRAINING_AUTHORIZED` is retained as a legacy refusal sentinel. It must
  remain `None`; any other value is itself a refusal, at the entry gate and on
  the permit path, so editing scientific source can never open the core.

The grant binds the explicit user authorization and its token and purpose, the
E10 experiment and protocol family, the exact arms `B4` and `B4r`, the exact
scales `train_40k` and `train_250k`, the exact seeds 0, 1 and 2, the frozen
pair-preserving twelve-cell order, the reviewed Phase-3 source and config
digests, the recipe-contract reference and all twelve recipe digests, the
pinned model repository, revision, weight digest and both frozen identities,
the 12.0-hour per-cell wall, the 40.0-hour effective identity ceiling, zero
automatic retries, an exact scope-exclusion map that keeps the clean test, F1,
F2, A5 and A8c closed, immutable creation metadata, and a `grant_id` self-hash
over its own body so any modified byte refuses.

Entry-gate integration follows the existing authoritative paths rather than
adding new ones. `assert_core_entry_authorized` now returns the validated
grant, and `authorize_cell_execution`, `phase1.assert_cell_entry_guarded`,
`ScientificCellGuard.__enter__`, `science.run_core_cell`, `run.core_cell` and
`charge_core_cell_hours` all reach the grant through it. The signed,
process-bound cell permit carries the grant's identity and bytes, so a permit
can exist only where a valid grant existed; `ScientificCellGuard.stage()`
revalidates the grant bytes at every stage boundary, and the per-optimizer-step
path stays free of filesystem work.

Because implementing the validator necessarily changes reviewed governance
source, one narrow Phase-3 amendment is chained onto the approved Phase-2
state, exactly as Phase 1 and Phase 2 were chained before it. It preserves every
approved Phase-0, Phase-1 and Phase-2 record byte for byte, binds the Phase-2
closure digests as its predecessor, binds the new live source digest, and is
the newest link in the strictly-newer amendment chain. The Phase-2 amendment is
demoted from live-binding to recorded-binding by the same rule that demoted the
Phase-1 repair record in Phase 2.

Commands run, from the project root with the environment sourced:

    python -B -m experiments.e10_capacity_360m.run phase3-record
    python -B -m experiments.e10_capacity_360m.run phase3-verify
    python -B tests/test_e10_phase3.py
    python -B tests/run_all.py

## Outputs

Source: `experiments/e10_capacity_360m/phase3.py` (new), and the authorization
grant, validator, Phase-3 amendment validator and entry-gate changes in
`experiments/e10_capacity_360m/e10_common.py`, with the stage-boundary
revalidation in `phase1.py` and the two new subcommands in `run.py`.

Tests: `tests/test_e10_phase3.py` (new, 21 checks), registered in
`tests/run_all.py`; the shared synthetic-grant and governance-mirror helpers in
`tests/test_e10.py`; and the authorised-state helpers in
`tests/test_e10_phase1.py` and `tests/test_e10_phase2.py`, which now install one
synthetic external grant instead of mutating the source constant.

Immutable records, written once:

- `results/experiments/e10_capacity_360m/phase3_binding_amendment_20260811.json`
- `results/experiments/e10_capacity_360m/phase3_authorization_binding_20260811.json`

No scientific artefact was written. `results/experiments/e10_capacity_360m_core/`
still does not exist.

## Results

The defect, reproduced at the Phase-2 closure HEAD by changing only
`E10_TRAINING_AUTHORIZED` from `None` to `"e10-core-matrix-approved"`:

- live source digest moved from
  `809320d75e0ae68f81bdd4ba6d7e33c8a6afcdf39db4afbe0d49343bf7b49baa` to
  `214fcc39f99c146609a8a863be3ad3f64d4879382bd565c06083ab2e9b462094`;
- `phase1-verify` failed with `E10 Phase-2 amendment: stale source_digest`;
- `phase2-verify` failed with the same refusal;
- `run core-cell B4 train_40k 0` failed on stale provenance rather than
  reaching the authorization decision at all;
- the config digest did not move.

The reproduction was reverted and the worktree confirmed clean before any
Phase-3 work began.

After the repair, at the final Phase-3 state:

- live source digest `9b1afd8359148ef7b63d08cddad48823e2568760c12341814b5e6ec1bdc99801`;
  config digest `4b9fa43cb74b511c0bb6fda3c7c043d6047ef5e5e220c1abad6e35138f3df9ab`,
  unchanged from Phase 2;
- creating, validating and opening a synthetic grant leaves the source digest
  and every per-file digest byte-identical, measured before, during and after;
- `phase1-verify`, `phase2-verify` and `phase3-verify` all exit zero;
- `tests/test_e10.py` 21 checks, `tests/test_e10_phase1.py` 15 checks,
  `tests/test_e10_phase2.py` 38 checks, `tests/test_e10_phase3.py` 21 checks,
  and `tests/run_all.py` reports all test modules passing;
- the Phase-2 synthetic end-to-end production run still reaches a reconciled
  COMPLETE cell, now under a synthetic external grant rather than a mutated
  source constant;
- with no grant, all twelve frozen cells refuse at every authoritative entry
  path; with one valid synthetic grant, exactly those twelve become
  authorisable and nothing else does;
- a thirteenth cell, a dropped pair, a reordered matrix, a wrong arm, scale,
  seed, recipe digest, recipe-contract reference, source digest, config digest,
  protocol family, experiment, model revision, wall, ceiling, retry count or
  authoriser, a broadened scope exclusion, a stale predecessor provenance
  reference, a tampered byte, a competing second record and a missing field all
  refuse;
- a mutated source constant is refused whether or not a valid grant exists, and
  an environment variable is not a route;
- the frozen E8A, E8B and E9 result trees are unchanged (111, 201 and 62 files,
  tree digests equal to the recorded baselines);
- all twelve recipe digests still equal the frozen Phase-0 contract, the
  cadence is still 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 19, 22, the canonical epoch
  is still 22, the wall is still 12.0 hours, the effective identity ceiling is
  still 40.0 hours and automatic retries are still zero;
- the known negative refuses with
  `E10 CORE REFUSED: scientific authorization grant is absent`;
- zero scientific cells, zero checkpoints, zero E10 accuracies, zero core
  GPU-hours charged, and no real authorization grant exists.

## Decisions and problems

The grant lives in shared state rather than in the repository. A repository
record would be tracked by git and, more importantly, would put the
authorization inside the tree whose digests it binds, which is the same class
of error the phase exists to repair. Shared state is also where the retry
authorizations already live, and the retry design is the precedent this one
follows: an out-of-band, immutable, strictly validated record that no project
source may write. `phase3.assert_no_self_granted_authorization` proves that by
scanning every project source for a write whose argument names the grant.

The source constant was kept rather than deleted, and made stricter rather than
merely ignored. Requiring it to stay `None` means a future editor who reaches
for the old mechanism gets an explicit refusal naming the reason, instead of a
silent no-op.

The per-optimizer-step path deliberately does not re-read the grant. The permit
is signed, process-bound and carries the grant bytes, the grant is fully
revalidated at cell entry, at each of the six stage boundaries and again inside
accounting, and a per-step read of a shared-filesystem record would add tens of
thousands of network round trips to a 250k cell for no additional guarantee.

The grant binds the model repository, revision, weight digest and both frozen
identities, and does not bind the parameter count. The count is a derived
property of the pinned checkpoint and is already enforced at cell setup by
`science.assert_model_identity` and the frozen parameter constants; binding it
in the authorization would have added no authorisation-relevant information.

Two problems were found and fixed during implementation. First, the Phase-1 and
Phase-2 tests redirect `OUT_DIR` into a temporary directory, and the entry gate
now resolves the provenance chain through `OUT_DIR`; those tests now mirror the
immutable governance tree into the temporary directory byte for byte instead of
starting from an empty one. Second, the Phase-2 production harness patched only
part of shared state, so the first run of the updated tests wrote a synthetic
grant to the real shared-state path. That file was removed, the harness now
redirects `CORE_AUTHORIZATION_DIR` with the rest of shared state, and
`write_core_authorization_grant` refuses outright if the real path is still
live, so the mistake cannot recur silently. The harness also now installs its
grant after the stand-in model identity is in force.

Phase 3 grants nothing. The one remaining gate is unchanged in substance and
only changed in form: one explicit user scientific authorization, created out of
band after an independent Phase-3 review returns PASS, and never by editing
scientific source.
