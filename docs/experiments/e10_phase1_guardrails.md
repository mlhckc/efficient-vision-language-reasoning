# E10 Phase 1: authorization guardrails

## Purpose

E10 Phase 0 implemented the registered B4/B4r SmolLM2-360M answer-readout
family, verified the pinned model, ran one bounded non-scientific calibration
and passed Gate 1. It deliberately left the two E10 resource constants unset
and recorded a recommendation instead of a decision.

The independent Phase-0 review returned PASS with one execution-readiness
finding: `build_recipe` drops the `wall_clock_halt_hours` field it inherits
from the E8B recipe, and no Phase-1 runner re-imposed an E10-specific wall, so
a future scientific cell would have had no executable operational halt.

This task records the user's Phase-1 resource decisions, makes them
enforceable, and repairs that single gap. It authorises no scientific cell.
The twelve-cell B4/B4r core matrix remains refused.

## Method

The user's decisions of 2026-08-10, taken after the Phase-0 PASS:

1. `E10_PER_CELL_WALL_CLOCK_HOURS = 12.0`, a hard executable operational halt
   rather than a projection field. A cell that reaches the wall fails closed,
   is recorded and charged as halted, and never continues.
2. `E10_PER_IDENTITY_CEILING_HOURS = 40.0`, never more permissive than the
   programme-wide frozen-model identity ceiling. Every gate uses
   `min(E10_PER_IDENTITY_CEILING_HOURS, PER_MODEL_IDENTITY_CEILING_HOURS)`.
3. No automatic retry. The previous nominal one-forced-retry behaviour is
   withdrawn: the Gate-1 stress projection leaves post-retry identity margins
   of 0.040641 h (pretrained) and 0.007927 h (random), which are operationally
   meaningless. A failed scientific cell now requires a fresh explicit user or
   reviewer retry authorization record.
4. A5/A8c keep the `OPEN_UNQUANTIFIED_RESERVE` disposition. No numeric reserve
   was invented. They stay outside E10 and re-deferred, but if they use the
   same pinned pretrained SmolLM2-360M frozen identity their future compute is
   charged against the same authoritative frozen-model identity accounting
   rather than treated as free independent budget.
5. Storage keeps its Phase-0 qualification: physical fit is established, a
   formal project storage allocation remains unestablished, and free space is
   not an allocation claim.

The wall is re-imposed outside the frozen recipe. `build_recipe` is unchanged,
so the twelve recipe digests and six pairing-normalised digests in the Phase-0
recipe contract are identical. Instead, `e10_common.CellExecutionPermit`
carries the wall as a signed, process-bound monotonic deadline derived from
`config.E10_PER_CELL_WALL_CLOCK_HOURS`, and `phase1.ScientificCellGuard` holds
one cell inside that deadline. `guarded_optimizer_step` remains the only
function in the E10 package that calls `optimizer.step()`; it now accepts the
cell permit as well as the calibration permit, so a scientific training loop
cannot step outside the wall.

Setting the two constants changes the live config digest, and adding the
Phase-1 sources changes the live source digest, so every immutable Phase-0
record would otherwise become unverifiable. The Phase-0 records are not
rewritten. One immutable amendment record pins the pre-amendment digests and
the SHA-256 of all fourteen Phase-0 records (twelve in the repository, two in
shared state), and `assert_recorded_binding` accepts a superseded binding only
for a record that amendment names and whose bytes still match. This mirrors
the mechanism the earlier Phase-0 source correction already established.

Commands run, from the project root with the environment sourced:

    python -B tests/test_e10.py
    python -B tests/test_e10_phase1.py
    python -B tests/run_all.py
    python -B -m experiments.e10_capacity_360m.run phase1-record
    python -B -m experiments.e10_capacity_360m.run phase1-verify
    python -B -m experiments.e10_capacity_360m.run core-order
    python -B -m experiments.e10_capacity_360m.run core-cell B4 train_40k 0

## Outputs

Source:

- `config.py`: the Phase-1 resource constants, including
  `E10_AUTOMATIC_RETRIES_PER_IDENTITY = 0` and
  `E10_MAX_AUTHORIZED_RETRIES_PER_IDENTITY = 1`.
- `experiments/e10_capacity_360m/phase1.py`: the guardrail module.
- `experiments/e10_capacity_360m/e10_common.py`: the effective ceiling, the
  mandatory wall accessors, the cell permit, core-cell accounting, the retry
  authorization contract and the binding chain.
- `experiments/e10_capacity_360m/run.py`: `phase1-policy`, `phase1-record` and
  `phase1-verify`, and the guarded core-cell entry point.
- `tests/test_e10_phase1.py`, plus the retry updates in `tests/test_e10.py`.

Records, all immutable and non-scientific, under
`results/experiments/e10_capacity_360m/`:

- `phase1_binding_amendment_20260810.json`, SHA-256
  `872591d2c89058ac843d12d3205cc29f1636ce63dbefeb9f2cd1a7caa114ef95`.
- `phase1_resource_policy_20260810.json`, SHA-256
  `0cf96a0465b6fdc88e5797d2322db5785c54414aa7b54d2dc55cb0f5d6852a4c`.
- `phase1_a5_a8c_identity_governance_20260810.json`, SHA-256
  `4ee2453cef6cb3497da8ead7f30577ed48d5b9bc217df53b90a4f18118fe8b9a`.

## Results

No scientific result was produced. This task ran no epoch, no scientific cell,
no evaluation and no optimizer step outside a unit test, created no checkpoint,
and charged zero GPU-hours to either frozen model identity.

Resolved policy, from `run phase1-verify`: per-cell wall 12.0 h; E10 identity
ceiling 40.0 h; programme-wide identity ceiling 40.0 h; effective identity
ceiling 40.0 h; headroom floor 1.0 h; automatic retries per identity 0;
authorised retries per identity at most 1.

Verification:

- The known negative still refuses. `run core-cell B4 train_40k 0` exits
  non-zero with `E10 CORE REFUSED: scientific authorization is None`, before
  any model, data, CUDA, output directory or optimizer initialisation.
- `run core-order` prints the frozen twelve-cell order unchanged, B4 before B4r
  within every scale and seed.
- All twelve live recipe digests and six pairing-normalised digests equal the
  Phase-0 recipe contract, so the scientific matrix, recipe, optimiser,
  schedule, epochs, evaluation cadence, model revision, tokenizer, vocabulary,
  manifests and seed set are untouched.
- The E8A, E8B and E9 result trees hash to their pre-implementation baselines
  (111, 201 and 62 files), and `git diff` against the base commit over those
  three directories is empty.
- 12 focused Phase-1 checks pass; `tests/test_e10.py` passes 21 checks; the
  full suite passes all fifteen modules.
- The source scan over 101 project files finds no code path that writes a
  retry authorization record and no call site that consumes a retry.

## Decisions and problems

The reviewer's finding could have been repaired by putting
`wall_clock_halt_hours` back into the recipe. That was rejected: the recipe
digest is the scientific identity of a cell, it is pinned in the Phase-0 recipe
contract, and a resource bound is not part of it. Imposing the wall through the
permit keeps the frozen matrix bit-identical and still makes the wall
mandatory, which the tests check from both directions.

Setting the constants necessarily moved both binding digests. Rewriting the
Phase-0 records to match was rejected as evidence tampering; leaving them
stale-bound was rejected because the Phase-0 evidence would silently stop
validating. The amendment record is the same device the Phase-0 source
correction used, with the same constraint that a superseded binding is accepted
only for a named record whose bytes still hash to the pinned value.

The retry policy is enforced where a retry is actually consumed. Rather than a
flag that a runner could ignore, `record_forced_retry` refuses unless a fresh,
single-use, out-of-band authorization record names exactly that arm, scale,
seed, reason and failure record, and is signed by the user or an independent
reviewer; the consumed nonce is written into the retry ledger so the same
record cannot be replayed. No project source can create such a record, and the
source scan is part of the guard rather than only of the tests.

The A5/A8c reserve remains unquantified by decision. Nothing here invents a
number for it, and the governance record states only the conditional: shared
frozen identity implies shared identity accounting.

E10 scientific execution remains refused. The twelve-cell B4/B4r core, the
clean test, F1 and F2 all remain unauthorised.
