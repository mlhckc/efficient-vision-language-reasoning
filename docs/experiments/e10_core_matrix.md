# E10: the 360M answer-readout capacity matrix

## Purpose

E10 asks whether the pretrained weights of a larger frozen small language model
contribute anything to a lightweight answer readout, and whether any such
contribution grows with training-set size. It compares B4, a frozen pretrained
SmolLM2-360M readout, against B4r, an architecture-matched frozen randomly
initialised SmolLM2-360M readout, at train_40k and train_250k with seeds 0, 1
and 2. The twelve cells form a fully crossed, pair-preserved design whose
registered primary output is the pretraining-by-scale difference in
differences.

The experiment is the 360M capacity point of the E8B readout family, which
established the same comparison at 135M. Every encoder and every language model
stays frozen; the only trainable parts are the inherited latent-query reasoner
and the linear projection into the language model's embedding width.

## Method

The scientific design, the guardrails, the pipeline and the authorisation
binding were implemented and independently reviewed in four earlier phases
(docs/experiments/e10_phase0_calibration.md through
e10_phase3_authorization_binding.md). This report covers execution only. No
scientific source, configuration, recipe, resource constant or approved
governance record was modified to run the matrix.

Execution was opened by one immutable external core-authorisation grant,
created out of band at
`<shared-state>/e10-core-authorization/e10-core-authorization-grant.json`,
grant id `7ce353d7ec9d93b61925e735fdef2817f18a6c051221c3208cc52e6e3b7589d2`.
The grant binds the twelve-cell matrix and its pair-preserving order, both arm
identities, the pinned model repository, revision and weight digest, all twelve
recipe digests, the reviewed live source and config digests, the 12.0 hour
whole-cell wall, the 40.0 hour effective identity ceiling, zero automatic
retries, and the closed scope exclusions for the clean test, F1, F2, A5 and
A8c. Creating it provably did not move the source digest
(`857ec44e…`) or the config digest (`4b9fa43c…`), both of which are unchanged
from before creation to after execution.

Before the grant was created the recorded start state was verified: branch
`v2-protocol` at `028531e`, local equal to origin with divergence 0/0, a clean
worktree, all three phase verifiers passing, the full regression suite passing,
the frozen E8A, E8B and E9 result trees matching their recorded digests, all
twelve recipe digests matching the frozen contract, no grant present, the
known-negative core-cell invocation still refusing, and zero scientific cells,
checkpoints or core GPU-hours. Immediately after creation a fifteen-point
acceptance test passed in full, including that all three verifiers still passed
with the grant present and that a wrong arm, a wrong scale and a wrong seed
each still refused.

The twelve cells ran strictly in the frozen pair-preserving order, one at a
time, on a single node, each through the reviewed production entry point
`run core-cell`. The recipe was untouched: 22 epochs with no early stopping,
epoch 22 as the sole primary checkpoint, the pinned development cadence
1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 19, 22, a cosine schedule over a 100-epoch
horizon, learning rate 3e-4, no warmup, dropout 0.1, weight decay 0.01 on
parameters of rank two or more, batch size 128, gradient clipping 1.0, bf16
autocast on the training path and FP32 for all canonical evaluation. Each cell
ran inside the whole-cell guard covering setup, training, development
evaluation, the G14 diagnostics, the final R1 evaluation and result
publication.

Between cells an independent check re-read the published evidence from disk
rather than trusting the runner: the COMPLETE record and its epoch policy, the
re-hashed canonical, secondary and training-state checkpoints, the re-hashed
per-row arrays and their row count, the ledger entry and its identity, the
grant's continued validity and unchanged identity, the source, config and
recipe digests, the wall and identity limits, and the headroom gate for the
next cell. Any non-zero exit from a cell or from that check would have stopped
the whole matrix; none occurred.

After the twelfth cell the frozen analysis ran unchanged and refused nothing.

## Outputs

- `results/experiments/e10_capacity_360m_core/e10_core_<arm>_<scale>_seed<n>.json`,
  twelve immutable cell records.
- `results/experiments/e10_capacity_360m_core/e10_core_analysis.json`, the
  frozen twelve-cell analysis.
- `results/experiments/e10_capacity_360m_core/e10_core_execution_20260813.json`,
  the execution and integrity evidence for this report. It is immutable and was
  not edited by the later correction packet.
- `results/experiments/e10_capacity_360m_core/e10_core_execution_correction_20260813.json`,
  the immutable addendum recording every correction the independent review
  required in this report and in the `CLAUDE.md` status entry, and stating that
  no scientific artefact or result changed.
- `results/experiments/e10_capacity_360m_core/ARTEFACT_MANIFEST.json`, the path,
  size and SHA-256 of every artefact, including the local binaries.
- `results/experiments/e10_capacity_360m/cell_completed_*.json`, twelve guard
  outcome records.
- Local, not version controlled: twelve per-row `.npz` arrays and
  thirty-six checkpoints under `checkpoints/`, each pinned by a SHA-256
  recorded inside its cell record and in the artefact manifest.

## Results

Development-set results only. The clean test was not read, and its embargo is
unchanged.

Primary metric: canonical FP32 development R1 accuracy of the epoch-22
checkpoint over 7,714 development rows on 768 images, closed 100-answer
vocabulary. The pinned G21 normalised exact match agrees with it to six decimal
places in every cell, and G21 raw exact equals G21 normalised in every cell.

| arm | scale | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|---|
| B4 | train_40k | 0.52515 | 0.53150 | 0.52243 | 0.52636 | 0.00466 |
| B4r | train_40k | 0.53474 | 0.52787 | 0.53695 | 0.53319 | 0.00473 |
| B4 | train_250k | 0.60215 | 0.58556 | 0.59450 | 0.59407 | 0.00831 |
| B4r | train_250k | 0.59256 | 0.59774 | 0.59904 | 0.59645 | 0.00343 |

The five preregistered contrasts, with per-seed differences and the
image-clustered 95 per cent interval:

| contrast | per seed | mean | interval | rule |
|---|---|---|---|---|
| B4 − B4r at train_40k | −0.00959, +0.00363, −0.01452 | −0.00683 | [−0.01508, +0.00114] | uncertain or mixed |
| B4 − B4r at train_250k | +0.00959, −0.01219, −0.00454 | −0.00238 | [−0.01071, +0.00588] | uncertain or mixed |
| B4 scale effect | +0.07700, +0.05406, +0.07208 | +0.06771 | [+0.05800, +0.07738] | positive directional |
| B4r scale effect | +0.05782, +0.06987, +0.06209 | +0.06326 | [+0.05287, +0.07296] | positive directional |
| difference in differences | +0.01919, −0.01582, +0.00998 | +0.00445 | [−0.00709, +0.01641] | uncertain or mixed |

The difference in differences is `(B4 − B4r)@250k − (B4 − B4r)@40k` at full
magnitude. Recomputing it from the twelve published accuracies gives
+0.01918, −0.01581, +0.00998 with mean 0.00445, matching the analysis to
within rounding; the Phase-2 factor-of-two defect is not present.

The `sd` column in the table above is the sample standard deviation across the
three training seeds (`ddof=1`). It is not a confidence interval, and
`mean ± sd` must not be read as one. The bracketed intervals in the contrast
table are a different quantity: they are image-clustered 95 per cent bootstrap
intervals that condition on the fixed trained seed set, propagating
evaluation-sampling variation over images and questions and not uncertainty
over possible training seeds. Training-seed variability is carried instead by
the per-seed columns and by `sd`. Neither quantity substitutes for the other,
and with three seeds the training-seed term is estimated very coarsely.

The interpretation the design registered in advance is therefore:

- Training-set size has a large and consistent effect at 360M. Both scale
  effects are positive in every seed, about +6.3 to +6.8 points, with intervals
  well clear of zero.
- No reliable positive pretrained-over-random advantage was detected at either
  training scale. Both pretraining-effect intervals include zero and do not
  establish equivalence or the absence of an effect. Both point estimates are
  slightly negative and the three seeds disagree in sign at both scales.
- There is no evidence that the pretrained-over-random contribution grows with
  scale. The difference in differences is +0.00445 with an interval containing
  zero, and its per-seed values span −0.01582 to +0.01919, a range four times
  its mean. This too is a failure to detect, not a demonstration of absence.

E10 qualitatively reproduces the earlier E8B finding that the tested answer-side
frozen-SLM interface does not exhibit a reliable positive pretrained-over-random
advantage, now at the larger 360M configuration. It does not reproduce E8B's
directional negative 40k effect. At 135M, E8B's B3 − B2 at train_40k was
−0.02463 with an image-clustered interval of [−0.03333, −0.01619] excluding
zero and all three seeds negative; E10's B4 − B4r at train_40k is −0.00683 with
an interval containing zero and mixed seed signs. At train_250k both
experiments are uncertain: E8B −0.00475, [−0.01208, +0.00297]; E10 −0.00238,
[−0.01071, +0.00588]. The comparison with E8B is context from an
already-published record, not part of the frozen E10 contrast set, and no new
statistic was computed for it.

The 135M to 360M comparison is whole-system capacity sensitivity, not an
isolated causal language-model-size effect. Changing the frozen language model
also changes its hidden size, and therefore the width of the trainable
projection into it, so trainable capacity moves with the model: 21,343,808
parameters at 135M against 21,540,800 at 360M, a difference of 196,992
parameters or about 0.923 per cent. The two configurations differ in more than
the frozen backbone, and nothing here isolates the backbone's contribution.

Mandatory gates, all twelve cells: G9 reload identity true; G11 repeat identity
true; G10 collapse did not fire (98 to 100 distinct answers, top-1 share 0.236
to 0.279); G14-FP32 pre-selection identical on 64 pinned rows and
post-selection identical on 146 to 152 rows across the P, L and O strata, with
clauses C1, C2 and C3 all identical and no numerical exemption taken. G8 is not
part of E10, and no visual-intervention or R2/R3 condition exists in the frozen
E10 contract, so E10 produces no new visual-reliance evidence; the analysis
records that explicitly rather than estimating one.

Resources. Total real E10 core execution 51.07838 GPU-hours across twelve
cells, plus the 0.03271 GPU-hours of approved Phase-0 calibration, for
51.11110 GPU-hours of E10 in total. Per identity, `pretrained_smollm2_360m`
25.56327 hours and `random_smollm2_360m` 25.54783 hours, leaving 14.43673 and
14.45217 hours of headroom against the 40.0 hour effective ceiling. Per cell
the six train_40k cells took 2.02273 to 2.03553 hours and the six train_250k
cells 6.46169 to 6.51955 hours; the longest cell used 54.3 per cent of the
12.0 hour wall. Zero retries of any kind were taken and the retry ledger is
empty. Peak reserved memory ran 0.659 to 0.677 of the device against the 0.80
ceiling, above the 0.512 to 0.530 the Gate-1 projection recorded but well
inside the gate, which did not fire in any cell.

Projected against actual. The Gate-1 projection budgeted 49.72765441864283
GPU-hours of core execution. Adding the known extra development pass described
below brings the honest projection to 50.48143730636344. Actual core execution
was 51.07838121415333, which is 2.72 per cent above the unadjusted projection
and 1.18 per cent above the adjusted one, leaving a residual of
0.5969439077898893 GPU-hours above the adjusted projection. Per identity the
overruns were +0.69944 and +0.65129 hours. Where this report shows shorter
figures such as 50.48144 or 0.59694, they are rounded from the canonical values
given here, which are the ones recorded in
`e10_core_execution_20260813.json`.

The stress comparison, with its basis stated because the earlier version of
this report did not state it: core actual execution of 51.07838121415333
GPU-hours is 82.17 per cent of the corresponding core stress budget of
62.15956802330354 GPU-hours, which is six cells at each scale priced at the
Gate-1 1.25-times-measured stress figure. That is a core-to-core comparison. It
is not a comparison against the 62.19228220657687 hour figure recorded as
`view_1_e10_only.stress_total_gpu_hours`, which additionally includes the
Phase-0 calibration charge.

Evaluation-pass accounting, stated as three separate numbers because they
differ. The Gate-1 projection budgeted 13 full-development passes per cell,
being the 12 pinned cadence evaluations plus the final R1 evaluation. The
implemented pipeline makes 14 full-development calls per cell, because the
mandatory G11 reproduction repeats the final development pass. The difference
is one pass per cell, 226.13 seconds each by the Gate-1 timing, so about
0.75378 GPU-hours across the matrix. That accounts for just over half of the
1.3507267955105036 hour total overrun; the remaining 0.5969439077898893 hours
is ordinary cell-to-cell variation of about 1.2 per cent. No resource constant
was altered to absorb it, and it had no wall or ceiling consequence.

## Decisions and problems

Execution itself was uneventful. Every cell completed, reconciled and published
on the first attempt; no gate halted, no wall was approached, no automatic
retry was available or taken, and no cell was descoped, reordered or repeated.

Three integrity observations should go to the reviewer. The first two were
understated in the first version of this report and are corrected here; the
correction is recorded separately and immutably in
`e10_core_execution_correction_20260813.json`.

The first is a state-coupling defect in the verifiers and tests, of exactly the
class the independent Phase-3 review already found once. Before execution all
three verifiers and the whole regression suite passed, both with no grant and
with the real grant present. The terminal state after legitimate completion is:

- `phase1-verify` passes.
- `phase2-verify` fails, because `phase2.pipeline_contract()` still contains
  pre-execution assumptions that are false once the matrix is complete:
  `assert_analysis_refuses_incomplete()` requires the real analysis to refuse,
  and `e10.assert_no_scientific_cells()` requires zero scientific artefacts.
- `phase3-verify` also fails, and not only transitively through phase2.
  `phase3.authorization_contract()` has its own direct call to the same stale
  invariant at `phase3.py:438`, confirmed against the live file at this HEAD.
  Phase 3 would therefore fail on its own terms even if phase2 were repaired
  alone.

The first version of this report claimed that every other clause of all three
phases passes individually. That claim is withdrawn: a subset of clauses was
checked individually and passed, which is not the same statement and does not
license the general one. What is established is the specific failing set named
above.

The test suite's terminal state, traced test by test rather than module by
module:

| module | failing | total |
|---|---|---|
| `test_e10` | 1 | 21 |
| `test_e10_phase1` | 4 | 15 |
| `test_e10_phase2` | 6 | 38 |
| `test_e10_phase3` | 4 | 23 |
| total | 15 | 97 |

So 15 E10 tests fail and 82 pass. All twelve non-E10 test modules pass. The
independent review recorded ten non-E10 modules; the verified count at this
HEAD is twelve, and the count is stated here as measured. The review's E10
figures reproduced exactly.

The independent reviewer traced all fifteen failures and classified them as
lifecycle and state-coupling failures rather than scientific regressions, which
this packet confirms. The principal mechanism is `mirror_governance()`: it
copies the real, now-completed governance tree, including the twelve real
`cell_completed_*.json` records, into the temporary test trees, so tests whose
intended state is pre-execution or pre-authorisation are invalidated by real
evidence rather than by their own fixtures. The remaining failures assert
directly that no grant exists, that no scientific cell exists, or that the
analysis refuses.

None of this affects the executed science. The source and config digests are
byte-identical to the reviewed ones before the grant, after the grant, after
all twelve cells and after this repair; the frozen E8A, E8B and E9 result trees
are unchanged; all twelve recipe digests, the pinned manifests, the token
stores and the model verification still match. The correct repair is to make
these assertions execution-aware, exactly as the Phase-3 repair made the
refusal invariant grant-aware, together with the `mirror_governance()` fixture.
That work is R3. It is deliberately not done in this packet, which is
documentation-only and digest-neutral, and it requires separate authorisation
and independent re-review.

The second observation is the memory figure above. Peak reserved memory was
consistently about 0.15 higher than the Gate-1 projection predicted. The gate
never fired and no cell came close to 0.80, but the projection basis is worth
revisiting before any future cell is sized against it.

The third observation is the grant's terminal status, recorded as a factual
observation from the independent review and not acted on here. The original
grant remains present and remains historically valid for the completed
execution. All twelve authorised cells now refuse further execution, because a
completed scientific cell is immutable and is never overwritten: invoking
`run core-cell B4 train_40k 0` at this HEAD returns
`E10 CELL REFUSED: ... already exists`. Non-authorised arms, scales and seeds
still refuse at the entry gate, for example `E10 CORE REFUSED: unknown cell
B3/train_40k/seed0`. Whether the grant should additionally be retired or
marked spent operationally is an R3 question and is not decided here; the grant
was not modified in this packet.

One operational note. The matrix was expected to take more than two days, so it
ran under a detached driver script outside the repository tree, which invoked
the reviewed production entry point once per cell and stopped the whole matrix
on any non-zero exit. The driver adds no scientific behaviour, is not project
source, and no reviewed source was modified to accommodate it. Persistence is
not retry: a failed cell would have halted the matrix and returned to the user
for a fresh explicit authorisation.

All results here are development-set results. The clean test was not read at
any point, F1 and F2 have not started, and nothing in this packet authorises
them.
