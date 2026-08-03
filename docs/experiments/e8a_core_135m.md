# E8A dissertation core: the SmolLM2-135M question encoder at two scales

## Purpose

Whether replacing the frozen CLIP question tokens with the frozen hidden
states of a small language model helps the latent-query reasoner, and whether
any benefit comes from the language model's pretraining rather than from its
architecture or its wider interface.

Three arms, all with the same 21.1M-parameter reasoner trunk and the same
frozen CLIP image tokens, differing only in where the question tokens come
from and in the width of the one trainable projection into the 512-dimensional
reasoner interface:

- **A0p** — frozen CLIP question tokens, Linear(512, 512). The
  interface-matched control. Image and question tokens share CLIP's pretrained
  space.
- **A1** — frozen pretrained SmolLM2-135M hidden states, Linear(576, 512).
- **A1r** — frozen randomly initialised SmolLM2-135M, identical architecture
  and tokenizer configuration, weights from the pinned seed 20260802,
  Linear(576, 512). The within-size non-pretrained causal control.

A1 and A1r are an inseparable pair and were never run at a scale where the
other did not.

## Method

Eighteen cells: three arms at train_40k and train_250k with seeds 0, 1 and 2.
Three are the reused Phase-1B pilots and were not rerun; the other fifteen ran
under the frozen execution manifest. The recipe is the inherited v3_01
reasoner recipe, unchanged and not searched: AdamW at 3e-4, batch 128, dropout
0.1, weight decay 0.01 on parameters of two or more dimensions, gradient clip
1.0, cosine after warmup stepped per optimizer step over 100 x steps_per_epoch,
at most 100 epochs, early stopping at patience 10, checkpoint at best
development accuracy with the earliest epoch winning a tie, bf16 autocast on
the training path and fp32 evaluation.

Every encoder is frozen. The only newly trainable component in the language
model path is the one linear projection. The projected interface is a learned
common width and is not a naturally shared pretrained embedding space; for A1
and A1r the image and question representations do not originate from one
model, and every statement about them says so.

Selection and every reported number use data/v2/dev.csv, 7,714 in-vocabulary
questions over 768 development images. The clean test was not read.

Before any statistic, each cell passes the assertions in
experiments/e8a_question_encoder/analysis_integrity.py: the stored prediction
rows are the canonical dev.csv rows compared elementwise by questionId, the
stored scoring targets equal the vocabulary labels, the record identifies
itself as the arm, scale and seed it is filed under, and every checkpoint and
prediction hash is re-measured at analysis time rather than trusted from the
run record. The reported accuracy is recomputed from the stored per-row
vectors and must agree to within 1e-5.

Intervals are the v3_03 image-clustered bootstrap: 2,000 draws from a fresh
default_rng(0), clusters are represented development imageIds, seeds combined
by the within-draw mean, 2.5 and 97.5 percentiles. The universal directional
rule is applied exactly: positive evidence requires mean > 0, interval lower
bound > 0 and at least two of three seed differences > 0; negative evidence
requires the mirror; everything else is uncertain or mixed.

The >=4-step deficit is the repaired v3_02a definition — the unweighted mean
of the four per-bucket lifts over the fixed train_40k prior, minus the
question-weighted combined >=4 lift — with the v2_05b per-bucket priors held
fixed at both scales. Those priors are reloaded from the stored addendum and
independently recomputed from train_40k; they reproduce it exactly
(0.20917, 0.14625, 0.27356, 0.51616).

## Outputs

results/experiments/e8a_question_encoder/: eighteen run records, eighteen
prediction-vector files, eighteen checkpoints, core_analysis.json,
core_analysis_train_40k.json, efficiency_resources.json,
invocation_status_train_250k.json and e8a_135m_core_frozen.json, which
collects all of it with every artefact hashed at freeze time.

tests/test_e8a.py now runs 187 checks, up from 149, the addition being the
four invocation-outcome cases and their two non-vacuity controls. All test
modules pass.

## Results

Development set only. No result here is a test result.

### Accuracy, three seeds

| arm | train_40k | train_250k |
| --- | --- | --- |
| A0p | 0.54044 +/- 0.00438 | 0.59446 +/- 0.00030 |
| A1  | 0.52861 +/- 0.00296 | 0.57316 +/- 0.00153 |
| A1r | 0.49041 +/- 0.00259 | 0.52480 +/- 0.00487 |

### The three contrasts

| contrast | scale | per-seed | mean | 95% CI | rule |
| --- | --- | --- | --- | --- | --- |
| A1 - A1r | 40k | 0.03565, 0.04304, 0.03591 | +0.03820 | [0.02947, 0.04680] | positive |
| A1 - A1r | 250k | 0.04472, 0.04874, 0.05159 | +0.04835 | [0.03935, 0.05803] | positive |
| A1 - A0p | 40k | -0.01037, -0.01322, -0.01193 | -0.01184 | [-0.01996, -0.00427] | negative |
| A1 - A0p | 250k | -0.01970, -0.02191, -0.02230 | -0.02130 | [-0.02983, -0.01253] | negative |
| A0p - A1r | 40k | 0.04602, 0.05626, 0.04784 | +0.05004 | [0.04144, 0.05864] | positive |
| A0p - A1r | 250k | 0.06443, 0.07065, 0.07389 | +0.06966 | [0.05951, 0.07920] | positive |

A1 minus A1r is the primary within-size pretrained-versus-random comparison.
It is positive at both scales and grows with scale. The degeneracy rule does
not fire in any of the six A1r cells, so the comparison is not an artefact of
a collapsed control: across those six cells A1r predicts 89 to 100 distinct
answers with entropy 0.680 to 1.023 nats and a maximum class share of 0.244 to
0.317, against collapse thresholds of 0.30 nats and 0.60.

A1 minus A0p is a matched **system** comparison, not an isolation of language
model semantics. It changes two things at once: the semantics of the question
representation, and whether the image and question tokens originate from
CLIP's shared pretrained space. The recipe was historically selected on the
CLIP-question-token reasoner, which favours A0p. It is negative at both
scales and becomes more negative with scale. No equivalence language is used
anywhere; a negative result is reported as a negative result.

The interval conditions on the fixed set of three trained seeds and does not
fully propagate training-seed uncertainty. The measured across-seed standard
deviation in the stored v3_01 runs is 0.0037 at 40k and 0.0074 at 250k.

### Scale

| arm | 250k - 40k accuracy | 95% CI | rule |
| --- | --- | --- | --- |
| A0p | +0.05401 | [0.04444, 0.06320] | positive |
| A1  | +0.04455 | [0.03531, 0.05418] | positive |
| A1r | +0.03440 | [0.02458, 0.04400] | positive |

Every arm improves with more data and the ordering of the arms is unchanged.
Convergence is early throughout: best epochs are 3 to 7 except A1 at 250k
seed 2, which peaked at 14. Time per epoch is 15.7-16.1 s at 40k and
88.8-89.6 s at 250k; peak allocated memory is 1122-1274 MiB against a
16,014 MiB ceiling.

### Multi-step reasoning

The pooled >=4-step deficit persists in every arm at every scale, between
0.082 and 0.109. Higher overall accuracy at 250k is not read as improved
multi-step reasoning; the deficit is reported separately.

| arm | 40k | 250k | change | 95% CI | rule |
| --- | --- | --- | --- | --- | --- |
| A0p | +0.09389 | +0.08212 | -0.01177 | [-0.02174, -0.00160] | negative |
| A1  | +0.10594 | +0.08826 | -0.01768 | [-0.02802, -0.00784] | negative |
| A1r | +0.09831 | +0.10921 | +0.01091 | [0.00075, 0.02143] | positive |

The two arms with a pretrained question representation reduce their deficit
with more data; the random control's deficit grows. At 250k the paired deficit
contrasts are A1 - A1r -0.02096 [-0.03185, -0.01088] and A0p - A1r
-0.02709 [-0.03898, -0.01558], both negative directional; A1 - A0p is
+0.00614 [-0.00272, 0.01548], uncertain. At 40k only A1 - A0p resolves
(+0.01204 [0.00238, 0.02166]).

### Reliance

Three-seed means of normal minus each intervention.

| arm | scale | fixed image | fixed question | shuffled image |
| --- | --- | --- | --- | --- |
| A0p | 40k | +0.08975 | +0.47744 | +0.11347 |
| A1  | 40k | +0.08880 | +0.30149 | +0.11002 |
| A1r | 40k | +0.09805 | +0.27945 | +0.10954 |
| A0p | 250k | +0.12911 | +0.54230 | +0.15093 |
| A1  | 250k | +0.10894 | +0.34435 | +0.13888 |
| A1r | 250k | +0.10396 | +0.47044 | +0.12397 |

Image reliance rises with scale in every arm. The largest single change is
A1r's question reliance, +0.19099 from 40k to 250k, far above A0p's +0.06486
and A1's +0.04286: with more data the random-feature arm comes to depend much
more on the question channel it cannot read semantically.

The fixed-question intervention changes both question content and sequence
length, and by different amounts across arms. It is **not** a pure semantic
intervention and the numbers above must not be read as one. The shuffled-image
condition is the imageId-level derangement with zero self-pairs.

### Efficiency

Cached-head boundary: the trained head on cached question states and cached
CLIP image tokens, no encoder forward pass.

| row | dev acc | trainable | total loaded | cached-head ms | ckpt MiB | GPU-h |
| --- | --- | --- | --- | --- | --- | --- |
| A0p/40k | 0.54044 | 21,362,276 | 84,790,372 | 1.3848 | 81.53 | 0.0678 |
| A0p/250k | 0.59446 | 21,362,276 | 84,790,372 | 1.3347 | 81.53 | 0.4451 |
| A1/40k | 0.52861 | 21,395,044 | 155,910,052 | 1.3844 | 81.66 | 0.0676 |
| A1/250k | 0.57316 | 21,395,044 | 155,910,052 | 1.3418 | 81.66 | 0.4533 |
| A1r/40k | 0.49041 | 21,395,044 | 155,910,052 | 1.3468 | 81.66 | 0.0659 |
| A1r/250k | 0.52480 | 21,395,044 | 155,910,052 | 1.3422 | 81.66 | 0.3454 |

A0p at 250k is the only point on the latency, trainable-parameter,
total-loaded-parameter and checkpoint-size fronts: it is both the most
accurate and the cheapest on every one of those axes. On training GPU-hours
the front is A1r/40k, A1/40k, A0p/40k and A0p/250k. The SLM arms carry
134,515,008 frozen encoder parameters that A0p does not, for lower accuracy.

Two boundaries are kept apart and a third is not reported. Offline extraction
cost 0.126813 and 0.126589 GPU-hours per 40k store and 0.602786 and 0.597567
per 250k store; A0p writes no store, reusing the v3_00 CLIP question tokens.
No **encoder-inclusive** per-query cost is given for A1 or A1r: no online
SmolLM2-135M forward pass has been timed in this project and E7a measured no
SmolLM2 tower, so the figure does not exist and is not estimated. The E8A head
latencies use the 20 warm-up / 200 repeat mean protocol that E7a explicitly
supersedes, so the front above is built only from E8A's own internally
consistent measurements; E7a's reasoner point (arm A0, 1.39683 ms head-only,
across-pass spread 0.0625 ms) appears as context and is not on this front.
Given that spread, the 0.05 ms differences between E8A rows are not resolved.

### Resources

Measured, not projected: 5.78927 GPU-hours in total, being 4.33551 training
over eighteen cells and 1.45376 extraction over four stores. Evaluation runs
inside each training invocation and is inside that wall clock, so it is not
double-counted and cannot be reported separately.

| gate | ceiling | measured | fires |
| --- | --- | --- | --- |
| per run | 8 GPU-h | 0.59183 (A1/250k/seed2) | no, margin 7.408 h |
| per model | 35 GPU-h | 5.78927 | no, margin 29.211 h |
| memory | 80 per cent, 16,014 MiB | 1274.1 MiB | no |
| storage | project allocation | 6.757 GiB written, 331.2 GiB free | no |

No cell failed, was interrupted or was retried. The per-model gate is a
**lower bound**: the canonical aggregate also counts the E8B readout arms,
which are not authorised and have not run, so it is not finally discharged.
The E8B runtime multiplier, any complete cross-E8 aggregate and the
min(180, three times expected) core ceiling all remain unresolved for the same
reason.

The 426-versus-420 discrepancy in the canonical resource tables is
presentational. The six evaluation passes counted in section 13.1 but not
itemised in 13.2 are the B1 classification evaluations, and the stated totals
already include them. No GPU-hour figure is wrong and none is changed here.

## Decisions and problems

The runner decided its exit code from every to_run cell at the scale rather
than from the cells the invocation requested, so a deliberately chunked run
returned 2 after training and validating everything asked of it, which under
`set -e` reads as failure. It now reports invocation_complete,
authorised_matrix_incomplete or actual_failure, and only the last is non-zero.
Training, evaluation, selection and every artefact a completed run wrote are
unchanged by that correction.

The earlier 40k analysis was produced from a dirty worktree. It was
regenerated from committed state 8d301d0 against the same frozen model
artefacts, with no retraining. Every pre-existing value is identical: all
three per-arm summaries, all three contrasts including their intervals and
directional outcomes, the degeneracy block and every shared per-cell field.
The only differences are the added fields and the clean provenance.

analyse_core.py's docstring promised the scale, reliance and >=4-step
analyses; none was implemented and STEP_ORDER was an unused constant. All
three are now implemented from the frozen definitions.

extraction_train_250k.json's storage_projection.train_40k_plus_dev block
repeats its own 250k figures rather than the 40k ones. The 40k storage numbers
are therefore taken from extraction.json, which is the only record that
measured them. The defective block is left as written rather than edited after
the fact.

Two record-shape drifts were handled rather than papered over: the Phase-1A
pilot spells the frozen parameter count `frozen_language_model_parameters`
where later runs spell it `frozen_question_encoder_parameters`, and the three
reused seed-0 prediction files predate the scale-qualified naming, so their
paths are read from the manifest and never derived. Either would have silently
dropped cells from an aggregator.

A0 is not A0p. The stored v3_01 reasoner has no interface-matching projection,
is 21,099,620 parameters, and appears only as labelled context.

The store-agreement gate was run standalone as well as in-run: for each of A1
and A1r the train_40k and train_250k question stores hold bitwise identical
states on all 47,714 shared questionIds, maximum absolute deviation 0.0, so
the 40k-versus-250k comparison is not confounded by different frozen question
features.

Nothing here authorises SmolLM2-360M, E8B, E9, F1, F2 or any clean-test
access, and none was run.
