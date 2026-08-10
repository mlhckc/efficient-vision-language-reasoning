# E8B core matrix: results and analysis

## Purpose

Report the frozen eighteen-cell E8B core matrix and the analysis that can
be computed from its artefacts. The matrix asks whether a frozen
pretrained small language model used as an answer readout (B3) beats a
within-size randomly initialised control (B2), and how both compare with
a language-model-free readout (B1), at two training scales.

## Method

Three arms, two scales, three seeds, executed in the pair-preserving
order at commit 1b4ea42. B2 and B3 use the frozen recipe (lr 3e-4, no
warmup, dropout 0.1, exactly 22 epochs, no early stopping, epoch 22
canonical). B1 keeps its own frozen section-7.3 classifier recipe: a
100-epoch budget with early stopping and best-on-development selection.
Training is bf16 autocast; the canonical scientific evaluation is fp32
from the trainable trunk onward.

The primary metric is canonical FP32 R1 accuracy on the 7,714
in-vocabulary development rows. Uncertainty follows the universal primary
directional rule of canonical plan section 5, applied through the
existing E8A implementation: an image-clustered paired bootstrap over 768
image clusters, 2,000 draws from a fresh `default_rng(0)`, a 95 per cent
percentile interval, and the three-condition rule requiring the mean
difference, the interval bound and at least two of three seeds to agree
in sign. No statistical test was added, substituted or tuned for E8B.

Every interval conditions on the fixed trained seed set and does not
fully propagate training variance. The three per-seed differences are
reported beside every interval.

## Outputs

- `results/experiments/e8b_readout_generation/e8b_core_analysis_20260809.json`
- `results/experiments/e8b_readout_generation/e8b_core_analysis_20260809.png`
- 18 core result records, canonical and resume checkpoints, per-row dumps
- `experiments/e8b_readout_generation/core_analysis.py`, read-only

## Results

Development-set results only. The clean test is embargoed and untouched.
Primary metric: canonical FP32 R1 accuracy on the 7,714 in-vocabulary
development rows, 768 image clusters.

### Table 1 (PRIMARY) — canonical FP32 development accuracy

| arm | 40k mean ± SD | 40k seeds | 250k mean ± SD | 250k seeds | 40k → 250k |
|---|---|---|---|---|---|
| B1 | 0.54507 ± 0.00390 | 0.54952, 0.54226, 0.54343 | 0.59394 ± 0.00486 | 0.58841, 0.59749, 0.59593 | +0.04887 |
| B2 | 0.53392 ± 0.00233 | 0.53215, 0.53656, 0.53306 | 0.60086 ± 0.00528 | 0.59593, 0.60021, 0.60643 | +0.06694 |
| B3 | 0.50929 ± 0.02795 | 0.53124, 0.47783, 0.51880 | 0.59610 ± 0.00270 | 0.59826, 0.59697, 0.59308 | +0.08681 |

### Table 2 (PRIMARY) — paired contrasts, universal directional rule

| contrast | scale | mean diff | per-seed | seeds > 0 | 95% CI | outcome | clean causal |
|---|---|---|---|---|---|---|---|
| B3 - B2 | 40k | -0.02463 | -0.00091, -0.05872, -0.01426 | 0/3 | [-0.03333, -0.01619] | negative directional evidence | yes |
| B2 - B1 | 40k | -0.01115 | -0.01737, -0.00570, -0.01037 | 0/3 | [-0.01965, -0.00299] | negative directional evidence | NO - selection rules differ |
| B3 - B1 | 40k | -0.03578 | -0.01828, -0.06443, -0.02463 | 0/3 | [-0.04420, -0.02749] | negative directional evidence | NO - selection rules differ |
| B3 - B2 | 250k | -0.00475 | +0.00233, -0.00324, -0.01335 | 1/3 | [-0.01208, +0.00297] | uncertain or mixed evidence | yes |
| B2 - B1 | 250k | +0.00691 | +0.00752, +0.00272, +0.01050 | 3/3 | [-0.00112, +0.01491] | uncertain or mixed evidence | NO - selection rules differ |
| B3 - B1 | 250k | +0.00216 | +0.00985, -0.00052, -0.00285 | 1/3 | [-0.00511, +0.00952] | uncertain or mixed evidence | NO - selection rules differ |

### Table 3 (PRIMARY) — within-arm scale effect, 250k minus 40k

| arm | mean diff | per-seed | 95% CI | outcome |
|---|---|---|---|---|
| B1 | +0.04887 | +0.03889, +0.05522, +0.05250 | [+0.03943, +0.05801] | positive directional evidence |
| B2 | +0.06693 | +0.06378, +0.06365, +0.07337 | [+0.05683, +0.07633] | positive directional evidence |
| B3 | +0.08681 | +0.06702, +0.11913, +0.07428 | [+0.07781, +0.09640] | positive directional evidence |

### Table 4 (SECONDARY DIAGNOSTIC) — stability, primary versus best-of-22

| arm/scale | primary mean | primary SD | seed range | best-of-22 − primary (mean) | (max) |
|---|---|---|---|---|---|
| B1/train_40k | 0.54507 | 0.00390 | 0.00726 | n/a (B1 rule) | n/a |
| B1/train_250k | 0.59394 | 0.00486 | 0.00908 | n/a (B1 rule) | n/a |
| B2/train_40k | 0.53392 | 0.00233 | 0.00441 | +0.00130 | +0.00376 |
| B2/train_250k | 0.60086 | 0.00528 | 0.01050 | +0.00453 | +0.00881 |
| B3/train_40k | 0.50929 | 0.02795 | 0.05341 | +0.02684 | +0.05160 |
| B3/train_250k | 0.59610 | 0.00270 | 0.00518 | +0.00315 | +0.00557 |

### Table 5 (PRIMARY) — resource close-out

| quantity | value |
|---|---|
| A. scientific successful-run compute | 35.46092 GPU-h |
| B. operational charged compute | 36.77751 GPU-h |
| difference (failed cell-1 attempt, never netted out) | 1.31659 GPU-h |
| identity `pretrained`: charged / projected / headroom | 27.9777 / 35.9417 / 4.0583 GPU-h, gate fires False |
| identity `random`: charged / projected / headroom | 18.6621 / 21.5631 / 18.4369 GPU-h, gate fires False |
| identity `none`: charged / projected / headroom | 2.119 / 3.2229 / 36.7771 GPU-h, gate fires False |
| trainable per LM arm (trunk + projection) | 21,343,808 |
| frozen language model | 134,515,008 |
| B1 trainable readout | 52,836 |
| checkpoint storage, 36 files | 7.133 GiB |

### Tables 6 and 7 — NOT PRODUCED

The R1/R2/R3 readout table and the intervention table require the
frozen readout evaluation, which has not run on these checkpoints.
See the `not_computed` block of e8b_core_analysis_20260809.json.


### Table 6 (PRIMARY) — readouts, normal condition, mean ± SD over 3 seeds

| arm | scale | readout | in-vocab raw | in-vocab norm. | raw-denom raw | raw-denom norm. |
|---|---|---|---|---|---|---|
| B1 | 40k | classifier | 0.54507 ± 0.00390 | 0.54507 ± 0.00390 | 0.42030 ± 0.00301 | 0.42030 ± 0.00301 |
| B1 | 250k | classifier | 0.59394 ± 0.00485 | 0.59394 ± 0.00485 | 0.45798 ± 0.00374 | 0.45798 ± 0.00374 |
| B2 | 40k | R1 | 0.53392 ± 0.00233 | 0.53392 ± 0.00233 | 0.41170 ± 0.00179 | 0.41170 ± 0.00179 |
| B2 | 40k | R2 | 0.53409 ± 0.00280 | 0.53409 ± 0.00280 | 0.41184 ± 0.00216 | 0.41184 ± 0.00216 |
| B2 | 40k | R3 | 0.53370 ± 0.00277 | 0.53370 ± 0.00277 | 0.41154 ± 0.00214 | 0.41154 ± 0.00214 |
| B2 | 250k | R1 | 0.60086 ± 0.00528 | 0.60086 ± 0.00528 | 0.46331 ± 0.00407 | 0.46331 ± 0.00407 |
| B2 | 250k | R2 | 0.60081 ± 0.00519 | 0.60081 ± 0.00519 | 0.46328 ± 0.00401 | 0.46328 ± 0.00401 |
| B2 | 250k | R3 | 0.60064 ± 0.00513 | 0.60064 ± 0.00513 | 0.46315 ± 0.00395 | 0.46315 ± 0.00395 |
| B3 | 40k | R1 | 0.50929 ± 0.02794 | 0.50929 ± 0.02794 | 0.39271 ± 0.02155 | 0.39271 ± 0.02155 |
| B3 | 40k | R2 | 0.50916 ± 0.02863 | 0.50916 ± 0.02863 | 0.39261 ± 0.02208 | 0.39261 ± 0.02208 |
| B3 | 40k | R3 | 0.50825 ± 0.02896 | 0.50825 ± 0.02896 | 0.39201 ± 0.02231 | 0.39201 ± 0.02231 |
| B3 | 250k | R1 | 0.59610 ± 0.00270 | 0.59610 ± 0.00270 | 0.45965 ± 0.00208 | 0.45965 ± 0.00208 |
| B3 | 250k | R2 | 0.59580 ± 0.00234 | 0.59580 ± 0.00234 | 0.45942 ± 0.00180 | 0.45942 ± 0.00180 |
| B3 | 250k | R3 | 0.59558 ± 0.00231 | 0.59558 ± 0.00231 | 0.45925 ± 0.00178 | 0.45925 ± 0.00178 |

### Table 7 (PRIMARY) — R3 emission validity, every row retained

| arm | scale | in-vocab | out-of-vocab | empty | overlong |
|---|---|---|---|---|---|
| B2 | 40k | 0.9790 | 0.0208 | 0.000133 | 0.000000 |
| B2 | 250k | 0.9898 | 0.0102 | 0.000033 | 0.000000 |
| B3 | 40k | 0.9766 | 0.0226 | 0.000633 | 0.000133 |
| B3 | 250k | 0.9913 | 0.0086 | 0.000000 | 0.000067 |

### Table 8 (PRIMARY) — interventions, drop from each arm's own normal

| arm | scale | condition | accuracy ± SD | drop |
|---|---|---|---|---|
| B1 | 40k | normal | 0.54507 ± 0.00390 | 0.00000 |
| B1 | 40k | fixed_image | 0.45497 ± 0.00757 | 0.09010 |
| B1 | 40k | fixed_question | 0.23114 ± 0.00259 | 0.31393 |
| B1 | 40k | shuffled_image | 0.42620 ± 0.00470 | 0.11887 |
| B1 | 250k | normal | 0.59394 ± 0.00485 | 0.00000 |
| B1 | 250k | fixed_image | 0.47403 ± 0.00871 | 0.11991 |
| B1 | 250k | fixed_question | 0.21364 ± 0.01997 | 0.38030 |
| B1 | 250k | shuffled_image | 0.44154 ± 0.00500 | 0.15241 |
| B2 | 40k | normal | 0.53392 ± 0.00233 | 0.00000 |
| B2 | 40k | fixed_image | 0.44901 ± 0.00273 | 0.08491 |
| B2 | 40k | fixed_question | 0.22708 ± 0.00746 | 0.30685 |
| B2 | 40k | shuffled_image | 0.42226 ± 0.00378 | 0.11166 |
| B2 | 250k | normal | 0.60086 ± 0.00528 | 0.00000 |
| B2 | 250k | fixed_image | 0.47580 ± 0.01232 | 0.12505 |
| B2 | 250k | fixed_question | 0.16014 ± 0.06376 | 0.44072 |
| B2 | 250k | shuffled_image | 0.43747 ± 0.00279 | 0.16338 |
| B3 | 40k | normal | 0.50929 ± 0.02794 | 0.00000 |
| B3 | 40k | fixed_image | 0.41081 ± 0.04742 | 0.09848 |
| B3 | 40k | fixed_question | 0.16511 ± 0.08946 | 0.34418 |
| B3 | 40k | shuffled_image | 0.39979 ± 0.02797 | 0.10950 |
| B3 | 250k | normal | 0.59610 ± 0.00270 | 0.00000 |
| B3 | 250k | fixed_image | 0.46811 ± 0.00271 | 0.12799 |
| B3 | 250k | fixed_question | 0.08759 ± 0.05346 | 0.50851 |
| B3 | 250k | shuffled_image | 0.43691 ± 0.00638 | 0.15919 |

### Table 9 (PRIMARY) — image reliance, normal − deranged, 777 clusters

| arm | scale | drop | per-seed | 95% CI | outcome |
|---|---|---|---|---|---|
| B1 | 40k | +0.09166 | +0.09906, +0.08637, +0.08956 | [+0.08153, +0.10227] | positive directional evidence |
| B1 | 250k | +0.11752 | +0.11485, +0.12305, +0.11465 | [+0.10654, +0.12915] | positive directional evidence |
| B2 | 40k | +0.08610 | +0.08507, +0.09086, +0.08237 | [+0.07515, +0.09776] | positive directional evidence |
| B2 | 250k | +0.12598 | +0.12005, +0.12545, +0.13245 | [+0.11477, +0.13750] | positive directional evidence |
| B3 | 40k | +0.08443 | +0.08816, +0.08507, +0.08007 | [+0.07447, +0.09506] | positive directional evidence |
| B3 | 250k | +0.12275 | +0.12025, +0.12215, +0.12585 | [+0.11146, +0.13443] | positive directional evidence |

### Table 10 (PRIMARY) — B3 − B2 on the raw 10,004-row denominator

| scale | mean diff | per-seed | 95% CI | outcome |
|---|---|---|---|---|
| 40k | -0.01899 | -0.00070, -0.04528, -0.01100 | [-0.02548, -0.01205] | negative directional evidence |
| 250k | -0.00367 | +0.00180, -0.00250, -0.01030 | [-0.00965, +0.00215] | uncertain or mixed evidence |

### Table 11 (PRIMARY) — compute close-out

| quantity | value |
|---|---|
| measured final-evaluation compute | 3.84396 GPU-h |
| projection line item (LM readouts + interventions) | 2.8 GPU-h |
| A. E8B-only successful training compute | 35.46092 GPU-h |
| B. E8B-only operational charged compute (incl. failed attempt) | 36.77751 GPU-h |
| C. programme-wide identity totals, **do not sum to B** | pretrained 27.9777 h, random 18.6621 h, none 2.119 h |

## Hypotheses

Re-tested against the complete evidence, including the readouts and
interventions. None is forced to be supported.

**H1, data scale is the dominant driver: SUPPORTED.** Every arm gains
under the directional rule (B1 +0.04887, B2 +0.06693, B3 +0.08681, all
three intervals excluding zero). Scale effects are larger than every
observed between-arm mean contrast and are substantially larger than the
between-arm differences at 250k.

**H2, B3 pretraining gives little or no primary accuracy advantage over
B2: SUPPORTED.** At 40k, B3 minus B2 is negative directional evidence
(-0.02463 in-vocabulary, -0.01899 on the raw denominator). At 250k it is
uncertain or mixed and close to parity (-0.00475 and -0.00367). There is
therefore no evidence of a positive pretraining advantage at either
scale. This is not a proof of absence at 250k: the interval there
includes zero and admits small effects in both directions.

**H3, B3 small-data instability is substantially reduced at 250k:
SUPPORTED.** B3's seed SD falls from 0.02795 to 0.00270 and its
best-of-22 minus primary gap from +0.02684 to +0.00315. The reduction is
specific to B3: B2's SD rises slightly with scale.

**H4, the language-model-backed architecture provides only a small
incremental advantage, if any, over B1 at large scale: SUPPORTED, as a
system-level statement.** B3 minus B1 at 250k is +0.00216 and B2 minus
B1 is +0.00691, both uncertain or mixed. The differing
checkpoint-selection rules may favour B1 relative to fixed-epoch
systems, so these are comparisons of whole configurations, not clean
causal architecture contrasts.

**Readouts agree with each other.** R1, R2 and R3 sit within about
0.0005 of one another on every arm and scale, so the choice of readout
is not what separates these systems. R3 stays valid: 97.7 to 99.1 per
cent of free emissions are in vocabulary, overlong output is at most
0.013 per cent, and every invalid emission stays in the denominator.

**All three arms rely on the image, and rely on it more at scale.**
Deranging the image costs 0.084 to 0.092 accuracy at 40k and 0.118 to
0.126 at 250k, positive directional evidence everywhere. Removing the
question is far more damaging than removing the image in every
configuration.

## Decisions and problems

**Two arms, one clean comparison.** B3 minus B2 is the only contrast
whose arms share a checkpoint-selection rule, so it is the only clean
causal test of pretraining in this matrix. Any comparison involving B1
confounds architecture with selection rule and is labelled as such
wherever it appears.

**A defect was found and repaired during execution.** The first
B3/train_40k/seed0 attempt trained all 22 epochs and then died in the G10
collapse diagnostic: `canonical_dev_predictions` returns a numpy array
while `mean_prediction_entropy_nats` required a torch tensor. All twelve
B2/B3 cells would have failed the same way. The consumer now accepts both
types and fails loudly on anything else, and the branch was extracted so
a test executes it. The cell was retrained from scratch rather than
resumed, because the source-digest resume gate correctly refuses to
splice a trajectory across implementations; the retrained epoch-22
checkpoint is bit-identical to the archived first attempt, which is
direct evidence the repair did not touch the learned trajectory.

**The failed attempt is charged, not netted out.** Its 1.31659 GPU-h
appears in the operational total and is excluded from the scientific
total. It consumed the pretrained identity's one forced-retry allowance,
recorded under `gate_requires_corrected_rerun`.

**This analysis is incomplete by design, not by omission.** The R1/R2/R3
readout evaluation and the three interventions have not run on these
checkpoints. `final_evaluation.py` implements and validates them, but
nothing in the repository drives it over the eighteen frozen
checkpoints. That driver is new executable scientific code producing
primary results and was not written unilaterally.

**No clean-test claim is made.** Every number here is a development-set
result. The clean test remains embargoed and untouched.

**The final evaluation over-ran its projection line item.** The measured
language-model readout and intervention compute is 3.84396 GPU-h against
a 2.8 GPU-h line item. It stayed inside every per-identity committed
reservation, fired no gate and touched no headroom; the over-run is
recorded rather than absorbed.

**Compute is reported in three separate scopes.** A, E8B-only successful
training compute, is 35.46092 GPU-h. B, E8B-only operational charged
compute, is 36.77751 GPU-h and includes the failed first attempt. C, the
programme-wide per-identity ledger, is a different question entirely and
does not sum to B, because it includes E8A, the abandoned search and
probe hours spent before this matrix existed.
