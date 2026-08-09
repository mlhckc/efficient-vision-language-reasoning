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
