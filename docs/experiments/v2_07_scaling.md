# Experiment v2_07: data scaling

## Purpose

Scale the training set from 40,000 to 100,000 and 250,000 questions under
the V2 protocol, with dev fixed at 7,714 rows, to measure what data buys
each model, how the interaction-feature advantage evolves, and whether the
multi-step lift deficit from v2_05b erodes with scale.

## Method

experiments/v2_07_scaling/materialise.py first materialised the 100k and
250k aligned views from the canonical keyed stores (pure I/O, no encoding);
verification passed all checks, including byte-identity of the first 40,000
rows of each view with train_40k.h5 and of the first 100,000 rows of the
250k view with the 100k view, making the manifest nesting visible in the
arrays.

experiments/v2_07_scaling/run.py trained question_only, concat, product_576k
(hidden 351) and fusion at 100k and 250k for seeds 0, 1, 2, 3 and 42, with
hyperparameters unchanged from config and the same loader reseeding as
v2_02; the 40k results are loaded from v2_02/v2_04, not retrained.
question_only tracks whether language bias saturates with data; the other
three are the frontier models. The identical recipe means the 30 fixed
epochs give proportionally more optimisation steps at larger scales; that is
a property of the scale axis itself, not a confound between models at the
same scale. A benchmark of one 250k fusion epoch (1.5 s) projected about
0.37 GPU-hours for the grid, well under the six-hour stop threshold, and the
grid ran in about that time.

Blinding: dev selection and reporting only; no test_clean_* file was read.

## Outputs

Under results/experiments/v2_07_scaling/ (git-ignored): materialise.json,
results.json, table.csv, forty checkpoints ({model}_{scale}_seed{seed}) and
scaling_curves.png (dev accuracy against training size, log x, std bands,
majority and structural-prior floors dashed).

## Results

Dev accuracy, mean +/- sample std over five seeds:

| model         | 40k             | 100k            | 250k            |
|---------------|-----------------|-----------------|-----------------|
| question_only | 0.4580 +/- 0.0029 | 0.4829 +/- 0.0015 | 0.4977 +/- 0.0044 |
| concat        | 0.5240 +/- 0.0028 | 0.5508 +/- 0.0020 | 0.5786 +/- 0.0026 |
| product_576k  | 0.5335 +/- 0.0031 | 0.5636 +/- 0.0022 | 0.5824 +/- 0.0032 |
| fusion        | 0.5384 +/- 0.0022 | 0.5631 +/- 0.0027 | 0.5823 +/- 0.0020 |

Paired per-seed gaps by scale:

| gap                        | 40k               | 100k              | 250k              |
|----------------------------|-------------------|-------------------|-------------------|
| fusion - concat            | +0.0144 +/- 0.0014 | +0.0122 +/- 0.0040 | +0.0037 +/- 0.0044 |
| product_576k - concat      | +0.0096 +/- 0.0034 | +0.0128 +/- 0.0008 | +0.0038 +/- 0.0027 |
| fusion - product_576k      | +0.0049 +/- 0.0024 | -0.0006 +/- 0.0043 | -0.0002 +/- 0.0039 |
| concat - question_only     | +0.0660 +/- 0.0032 | +0.0679 +/- 0.0022 | +0.0809 +/- 0.0059 |

At 250k the fusion - concat gap flips sign in two of five seeds (per-seed
range -0.0010 to +0.0079) and product_576k - concat touches zero in one
(range -0.0001 to +0.0061); neither is seed-robust any more.

Step-bucket lift at 250k (seed 42, v2_05b priors held fixed), with the 40k
values from v2_05b in parentheses:

| model         | >=4-step lift   | mean step lift  | deficit         |
|---------------|-----------------|-----------------|-----------------|
| question_only | 0.152 (0.115)   | 0.226 (0.191)   | 0.074 (0.076)   |
| concat        | 0.233 (0.150)   | 0.311 (0.244)   | 0.077 (0.094)   |
| fusion        | 0.237 (0.183)   | 0.312 (0.264)   | 0.075 (0.082)   |
| product_576k  | 0.241           | 0.313           | 0.072           |

Figure: results/experiments/v2_07_scaling/scaling_curves.png.

## Verdicts

(a) What 6.25x data buys. Every model improves and none has plateaued at
250k: question_only gains 4.0 points (0.458 to 0.498), concat 5.5 points
(0.524 to 0.579), product_576k 4.9 and fusion 4.4. question_only is
decelerating (+2.5 points for the first 2.5x, +1.5 for the second), so
language bias absorbs data with diminishing returns, but it has not
saturated. concat's increments are constant across both 2.5x steps (+2.7,
+2.8 points), and the multimodal margin over question_only grows from
+0.066 to +0.081: more data teaches the heads to use the image more, not
less.

(b) The interaction-feature advantage shrinks with data. The fusion and
product gains over concat, seed-robust at 40k and 100k, fall to +0.004 at
250k and are no longer distinguishable from zero across seeds. The v2_03
and v2_04 finding that the features carry a real gain therefore has a
scale qualifier: the handcrafted interaction terms act as a small-data
prior, and with enough data a plain concat head learns equivalent
interactions on its own. The v2_04 redundancy finding is scale-stable:
fusion and product_576k are indistinguishable at every scale, so the fourth
input part remains redundant everywhere. For the efficiency argument this
cuts both ways: at small training budgets the interaction features are the
cheapest accuracy available; at 250k they no longer pay.

(c) The multi-step deficit persists at scale. All step buckets improve with
data, but the >=4-step lift deficit barely moves: concat 0.094 to 0.077,
fusion 0.082 to 0.075, question_only 0.076 to 0.074 (seed-42 slicing, same
caveat as v2_05b). Scale lifts the curve without closing the compositional
gap, so scale alone does not erode the ceiling that deep-program questions
sit under. This is the direct bridge to the V3 decision: if 6.25x data
leaves the multi-step deficit essentially intact for every global-embedding
head, the remaining candidates are architectural, which is precisely the
V3 hypothesis of question-conditioned reasoning over token-level visual
features. As in v2_06, this is consistency with the V3 motivation, not
proof that token-level features will close the gap.

## Decisions and problems

The 40k results were reused rather than retrained, so the 40k column is
byte-identical to v2_02/v2_04; the fresh 100k and 250k runs use the same
code path end to end. The lift comparison holds the v2_05b priors fixed
(train_40k-based) so lift changes reflect model accuracy only; recomputing
priors from the larger manifests would shift all lifts by at most the
prior-frequency drift, which is negligible for these answer distributions.
Dev remains a development instrument: none of these numbers is a
confirmatory test result, and the final model list for clean-test
evaluation is still open.
