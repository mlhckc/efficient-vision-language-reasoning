# Experiment v3_03 (E1): reasoner scaling to 100k and 250k

## Purpose

Test whether the negative 40k result of v3_01 holds at scale: under the
identical frozen recipe, does the 21.1M latent-query reasoner beat the
small global heads at 100k and 250k training questions, and does the
question-weighted pooled >=4-step deficit change? Authorized by the user
on 30 July 2026 (E1) and launch-instructed on 31 July 2026.

## Method

experiments/v3_03_scaling/run.py imports the v3_01 training function,
optimizer, scheduler and every recipe constant unchanged
(experiments/v3_01_reasoner/run.py: AdamW, weight decay 1e-2 on
non-bias/non-LayerNorm parameters, batch 128, gradient clip 1.0, cosine
schedule, patience 10, max 100 epochs, bf16 autocast training with fp32
evaluation, best-on-dev checkpointing) and reads the frozen selected
configuration from the stored v3_01 results (learning rate 3e-4, no
warmup, dropout 0.1). The only runtime override redirects checkpoint
output to results/experiments/v3_03_scaling/checkpoints/; no v3_01 or
v2_07 artifact is written. Six final runs: seeds {0, 1, 2} at train_100k
and train_250k. Nothing was tuned on the new results.

Gates, all passed before training: frozen-recipe identity against the
live v3_01 module and its stored record; token-store coverage of both
manifests and dev (zero missing IDs); one-batch forward and padding-mask
corruption per scale (corruption changed logits by exactly 0.0);
1,000-example overfit (0.9970 at epoch 9); per-scale memory/throughput
pilots (projected 36.9 s and 76.9 s per epoch; worst case 1.03 h and
2.14 h per run against the 12 h abort gate; peak about 1,060 MiB);
gradient hygiene. A fresh-context reviewer approved the script for
recipe fidelity before launch; the reviewed code was pinned as commit
cf3d948 (script SHA-256 ec4769f7...) and the run was launched detached
(re-parented to init) with an unbuffered log.

Comparisons: paired same-seed gaps against the stored v2_07
question_only/concat/product_576k/fusion accuracies at the matching
scale, and per-question dev correctness for those heads re-evaluated
from the stored v2_07 checkpoints (evaluation only). The step analysis
uses the binding v3_02a definition: pooled >=4 lift is the
question-weighted pooling of the 4 and >=5 buckets under the fixed
v2_05b per-bucket priors, and the deficit is the mean of the four
bucket lifts minus that pooled lift. The paired reasoner-fusion deficit
difference carries an image-clustered bootstrap (2,000 draws, NumPy
seed 0, 768 represented dev-image clusters, paired within every draw
across seeds 0/1/2). Three method notes from the pre-launch review:
the "mean" reported with each bootstrap interval is the full-sample
point estimate, not the mean over draws; the bootstrap holds the
per-bucket priors constant within draws, which cancels exactly in the
paired difference but would not be valid for per-model deficit
intervals; the trainable-parameter count (21,099,620) is the value
measured and printed by the v3_01 constructor. The paired gaps inherit
the seed-pairing clarification recorded in the v2 reports on 31 July
2026: v2_07 trained its models sequentially against one seeded loader
per (scale, seed), so these are same-seed, same-data comparisons, not
matched-stream comparisons.

Blinding: all selection and reporting on dev only;
test_clean_targets.csv was never read.

## Outputs

Under results/experiments/v3_03_scaling/ (git-ignored): preflight.json,
results.json, run.log and six checkpoints
reasoner_{100k,250k}_seed{0,1,2}.pt (84.4 MB each). preflight.json and
results.json are exported byte-identically to
artifacts/results_export/v3_03_scaling/; the checkpoints are hash-pinned
but not exported in artifacts/results_export/MANIFEST.json, and
run.log's hash is recorded in the task packet.

## Results

Dev accuracy per scale (best epoch in parentheses; early stopping in
every run):

| scale | seed 0 | seed 1 | seed 2 | mean | std |
|---|---|---|---|---|---|
| 100k | 0.57078 (4) | 0.57065 (11) | 0.57039 (5) | 0.5706 | 0.0002 |
| 250k | 0.59139 (6) | 0.59165 (12) | 0.60436 (7) | 0.5958 | 0.0074 |

The 250k standard deviation is driven by seed 2 (0.60436), the first
model in the project above 0.60 on dev; seeds 0 and 1 agree to 0.0003.

Paired same-seed gaps against the stored v2_07 heads (mean over seeds
0/1/2; minimum per-seed gap in parentheses; the reasoner beats every
head in every seed at both scales):

| gap | 100k | 250k |
|---|---|---|
| reasoner - fusion | +0.0065 (+0.0054) | +0.0136 (+0.0070) |
| reasoner - product_576k | +0.0083 (+0.0071) | +0.0126 (+0.0087) |
| reasoner - concat | +0.0211 (+0.0201) | +0.0173 (+0.0146) |
| reasoner - question_only | +0.0883 (+0.0866) | +0.0976 (+0.0874) |

For scale context, the v2_07 head means are concat 0.5508/0.5786,
product_576k 0.5637/0.5824 and fusion 0.5631/0.5823 at 100k/250k, and
the 40k v3_01 result was reasoner 0.5422 +/- 0.0037 against fusion
0.5384 (gap +0.0031 +/- 0.0062, negative in two of three seeds).

Question-weighted pooled >=4 step deficits (mean over seeds 0/1/2,
fixed priors):

| model | 100k | 250k |
|---|---|---|
| question_only | 0.0690 | 0.0638 |
| concat | 0.0958 | 0.0842 |
| product_576k | 0.0874 | 0.0771 |
| fusion | 0.0864 | 0.0803 |
| reasoner | 0.0860 | 0.0775 |

Paired reasoner - fusion deficit difference (image-clustered bootstrap):
100k -0.00037, 95% CI [-0.01211, +0.01057]; 250k -0.00275, 95% CI
[-0.01323, +0.00760]. Both intervals include zero.

Efficiency: 35.6 s/epoch at 100k and 86.2 s/epoch at 250k (measured);
total wall time 1.82 h for gates, six runs and the analysis; 21,099,620
trainable parameters; peak allocation about 1,060 MiB; the architecture
is unchanged from v3_01, so its stored cached-feature latency applies
(about 30 times the fusion head).

## Decisions and problems

(a) The 40k negative result was data-limited, not architectural. With
100k training questions the same frozen recipe puts the reasoner ahead
of every global head in every seed (+0.0065 over fusion), and at 250k
the margin roughly doubles (+0.0136) while fusion's own advantage over
concat has decayed to noise (v2_07). Token-level access plus the
latent-query head extracts information at scale that the pooled global
embedding path does not provide. This is consistent with the published
low-data connector findings (DePALM, Vallaeys 2024, arXiv:2403.13499;
see docs/RELATED_WORK.md) and sharpens the project's scoping: the
v3_02a conclusion that the head "leans on CLS" described the 40k
regime, and the representation-sufficiency framing must now be scoped
to that regime for overall accuracy.

(b) The compositional conclusion stands unchanged. The pooled >=4-step
deficit persists for every model at every scale (0.064-0.096), and the
reasoner's deficit remains statistically indistinguishable from
fusion's at both new scales (both bootstrap intervals include zero).
More data and token-level access buy general accuracy, not a
disproportionate multi-step gain; the compositional bottleneck claim
should be stated specifically about the multi-step axis, where it is
now supported at three training scales.

(c) Scope. These are development-set results under the V2 protocol for
this dataset, encoder, architecture and recipe; nothing here is
confirmatory, and the clean-test embargo is untouched. The 250k seed-2
outlier (0.60436) is reported as measured; no seed was added, dropped
or reweighted.

(d) Consequence for the programme: E2 (frozen SigLIP-B/16 swap on the
global path) and E3 (1000-answer vocabulary) proceed as authorized; the
final model list for the eventual blinded clean-test evaluation should
now plausibly include the 250k reasoner alongside the global heads,
which makes the freeze decision (F1) more consequential than it was
under the 40k-parity picture.
