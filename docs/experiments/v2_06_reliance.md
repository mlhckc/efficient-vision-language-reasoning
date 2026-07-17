# Experiment v2_06: visual reliance of the trained heads

## Purpose

Measure how much of each model's dev accuracy actually depends on the
correct image, by evaluating the trained checkpoints with the image
embedding left intact, permuted across rows, or zeroed, with question
embeddings untouched. Evaluation only; no training.

## Method

experiments/v2_06_reliance/run.py evaluates five models (question_only,
image_only, concat, fusion from v2_02; product_576k from v2_04), all five
seeds each, under three dev conditions: normal; shuffled, where the image
embeddings are permuted across dev rows with a fixed recorded permutation
seed (42, numpy default_rng); and zeroed, where they are replaced by zeros.
question_only is the control: its predictions were asserted identical across
conditions in every seed, and they were. For seed 42, the drops are also
sliced by structural type for concat and fusion. Blinding: dev data and the
metadata join only; no test_clean_* file was read.

## Outputs

Under results/experiments/v2_06_reliance/ (git-ignored): results.json,
table.csv and reliance.png (grouped bars, three conditions per model, std
error bars over seeds).

## Results

Mean dev accuracy over five seeds, and reliance drops (mean +/- std):

| model         | normal | shuffled | zeroed | drop shuffled     | drop zeroed       |
|---------------|--------|----------|--------|-------------------|-------------------|
| question_only | 0.4580 | 0.4580   | 0.4580 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| image_only    | 0.2344 | 0.2108   | 0.2259 | 0.0236 +/- 0.0102 | 0.0085 +/- 0.0027 |
| concat        | 0.5240 | 0.4069   | 0.4241 | 0.1171 +/- 0.0049 | 0.0999 +/- 0.0040 |
| fusion        | 0.5384 | 0.3957   | 0.4196 | 0.1427 +/- 0.0024 | 0.1188 +/- 0.0052 |
| product_576k  | 0.5335 | 0.4054   | 0.4211 | 0.1282 +/- 0.0033 | 0.1124 +/- 0.0045 |

Seed-42 drops by structural type (shuffled / zeroed):

| type    | concat drop       | fusion drop       |
|---------|-------------------|-------------------|
| verify  | +0.0251 / +0.0379 | +0.0680 / +0.0847 |
| query   | +0.2249 / +0.1804 | +0.2233 / +0.1943 |
| choose  | +0.1123 / +0.0758 | +0.1267 / +0.0576 |
| logical | +0.0332 / +0.0000 | +0.0906 / +0.0688 |
| compare | +0.0000 / +0.0232 | -0.0033 / -0.0331 |

Figure: results/experiments/v2_06_reliance/reliance.png.

## Interpretation

(a) Dependence on the correct image. The multimodal heads route a real but
bounded share of their accuracy through the image vector: shuffling the
images costs concat 0.117, product_576k 0.128 and fusion 0.143 of accuracy.
Notably, all three fall below the question_only floor (0.458) under
shuffling (0.396 to 0.407): a wrong real image does not merely remove
information, it actively misleads, so the models are genuinely reading the
image vector rather than ignoring it. image_only, already near the majority
floor, has little to lose (0.024).

(b) Shuffled against zeroed. Zeroed drops are systematically smaller
(concat 0.100 against 0.117, fusion 0.119 against 0.143), even though zero
vectors are out-of-distribution inputs the heads never saw during training
(for fusion, zeros degenerate the input to [0, q, 0, q]). Permuted real
vectors stay within the training input distribution while destroying the
image-question pairing, and they inject misleading rather than absent
evidence, which explains the larger drop. Shuffling is the fairer reliance
measure: it isolates the value of the correct pairing without a
distribution shift. The model ordering is the same under both measures, so
the conclusions do not depend on this choice.

(c) Does fusion rely on the image more than concat? Yes, consistently:
+0.026 more drop under shuffling and +0.019 under zeroing, with small
seed-to-seed stds, and product_576k sits in between. Sliced by type, the
prediction from the v2_04/v2_05 agreement-detector account is only partly
confirmed: verify is not the largest absolute drop for fusion (query
dominates absolute drops for both models, about 0.22). What the account
does get right is where fusion's extra reliance over concat sits: the
excess drop is largest on logical (+0.057) and verify (+0.043), the two
yes/no formats where fusion's accuracy gains are, and near zero on query
and compare. So fusion's additional image use is concentrated where the
agreement signal helps, but verification is not where the image matters
most overall. The compare slice (n = 302) shows a negative drop for fusion,
consistent with its compare behaviour being noise-level.

(d) Implication for the V3 motivation, stated carefully. The largest image
dependence for every model is on open query questions (drop about 0.22),
which are also the weakest absolute slice for every trained model (about
0.48 in v2_05). So the single global vector is doing most of its work
precisely where models remain weakest, which is consistent with the V3
hypothesis that finer, token-level visual features have headroom on open
questions. This experiment measures current dependence, not attainable
headroom: it cannot show that token-level features will close the gap, only
that the global vector is the binding visual channel where the gap is
widest. The V3 case rests on that consistency, not on proof.

## Decisions and problems

The permutation seed (42) is recorded in results.json, so the shuffled
condition is exactly reproducible. Per-type drops use seed 42 only, with
the same small-slice caveat as v2_05; the aggregate drops are five-seed
quantities. One incidental observation: concat's logical slice loses
nothing under zeroing at seed 42 (+0.0000) yet 0.033 under shuffling,
a clean example of zeros understating reliance relative to misleading
real vectors.
