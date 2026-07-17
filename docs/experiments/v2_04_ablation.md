# Experiment v2_04: fusion-feature ablation, capacity-controlled

## Purpose

v2_03 showed the fusion features carry a real gain at matched capacity. This
experiment attributes that gain between the two interaction terms, the
elementwise product and the absolute difference, by training
single-interaction variants under the same capacity control.

## Method

Two variants were defined locally in experiments/v2_04_ablation/run.py (they
are single-use ablation models, so they are kept out of src/models.py to
leave the shared model set stable):

- ProductFusion: [image ; question ; image * question], input 1536.
- DifferenceFusion: [image ; question ; |image - question|], input 1536.

Each was trained at two widths across the same five seeds, loaders,
hyperparameters and shuffle reseeding as v2_02, with best-on-dev checkpoints
under results/experiments/v2_04_ablation/checkpoints/:

- matched width, computed programmatically: hidden 351, giving 574,687
  parameters, the largest 1536-input head at or under the concat budget of
  576,100 (off by 0.245%);
- natural width config.HIDDEN_DIM = 512, giving 838,244 parameters, to
  complete the natural-width ladder concat -> single interaction -> fusion.

concat and fusion per-seed results come from v2_02 and fusion_narrow from
v2_03; none was retrained. Blinding: only train_40k.h5, dev.h5, the
vocabulary and the v2_02/v2_03 results files were read; no test_clean_* file
was touched; all selection and reporting are on dev.

## Outputs

Under results/experiments/v2_04_ablation/ (git-ignored): results.json,
table.csv, twenty checkpoints, and ablation_576k.png (the four equal-budget
models with std error bars and parameter counts).

## Results

Dev accuracy over the five seeds:

| model              | params    | mean   | std    | min    | max    |
|--------------------|-----------|--------|--------|--------|--------|
| concat             | 576,100   | 0.5240 | 0.0028 | 0.5213 | 0.5272 |
| product_576k       | 574,687   | 0.5335 | 0.0031 | 0.5306 | 0.5372 |
| difference_576k    | 574,687   | 0.5334 | 0.0011 | 0.5320 | 0.5350 |
| fusion_narrow      | 576,032   | 0.5334 | 0.0014 | 0.5320 | 0.5355 |
| product_natural    | 838,244   | 0.5367 | 0.0010 | 0.5354 | 0.5381 |
| difference_natural | 838,244   | 0.5340 | 0.0003 | 0.5336 | 0.5342 |
| fusion             | 1,100,388 | 0.5384 | 0.0022 | 0.5362 | 0.5408 |

Equal-budget gaps at about 576k parameters (per-seed paired):

| gap                                  | mean    | std    | min     | max     |
|--------------------------------------|---------|--------|---------|---------|
| product_576k - concat                | +0.0096 | 0.0034 | +0.0061 | +0.0152 |
| difference_576k - concat             | +0.0094 | 0.0036 | +0.0053 | +0.0137 |
| fusion_narrow - product_576k         | -0.0001 | 0.0029 | -0.0044 | +0.0030 |
| fusion_narrow - difference_576k      | -0.0000 | 0.0020 | -0.0030 | +0.0021 |
| product_576k - difference_576k       | +0.0001 | 0.0028 | -0.0030 | +0.0038 |

Natural-width ladder gaps:

| gap                                  | mean    | std    | min     | max     |
|--------------------------------------|---------|--------|---------|---------|
| product_natural - concat             | +0.0127 | 0.0028 | +0.0092 | +0.0157 |
| difference_natural - concat          | +0.0100 | 0.0028 | +0.0067 | +0.0130 |
| fusion - product_natural             | +0.0017 | 0.0023 | -0.0017 | +0.0040 |
| fusion - difference_natural          | +0.0044 | 0.0022 | +0.0022 | +0.0069 |

Figure: results/experiments/v2_04_ablation/ablation_576k.png.

## Decisions and problems

Verdict. At equal capacity, either interaction term alone carries the whole
gain, and the two terms are redundant with each other, not complementary.
Each single-interaction variant beats concat in every seed (means +0.0096
and +0.0094, minima +0.0061 and +0.0053), and the three interaction models
at the 576k budget are indistinguishable: fusion_narrow does not beat either
single term (mean gaps -0.0001 and -0.0000, per-seed signs in both
directions), and product and difference are equal to within noise (+0.0001).
So the answer to "which term carries the gain" is: either one does; neither
is specifically required; adding the second term on top of the first buys
nothing at this budget.

At natural widths the picture is consistent. Both single-term models beat
concat; fusion's edge over product_natural (+0.0017, straddling zero across
seeds) is within noise and fusion also has 31% more parameters, so no claim
can be made that the full four-part input outperforms the three-part product
variant. Fusion's edge over difference_natural (+0.0044, positive in every
seed) is small but consistent; note product_natural and fusion differ from
difference_natural in opposite directions, which again suggests the product
and difference terms are interchangeable carriers of the same signal rather
than additive contributors.

Practical implication for the efficiency question: a three-part input
[image ; question ; image * question] at 574,687 parameters reaches the same
dev accuracy as the four-part fusion at its own budget (0.5335 against
fusion_narrow 0.5334), and at 838,244 parameters (0.5367) it exceeds
v2_03's concat_wide at 1.10M (0.5326). If a compact head is wanted, one
interaction term is enough; the four-part concatenation is not harmful but
carries a redundant term.

These conclusions are dev-set findings under the V2 protocol at the 40k
training scale, with n = 5 seeds; they inform model selection and are not
confirmatory test results.
