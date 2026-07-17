# Experiment v2_03: two-sided parameter-matched controls

## Purpose

The v2_02 fusion-vs-concat comparison is capacity-confounded: the fused input
is twice as wide, so the standard fusion head has about twice the trainable
parameters. This experiment separates the contribution of the fusion features
from the contribution of capacity by matching parameters in both directions.

## Method

Matched hidden widths were computed programmatically from instantiated
parameter counts, not hardcoded: concat_wide is a ConcatModel with the
smallest hidden width whose count reaches the standard fusion count, and
fusion_narrow is a FusionModel with the largest hidden width whose count
stays within the standard concat count.

- concat_wide: hidden 979, 1,101,475 parameters (target 1,100,388, off by
  0.099%).
- fusion_narrow: hidden 268, 576,032 parameters (target 576,100, off by
  0.012%).

Both controls were trained across the same seeds (0, 1, 2, 3, 42) with the
same train_40k/dev loaders, hyperparameters and shuffle reseeding as v2_02
(experiments/v2_03_param_match/run.py); best-on-dev checkpoints under
results/experiments/v2_03_param_match/checkpoints/. The standard concat and
fusion per-seed results are loaded from v2_02 and were not retrained. The
only source change is a backwards-compatible optional hidden_dim on MLPHead
and the four model classes (None keeps config.HIDDEN_DIM, so all existing
code is unchanged).

Blinding: only train_40k.h5, dev.h5, the vocabulary and the v2_02 results
file were read. No test_clean_* file was touched; all selection and
reporting are on dev.

## Outputs

Under results/experiments/v2_03_param_match/ (git-ignored): results.json,
table.csv, ten checkpoints, and params_matched.png (six models' mean dev
accuracy with std error bars, annotated with exact parameter counts).

## Results

Dev accuracy over the five seeds:

| model         | hidden | params    | mean   | std    | min    | max    |
|---------------|--------|-----------|--------|--------|--------|--------|
| concat        | 512    | 576,100   | 0.5240 | 0.0028 | 0.5213 | 0.5272 |
| fusion_narrow | 268    | 576,032   | 0.5334 | 0.0014 | 0.5320 | 0.5355 |
| concat_wide   | 979    | 1,101,475 | 0.5326 | 0.0026 | 0.5293 | 0.5356 |
| fusion        | 512    | 1,100,388 | 0.5384 | 0.0022 | 0.5362 | 0.5408 |

Paired per-seed decision gaps:

| gap                        | mean    | std    | min     | max     |
|----------------------------|---------|--------|---------|---------|
| fusion - concat_wide       | +0.0058 | 0.0030 | +0.0005 | +0.0076 |
| fusion_narrow - concat     | +0.0094 | 0.0018 | +0.0070 | +0.0112 |
| concat_wide - concat       | +0.0086 | 0.0030 | +0.0056 | +0.0134 |
| fusion - fusion_narrow     | +0.0050 | 0.0017 | +0.0029 | +0.0071 |

Figure: results/experiments/v2_03_param_match/params_matched.png.

## Decisions and problems

Verdict. The fusion advantage survives parameter matching in both directions,
but at roughly half its apparent size. At concat's budget (576k parameters),
fusion_narrow beats concat in every seed (mean +0.0094, minimum +0.0070), so
the fusion features genuinely help when capacity is held at the smaller
budget. At fusion's budget (1.10M parameters), fusion beats concat_wide in
every seed on paper, but the margin is smaller and less robust: mean +0.0058
with a minimum per-seed gap of +0.0005, a near-tie in the weakest seed. So
the honest reading of the v2_02 gap of +0.0144 is that it was roughly half
capacity and half features: extra capacity alone lifts concat by +0.0086,
and the features add about +0.0050 to +0.0094 on top, depending on the
budget at which the comparison is made. The feature effect is real at the
small budget and present but thin at the large budget; a claim that the
handcrafted fusion is worth 1.4 points would be an overstatement, and a
claim that it is pure capacity would also be wrong.

A secondary observation: fusion_narrow (576k) matches concat_wide (1.10M)
within noise (0.5334 against 0.5326), meaning the fusion features buy about
as much as doubling the concat head's parameters. Capacity also shows
diminishing returns for fusion itself (+0.0050 from 576k to 1.10M).

These numbers inform the v2_04 ablation: with the capacity question settled,
the next question is which of the two interaction terms (elementwise product,
absolute difference) carries the feature effect.
