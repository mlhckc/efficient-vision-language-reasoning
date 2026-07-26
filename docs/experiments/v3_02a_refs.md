# Experiment v3_02a: reference baselines, step-statistics repair and reasoner diagnostics

All numbers below are measured; no projections remain. Cross-references:
v2_02 (baselines), v2_04 (product ablation), v2_05b (lift machinery and
priors), v2_06 (reliance protocol), v3_01 (reasoner). None of those
artefacts was modified; the statistical addenda live only in this report.
test_clean_targets.csv was never opened; no 100k/250k scaling was run.

## 1. Preflight summary

All A1-A6 validations passed: manifests (dev 7,714 rows over 768 images;
train_40k 40,000 over 27,622; string ids; unique questionIds), metadata
coverage aligned row-for-row, aligned global stores with exact label
alignment, the image-token store at (63,599, 50, 512) fp16 with token 0
confirmed as CLS against the V2 globals for five sampled ids (max
deviations 5.4e-05 to 8.8e-05), all 23 checkpoints loadable (concat,
fusion, question_only, product_576k at seeds 0/1/2/3/42; reasoner at
0/1/2), and all required per-seed result fields present. Path note: the
task referenced data/v2/manifests/; the actual v2_00 layout is data/v2/
and was verified before use.

## 2. Measured wall-clock per phase

| phase | measured |
|-------|----------|
| A preflight | 2.0 s |
| B1 mean-patch build | 6.5 s |
| B2 reference training (10 runs) | 73.5 s |
| C statistics | 9.3 s |
| D diagnostics | 18.7 s |
| E patches-only probe | 213.9 s |

## 3. Direct-linear results

Linear(1024, 100) on the existing global embeddings, exactly 102,500
trainable parameters, five seeds: dev accuracy 0.4813 +/- 0.0011
(min 0.4803, max 0.4830); checkpoint 0.4 MB; about 7 s training per run;
0.014 ms single-example latency.

## 4. Meanpatch-concat results

src.models.ConcatModel (576,100 parameters, unchanged src/) on the
mean-of-49-patches image views (CLS excluded, fp32 reduction, unit-norm,
byte-identical deterministic rebuild): dev accuracy 0.5145 +/- 0.0019
(min 0.5124, max 0.5167); 0.027 ms latency.

## 5. Paired baseline comparisons (same seeds 0,1,2,3,42)

| comparison | mean | std | min | max |
|---|---|---|---|---|
| direct_linear - concat | -0.0427 | 0.0032 | -0.0461 | -0.0393 |
| meanpatch_concat - concat | -0.0095 | 0.0031 | -0.0134 | -0.0049 |

## 6. Step-bucket counts (dev)

| bucket | n questions | n unique images | fixed prior accuracy |
|---|---|---|---|
| <=2 | 2,400 | 674 | 0.2092 |
| 3 | 3,330 | 658 | 0.1462 |
| 4 | 870 | 396 | 0.2736 |
| >=5 | 1,114 | 531 | 0.5162 |
| combined >=4 | 1,984 | 620 | 0.4098 |

The >=5 bucket keeps its separate result (n = 1,114 stated); combined >=4
(question-weighted pooling under the fixed per-bucket prior) is the
primary interpretive quantity. This pooled definition supersedes v3_01's
unweighted two-bucket mean; there is one combined analysis, not two.
Recomputed per-bucket priors matched the stored v2_05b values exactly, and
every model reproduced its stored per-seed accuracy within 5e-06 (33/33
reproduction gates passed).

## 7. Image-clustered bootstrap (2,000 draws, NumPy seed 0, 768 clusters)

Seed-mean pooled ge4 deficits (mean step lift - combined >=4 lift) with
95% percentile intervals:

| model | deficit | 95% CI |
|---|---|---|
| question_only | 0.0809 | - (context) |
| fusion | 0.0918 | [0.0755, 0.1077] |
| reasoner | 0.0980 | [0.0824, 0.1143] |
| product_576k | 0.0978 | - |
| direct_linear | 0.1003 | [0.0825, 0.1183] |
| concat | 0.1050 | [0.0892, 0.1207] |
| meanpatch_concat | 0.1055 | [0.0890, 0.1225] |

Same-seed-paired reasoner - fusion deficit (seeds 0/1/2, paired within
every draw): +0.0047, 95% CI [-0.0067, +0.0170]. The interval includes
zero: with development-set uncertainty accounted for, the reasoner's
compositional deficit is statistically indistinguishable from fusion's.
Seed mean/std (training variability) and the clustered interval (finite
dev-set uncertainty) are reported separately and not mixed.

## 8. D1: attention-mass audit

Gate passed for every seed: instrumented path max absolute logit
difference 5.0e-06 to 5.7e-06, 100.00% prediction agreement, identical
accuracies. Visual cross-attention mass on CLS (uniform share would be
1/50 = 2%): block means 0.35-0.40 across seeds, per-head means ranging
0.13 to 0.89, per-example p95 up to 0.92, per-latent medians near the
block means. CLS-dominant attention is consistent with the model relying
strongly on the pooled CLS representation and may help explain the lower
shuffled-image reliance seen in v3_01; attention weights alone are not
causal evidence.

## 9. D2: CLS-masked evaluation (test-time intervention)

With-CLS reproduction gate passed for every seed (wrapper equals stored
accuracy). Removing CLS at test time (exactly 49 patches; question masks
unchanged):

| seed | normal | CLS-removed | change | ge4 lift normal | ge4 lift removed |
|---|---|---|---|---|---|
| 0 | 0.5464 | 0.4962 | -0.0502 | 0.1598 | 0.1381 |
| 1 | 0.5406 | 0.5122 | -0.0284 | 0.1653 | 0.1472 |
| 2 | 0.5397 | 0.4981 | -0.0416 | 0.1825 | 0.1507 |

The trained reasoner depends on CLS at inference (mean accuracy change
-0.0401). This does not show how a model trained without CLS would behave.

## 10. D3: patches-only training probe (training-time intervention)

Completed within budget (213.9 s measured, budget 20 min): seed 0, frozen
v3_01 recipe, CLS removed in training and evaluation. Dev accuracy 0.5359
(best epoch 5; the full-token seed-0 reasoner reached 0.5464), combined
>=4 lift 0.1462 (full-token seed 0: 0.1598), shuffled-image drop +0.1155
(full-token seed 0: +0.1101).

## 11. Scientific interpretation

Direct linear (F1): a single linear map reaches 0.4813 against concat's
0.5240, an absolute gap of 0.0427, i.e. 91.9% of concat's accuracy with
17.8% of its parameters. This is the measured answer to what performance
costs without a nonlinear head.

Mean-patch (F2): meanpatch_concat sits 0.95 points below CLS concat with
overlapping-magnitude seed noise but a consistently negative sign. Near
parity is consistent with simple mean pooling carrying similar global
information to CLS; the small consistent shortfall may indicate loss of
spatial structure, information loss from averaging or an unsuitable
pooling operation. No causal mechanism is claimed.

D1 (F3): CLS-dominant attention may be consistent with a shortcut
hypothesis in which the reasoner mostly re-reads the pooled global
vector; attention weights alone are not causal evidence.

Intervention hierarchy (F4): D1 and the mean-patch comparison are
observational or comparative diagnostics; D2 is a test-time intervention;
D3 is the stronger training-time intervention. Together they point the
same way: the trained reasoner leans on CLS (D1, D2), and when trained
without CLS it recovers most but not all accuracy (D3, -0.011) without
improving the compositional lift. Every conclusion is specific to this
dataset, the 40k scale, this frozen encoder, this architecture and this
recipe.

## 12. Supervisor update block

Measured values only.

- n 4-step questions: 870
- n >=5-step questions: 1,114
- n combined >=4: 1,984
- combined >=4 deficit (reasoner): 0.0980, 95% CI [0.0824, 0.1143];
  paired reasoner - fusion deficit +0.0047, 95% CI [-0.0067, +0.0170]
- direct_linear accuracy: 0.4813
- concat accuracy: 0.5240
- direct_linear vs concat gap: -0.0427
- meanpatch_concat accuracy: 0.5145
- meanpatch_concat vs concat gap: -0.0095
- D1: CLS receives 35-40% mean visual cross-attention mass (uniform share
  2%); instrumented-path gate 5.7e-06 max logit diff, 100% agreement
- D2: test-time CLS removal changes accuracy by -0.0502/-0.0284/-0.0416
  (seeds 0/1/2)
- D3: complete in 213.9 s; patches-only dev 0.5359, combined >=4 lift
  0.1462, shuffled drop +0.1155

STATUS: 100k/250k scaling remains on hold pending V3-design feedback.
