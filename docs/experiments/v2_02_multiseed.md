# Experiment v2_02: multi-seed training under the V2 protocol

## Purpose

Train the four models (question-only, image-only, concat, fusion) on the V2
train_40k manifest across five seeds, selecting and reporting on the
image-disjoint dev split, to establish whether the V1 single-seed ordering
and the fusion gain survive seed variation under the clean V2 protocol.

## Method

experiments/v2_02_multiseed/run.py trains every model for seeds 0, 1, 2, 3
and 42 with the shared loop in src/train.py and hyperparameters unchanged
from V1 (AdamW, learning rate 1e-3, weight decay 1e-4, batch size 256, 30
epochs, dropout 0.3), reading the cached V2 embeddings
(data/v2/embeddings/train_40k.h5 and dev.h5) through src/data.py. For each
run the checkpoint with the best dev accuracy is kept
(results/experiments/v2_02_multiseed/checkpoints/). Per seed,
utils.set_seed(seed) fixes model initialisation and dropout, and the training
loader's shuffle generator is reseeded with the same seed so the data order
also varies by seed. The majority reference predicts the most frequent
train_40k label ("no") and is seed-independent.

Two backwards-compatible extensions were made for this: train_model gained an
optional checkpoint_dir argument and make_loaders gained optional path
arguments; with the arguments omitted both behave exactly as in V1.

Blinding: only train_40k.h5, dev.h5 and the V2 vocabulary were read. No
test_clean_* file was touched; all selection and reporting are on dev.

## Outputs

Under results/experiments/v2_02_multiseed/ (git-ignored): results.json (all
per-run metrics and histories with run metadata), table.csv, twenty
checkpoints, and the figure accuracy_multiseed.png (mean dev accuracy per
model with sample-std error bars and the majority reference as a dashed
line).

## Results

Dev accuracy over five seeds (mean, sample std, min, max, then per-seed
values in seed order 0, 1, 2, 3, 42):

| model         | mean   | std    | min    | max    | per-seed |
|---------------|--------|--------|--------|--------|----------|
| majority ref  | 0.2247 | -      | -      | -      | (seed-independent) |
| image-only    | 0.2344 | 0.0003 | 0.2340 | 0.2349 | 0.2343, 0.2349, 0.2344, 0.2340, 0.2344 |
| question-only | 0.4580 | 0.0029 | 0.4531 | 0.4607 | 0.4531, 0.4607, 0.4581, 0.4590, 0.4589 |
| concat        | 0.5240 | 0.0028 | 0.5213 | 0.5272 | 0.5223, 0.5267, 0.5272, 0.5213, 0.5224 |
| fusion        | 0.5384 | 0.0022 | 0.5362 | 0.5408 | 0.5362, 0.5408, 0.5405, 0.5381, 0.5364 |

Paired per-seed differences (the decision quantities):

| gap                        | mean    | std    | min     | max     |
|----------------------------|---------|--------|---------|---------|
| fusion - concat            | +0.0144 | 0.0014 | +0.0132 | +0.0169 |
| concat - question-only     | +0.0660 | 0.0032 | +0.0622 | +0.0692 |

Figure: results/experiments/v2_02_multiseed/accuracy_multiseed.png.

## Decisions and problems

(a) The fusion-concat mean gap (+0.0144) exceeds its per-seed standard
deviation (0.0014) by an order of magnitude. (b) The minimum per-seed gap is
+0.0132, still positive: fusion beat concat in every one of the five seeds.
The gain is therefore stable under seed variation, though its size remains
modest (about 1.4 points) and the capacity confound noted in src/models.py
still applies; whether the gain survives parameter matching is exactly the
question for v2_03.

(c) Compared with the legacy V1 single-seed numbers (question-only 0.4582,
concat 0.5254, fusion 0.5414, gap +0.0160, majority "yes" 0.2339), the V2
multi-seed numbers are qualitatively consistent: the ordering image-only <
question-only < concat < fusion is unchanged, and the V1 fusion gap lies
near the top of the V2 per-seed range. The numbers are not directly
comparable, because the protocols differ: V2 selects and evaluates on the
image-disjoint dev split rather than the reused V1 validation set, the V2
train_40k manifest is a different sample (a permuted prefix of the eligible
pool under the V2 vocabulary), and the majority label is now "no" (dev
accuracy 0.2247) where V1's was "yes" (0.2339). These are different
experiments that happen to agree qualitatively; neither confirms the other.

Image-only sits closer to the majority reference than in V1 (+0.010 here),
with near-zero seed variance, consistent with the image alone carrying
almost no answer signal at this vocabulary. Best epochs vary across runs
(image-only as early as epoch 1, fusion as late as 28), so best-on-dev
checkpoint selection is doing real work; dev is a development split, so this
selection is legitimate under the protocol.
