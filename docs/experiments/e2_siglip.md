# Experiment E2: frozen SigLIP-B/16 encoder swap on the global path

## Purpose

Test whether the project's findings move with the frozen representation:
one frozen open_clip ViT-B-16-SigLIP (webli) encoder replaces frozen
CLIP ViT-B/32 (laion2b) for both modalities on the global-embedding
path, with everything else held fixed. Two questions: does overall
development accuracy change, and does the question-weighted pooled
>=4-step deficit move? Approved by the user on 30 July 2026 (E2);
dependency addition (transformers 5.14.1, sentencepiece 0.2.2, for the
SigLIP tokenizer only) authorized and recorded on 31 July 2026, commit
b2e5873.

## Method

experiments/e2_siglip/run.py (pinned commit da0a127 after a three-round
pre-launch review). Extraction mirrors v2_01: the canonical 63,599-image
and 265,727-question union (train pool, dev and clean-test inputs —
targets never touched) is encoded once with the frozen SigLIP encoder
into unit-normalised 768-d float32 keyed stores under
data/v2_siglip/embeddings/, with atomic writes, canonical CLIP-store id
ordering, and aligned train_40k/train_250k/dev views whose label arrays
were asserted identical to the CLIP aligned views row-for-row (row
alignment proven). The tokenizer's 64-token context truncates nothing:
the longest of the 184,432 unique questions is 32 raw tokens.

Training replicates the v2_02 recipe verbatim: AdamW, learning rate
1e-3, weight decay 1e-4, batch 256, 30 epochs, dropout 0.3, hidden 512,
best-on-dev checkpointing; per-seed utils.set_seed(seed) plus one
loader-generator reseed before the sequential model loop (order
question_only, concat, product, fusion; the 31 July seed-pairing
clarification applies). Heads are the same MLPHead family over 768-d
inputs; seeds {0, 1, 2, 3, 42}; scales train_40k and train_250k; 40
runs. Nothing was tuned on E2 results.

Comparisons: the SigLIP-minus-CLIP values are same-seed-label
differences over independent training streams, not paired gaps —
identical initialisation and shuffle streams are impossible across
encoders because the input widths differ. The CLIP references are the
stored v2_02 (40k) and v2_07 (250k) per-seed accuracies; the 40k
product reference is product_natural (hidden 512, the same
natural-width family), while the only stored 250k product head is
product_576k (hidden 351), so the 250k product row is doubly
capacity-mismatched (input and hidden width) and is excluded from every
headline statement. SigLIP heads carry more parameters at the same
hidden width because the inputs are 768-d (reported per model below);
this capacity confound is stated, not matched, consistent with the
project's v2_02/v2_03 treatment.

The step analysis uses the binding v3_02a pooled >=4 definition with
the fixed v2_05b priors (label-based, hence encoder-independent),
applied as per-bucket scalar fills of the stored 5-decimal prior
accuracies — equivalent to the per-question construction up to about
1e-5 in the deficit, disclosed here per the pre-launch review.

Gate history, recorded for traceability: the re-encode consistency gate
was pre-registered at 1e-5, relaxed to 1e-4 in review round 01, and the
first launch aborted on it — measured batch-1 versus batch-128
deviations were 4.9e-05 to 3.4e-04 (cosines at least 0.999998), caused
by cuDNN conv TF32 selecting batch-dependent algorithms for the patch
embedding. The gate was raised to the v2_01-precedent 1e-3 (review
round 03), which preserves misalignment detection by two orders of
magnitude; TF32-level extraction noise is shared with the v2_01 CLIP
stores, extracted under the same defaults on the same hardware, so
accepting it is the encoder-consistent choice. Row alignment does not
rest on this gate: deterministic id-order asserts and the label-identity
gate carry it.

Blinding: dev selection only; test_clean_targets.csv was never read.

## Outputs

Under results/experiments/e2_siglip/ (git-ignored): preflight.json,
results.json, run.log and 40 checkpoints {model}_{scale}_seed{seed}.pt.
results.json and preflight.json are exported to
artifacts/results_export/e2_siglip/; the checkpoints are hash-pinned but
not exported in the MANIFEST. New stores under data/v2_siglip/
(about 1.5 GB).

## Results

Dev accuracy over five seeds (SigLIP-B/16 heads; the SigLIP-minus-CLIP
column is the same-seed-label mean with the per-seed minimum in
parentheses):

| scale | model | mean | std | params | vs CLIP |
|---|---|---|---|---|---|
| 40k | question_only | 0.4661 | 0.0022 | 445,028 | +0.0081 (+0.0056) |
| 40k | concat | 0.5394 | 0.0038 | 838,244 | +0.0154 (+0.0123) |
| 40k | product | 0.5503 | 0.0024 | 1,231,460 | +0.0136 (+0.0087) |
| 40k | fusion | 0.5532 | 0.0025 | 1,624,676 | +0.0147 (+0.0114) |
| 250k | question_only | 0.5024 | 0.0045 | 445,028 | +0.0047 (-0.0030) |
| 250k | concat | 0.5918 | 0.0026 | 838,244 | +0.0132 (+0.0083) |
| 250k | product | 0.5980 | 0.0036 | 1,231,460 | (capacity-mismatched reference; excluded from headlines) |
| 250k | fusion | 0.5939 | 0.0032 | 1,624,676 | +0.0116 (+0.0057) |

The multimodal SigLIP heads beat their CLIP counterparts in every seed
at both scales (concat and fusion: minimum per-seed differences +0.0057
to +0.0154). The question-only difference is small and not seed-robust
at 250k (minimum -0.0030): the encoder swap helps through the visual
and joint representation, not the language side. For context, the
strongest SigLIP global heads at 250k (0.592-0.598) sit at parity with
the 21.1M CLIP-token reasoner of v3_03 (0.5958 +/- 0.0074) while being
about 13-18 times smaller.

Question-weighted pooled >=4 deficits (five-seed means, fixed priors;
CLIP references with their seed sets noted):

| scale | model | SigLIP | CLIP reference |
|---|---|---|---|
| 40k | question_only | 0.0838 | 0.0809 (v3_02a, 5 seeds) |
| 40k | concat | 0.1024 | 0.1050 (v3_02a, 5 seeds) |
| 40k | product | 0.0922 | 0.0978 (v3_02a product_576k, 5 seeds) |
| 40k | fusion | 0.0866 | 0.0918 (v3_02a, 5 seeds) |
| 250k | question_only | 0.0628 | 0.0638 (v3_03, 3 seeds) |
| 250k | concat | 0.0826 | 0.0842 (v3_03, 3 seeds) |
| 250k | product | 0.0793 | 0.0771 (v3_03 product_576k, 3 seeds) |
| 250k | fusion | 0.0838 | 0.0803 (v3_03, 3 seeds) |

Efficiency: extraction 63,599 images in 410 s and 184,432 unique texts
in under 3 minutes (one-off, fp32, frozen encoder); the relaunched
gates-to-analysis pass took 0.31 h; head training times match the
global-head pattern (seconds at 40k, about a minute at 250k per run).

## Decisions and problems

(a) The encoder matters for absolute accuracy. SigLIP-B/16 lifts every
multimodal global head by about +1.2 to +1.5 points at both scales,
seed-robustly, while leaving question_only nearly unchanged — the
improvement is visual/joint, as an encoder swap should be. The
strongest SigLIP global heads reach the accuracy of the much larger
CLIP-token reasoner, which sharpens the efficiency story: a better
frozen global representation buys what 21.1M parameters of token-level
reasoning buys over the weaker one.

(b) The compositional deficit does not move. Under SigLIP the pooled
>=4 deficit spans 0.063-0.102 across models and scales, the same range
as CLIP (0.064-0.105), with differences of at most about 0.005 in
either direction and no consistent sign. The multi-step deficit is now
observed across two frozen contrastive dual-encoders, three training
scales and heads from 0.1M to 21.1M parameters. This strengthens the
central claim while refining it: the bottleneck looks like a property
of the frozen contrastive global-embedding representation family, not
of one specific encoder — and, per v3_03, token-level access at scale
raises overall accuracy without closing the multi-step gap either.

(c) Scope. Development-set results under the V2 protocol; no
confirmatory claim; the clean-test embargo is untouched. The capacity
confound (768-d inputs) is stated with exact parameter counts, not
matched; a v2_03-style two-sided matching under SigLIP would be a
separate task if wanted. The 250k product comparison is excluded from
headlines (doubly capacity-mismatched reference). CLIP 250k deficit
references come from three seeds (v3_03); the SigLIP values are
five-seed means.

(d) Consequence for the programme: E3 (1000-answer vocabulary) remains
queued. The freeze decision (F1) now has a richer candidate list:
SigLIP global heads rival the CLIP-token reasoner at a fraction of the
size, which is directly relevant to the dissertation's efficiency
question.
