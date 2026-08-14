# Experiments and results

This is the detailed experiment-by-experiment record of the project for human
readers. Every number below is a development-set result under the V2 protocol
unless explicitly marked as V1 legacy. No confirmatory clean-test result
exists anywhere in the project: the model-list freeze (F1) is unstarted and
the blinded clean-test evaluation (F2) is unstarted and unauthorised.

Numbers are quoted at the precision of their canonical record: four decimal
places for the V2/V3/E2/E3 families, five for E8A/E8B/E10 and the closure
records, six for the E9 primary scores. Nothing is re-rounded here; where a
record stores more precision, the record wins. Field validity for corrected or
withdrawn values resolves through, in order:
`results/closure/pre_f1_evidence_metadata_repair_20260814.json` (its named
fields), `results/closure/e7b_evidence_supersession.json` (named E7b fields),
`results/ve0/supersession_map.json` (artefact groups), then historical stored
fields. Lifecycle and status resolve through
`results/closure/pre_f1_status_supersession_20260814.json`.

Three accuracy denominators appear and must not be mixed silently:

- **In-vocabulary dev accuracy**: the 7,714 dev questions whose answer is in
  the top-100 vocabulary (768 images); the top-1000 dev view has 9,823 rows.
- **Raw-distribution dev accuracy**: correct answers over all 10,004 raw dev
  questions (777 images), either measured directly on that denominator or
  computed as in-vocabulary accuracy times coverage.
- **G21 normalised exact match**: the pinned VQA-style normalised scorer; on
  the top-100 vocabulary it provably coincides with raw exact match (zero
  normalised collisions).

## V1 — legacy prototype pipeline

### Question
Can closed-set VQA over frozen CLIP global embeddings with small MLP heads
work at all, and does a handcrafted fusion input beat plain concatenation?

### Setup
40,000 training questions, 8,000 legacy validation questions
(legacy_v1_validation), top-100 answer vocabulary, one seed, frozen CLIP
ViT-B-32, four heads sharing one design: image-only, question-only, concat,
and fusion (image, question, elementwise product, absolute difference).

### Result
Validation accuracy: majority 0.234, image-only 0.243, question-only 0.458,
concat 0.525, fusion 0.541.

### Statistical evidence
None: a single seed, and the validation set was reused for checkpoint
selection, so the numbers are optimistically biased.

### Interpretation
The task is learnable from frozen global embeddings; language carries most of
the signal; the fusion input helps.

### Limitation
Prototype only. Single seed, selection-biased evaluation, and concat and
fusion differ in head capacity, so the fusion comparison is confounded. V1
results must never be presented as final findings. The V1 and V2 vocabularies
contain the same 100 answers but 11 answers differ in index, so V1 label
indices must never be mixed with V2 manifests.

### Why it led to the next step
Every weakness above is a protocol weakness, not a model weakness. V2 exists
to rebuild the evaluation so the same questions can be answered defensibly.

## v2_00 — the V2 evaluation protocol

### Question
Can a selection-unbiased, vocabulary-unconditioned, image-disjoint evaluation
protocol be built from the raw balanced GQA release and verified
independently?

### Setup and result
An image-disjoint development partition: 777 dev images carrying 10,004 raw
questions, 7,714 in-vocabulary (coverage 0.7711), spanning 768 images. A
training pool of 71,363 images with 724,074 eligible in-vocabulary questions
(margin 474,074 over the required 250,000). Strict nested question subsets
train_40k / train_100k / train_250k (27,622 / 46,158 / 61,859 unique images),
row-for-row prefixes of one seeded permutation. A clean test of 8,013 raw
questions on 972 images built from validation images never touched by
legacy_v1_validation, split into an inputs file and an embargoed targets
file. The vocabulary (top 100) is computed from the training pool only and
contains the same 100 answers as V1 with 11 indices changed. An independent
verifier re-derives the whole protocol from the raw GQA files: 99 checks, 0
failures; idempotence proven (13 generated files byte-identical across
rebuilds).

### Statistical evidence
Deterministic set, count and hash equality; no models trained.

### Interpretation and limitation
The protocol supports everything after it. Its stated limitation matters for
scaling claims: larger question prefixes add both new images and more
questions on seen images, so scaling conclusions are about question count,
not image count. Clean-test reporting is structural only (counts above); no
label statistic of any kind has been computed.

## v2_01 — embedding extraction and the zero-shot floor

### Question
What do the frozen representations cost to build, and what does CLIP achieve
on this task with no training at all?

### Setup
Frozen CLIP ViT-B-32 (laion2b_s34b_b79k), zero trainable parameters,
L2-normalised 512-d embeddings cached once: 63,599 images and 265,727
question texts.

### Result
Majority reference ("no") dev accuracy 0.2247. Zero-shot image-to-answer
matching: 0.0770 with the raw answer prompt, 0.0795 with "a photo of
{answer}", 0.0786 ensembled.

### Interpretation and limitation
The zero-shot floor (about 0.08) is far below the majority reference because
a large share of dev is yes/no, where image-answer matching carries almost no
signal. It is a training-free lower anchor, not a competitive baseline; it
uses the image only.

## v2_02 — five-seed baselines at 40k

### Question
Does the V1 ordering, and specifically the fusion-over-concat gain, survive
seed variation under the clean protocol?

### Setup
Seeds 0, 1, 2, 3, 42; train_40k; V1 hyperparameters unchanged (AdamW, lr
1e-3, weight decay 1e-4, batch 256, 30 epochs, dropout 0.3); best-on-dev
checkpoint per run.

### Result
Dev accuracy, mean over five seeds: image-only 0.2344 +/- 0.0003,
question-only 0.4580 +/- 0.0029, concat 0.5240 +/- 0.0028, fusion 0.5384 +/-
0.0022. Paired per-seed gaps: fusion minus concat +0.0144 +/- 0.0014
(minimum +0.0132, positive in all five seeds); concat minus question-only
+0.0660 +/- 0.0032.

### Statistical evidence
Five seeds, sample standard deviation, paired same-seed gaps with sign
consistency. No intervals in this report; image-clustered intervals for the
claim-bearing contrasts were added later by the statistical closure. The
per-seed pairing is same-seed, same-data, not a variance-reduced
matched-stream comparison (clarification of 31 July 2026).

### Interpretation and limitation
The ordering and the fusion gain are seed-stable, but the gain is modest
(about 1.4 points) and the capacity confound is untested here.

### Why it led to the next experiment
The fused input is twice as wide, so fusion has about twice concat's
parameters. v2_03 separates features from capacity.

## v2_03 — two-sided parameter matching

### Question
How much of the fusion gain is explained by extra capacity rather than the
interaction features?

### Setup
Programmatically matched widths in both directions: concat_wide (hidden 979,
1,101,475 parameters) against fusion (1,100,388), and fusion_narrow (hidden
268, 576,032) against concat (576,100). Same five seeds and recipe.

### Result
fusion_narrow 0.5334 +/- 0.0014; concat_wide 0.5326 +/- 0.0026. Paired gaps:
concat_wide minus concat +0.0086 (capacity alone); fusion_narrow minus concat
+0.0094 (features at the small budget, positive in every seed); fusion minus
concat_wide +0.0058 with a minimum per-seed gap of +0.0005 (a near-tie in the
weakest seed).

### Interpretation
The honest reading of the v2_02 gap of +0.0144 is roughly half capacity and
half features. Claiming the handcrafted fusion is worth 1.4 points would be
an overstatement; claiming it is pure capacity would also be wrong.

### Limitation
Five seeds, 40k scale, dev only. The closure later attached an
image-clustered interval to fusion minus concat_wide at 40k that does not
exclude zero (+0.00578 [-0.00116, +0.01226]), so the equal-budget advantage
at the large budget is not established once dev-set uncertainty is included.

### Why it led to the next experiment
If features carry half the gain, which of the two interaction terms carries
it? v2_04 ablates them under the same capacity control.

## v2_04 — interaction-term ablation

### Question
Which interaction term, the elementwise product or the absolute difference,
carries the feature gain?

### Setup
Single-interaction variants at the concat budget (product_576k and
difference_576k, hidden 351, 574,687 parameters) and at natural width
(838,244 parameters); same five seeds and recipe.

### Result
product_576k 0.5335 +/- 0.0031 and difference_576k 0.5334 +/- 0.0011, each
beating concat in every seed (+0.0096 / +0.0094); fusion_narrow minus either
single term is -0.0001 / -0.0000; product minus difference +0.0001.

### Interpretation
At equal capacity, either interaction term alone carries the whole gain, and
the two terms are redundant with each other, not complementary. The
three-part product variant at 574,687 parameters matches the four-part
fusion at its own budget.

### Limitation
Dev-set findings at 40k with five seeds; they inform model selection and are
not confirmatory.

### Why it led to the next experiment
With the mechanism narrowed to "one interaction term suffices", the next
questions are where the gain lives (v2_05) and whether the head actually
uses the image (v2_06).

## v2_05, v2_05b, v2_05c — question types, steps, and the multi-step deficit

### Question
Where does the fusion gain live across question types, and do relational or
multi-step questions expose a ceiling of the global-embedding representation?

### Setup
GQA structural/semantic types and program-step counts joined onto dev
(7,714/7,714 matched) and train_40k; per-type priors; slicing of the trained
checkpoints. Main slicing at seed 42; five-seed gaps and clustered intervals
in the addenda. Step buckets on dev: less-than-or-equal-2: 2,400 questions;
3: 3,330; 4: 870; 5-or-more: 1,114.

### Result
Fusion-over-concat gains concentrate in verify +0.046 +/- 0.009, obj +0.043
+/- 0.009, logical +0.038 +/- 0.008, rel +0.018 +/- 0.005 (all positive in
every seed); choose is a real small cost (-0.023 +/- 0.011, negative in all
five seeds). Under image-clustered intervals (2,000 draws), verify, logical,
obj and rel still exclude zero; no per-type loss is statistically
established. The lift analysis found the multi-step weakness: mean lift on
4-or-more-step buckets sits well below each model's mean step-bucket lift
(question_only 0.115 against 0.191; concat 0.150 against 0.244; fusion 0.183
against 0.264), a deficit of about 0.08.

### Statistical evidence
v2_05c (5 August 2026) superseded the row-level normal-approximation
intervals with image-clustered bootstrap intervals (2,000 draws,
default_rng(0)), because the 7,714 rows share 768 images. The clustered
intervals changed no exclusion conclusion.

### Interpretation and limitation
The interaction features help most where agreement between image content and
a proposition must be judged (verify/logical/obj). The step-count slicing is
confounded with question format, and the small slices (compare n=302, global
n=163) carry wide uncertainty. The multi-step deficit is the project's
central negative observation from here on.

### Why it led to the next experiment
If fusion's gain is an agreement signal, fusion should depend on the image
more than concat. v2_06 tests reliance directly.

## v2_06 — visual-reliance interventions

### Question
How much of each head's accuracy depends on the correct image?

### Setup
Trained checkpoints evaluated with the image embedding intact, permuted
across dev rows (recorded permutation seed 42), or zeroed; five models, five
seeds; questions untouched; question_only asserted invariant as a control.

### Result
Accuracy drop under shuffling: fusion 0.1427 +/- 0.0024 > product_576k
0.1282 +/- 0.0033 > concat 0.1171 +/- 0.0049 >> image_only 0.0236 >
question_only 0.0000. All multimodal heads fall below the question-only
floor (0.4580) under shuffling (0.396 to 0.407): a wrong real image actively
misleads rather than merely removing information. Fusion's excess drop over
concat concentrates in logical (+0.057) and verify (+0.043) at seed 42.

### Statistical evidence
Five seeds, mean and standard deviation; per-type drops at seed 42 only.

### Interpretation and limitation
The models use the image, and fusion uses it most, consistent with the
agreement-signal reading. Reliance is not evidence of deep visual reasoning:
it measures dependence, not reasoning quality. Supersession note (5 August
2026): the row-level permutation leaves 11 self-pairs (0.14 per cent of
rows), so from E8A onward the operative wrong-image control is the
imageId-level derangement with zero self-pairs; where both exist, the
derangement figures supersede the row-level shuffle. The v2_06 drops are
retained as descriptive quantities with the dilution stated.

### Why it led to the next experiment
Both remaining explanations of the deficit, data volume and capacity, are
testable by scaling. v2_07 scales the data.

## v2_07 — data scaling at 40k / 100k / 250k

### Question
What does 6.25 times more data buy each model, does the interaction-feature
advantage survive scale, and does the multi-step deficit erode?

### Setup
question_only, concat, product_576k and fusion at 100k and 250k, five seeds,
hyperparameters unchanged; 40k results reused from v2_02/v2_04.

### Result
Dev accuracy (40k / 100k / 250k): question_only 0.4580 / 0.4829 / 0.4977;
concat 0.5240 / 0.5508 / 0.5786; product_576k 0.5335 / 0.5636 / 0.5824;
fusion 0.5384 / 0.5631 / 0.5823. The fusion-minus-concat gap decays from
+0.0144 to +0.0122 to +0.0037 +/- 0.0044 (sign flips in two of five seeds at
250k); the multimodal margin over question_only grows from +0.066 to +0.081.
Under the binding question-weighted pooled definition (addendum of 31 July
2026, superseding the unweighted two-bucket mean), the 250k seed-42 pooled
4-or-more-step deficits are question_only 0.07489, concat 0.08443, fusion
0.08075, product_576k 0.07780.

### Statistical evidence
Five seeds per cell with paired per-seed gaps. Note from the later
statistical closure: the v2_07 family could not be row-level reconstructed
(one cell, question_only 250k seed 2, recomputes 0.49754 against the stored
0.49767, about one row in 7,714), so v2_07 carries no image-clustered
intervals and the E2-minus-CLIP contrast at 250k cannot be formed. The
stored v2_07 aggregates remain the canonical accuracy record.

### Interpretation
The handcrafted interaction terms act as a small-data prior: at 40k they are
the cheapest accuracy available; at 250k they no longer pay, while a plain
concat head learns equivalent interactions from data. More data teaches the
heads to use the image more, not less. Scale lifts the whole curve without
closing the compositional gap.

### Why it led to the next experiment
Scale, capacity and handcrafted interactions were now exhausted as
explanations of the deficit; the remaining candidates were architectural.
V3 tests token-level access with a real reasoning head.

## v3_00 — token-level stores

### Setup and result
Token-level frozen-CLIP features cached for the whole extraction union:
image_tokens.h5 (63,599 x 50 x 512 fp16: CLS plus 49 patches, all projected
into the 512-d joint space; 3.26 GB) and question_tokens.h5 (2,435,691
packed word tokens; 2.51 GB). Stored CLS reproduces encode_image and the
stored EOT state reproduces encode_text; tokens are stored unnormalised so
normalisation stays a modelling choice. All pairwise image-set intersections
between train_250k, dev and the clean test are zero.

## v3_01 — the latent-query reasoner at 40k

### Question
Does token-level reasoning shrink the 4-or-more-step deficit that every
global head shows?

### Setup
32 learned latent queries; four pre-LN blocks of cross-attention to question
tokens, cross-attention to the 50 image tokens, latent self-attention and a
4x GELU FFN; 8 heads, d_model 512; mean-pooled latents into Linear(512, 100).
Exactly 21,099,620 trainable parameters. Pre-registered search over lr,
warmup and dropout only (selected: 3e-4, no warmup, 0.1); AdamW, batch 128,
cosine, patience 10, best-on-dev; seeds 0, 1, 2.

### Result
Dev accuracy 0.5422 +/- 0.0037. Gap to fusion +0.0031 +/- 0.0062, negative
in two of three seeds: noise. The reasoner's accuracy profile is essentially
fusion's, achieved with 19 times the parameters (21.1M against 1.1M), about
30 times the head latency and roughly 60 times the per-epoch training cost.
By the shuffle measure it relies on the image less than fusion (drop 0.110
against 0.143), not more.

### Statistical evidence
Three seeds, paired same-seed gaps. The binding deficit statistics for this
scale are v3_02a's (below).

### Interpretation and limitation
A negative result at 40k: token-level latent-query reasoning at this scale
and size closed none of the compositional gap. One architecture point, three
seeds, 40k only.

## v3_02a — reference baselines, statistics repair, diagnostics

### Question
What do minimal references achieve, is the pooled step statistic correct,
and what is the trained reasoner actually attending to?

### Setup and result
direct_linear (Linear(1024, 100), 102,500 parameters, five seeds) reaches
0.4813 +/- 0.0011: 91.9 per cent of concat's accuracy with 17.8 per cent of
its parameters. meanpatch_concat (mean of 49 patches, CLS excluded) reaches
0.5145 +/- 0.0019, a consistent -0.0095 below CLS concat. The report defines
the binding question-weighted pooled 4-or-more-step deficit (superseding
v3_01's unweighted mean) and attaches image-clustered bootstrap intervals
(2,000 draws, 768 clusters): fusion 0.0918 [0.0755, 0.1077]; reasoner 0.0980
[0.0824, 0.1143]; paired reasoner-minus-fusion deficit +0.0047 [-0.0067,
+0.0170], including zero. Diagnostics: the reasoner's attention gives CLS
0.26 to 0.38 mean mass against a uniform share of 0.02 (observational);
masking CLS at test time costs -0.0401 mean accuracy (test-time
intervention); a patches-only training probe (seed 0) recovers most but not
all accuracy (0.5359 against 0.5464) without improving the compositional
lift (training-time intervention).

### Interpretation and limitation
The trained reasoner leans on the pooled global (CLS) signal. Attention
weights alone are not causal evidence; conclusions are specific to this
dataset, scale, encoder, architecture and recipe.

## v3_03 (E1) — reasoner scaling to 100k and 250k

### Question
Does the negative 40k result hold at scale under the identical frozen
recipe, and does the pooled deficit change?

### Setup
Six runs: seeds 0, 1, 2 at train_100k and train_250k; the v3_01 recipe
frozen (lr 3e-4, no warmup, dropout 0.1, max 100 epochs); paired same-seed
comparisons against the stored v2_07 heads.

### Result
Dev accuracy 0.5706 +/- 0.0002 at 100k and 0.5958 +/- 0.0074 at 250k (seed 2
reaches 0.60436, the first model in the project above 0.60 on dev). The
reasoner beats every stored head in every seed at both scales: over fusion
+0.0065 (minimum +0.0054) at 100k and +0.0136 (minimum +0.0070) at 250k.
Pooled 4-or-more-step deficits: reasoner 0.0860 / 0.0775 against fusion
0.0864 / 0.0803; the paired reasoner-minus-fusion deficit difference is
-0.00037 [-0.01211, +0.01057] at 100k and -0.00275 [-0.01323, +0.00760] at
250k. Both intervals include zero.

### Statistical evidence
Three seeds per scale; image-clustered bootstrap, 2,000 draws, paired within
draw.

### Interpretation
A configuration-level finding (correction note of 5 August 2026): the
reasoner differs from the global heads jointly in architecture, token-level
access and trainable capacity (21.1M against about 1.1M), and no experiment
isolates one factor. The compositional conclusion is unchanged: the deficit
persists at every scale and remains statistically indistinguishable from
fusion's. Accuracy flips with scale; the compositional story does not.

### Why it led to the next experiments
Two authorized follow-ups probed the representation rather than the head:
a different frozen encoder (E2) and a larger answer space (E3).

## E2 — frozen SigLIP-B/16 encoder swap (global path)

### Question
Is the global-embedding ceiling a property of CLIP specifically or of the
frozen contrastive dual-encoder family?

### Setup
SigLIP-B/16 replaces CLIP on the global path under the identical v2_02
recipe; five seeds; 40k and 250k. SigLIP heads carry more parameters at
equal hidden width (768-d inputs); the confound is stated, not matched. The
comparisons are same-seed-label differences over independent training
streams, not paired gaps.

### Result
Every multimodal head gains about +1.2 to +1.5 points over its CLIP
counterpart in every seed: at 40k concat 0.5394 (+0.0154), product 0.5503
(+0.0136), fusion 0.5532 (+0.0147); at 250k concat 0.5918 (+0.0131), fusion
0.5939 (+0.0116). question_only is nearly unchanged (+0.0081 / +0.0047).
The 250k product row is doubly capacity-mismatched and is excluded from
every headline statement. The strongest SigLIP 250k global heads sit at
parity with the 21.1M CLIP-token reasoner (0.5958 +/- 0.0074) while being
about 13 to 17 times smaller. The pooled 4-or-more-step deficit spans
0.063-0.102 across models and scales, the same range as CLIP (0.064-0.105).

### Statistical evidence
Five seeds. The closure attached image-clustered intervals at 40k:
SigLIP-minus-CLIP concat +0.01540 [+0.00592, +0.02481], fusion +0.01475
[+0.00601, +0.02401], product +0.01356 [+0.00403, +0.02282], all excluding
zero; question_only does not. The 250k contrast could not be formed because
its CLIP partner is the stopped v2_07 reconstruction family.

### Interpretation
Encoder quality moves absolute accuracy; the multi-step deficit does not
move. The bottleneck looks like a property of the frozen contrastive
global-embedding representation family, not of one specific encoder.

## E3 — the top-1000 answer vocabulary

### Question
What does growing the closed answer set from 100 to 1000 answers cost on the
shared rows, and buy on the raw distribution?

### Setup
Identical protocol and partition; the first 100 vocabulary entries gated
identical to answer_vocab_v2.json; five seeds; 40k and 250k. Dev coverage
rises from 0.7711 to 0.9819.

### Result
The 1000-way heads lose about 2 points on the shared head rows at 250k
(fusion 0.5621 seed-0 against the stored 0.5823 five-seed mean); class
competition and the smaller head-answer share of the fixed budget are
confounded. On the raw distribution (in-vocabulary accuracy times coverage,
row-fair over the same 10,004 questions) they win about +3.6 to +4.1 points
at 250k: top-1000 product 0.4904, fusion 0.4866, concat 0.4822 against
top-100 fusion 0.4490. The tail (ranks 101-1000) is data-hungry: fusion tail
accuracy 0.1588 at 40k rising to 0.2504 at 250k (seed 0).

### Interpretation
On the deployment-relevant raw distribution the larger vocabulary strictly
helps; the shared-row loss is real but confounded between two mechanisms.

## E7a — component efficiency accounting (partly superseded)

### Status, binding
E7a measured, under one protocol, the cost of every stored head plus both
frozen encoder towers. Its additive end-to-end columns
(`gpu_encoder_plus_head_ms`, `full_pipeline_ms`, `amortised_ms`) and every
Pareto front derived from those sums are SUPERSEDED: they are sums of stage
medians timed in isolation, no serial pass was timed in E7a, and they must
not be used as current end-to-end latency evidence in any comparison,
ranking or "cheaper" claim. They are not replaced by E7b values; the two
experiments measured different quantities.

### Retained and valid
The isolated component latencies: CLIP image tower 2.2510 ms and text tower
1.7309 ms on GPU (single query), SigLIP 5.4818 / 2.1355 ms, CPU decode and
preprocess 2.295 ms, against head-only costs of 0.0162-0.0469 ms for the
global heads and 1.3968 ms for the reasoner. The peak-memory components, the
parameter counts, and the accuracy column (raw-distribution at 250k:
vocab1000_product 0.4904; reasoner 0.4594; fusion 0.4490; concat 0.4462).
Bounded statement drawn only from valid fields: the top-1000 product head at
250k is more accurate (0.4904 against 0.4594) than the 21.1M-parameter
reasoner, using 16 times fewer parameters. Parameter count is a poor latency
proxy: fusion (1,100,388 parameters, 0.0450 ms) is 48 per cent slower than
concat_wide (1,101,475 parameters, 0.0304 ms) at equal parameters.

## E7b — measured serial end-to-end efficiency

### Question
What does one batch-1 query actually cost, measured serially from raw image
and raw question to answer?

### Setup
Node otter155; eight systems, every timed checkpoint seed 0 at train_250k;
three pass-major passes, fresh subprocess per system per pass, 50 warm-up
plus 300 timed serial queries, synchronisation around every call; a pinned
64-row dev sample with per-image SHA-256 gating and 64/64 exact answer
reproduction in every pass. Accuracies are the frozen stored values on the
common denominator (all 10,004 raw dev questions).

### Result (warm serial median, common-denominator accuracy)
question_only 1.960 ms (0.38864, labelled non-multimodal floor); concat
7.610 ms (0.44462); fusion 7.635 ms (0.45062); vocab1000_product 7.671 ms
(0.49050); reasoner 9.172 ms (0.45602); e8a_A0p 9.212 ms (0.45852);
siglip_fusion 9.843 ms (0.45792); e8a_A1 20.146 ms (0.44332; the frozen
SmolLM2-135M forward alone is 13.46 ms, about two-thirds of the query).
Primary multimodal Pareto frontier (common-denominator accuracy against
measured serial latency): concat, fusion, vocab1000_product. The measured
CLIP-global serial cost sits about 20 per cent above E7a's additive sums,
which is why the additive columns were withdrawn. Cached-image (1.91-1.95 ms
for CLIP globals) and cached-feature (0.024-0.048 ms) regimes are separately
labelled partial-pipeline measurements, never end-to-end costs.

### Withdrawn fields, binding
`peak_allocated_mib` and `peak_reserved_mib` are INVALID_WITHDRAWN
(measurement-boundary defect: the window did not isolate the batch-1 serial
envelope). `cold_first_query_ms` is SUPERSEDED_WITHDRAWN (the accepted
source repair standardised the cold query to torch.no_grad(); the stored
values measure the older path). The two causes are distinct. No replacement
value exists and none may be estimated or substituted; remeasurement is not
authorised (the fusion bridge control between otter155 and otter159 failed
its pre-registered 10 per cent tolerance at -18.69 per cent, so cross-node
substitution is forbidden and no adjustment factor is permitted). Warm
serial latency, spreads, the accuracy column, the parameter counts and the
Pareto frontier are retained; neither withdrawn field is a frontier axis.
E7b is CLOSED (independent review
E7B_CANONICAL_SUPERSESSION_REVIEW_PASS, 14 August 2026).

## E8A — frozen small language model as question encoder

### Question
Does a frozen pretrained SmolLM2-135M question representation, projected into
the reasoner interface, beat an architecture-matched random-initialised
control, and how does the system compare with an interface-matched CLIP
control?

### Setup
Three arms sharing the 21.1M reasoner trunk and the frozen CLIP image
tokens, differing only in question-token source and projection width: A0p
(frozen CLIP question tokens, Linear(512, 512): the interface-matched
control), A1 (frozen pretrained SmolLM2-135M states, Linear(576, 512)), A1r
(frozen random-initialised SmolLM2-135M from a pinned seed: the within-size
causal control). 18 cells: three arms, 40k and 250k, seeds 0, 1, 2. The
projected interface is a learned common width, not a naturally shared
pretrained embedding space; every E8A disclosure must state this. Other
authorized E8A arms (A2/A2r, A4, A5, A7c, A8c, FLAN-T5-small) were never
executed and have no results.

### Result
A0p 0.54045 +/- 0.00438 / 0.59446 +/- 0.00030; A1 0.52861 +/- 0.00296 /
0.57316 +/- 0.00154; A1r 0.49041 +/- 0.00259 / 0.52480 +/- 0.00487 (40k /
250k). Primary causal contrast A1 minus A1r: +0.03820 [0.02947, 0.04680] at
40k and +0.04835 [0.03935, 0.05803] at 250k, positive at both scales and
growing with scale; the A1r control is non-degenerate in all six cells.
System comparison A1 minus A0p: -0.01184 [-0.01996, -0.00427] at 40k and
-0.02130 [-0.02983, -0.01253] at 250k, negative at both scales and more
negative with scale.

### Statistical evidence
Three seeds; image-clustered 95 per cent bootstrap intervals conditioning on
the fixed seed set; no multiplicity correction (disclosed).

### Interpretation and limitation
Question-side pretraining is genuinely beneficial against the matched random
control. But A1 minus A0p is a matched system comparison, not an isolation of
language-model semantics: it changes the question semantics and whether both
modalities share CLIP's pretrained space at once, and the recipe was
historically selected on the CLIP-question-token reasoner, which favours A0p.
The SLM question path buys no accuracy over the interface-matched CLIP
control and, per E7b, roughly doubles the serial query cost.

## E8B — frozen small language model as answer readout (135M)

### Question
Does a frozen pretrained SmolLM2-135M used as an answer readout (B3) beat a
within-size random-initialised control (B2), and how do both compare with a
language-model-free readout (B1)?

### Setup
18 cells: B1, B2, B3 at 40k and 250k, seeds 0, 1, 2, in a pair-preserving
order. B1 is a classifier readout over the 32 reasoner latents (52,836
trainable readout parameters); B2/B3 project the latents into the frozen
SmolLM2-135M and read answers out (21,343,808 trainable per LM arm). Frozen
recipe lr 3e-4, no warmup, dropout 0.1, retained solely because it was the
pre-result default; the eight-point recipe search was permanently abandoned
and grids 1-3 are exploratory evidence only, supporting no winner claim.
Canonical evaluation is FP32; primary metric is R1 accuracy on the 7,714
in-vocabulary dev rows.

**Checkpoint selection, binding**: B1 is BEST_ON_DEVELOPMENT (up to 100
epochs, early stopping patience 10); B2 and B3 are FIXED_EPOCH_22 (exactly
22 epochs, no early stopping, epoch 22 canonical). B3 minus B2 is therefore
the only selection-matched clean causal contrast; every comparison involving
B1 confounds architecture with selection rule and carries the
MIXED_WITHIN_ROW label in the evidence inventory (corrected by the pre-F1
metadata repair).

### Result
B1 0.54507 +/- 0.00390 / 0.59394 +/- 0.00486; B2 0.53392 +/- 0.00233 /
0.60086 +/- 0.00528; B3 0.50929 +/- 0.02795 / 0.59610 +/- 0.00270 (40k /
250k). B3 minus B2: -0.02463 [-0.03333, -0.01619] at 40k (negative
directional, all three seeds negative) and -0.00475 [-0.01208, +0.00297] at
250k (uncertain or mixed, near parity). Scale effects: B1 +0.04887, B2
+0.06693, B3 +0.08681, all positive directional and larger than every
between-arm contrast. B3 at 40k is unstable (seed 1 reaches 0.47783; seed SD
0.02795 falling to 0.00270 at 250k). The R1/R2/R3 readout modes (canonical
scoring, trie-constrained generation, free bounded generation) agree within
about 0.0005 on every arm and scale, so output mechanism is not what
separates these systems. All arms rely on the image (derangement drops
+0.084 to +0.092 at 40k, +0.118 to +0.126 at 250k, all directional);
removing the question is far more damaging than removing the image
everywhere.

### Interpretation
Data scale is the dominant driver. Answer-side pretraining shows no positive
advantage: directionally negative at 40k, near parity with an interval
including zero at 250k. This is not a proof of absence at 250k. The LM-backed
architecture offers at most a small, selection-confounded system-level
advantage over the language-model-free readout.

## E9 — compact integrated VLM in context (evaluation only)

### Question
Where does a frozen off-the-shelf compact integrated VLM land, given no GQA
supervision, relative to the trained lightweight systems, and at what
measured serial cost?

### Setup
SmolVLM-256M-Instruct (primary) and SmolVLM-500M-Instruct (secondary),
evaluation only, zero trainable parameters, greedy decoding, protocol frozen
and hashed before any score was observed. Primary metric: the pinned G21
normalised exact match on the full 10,004-row raw development partition.
This is contextual positioning, not a matched causal comparison: training
history, multimodal pretraining, answer support and output format all
differ, and the 256M text backbone is the Instruct checkpoint where E8A/E8B
use base.

### Result
SmolVLM-256M reaches 0.436525: above every E8B arm at train_40k and below
every one at train_250k, all six paired contrasts directional. SmolVLM-500M
reaches 0.489004, exceeding every E8B arm at both scales (+0.026 to +0.031
over the 250k arms). Constraining generation to the top-1000 answer support
moves the score by at most about 0.005 with every interval containing zero,
so output format is not what limits the compact VLM; strict raw exact match
is near zero only because the model emits " Yes." against gold "yes". The
matched image-partner derangement costs SmolVLM-256M +0.16763, about a third
more than any E8B arm (+0.084 to +0.126): it relies on the image more. On
one node (otter159) under the frozen E7b serial protocol, SmolVLM-500M
measured 151.380 ms for its open-readout row (accuracy 0.489004) and
152.854 ms for its constrained row (accuracy 0.485706), against 6.199 ms
for the lightweight top-1000 product head (accuracy 0.49050):
approximately 24.4 and 24.7 times the warm serial latency respectively,
under the same-node contextual protocol. That is a
latency statement only; no energy, power, carbon, monetary-cost or general
computational-cost claim follows from it. The fusion bridge control failed
its pre-registered 10 per cent tolerance against the historical E7b node
(-18.7 per cent), so the otter155 and otter159 latency sets are kept as two
frontiers, never merged, with no adjustment factor.

### Statistical evidence
Image-clustered paired bootstrap (2,000 draws) for the contrasts; E9 has no
training seeds (single evaluation run, bitwise-replicated in a fresh
process).

## E10 — answer-side capacity sensitivity at 360M

### Question
Does the E8B answer-side finding change when the frozen readout grows from
SmolLM2-135M to SmolLM2-360M?

### Setup
B4 (frozen pretrained SmolLM2-360M readout) and B4r (architecture-matched
random control), 40k and 250k, seeds 0, 1, 2: twelve pair-preserved cells,
executed 11-13 August 2026 under a single-use external authorisation grant
(now operationally spent) with a 12.0-hour whole-cell wall and zero retries.
Fixed epoch 22, FP32 canonical evaluation, primary metric on the 7,714-row
dev partition. The registered primary output is the pretraining-by-scale
difference in differences.

### Result
B4 0.52636 +/- 0.00466 / 0.59407 +/- 0.00831; B4r 0.53319 +/- 0.00473 /
0.59645 +/- 0.00343 (40k / 250k; the +/- figures are the sample standard
deviation across the three training seeds, ddof=1, and are not confidence
intervals). Scale effects: B4 +0.06771 [+0.05800, +0.07738], B4r +0.06326
[+0.05287, +0.07296], both positive directional. Pretraining effects: -0.00683
[-0.01508, +0.00114] at 40k and -0.00238 [-0.01071, +0.00588] at 250k, both
intervals containing zero, seeds disagreeing in sign at both scales.
Difference in differences +0.00445 [-0.00709, +0.01641], also not
directional. The bracketed intervals are image-clustered bootstrap intervals
conditioning on the fixed trained seed set: they carry evaluation-sampling
variation, not training-seed variation.

### Interpretation
No reliable positive pretrained-over-random advantage was detected at either
scale; the intervals establish neither equivalence nor the absence of an
effect. Training-set size is what moves accuracy (+6.3 to +6.8 points in
every seed). E10 qualitatively reproduces E8B's finding at 360M; it does not
reproduce E8B's directional negative 40k effect (E10's -0.00683 is uncertain
with mixed seed signs against E8B's -0.02463 with all seeds negative).

### Limitation
The 135M-to-360M comparison is whole-system capacity sensitivity, not an
isolated causal language-model-size effect: the projection width follows the
hidden size, so trainable capacity moves too (21,343,808 to 21,540,800
parameters, about 0.923 per cent). E10 produces no new visual-reliance
evidence. E10 is CLOSED at E10_PASS and must not be reopened.

## Evidence infrastructure (not experiments)

The statistical and efficiency closure (13 August 2026) re-verified all 460
hash-manifested artefact entries, reconstructed per-question correctness for
205 of 245 conditions (the v2_07 family stopped on a one-row reproduction
mismatch and contributes no interval), and attached image-clustered
intervals to the claim-bearing contrasts; its outputs are under
`results/closure/`. VE-0 froze the 428-row canonical evidence inventory;
VE-1 rendered the dissertation's quantitative figures (effective counts: 9
main-text, 6 appendix) and 11 tables from that contract; VE-2 produced the
deterministic qualitative gallery (18 records, selection by salted hash).
The pre-F1 status supersession consolidates lifecycle (five packets closed:
E7b, VE-0, VE-1, VE-2, E10); the pre-F1 evidence metadata repair is the
field-level successor correcting the nine B1-bearing checkpoint-selection
rows, five presentation-precision fields, the E7a locator and the
VE1-FIG-04-FULL placement.

## Master experiment table

| Experiment | Scientific question | Main result | Interpretation | Status |
|---|---|---|---|---|
| V1 | Is closed-set VQA over frozen CLIP learnable? | fusion 0.541 vs concat 0.525 (single seed) | Learnable; language dominates | Legacy prototype, non-confirmatory |
| v2_00 | Can a defensible protocol be built? | 99/99 verifier checks; byte-identical rebuilds | Protocol sound | Complete, verified |
| v2_01 | Zero-shot floor? | 0.0770-0.0795 vs majority 0.2247 | Training-free anchor only | Complete |
| v2_02 | Does the fusion gain survive seeds? | fusion-concat +0.0144, positive all 5 seeds | Stable but modest | Complete |
| v2_03 | Features or capacity? | capacity +0.0086; features +0.0050 to +0.0094 | Roughly half and half | Complete |
| v2_04 | Which interaction term? | either term alone: +0.0096/+0.0094 | Terms redundant, not complementary | Complete |
| v2_05/b/c | Where does the gain live? | verify/logical/obj/rel; deficit about 0.08 on >=4-step | Agreement signal; multi-step weakness found | Complete (clustered intervals) |
| v2_06 | Do the heads use the image? | shuffle drops: fusion 0.143 > product 0.128 > concat 0.117 | Real reliance; not proof of reasoning | Complete (descriptive; derangement supersedes where both exist) |
| v2_07 | What does 6.25x data buy? | feature gap +0.0144 to +0.0037; deficit persists | Interactions are a small-data prior | Complete (no clustered intervals; reconstruction stopped) |
| v3_01 | Does token-level reasoning close the deficit at 40k? | reasoner-fusion +0.0031 +/- 0.0062 (noise) | Negative result | Complete |
| v3_02a | References and diagnostics | direct_linear 0.4813; reasoner leans on CLS | Head re-reads the global signal at 40k | Complete |
| v3_03 (E1) | Does the 40k result hold at scale? | reasoner beats every head (+0.0065 / +0.0136 over fusion); deficit unchanged | Configuration-level accuracy win; compositional deficit persists | Complete |
| E2 | Encoder-specific or family-wide? | +1.2 to +1.5 points; deficit range unchanged | Family-wide bottleneck | Complete |
| E3 | Cost/benefit of 1000 answers? | -2 points shared rows; +3.6 to +4.1 points raw distribution | Raw-distribution win | Complete |
| E7a | Component costs? | heads 0.016-0.047 ms; CLIP towers 2.25/1.73 ms | Components valid; additive sums withdrawn | Complete, partly superseded |
| E7b | Measured serial end-to-end cost? | globals 7.61-7.67 ms; frontier concat/fusion/vocab1000_product | Authoritative end-to-end evidence | CLOSED (3 fields withdrawn) |
| E8A | Question-side SLM: pretrained vs random? | A1-A1r +0.03820/+0.04835 (positive); A1-A0p negative | Pretraining helps vs random; CLIP control wins as a system | Complete (core 135M) |
| E8B | Answer-side SLM: pretrained vs random? | B3-B2 -0.02463 at 40k; -0.00475 (interval spans zero) at 250k | No positive pretraining advantage | Complete |
| E9 | Compact VLM in context? | 256M 0.436525; 500M 0.489004; approx 24-25x latency | Contextual positioning only | Complete, closed |
| E10 | Does 360M change the answer-side finding? | pretraining effects -0.00683/-0.00238, intervals span zero; DiD +0.00445 | Capacity does not rescue answer-side pretraining | CLOSED at E10_PASS |

Accuracy denominators in this table: V2/V3/E2/E8A/E8B/E10 rows are
in-vocabulary dev accuracy (7,714 rows); E3 quotes raw-distribution values
where stated; E9 is G21 normalised on the 10,004-row raw denominator; E7b
accuracies are common-denominator raw-distribution values.
