# Experiment E3: the 1000-answer vocabulary

## Purpose

Measure what the top-100 answer restriction costs and what changes when
the closed set grows to the top 1000 under the same frozen CLIP
ViT-B/32 encoder, protocol and recipe: coverage of the development
distribution, absolute accuracy, and the head/tail composition of that
accuracy. Approved by the user on 30 July 2026 (E3); launch-instructed
on 1 August 2026.

## Method

experiments/e3_vocab1000/run.py (pinned commit 0b3b61c after a
two-round pre-launch review). The recorded v2_00 image partition is
reused verbatim via the tracked membership artifacts (777 dev images,
71,363 pool images; counts, disjointness and the 932,996/10,004 raw row
counts hard-asserted; nothing re-derived). The top-1000 vocabulary uses
the identical V1/V2 rule (count-descending, lexicographic tie-break)
over raw training-pool answers only; because that ranking is
prefix-stable, the first 100 entries were gated to equal
data/v2/answer_vocab_v2.json exactly (answers and indices) — they do.
The eligible pool (921,335 in-vocabulary questions, coverage 0.9875)
was canonically sorted and permuted once with the same stream design
and seed role as v2_00 (numpy default_rng(43)); nested 40k/100k/250k
prefixes were written under data/v2_1000/ (training uses 40k and 250k).
The build ran twice and all seven generated files were byte-identical.

Embeddings reuse the canonical frozen-CLIP stores row-for-row; only the
delta was encoded (5,667 images; 183,987 question ids over 126,536
unique texts) with v2_01 conventions (fp32, unit-normalised, atomic
writes, canonical provenance attrs). Gates: unit norms; five sampled
re-encodes within the v2_01-precedent 1e-3; five reused dev rows equal
to the canonical store rows exactly; labels in [0, 1000); the stored
v2_02 concat seed-0 accuracy reproduced at 0.52230 before training.

Training is the v2_02 recipe verbatim over 1000-way heads of the same
shape (Linear-ReLU-Dropout-Linear, hidden 512, dropout 0.3; config.py
untouched): question_only, concat, product, fusion; AdamW lr 1e-3,
weight decay 1e-4, batch 256, 30 epochs, best-on-dev; per-seed
utils.set_seed plus one loader-generator reseed before the sequential
model loop (the 31 July seed-pairing clarification applies); seeds
{0, 1, 2, 3, 42}; scales 40k and 250k; 40 runs; nothing tuned on E3
results. All run records were persisted before the analysis gates
(runs_interim.json), per the pre-launch review.

Provenance note: the top-level run_metadata block in results.json
carries the shared config defaults (seed 42, top_k_answers 100); the
nested e3_vocab1000 block is authoritative for the seeds and the
1000-answer vocabulary.

Blinding: all selection on the E3 dev view; the clean test was
completely untouched — no clean-test file read or written, and no
statistic under the new vocabulary involves it.

## Outputs

data/v2_1000/ (git-ignored): the vocabulary, manifests, build summary
and hash record, plus embeddings/{images_extra.h5, questions_extra.h5,
train_40k.h5, train_250k.h5, dev.h5}. Under
results/experiments/e3_vocab1000/: preflight.json, runs_interim.json,
results.json, run.log and 40 checkpoints (hash-pinned, not exported;
the JSONs are exported to artifacts/results_export/e3_vocab1000/).

## Results

Coverage of the raw development distribution (10,004 questions on the
777-image partition, identical for both vocabularies): top-1000 covers
9,823 questions (0.9819) against 7,714 (0.7711) under top-100; pool
coverage 0.9875 against 0.7761. The majority reference under the new
vocabulary ("no") scores 0.17642 on the E3 dev view.

Dev-view accuracy over five seeds (1000-way heads):

| scale | model | mean | std | params |
|---|---|---|---|---|
| 40k | question_only | 0.3674 | 0.0008 | 775,656 |
| 40k | concat | 0.4326 | 0.0024 | 1,037,800 |
| 40k | product | 0.4438 | 0.0022 | 1,299,944 |
| 40k | fusion | 0.4469 | 0.0025 | 1,562,088 |
| 250k | question_only | 0.4062 | 0.0036 | 775,656 |
| 250k | concat | 0.4911 | 0.0018 | 1,037,800 |
| 250k | product | 0.4995 | 0.0025 | 1,299,944 |
| 250k | fusion | 0.4955 | 0.0023 | 1,562,088 |

Head/tail decomposition (seed 0 only, reproduction-gated; "head" means
labels 0-99, which — by the prefix-identity gate — is row-identical to
the 7,714-question top-100 dev view):

| scale | model | head accuracy | tail accuracy |
|---|---|---|---|
| 40k | fusion | 0.5215 | 0.1588 |
| 40k | product | 0.5228 | 0.1697 |
| 250k | concat | 0.5537 | 0.2679 |
| 250k | product | 0.5634 | 0.2660 |
| 250k | fusion | 0.5621 | 0.2504 |

Context against the stored top-100 results — side-by-side only, never
paired: the top-100 numbers use the 7,714-row dev view, different
training rows and a 100-way output; the 250k product reference there is
the parameter-matched 576k head (hidden 351), not
architecture-comparable to E3's natural-width product. Stored top-100
means: 40k 0.4580/0.5240/0.5367 (product_natural)/0.5384; 250k
0.4977/0.5786/0.5824 (product_576k)/0.5823.

Raw-distribution accuracy (derived: in-vocabulary accuracy times
coverage, since out-of-vocabulary questions are necessarily wrong under
a closed set; both protocols share the identical 10,004 raw dev
questions, so this comparison is row-fair):

| system (250k) | in-vocab acc x coverage | raw-distribution accuracy |
|---|---|---|
| top-100 fusion (v2_07) | 0.5823 x 0.7711 | 0.4490 |
| top-100 concat (v2_07) | 0.5786 x 0.7711 | 0.4462 |
| top-1000 fusion (E3) | 0.4955 x 0.9819 | 0.4866 |
| top-1000 concat (E3) | 0.4911 x 0.9819 | 0.4822 |
| top-1000 product (E3) | 0.4995 x 0.9819 | 0.4905 |

The derived cells use the rounded factors shown; recomputing from the
stored five-decimal values gives gains of +0.0361 (concat), +0.0375
(fusion) and +0.0413 (product), so the honest range is about +3.6 to
+4.1 points.

Efficiency: total wall time 0.34 h for the build, both idempotence
passes, delta extraction, 40 runs and the analysis; head training
times remain seconds (40k) to about a minute (250k) per run.

## Decisions and problems

(a) Coverage, not per-question accuracy, is where the vocabulary axis
pays. Growing the closed set to 1000 answers raises dev-distribution
coverage from 77.1% to 98.2%. On the shared head rows (the exact
top-100 dev view), the 1000-way fusion head loses about 2.0 points at
250k (0.5621 seed-0 against the stored 0.5823 five-seed mean) —
consistent with competing over ten times as many classes and with the
smaller head-answer share of the fixed training budget; the two
mechanisms are confounded here — but the recovered tail (2,109 dev
questions, accuracy 0.25-0.27 at 250k) more than compensates: on the
full raw distribution the top-1000 heads answer about 3.6 to 4.1
points more of all dev questions correctly than their top-100
counterparts. The dissertation's efficiency claim should be
stated on this raw-distribution basis, where the larger vocabulary
strictly helps.

(b) The tail is data-hungry. Tail accuracy roughly jumps by half again
from 40k to 250k (fusion 0.1588 to 0.2504) while head accuracy moves
about 4 points; rank-centile accuracies (stored per centile in
results.json) fall steeply with rank. Larger training budgets
disproportionately benefit rare answers, which matters for any future
scale-up.

(c) Scope and caveats. Development results only; the clean-test
embargo is untouched. Head/tail and centile numbers are seed-0
(reproduction-gated); the five-seed aggregates carry the headline. The
top-100 context is never paired; the 250k product context reference is
capacity-mismatched as noted. The raw-distribution derivation is
arithmetic over stored values (formula stated above), not a new
measurement.

(d) Consequence for the programme: the strengthening experiments
(E1-E3) are complete. The freeze decision (F1) now weighs: encoder
(SigLIP global heads rival the CLIP-token reasoner at a fraction of its
size), head (reasoner wins at scale on CLIP tokens), and vocabulary
(top-1000 wins on the raw distribution). These axes are complementary
and the final model list should be assembled with the supervisor's
design feedback in view.
