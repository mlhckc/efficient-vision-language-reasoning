# VE-2: deterministic qualitative development evidence

## Purpose

VE-2 populates VE0-FIG-10, the one VE-0 specification VE-1 recorded as
`DEFERRED_TO_VE2`. It shows individual GQA development questions beside the
real photograph, the gold answer and what each compared system actually
answered, so a reader can see what the aggregate findings look like on single
rows.

It is a reporting task, not an experiment. Nothing was trained, evaluated,
timed, scored or selected as a checkpoint; no checkpoint or embedding store was
opened; zero GPU hours were charged; no quantitative result was recomputed; and
the embargoed clean test was neither read nor resolved.

The problem VE-2 exists to solve is narrower than it looks. A qualitative panel
is easy to produce and easy to produce dishonestly: choose the example that
reads best, show the model in its good light, and let a picture carry an
argument the statistics do not support. The frozen VE-0 qualitative protocol
removes the choice from the author. VE-2's job is to execute that protocol
exactly, prove that every string on every panel came from a frozen artefact,
and record what it could not show.

## Method

The build is one deterministic pass:

    python -m experiments.ve2.run_ve2

### The frozen contract

The fourteen VE-0 outputs under `results/ve0/` are hashed at load time and the
build refuses if any output's `content_sha256` no longer matches
`VE0_MANIFEST.json`. The qualitative protocol is artefact H; its content digest
is `c6ca3aece6f877afefd6d46afd99ab05b60e8d1a5afa172568666dae7150b76d` and the
build ran against exactly that.

The selection salt is `P2607-VE0-QUALITATIVE-SALT-20260813-v1`. It is READ from
the contract at build time; no VE-2 module holds a copy, and a test asserts
that no copy exists in VE-2 source. The rank rule is the recorded one:

    rank_key = SHA256(salt + '|' + category_id + '|' + question_id)

Candidates sort ascending by rank key, ties break by ascending question
identifier. `assert_rank_formula_matches` refuses to select at all if the
protocol's recorded formula is not the rule the code implements.

Nothing about a candidate other than its question identifier enters the key.
`order_pool` is given indices and question identifiers only, so it has no
access to the image, the model, the prediction or the outcome, and the
candidate pool is built before the order is computed and the order before any
image file is opened.

### Proving prediction identity

A qualitative example is one row of one experiment's stored predictions, so the
whole risk is that the row shown under a model's name is a different row or a
different model's. Every source proves four things before it yields a value.

The bytes hash to what a frozen manifest recorded, and the manifest is reached
from the protocol's own `prediction_evidence_sources[...].hash_bound_in` field
rather than from a path typed into VE-2. The artefact is the development split,
checked against the 7,714-row in-vocabulary view. `question_id`, `image_id` and
the gold answer or gold label are compared element-by-element against
`data/v2/dev.csv`; no source is used positionally on the strength of a row
count. Arm, scale, seed and condition are read from the manifest entry or from
the file's own columns, never parsed out of a filename. For the E8A tables the
recorded checkpoint and scorer digests travel into the record as well.

Three sources are used: the closure reconstructed row evidence for the
global-embedding families (v2_02, v2_06, e2), which stores prediction strings
directly; the E8A G21 prediction tables for A1 and A1r; and the E10 core
per-row arrays for B4 and B4r, whose vocabulary indices are resolved to answer
strings through `data/v2/answer_vocab_v2.json`, which the protocol states
explicitly is a lookup and not a re-evaluation. E10 is closed and was read only:
VE-2 wrote nothing into the E10 tree and ran no E10 code.

### The wrong-image partner

VE0-QC07 requires showing the image that replaced the question's own, and the
stored v2_06 row evidence records each row's own image under both conditions,
so the partner is written down nowhere. It is recoverable, because the
intervention permutes the cached image embeddings under a recorded seed, but
recovering it by re-running the shuffle would be taking the original builder's
word for it.

Instead the permutation is derived and then PROVED from stored evidence alone.
The `image_only` head consumes nothing but an image embedding, so under a
correct permutation its shuffled prediction at row i must equal its normal
prediction at row perm[i]. That identity holds on 7,714 of 7,714 rows. A
deliberately different permutation, as a negative control against a head that
might have predicted one answer everywhere, explains 3,131. The recovered
mapping is therefore a checked fact, and the build refuses to name a partner it
cannot prove.

### Skips

The frozen substitution policy allows a skip for exactly two mechanical
reasons: the image file is unreadable, or the row fails the category predicate
on re-check. `select` accepts only those two strings and raises on anything
else, so an aesthetic, rhetorical or outcome-related skip cannot be recorded
even by accident. A negative test confirms the refusal fires.

## Outputs

Under `results/ve2/`:

- `qualitative_records.json` — 18 records, each carrying exactly the nineteen
  fields the frozen VE-0 record schema declares.
- `selection_registry.json` — every category's pool size, the rank of every
  example taken, every skip with its reason, the balance directions, the
  rank-collision check, the dropped category with its finding, and the unused
  E8B source with the reason it is unused.
- `galleries/ve2_gallery_main.pdf` — five pages, one per main-text category,
  two example rows each, with a preview PNG per page.
- `galleries/ve2_gallery_appendix.pdf` — four pages on the same footing.
- `galleries/*.data.json` — exactly what was drawn on each page, which is what
  the tests compare against the records.
- `captions.json` — 28 caption records: one per gallery page, one per selected
  example, one for the dropped category.
- `visual_qa.json` — the inspection record, findings kept after repair.
- `provenance_registry.json` and `VE2_MANIFEST.json`.

Builders under `experiments/ve2/`; tests in `tests/test_ve2.py`, run by
`tests/run_ve2.py`.

## Results

Nine of the ten frozen categories were populated, two examples each, 18
examples over 18 distinct questions and 20 distinct images. No skip occurred:
every category's lowest-ranked candidates satisfied the predicate on re-check
and every image resolved and decoded.

| Category | Placement | Pool | Ranks taken | Skips |
|---|---|---|---|---|
| QC01_all_correct | APPENDIX | 2,236 | 1, 2 | 0 |
| QC02_all_wrong | MAIN_TEXT | 2,323 | 1, 2 | 0 |
| QC03_fusion_right_concat_wrong | MAIN_TEXT | 774 | 1, 2 | 0 |
| QC04_representation_improvement | APPENDIX | 1,130 | 1, 2 | 0 |
| QC05_question_side_pretraining | MAIN_TEXT | 1,467 | 1, 2 | 0 |
| QC06_answer_side_disagreement | APPENDIX | 2,756 | 1, 3 | 0 |
| QC07_visual_reliance | MAIN_TEXT | 2,142 | 1, 2 | 0 |
| QC08_short_reasoning | APPENDIX | 2,400 | 1, 2 | 0 |
| QC09_deep_reasoning_failure | MAIN_TEXT | 838 | 1, 2 | 0 |

Pool sizes are candidate counts within the 7,714-row in-vocabulary development
view. They are a property of the frozen predicates, not a new result: each is
the count of rows satisfying a condition whose aggregate consequences the
closure and VE-1 already report.

QC06 takes ranks 1 and 3 rather than 1 and 2 because its frozen balance
requirement asks for equal numbers in each direction and ranks 1 and 2 both
fall the same way. Rank 1 is a question the pretrained 360M readout answers
correctly and its random control does not; rank 3 is the reverse. The aggregate
answer-side contrast is not directional and its interval contains zero, so
showing two rows in one direction would have suggested a direction the
statistics do not support.

### The dropped category

QC10, the compact-VLM disagreement case, is recorded `DROPPED_BY_PROTOCOL`. The
protocol marked it `CONDITIONAL_REQUIRES_VE2_VERIFICATION` with
`drop_if_unverified: true` and required VE-2 to establish, from the frozen E9
artefacts alone, whether a per-row correctness vector is recoverable. What the
verification found:

- `e9_smolvlm_*.tokens.npz` holds `tokens`, `lengths` and `questionIds` over
  the 10,004-row raw development view. No correctness array.
- The per-condition JSON holds `texts`, the already-decoded generations,
  alongside `questionIds`. Decoding is therefore not even necessary; the answer
  strings are stored. Still no correctness array.
- `e9_results.json`, `e9_efficiency.json`, `delta_ledger.json` and the artefact
  manifest hold aggregate scores only. No artefact anywhere in the E9 tree
  stores per-row correctness.

So the only route to the vector the category predicate needs is to apply the
pinned G21 normalised scorer to the stored generations against the gold
answers. That produces a per-row scientific quantity no frozen artefact holds,
and the protocol names that exact operation, re-scoring, as a new evaluation
and puts it out of scope without fresh explicit user authorisation, which has
not been given. A second obstacle stands behind the first: the compact VLM was
evaluated over the 10,004-row raw view and the global-embedding head over the
7,714-row in-vocabulary view, so the two correctness vectors do not share a
denominator.

No SmolVLM weight was loaded, no generation was re-run, no scorer was applied
to any stored generation, and no substitute compact-VLM category was invented.

The E8B per-row arrays are listed by the protocol as an available
correctness-only source but are named by no category, so no E8B row was read,
no positional alignment was assumed, and no E8B prediction appears in any VE-2
artefact. This is recorded rather than left implicit.

### Visual quality assurance

Every rendered page was opened and inspected. Five findings, all repaired and
kept in the record:

1. Blocking. The page title ran off the right edge, cut mid-word. The title now
   carries the category identifier alone, the description moved to a wrapping
   subtitle, and every string is measured against the space available before it
   is drawn.
2. Blocking. On the visual-reliance page the prediction values were drawn on
   top of their own labels. The value column is now placed after the widest
   label actually being drawn, measured through the font, and the build refuses
   a row whose columns would collide.
3. Blocking, and found by cross-checking the rendered pages against the records
   rather than by eye. Both QC06 examples carried the identifier
   `VE2-QUAL-QC06_answer_side_disagreement-01`, because the two balanced
   directions were ranked separately and each produced a rank 1. Two questions
   under one identifier is a provenance failure. The category's ordering is now
   computed once over its whole pool and each direction selects by filtering
   it, so every example keeps its rank in the single ordering; a guard refuses
   any build in which two records share an identifier or a category selects one
   question twice.
4. Cosmetic. The two images in a visual-reliance row did not share a top edge
   and their labels sat at different heights. Each image is now drawn into a
   fixed square box anchored at the top, keeping its own aspect ratio.
5. Cosmetic. A landscape photograph's identifier label floated below its image.
   The label now follows the rendered image rather than its box.

### Validation

`python -B tests/run_ve2.py` runs 35 checks, all passing. They recompute every
rank key, re-derive every candidate pool independently of the builder, confirm
each published example is the lowest-ranked eligible row of its category's
ordering, read every prediction and correctness value back out of its source
file, hash every image and confirm it is the file for its identifier, check
that every drawn string matches its record and that no prediction sits under
the wrong label, and confirm the rebuild reproduces the gallery bytes.

Four deliberate tamperings were applied to a copy and all four were caught:
swapping two systems' predictions, changing a gold answer, pointing an image at
a different file, and altering a selection rank.

### Scientific invariance

No quantitative value changed. VE-0 and VE-1 artefacts still hash to their own
recorded content digests; `tests/run_ve0.py` (3,875 checks),
`tests/run_ve1.py` (14,013 checks) and `tests/run_closure.py` (4,190 checks)
all pass unchanged. The three E10 phase verifiers pass and
`run core-cell B4 train_40k 0` still refuses with the operationally-SPENT grant
message. No claim status changed, no checkpoint or model changed, no scorer
changed, and no new aggregate result was introduced.

## Decisions and problems

**One page per category rather than one page.** The specification asks for "one
row per selected example, grouped by selection category". Ten main-text
examples on a single page would either overflow a dissertation page or shrink
to an unreadable font, so the gallery is a multi-page document with one page
per category and one row per example on it. That satisfies the grouping
literally and keeps each page readable. The deviation is recorded in each
gallery sidecar as `grouping_note`.

**Layout on an inch grid, with overflow as a build failure.** The first
implementation positioned text in axes fractions and produced two blocking
defects immediately. Both are now impossible to reintroduce silently: strings
are measured through the font manager and the build stops rather than clipping
text or overprinting a label. A character count was the wrong unit for a
proportional face, so wrapping is by measured width.

**Short labels on the panel, full expansions in the header.** The style
contract requires every arm code to be expanded at first use. Putting the full
expansion against each prediction row made the label column wider than the page
could carry, so the panel carries a short label and the header carries the
expansion.

**The program-step count appears on two categories only.** QC08 and QC09 are
defined by program length, so the count belongs on their panels. The trigger is
the frozen predicate naming `n_steps`, not a hand-picked list. It is labelled
"GQA program steps" because the style contract requires program steps, never
reasoning steps performed by the model.

**Interpretation text is assembled, not written per example.** Each panel
carries the mandatory illustrative marker followed by its category's own
`interpretation_bound`, copied verbatim, so a caption cannot say more than the
frozen category allows. A guard refuses the specific overclaims the style
contract names — grounding, understanding, reasoning about an image, proof, and
out-of-distribution compositional generalisation — in any reader-facing string,
with the sanctioned disclaimers exempted and negative tests on both sides.

**Clean-test governance.** The clean-test contents were never inspected or used
for development, model selection, or reporting decisions. Its bytes were
mechanically read once by an independent reviewer integrity-hash command on
13 August 2026. VE-2 did not open, hash, stat or parse the clean-test target,
and no VE-2 module resolves its path. Image eligibility is established
positively, by membership in `data/v2/dev_image_ids.json`, rather than by
absence from an embargoed list, so no embargoed identifier file is opened
either. The path and payload guards both refuse the embargoed stem and both
have negative tests.

**What a reader must not take from these panels.** Each example illustrates an
aggregate finding established by the quantitative evidence and is not itself
population-level statistical evidence. The visual-reliance panels show that a
prediction depends on the image input for that row; they are not proof of
grounding. The deep-reasoning panels show where the pooled four-or-more-step
deficit lives; they do not show that the model failed because the question was
deep, and GQA program length is not out-of-distribution compositional
generalisation. The fusion-versus-concatenation panels illustrate a contrast
that is itself capacity-confounded. The representation panels change an encoder
and two widths together. The answer-side panels sit under a contrast whose
interval contains zero, which is an absence of a detected effect and not
equivalence.
