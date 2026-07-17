# Experiment v2_05: GQA metadata re-join and question-type analysis

## Purpose

Join the GQA structural type, semantic type and program-step count back onto
the V2 dev and train_40k manifests, build per-type prior baselines, and slice
the trained models' dev accuracy by type, to see where the fusion gain lives,
to corroborate the v2_04 redundancy finding, to map which types are
answerable from language statistics alone, and to test whether relational and
multi-step questions expose the global-vector ceiling. Analysis only; no
training.

## Method

experiments/v2_05_types/join_metadata.py extracts, per string questionId,
types.structural, types.semantic and len(semantic) from the raw GQA
train_balanced and val_balanced JSONs, joins them onto dev.csv and
train_40k.csv (both joins matched 7,714/7,714 and 40,000/40,000, 100%), and
writes data/v2/metadata/dev_types.csv and train_40k_types.csv aligned to the
manifest row order. Per-type priors follow the VQA/GQA-paper methodology:
the most frequent train_40k answer per structural and per semantic type,
evaluated on dev. No metadata was built for the clean test and no clean-test
statistic was computed (clean-test question-type statistics are embargoed).

experiments/v2_05_types/analyze.py loads the seed-42 checkpoints of the
seven trained models (v2_02: question_only, image_only, concat, fusion;
v2_03: fusion_narrow; v2_04: product_576k, difference_576k), verifies each
recomputed dev accuracy against the stored value (all matched), and adds the
v2_01 zero-shot with the "a photo of {answer}" prompt, whose per-row dev
predictions were recomputed from the cached embeddings (v2_01 saved only
accuracies; recomputation is free and reproduced 0.0795 exactly). As
documented in v2_01, a question-conditioned zero-shot is not possible with
plain CLIP, so the zero-shot here is image-only. Accuracy is sliced by
structural type, semantic type and step buckets (<=2, 3, 4, >=5).

Seed limitation: per-type slicing uses seed 42 only. Per-type deltas across
seeds are second-order relative to slice sizes, but this is untested here;
per-type numbers therefore carry more uncertainty than the multi-seed
overall numbers, especially for small slices (compare n=302, global n=163).

## Outputs

data/v2/metadata/ (git-ignored): dev_types.csv, train_40k_types.csv. Under
results/experiments/v2_05_types/ (git-ignored): metadata_summary.json,
results.json, per_type_accuracy.csv, structural_types.png (grouped bars,
four main models by structural type) and fusion_gap_semantic.png
(fusion-concat gap by semantic type).

## Results

Dev type distributions (7,714 questions): structural choose 1,042, compare
302, logical 1,236, query 3,104, verify 2,030; semantic obj 1,170, attr
2,964, cat 439, rel 2,978, global 163; steps 2: 2,400, 3: 3,330, 4: 870,
5: 824, 6: 6, 7: 284.

Per-type priors on dev (train_40k priors):

| predictor        | overall | notable per-type values |
|------------------|---------|--------------------------|
| global majority  | 0.2247  | "no" everywhere |
| structural prior | 0.3053  | verify "no" 0.5099, logical "yes" 0.5275, compare "yes" 0.4404, choose "right" 0.2783, query "left" 0.0789 |
| semantic prior   | 0.2350  | obj "yes" 0.5171, attr "yes" 0.1943, rel "no" 0.1924, cat "table" 0.0547, global "yes" 0.2147 |

Per-model dev accuracy by type: full table in per_type_accuracy.csv;
overall (seed 42): question_only 0.4589, image_only 0.2344, concat 0.5224,
fusion 0.5364, fusion_narrow 0.5336, product_576k 0.5306, difference_576k
0.5336, zero_shot_photo 0.0795.

## The four analyses

(a) Where does fusion gain over concat? The gain is not uniform, and it is
not purely relational. Fusion beats concat most on verify (+0.041), logical
(+0.040), obj (+0.039), steps >=5 (+0.040), steps 4 (+0.026) and rel
(+0.026); it is flat on query (+0.001) and attr (-0.004); and it loses on
compare (-0.043), choose (-0.016) and global (-0.018). So the gain
concentrates in verification and logical yes/no formats, object questions,
relational questions and deeper programs, at a real cost on choose and
compare. A mechanistic reading, offered as interpretation only: the product
term is a per-dimension image-question agreement signal, which is exactly
what verification-style questions need. The small slices (compare n=302,
global n=163) carry the seed-42 caveat most strongly.

(b) Redundancy corroboration. Across the 14 type slices, the per-type gains
over concat of product_576k and difference_576k correlate at Pearson r =
0.578 (p = 0.031) and Spearman rho = 0.767 (p = 0.001). The rank correlation
is strong and supports the v2_04 one-signal interpretation; the moderate
Pearson value says the profiles agree in shape more than in exact magnitude,
so the corroboration is positive but not perfect.

(c) Language-prior map. Types substantially answerable from text statistics
alone: compare (question_only 0.616 against prior 0.440), logical (0.609
against 0.528) and choose (0.531 against 0.278; choose questions name both
candidate answers in the text). Types that need the image: query
(question_only 0.318, concat 0.476), cat (0.255 against 0.565) and global
(0.534 against 0.804, the largest image benefit). On verify, question_only
(0.523) barely improves on always answering "no" (0.510), so verification
is essentially unanswerable from text alone, which matches it being the type
where fusion helps most. Zero-shot is strongest on cat (0.321), consistent
with CLIP acting as an image classifier over category names.

(d) The limitation test: the expected ceiling did not appear in this
slicing. Every trained multimodal model is above its own overall mean on
relation questions (concat +0.006, product_576k +0.011, fusion_narrow
+0.013, difference_576k +0.015, fusion +0.018) and on >=4-step questions
(+0.027 to +0.047); only the unimodal baselines fall below on rel
(question_only -0.030, image_only -0.015). The step-count slice is
confounded with question format: long programs are dominated by verify and
logical questions with small (mostly binary) answer spaces, so absolute
per-slice accuracy mixes reasoning difficulty with chance level. The honest
conclusion is that this slicing does not make a global-vector ceiling
visible, not that no ceiling exists; a format-controlled comparison (for
example, relational against non-relational within verify questions only)
would be the sharper instrument and is left for later work. What the slices
do show clearly is that open query questions are the weakest slice for
every trained model (concat 0.476, fusion 0.477).

## Decisions and problems

The v2_01 zero-shot per-row predictions were not saved, so they were
recomputed from the cached embeddings; the recomputed overall accuracy
reproduced the stored 0.0795 exactly, confirming the recomputation path.
All seven checkpoint accuracies also matched their stored seed-42 values
exactly, so the per-type slices decompose exactly the numbers already
reported. These are dev-set analyses under the V2 protocol at the 40k scale;
they guide model selection and question-type expectations and are not
confirmatory test results.
