# V2 Day 1 completion summary

## Purpose

Day 1 replaced the legacy V1 evaluation data with a defensible V2 protocol:
an image-disjoint development set, nested training manifests of increasing
size, and a blinded clean test set, all built deterministically and verified
independently. Day 1 performed no model training, no embedding extraction and
no clean-test evaluation.

## Legacy V1 limitation

The V1 validation set (data/val.csv, called legacy_v1_validation) was used to
select checkpoints, its numbers influenced development decisions, and its rows
were sampled after out-of-vocabulary filtering under the V1 vocabulary. It is
therefore unusable as a clean evaluation set and is retained only as a legacy
artefact and an image-exclusion list.

## The V2 protocol

- Development split: the 72,140 unique GQA train_balanced image IDs were
  sorted and permuted with an independent seeded stream, and complete images
  were assigned until the raw question count first reached 10,000. Result:
  777 dev images carrying 10,004 raw questions, of which 7,714 are
  in-vocabulary (dev.csv spans 768 images; nine dev images have no
  in-vocabulary question).
- Training pool: the remaining 71,363 images with 932,996 raw questions. The
  V2 vocabulary is the top 100 answers of the raw pool questions only.
  Eligible (in-vocabulary) pool questions: 724,074, a margin of 474,074 over
  the required 250,000.
- Nested subsets: the eligible questions were sorted by questionId and
  permuted once; the 40,000 / 100,000 / 250,000 prefixes were written in
  permuted order as train_40k, train_100k and train_250k. They cover 27,622 /
  46,158 / 61,859 unique images respectively, and each smaller manifest is
  row-for-row the prefix of the larger one.
- Clean test: from the 10,234 unique val_balanced images, the 4,928 images in
  legacy_v1_validation were excluded; the rest were sorted, permuted with a
  third independent stream, and complete images assigned until the raw
  question count first reached 8,000. Result: 8,013 questions on 972 images
  (structural counts only). Inputs and targets are separate files; the targets
  are embargoed and no clean-test vocabulary coverage, OOV count, answer
  distribution, yes/no share or question-type statistic has been computed.

## Independent verification

verify_protocol.py reloads the raw GQA JSONs with its own inline loader and
recomputes the partitions, vocabulary, counts, hashes and manifest contents
without importing any build code. Result: 99 checks, 0 failed. Missing image
files: 0 across all five manifests. Duplicate questionId rows: 0 in every
manifest. Train/dev/test disjointness: all questionId and imageId overlaps are
0, including legacy_v1_validation images against the clean test. Nested
40k/100k/250k intersections are exactly 40,000 / 100,000 / 40,000 with
row-for-row prefix equality.

## Preservation and idempotence

- Preservation: a full-repository inventory before and after (185,483 files
  before) shows no file outside the declared output allowlist was created,
  changed or deleted; V1 data, checkpoints and historical results are
  byte-identical.
- Idempotence: rebuilding with the same data and seeds reproduced all 13
  generated files byte-identically (IDEMPOTENCE PASS), so the recorded sha256
  hashes pin the protocol exactly.

## V1/V2 vocabulary comparison

Both vocabularies contain the same 100 answers. The answer-to-index mappings
differ for 11 answers (adjacent frequency-rank swaps caused by removing the
777 dev images from the counting pool); see
v1_v2_vocab_index_comparison.json. Consequently V1 label indices must never be
mixed with V2 manifests; all V2 work uses answer_vocab_v2.json.

## Status and next stage

Day 1 is complete and verified. Day 1 performed no model training, no
embedding extraction and no clean-test evaluation. The next permitted stage is
V2 global CLIP embedding extraction for the new manifests, followed by
integrity verification of the extracted embeddings; model training does not
start before both are complete.
