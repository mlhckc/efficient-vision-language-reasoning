# Experiment v2_00: the V2 evaluation protocol

This document defines the V2 data protocol, records how it was built and
verified, and states its guarantees and limits. Day 1 produced manifests only:
no embeddings were extracted, nothing was trained, and no test prediction was
made.

## Why the legacy V1 validation set is disqualified

data/val.csv, called legacy_v1_validation from here on, cannot serve as an
evaluation set for V2 experiments, for three reasons.

1. Checkpoint selection. Stages 3 and 4 selected each model's best checkpoint
   by its accuracy on this set, so reported accuracies on it are optimistically
   biased model selections, not clean estimates.
2. Influence on research decisions. Its numbers were read repeatedly and shaped
   decisions about models and reporting, so it has leaked into the development
   loop.
3. Vocabulary-conditioned sampling. Its 8,000 rows were sampled after dropping
   out-of-vocabulary answers under the V1 vocabulary, so its composition is
   conditioned on a vocabulary choice rather than reflecting the raw question
   distribution.

legacy_v1_validation remains untouched on disk and is still used by the V1
stage scripts; V2 code reads it only to exclude its images from the clean test.

## The V2 partition

All partitioning operates on the raw balanced GQA v1.2 question files.

- Development set. The unique train_balanced image IDs (72,140) are sorted
  lexicographically and permuted with the DEV_IMAGE_SEED stream. Complete
  images are assigned in permuted order until the accumulated raw question
  count first reaches 10,000. Result: 777 dev images carrying 10,004 raw
  questions. The remaining 71,363 images are the training pool (932,996 raw
  questions). Partitioning by complete images means no image is shared between
  dev and the pool.
- V2 vocabulary. The top 100 answers by frequency over the raw training-pool
  questions only, ties broken alphabetically (the same rule as V1). The dev
  images contribute nothing to the vocabulary. Empirically the V2 vocabulary
  contains exactly the same 100 answers as the V1 vocabulary (symmetric
  difference empty), which is expected, since removing 777 of 72,140 images
  perturbs the frequency ranking only slightly.
- dev.csv. The dev-image questions with in-vocabulary answers: 7,714 of the
  10,004 raw dev questions (coverage 0.771092), with integer V2 labels, sorted
  by questionId. Nine dev images have no in-vocabulary question, so dev.csv
  spans 768 of the 777 dev images; the membership artefact records all 777.
- Nested training manifests. The eligible pool is the in-vocabulary
  training-pool questions: 724,074 (coverage 0.776074), a margin of 474,074
  over the required 250,000. They are sorted by questionId, the row order is
  permuted once with the TRAIN_QUESTION_SEED stream, and the prefixes of
  40,000, 100,000 and 250,000 rows are written as train_40k.csv,
  train_100k.csv and train_250k.csv in permuted-prefix row order, never
  re-sorted. The row order is the nesting proof: train_40k is row-for-row the
  first 40,000 rows of train_100k, which is row-for-row the first 100,000 rows
  of train_250k.
- Clean test. From the 10,234 unique val_balanced images, every image that
  appears in legacy_v1_validation (4,928 images) is excluded. The remaining
  images are sorted, permuted with the TEST_IMAGE_SEED stream, and complete
  images are assigned until the raw question count first reaches 8,000.
  Result: 972 images carrying 8,013 raw questions. The empirical raw
  train/val image overlap is 0, so no clean-test image occurs anywhere in
  training or development data.

## Test blinding

The clean test is split into two files, both sorted by questionId:

- data/v2/test_clean_inputs.csv with columns questionId, imageId, question;
- data/v2/test_clean_targets.csv with columns questionId, answer, label, where
  label is the V2 vocabulary index or -1 for an out-of-vocabulary answer.

Training and development code must never load test_clean_targets.csv before
final evaluation. The same rule is stated in the header comments of the
generation code. The verifier reads the targets file only for row-level
consistency checks and prints no aggregate about it.

## The three RNG streams and their independence

- DEV_IMAGE_SEED = config.RANDOM_SEED = 42
- TRAIN_QUESTION_SEED = config.RANDOM_SEED + 1 = 43
- TEST_IMAGE_SEED = config.RANDOM_SEED + 2 = 44

Each stream is a separate numpy.random.default_rng instance, constructed
independently and used exactly once. Because the streams share no state, a
change to the dev partition code can never alter the clean-test permutation,
and vice versa. No pandas DataFrame.sample is used anywhere; all randomness is
a numpy permutation applied after canonical sorting.

## Canonical rules

- questionId and imageId are strings everywhere; every pandas read of any
  manifest uses dtype={"questionId": str, "imageId": str}.
- All ID sorting is lexicographic string sort.
- All CSVs are UTF-8 with "\n" line endings and no index column.
- Image paths resolve exactly as the V1 extractor does:
  config.GQA_IMAGES_DIR / f"{image_id}.jpg".
- The threshold constants (10,000 dev questions, 8,000 test questions, the
  40k/100k/250k prefix sizes) live in the experiment scripts rather than
  config.py, because config.py is outside this run's output allowlist; the
  seeds derive from config.RANDOM_SEED.

## Embargo

The following rule is in force, quoted verbatim from the protocol
specification:

> EMBARGO (hard rule): no output of either script may state clean-test
> vocabulary coverage, OOV counts, answer distribution, yes/no share, or
> question-type statistics. Clean-test reporting is structural only: question
> count, unique image count, overlaps, duplicates, hashes, image availability.

Accordingly, this document and every artefact of this run state only that the
clean test has 8,013 questions on 972 unique images.

## Out-of-vocabulary rules reserved for final evaluation

At final evaluation, and only then, the following will be computed on the
clean test: vocabulary coverage; overall accuracy with out-of-vocabulary
questions counted as wrong; and in-vocabulary accuracy. None of these numbers
has been computed now, and the -1 labels in the targets file have not been
aggregated.

## Caveats stated up front

- Question-level scaling. The nested manifests scale the number of training
  questions. Because questions are drawn from a shared pool of images, a larger
  prefix both adds questions on previously unseen images and adds further
  questions on images already seen (empirically 27,622 unique images at 40k,
  46,158 at 100k, 61,859 at 250k). These two effects are not separable in this
  design; scaling conclusions are about question count, not image count.
- Legacy exclusion and the test pool. Excluding the 4,928 legacy-touched
  images removes about half of the val_balanced image pool. If those images
  differ systematically from the rest (for example in questions per image),
  the remaining pool's questions-per-image distribution may shift accordingly.
  The clean test is a random sample of the remaining pool, not of the full
  val_balanced pool.
- What "clean" means. The clean test is clean with respect to model selection
  and vocabulary conditioning: no checkpoint was ever selected on it, and its
  composition does not depend on any vocabulary. It is not claimed to be
  "never influenced by prior observation": it comes from the same public
  val_balanced release that produced legacy_v1_validation, and coarse facts
  about that split (for example its overall size) were known before this
  protocol was designed.

## Verifier independence and its residual limit

verify_protocol.py reloads the raw GQA JSONs with its own inline loader,
recomputes the partitions, the vocabulary, the counts and the hashes with
inline implementations, and compares them to the artefacts. It imports nothing
from build_manifests.py and does not import 1_prepare_gqa.py (whose numeric
filename is not importable in any case; the build script loads it via
importlib to reuse the V1 loading and vocabulary code). The residual limit is
that both scripts implement the same written specification, so a
misunderstanding shared by the author of both would not be caught; the
independence is of implementation, not of specification.

## Preservation scope and the bytecode safeguard

Preservation is checked against the exact output allowlist of this experiment,
not by excluding directories. Before any file was created, a full-repository
inventory (185,483 files) was captured in memory and written to
preservation_before.json; after verification, the inventory was rebuilt
(preservation_after.json) and compared, excluding exactly the allowlisted
output files. Scope: every regular file under the project root excluding only
.git; files under data/gqa/images/ and files larger than 100 MiB are recorded
by size and mtime_ns instead of sha256 (the two raw question JSONs fall under
the size rule in the inventory, but their full sha256 is recorded separately
in manifest_hashes.json). The preservation claim covers exactly this
inventoried scope. Result: no non-allowlisted file was changed, deleted or
created. git status and git diff were captured before and after
(git_before.txt, git_after.txt); the tracked tree was clean both times.

Every Python invocation of this run, including the build, the verifier and the
idempotence rerun, used python -B with PYTHONDONTWRITEBYTECODE=1, so no
__pycache__ directory or .pyc file was created anywhere in the repository; the
preservation check would have failed otherwise.

## Build and verification outcome

The build ran once and wrote all artefacts under data/v2 and
results/experiments/v2_00_protocol. The verifier then ran 99 checks and all
passed (status PASS): set equalities for the three image partitions,
vocabulary recomputation including indices, every summary count and
distribution recomputed from raw data and manifest files, row-level integrity
of every manifest against the raw records, correct labels everywhere, zero
questionId and imageId overlap between every training manifest and dev and the
clean test, zero overlap between dev and the clean test and between
legacy_v1_validation images and clean-test images, exact 40,000/100,000/40,000
nested intersections with row-for-row prefix equality, no duplicate questionId
in any manifest, all referenced image files present on disk for all five
manifests, and all 18 recorded hashes matching on disk.

Duplicate diagnostics (reported, nothing deleted): within-manifest duplicate
(imageId, question) rows are 6 in dev, 51 in train_250k and 5 in the clean
test, of which 5, 47 and 4 respectively also share the answer; these are
repeated question texts on the same image under different questionIds in the
raw GQA release. Across manifests all duplicate counts are zero.

## Idempotence

After build and verification passed, build_manifests.py was run a second time
with python -B on the same data and seeds, and every generated-group file was
re-hashed. IDEMPOTENCE PASS: all 13 generated files are byte-identical across
the two builds, and manifest_hashes.json is byte-identical as well.
build_run_metadata.json is volatile (timestamps, environment) and excluded
from the comparison by design. The result is appended to protocol_report.json.

## Files produced by this run

Exactly the output allowlist:

    experiments/v2_00_protocol/build_manifests.py
    experiments/v2_00_protocol/verify_protocol.py
    data/v2/dev.csv
    data/v2/train_40k.csv
    data/v2/train_100k.csv
    data/v2/train_250k.csv
    data/v2/test_clean_inputs.csv
    data/v2/test_clean_targets.csv
    data/v2/dev_image_ids.json
    data/v2/train_pool_image_ids.json
    data/v2/test_clean_image_ids.json
    data/v2/dev_raw_question_ids.json
    data/v2/test_clean_raw_question_ids.json
    data/v2/answer_vocab_v2.json
    data/v2/protocol_build_summary.json
    data/v2/manifest_hashes.json
    results/experiments/v2_00_protocol/build_run_metadata.json
    results/experiments/v2_00_protocol/protocol_report.json
    results/experiments/v2_00_protocol/preservation_before.json
    results/experiments/v2_00_protocol/preservation_after.json
    results/experiments/v2_00_protocol/git_before.txt
    results/experiments/v2_00_protocol/git_after.txt
    docs/experiments/v2_00_protocol.md

git_after.txt was captured during verification, before this document existed,
so the document does not appear in it; the document is on the allowlist and
covered by the preservation rules.
