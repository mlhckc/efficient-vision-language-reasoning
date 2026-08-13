# Closure: statistical and efficiency evidence

## Purpose

This closure produces no new scientific result. It exists so that the Visual
Evidence layer, the model-list freeze (F1) and the one-shot clean-test
evaluation (F2) rest on a frozen, hash-identified and correctly quantified
base.

The controlling question is narrow: for every result that will plausibly
enter the dissertation, is the point estimate reproducible from a frozen
artefact, is its uncertainty model correct for the dependence structure of
the data, and is the human-readable claim attached to it inside the boundary
the evidence supports?

The planning audit found the statistical gap confined to families that never
stored row-level correctness arrays. V2, E2 and E3 recorded aggregates,
per-seed accuracies and some slice tables, but not the per-question
correctness vectors those aggregates were computed from. Without those
vectors an image-clustered interval cannot be computed at all, so several
claim-bearing contrasts carried only a mean and an across-seed standard
deviation. Development rows share images — 7,714 questions over 768
represented images — so a row-level interval on them would assume an
independence the data does not have.

E10 is closed at E10_PASS and was not reopened. Nothing here trains,
fine-tunes, selects a checkpoint, chooses an epoch or touches the clean test.

## Method

All work is CPU-only and evaluation-only, under the user's authorisation of
13 August 2026 for deterministic forward passes of already-frozen checkpoints
over already-frozen cached embeddings, solely to reconstruct missing
row-level evidence and dependence-aware uncertainty for existing results.

Pre-execution integrity check. HEAD 8cce6a6, branch v2-protocol, clean
worktree. All four existing hash manifests were re-verified entry by entry:
263 results-tree entries from `artifacts/results_export/MANIFEST.json`, 58
from E8A, 59 from E9 and 80 from E10, total 460, with 0 mismatches and 0
missing files. Every checkpoint directory and cached embedding store the
reconstruction needed was confirmed present.

Reconstruction. `experiments/closure/reconstruct_rows.py` reloads each frozen
checkpoint on CPU with the model constructor imported from the experiment
that trained it, and evaluates it over the cached frozen development
embeddings. No optimizer is constructed anywhere in this closure; every
checkpoint used is the one the frozen record already designates for that
cell.

The point-estimate reproduction gate is the point of that script. Before any
row-level array is written, the recomputed aggregate must equal the stored
canonical value exactly at the canonical reported precision of five decimal
places. A failing cell stops its whole family: that family writes no array,
contributes no interval and supports no claim in these outputs. Nothing is
adjusted, re-rounded, re-fitted, re-seeded or replaced, and no historical
value is touched. Every cell is nevertheless evaluated, even after a failure,
so a blocker is reported at its true extent rather than at whatever the
iteration order happened to reach first.

Statistics. `experiments/closure/build_intervals.py` computes image-clustered
percentile bootstrap intervals with the project's single existing procedure:
cluster unit "represented development imageId", 2,000 draws,
`numpy.random.default_rng(0)`, percentiles 2.5 and 97.5, multi-seed rows
combined by the within-draw mean. This is byte-for-byte the v2_05c method
string and the E8A/E8B/E10 procedure. It was validated before use by
reproducing all 42 stored v2_05c interval and count values exactly.

The pooled four-or-more-step deficit is a combination of bucket means rather
than a row mean, so it uses a custom-statistic form of the same bootstrap:
the same clusters, generator, draw count and percentiles, with the statistic
recomputed from the drawn rows on every draw.

Families that already carry a correct image-clustered interval — E8A, E8B,
E10, v2_05c, v3_02a, v3_03 — are READ, never recomputed. This closure
deliberately produces no competing re-analysis of a settled question.

Commands, all from the project root with the venv active:

    python -m experiments.closure.run_closure
    python -B tests/run_closure.py
    python -B tests/run_all.py

## Outputs

Under `results/closure/`, with the small JSON and CSV records tracked and the
reconstructed per-row binaries local and manifested:

| output | file | content |
|---|---|---|
| A | `A_evidence_registry.json` / `.csv` | 18 canonical artefacts: hash, family, arms, scale, seeds, split, metric, row-level availability, uncertainty model actually carried |
| B | `B_primary_contrasts.json` / `.csv` | 70 claim-bearing contrasts in one schema |
| C | `C_reconstructed_evidence_manifest.json` | every reconstructed condition, its gate result and its array hash |
| D | `D_slice_statistics.json` / `.csv` | 195 slice rows with n_questions, n_unique_images, interval and cluster unit |
| E | `E_seed_variability.json` / `.csv` | 66 rows of across-training-seed spread |
| F | `F_multiplicity_status.json` | interval counts and the uncorrected-multiplicity disclosure |
| G | `G_claim_evidence.json` / `.md` | 17 claims, each with supporting artefacts, figures and a non-empty limitation |
| H | `H_evidence_readiness.json` / `.csv` | readiness status per claim plus the optional items |
| — | `efficiency_closure_table.json` / `.csv` | 36 efficiency rows with explicit timing class |
| — | `manifest_e7b_serial_efficiency.json` | 38 entries |
| — | `manifest_e8b_readout_generation.json` | 201 entries |
| — | `row_evidence/` | 205 reconstructed per-row `.npz` arrays |

Code is under `experiments/closure/`; tests are `tests/test_closure.py` with
the runner `tests/run_closure.py`.

## Results

Reconstruction and the gate. 245 conditions were evaluated across seven
families. 205 reproduced their stored point estimate exactly and were
written; 40 belong to one stopped family. Six families reproduced completely,
with a maximum absolute difference below 5e-06 in every cell, which is pure
five-decimal rounding:

| family | cells | reproduced | max absolute difference |
|---|---|---|---|
| v2_02 | 20 | 20 | 4.85e-06 |
| v2_03 | 10 | 10 | 4.95e-06 |
| v2_04 | 20 | 20 | 4.92e-06 |
| v2_06 | 75 | 75 | 4.92e-06 |
| E2 | 40 | 40 | 4.79e-06 |
| E3 | 40 | 40 | 4.95e-06 |
| v2_07 | 40 | 39 | 1.33e-04 |

The one failure, and the family it stopped. `v2_07 question_only train_250k
seed 2` recomputes to 0.49754 against a stored 0.49767, a difference of
0.00013305, which is exactly 1.03 rows of 7,714. The stored value is unique
in that run's history at epoch 19, so there is no tie between epochs. A
recorded diagnostic shows why: the cell has exactly one development row whose
top-1 and top-2 logits differ by 4.77e-07, at float32 epsilon scale, and the
gold label is the runner-up. A different linear-algebra backend resolves that
argmax the other way, which accounts for exactly one row and exactly this
difference.

That diagnosis is recorded, and it does not override the gate. The difference
is more than twenty times the 5e-06 deterministic-scorer tolerance E2 and E3
established, so v2_07 is stopped: it contributes no row-level array, no
interval and no claim here. Keeping the 39 passing cells after seeing which
one failed would be outcome-conditioned selection, so the whole family is
stopped and the decision is left to the reviewer.

Two consequences follow and are recorded rather than worked around. The
v2_07 scaling contrasts have no clustered interval, so claim C02, that the
fusion advantage narrows with scale, keeps its 40k interval and has none at
250k. The E2 SigLIP-minus-CLIP contrast at train_250k cannot be formed at
all, because its CLIP partner is v2_07; the 40k version is unaffected.

Cross-checks. Beyond the per-cell gate, the reconstructed contrasts were
compared against the contrast values the source experiments published: all 29
comparable contrasts reproduce exactly, seed by seed. The maximum difference
in the aggregate mean is 1e-05, which arises because the stored tables
subtract two accuracies already rounded to five decimals and then round
again, while the reconstruction rounds once from the per-row differences.
Both forms are carried so the comparison is exact rather than approximate.
Separately, v2_06's normal condition re-evaluates the v2_02 and v2_04
checkpoints, and all 25 of its normal-condition arrays are byte-identical to
those families' arrays for the same checkpoint.

Priority contrast A, capacity (v2_03). Capacity explains part of the fusion
gain but not all of it, and the two matched contrasts disagree on whether the
remainder is distinguishable from zero:

| contrast | effect | 95% image-clustered | excludes zero | seed sd |
|---|---|---|---|---|
| `v2_02/fusion_minus_concat/train_40k` | +0.01442 | [+0.00794, +0.02095] | yes | 0.00141 |
| `v2_03/concat_wide_minus_concat/train_40k` | +0.00863 | [+0.00376, +0.01378] | yes | 0.00302 |
| `v2_03/fusion_narrow_minus_concat/train_40k` | +0.00944 | [+0.00285, +0.01592] | yes | 0.00175 |
| `v2_03/fusion_minus_fusion_narrow/train_40k` | +0.00498 | [+0.00025, +0.00967] | yes | 0.00170 |
| `v2_03/fusion_minus_concat_wide/train_40k` | +0.00578 | [-0.00116, +0.01226] | no | 0.00298 |

The unmatched v2_02 gain of +0.01442 is capacity-confounded. At the 576k
budget the fusion features still help (+0.00944, excluding zero); at the 1.1M
budget the interval includes zero. This is the first interval of any kind on
these contrasts.

Priority contrast B, visual reliance (v2_06). Every multimodal head loses
accuracy when the image is replaced or removed, and the question-only control
is provably unaffected:

| contrast | effect | 95% image-clustered | excludes zero |
|---|---|---|---|
| `v2_06/fusion/drop_shuffled/train_40k` | +0.14270 | [+0.12769, +0.15759] | yes |
| `v2_06/product_576k/drop_shuffled/train_40k` | +0.12816 | [+0.11355, +0.14209] | yes |
| `v2_06/concat/drop_shuffled/train_40k` | +0.11709 | [+0.10295, +0.13035] | yes |
| `v2_06/fusion/drop_zeroed/train_40k` | +0.11880 | [+0.10403, +0.13305] | yes |
| `v2_06/concat/drop_zeroed/train_40k` | +0.09987 | [+0.08697, +0.11228] | yes |
| `v2_06/image_only/drop_zeroed/train_40k` | +0.00850 | [-0.00093, +0.01774] | no |
| `v2_06/question_only/drop_shuffled/train_40k` | 0.00000 | [0.0, 0.0] | no |

Fusion relies on the image most. The question-only drop is exactly zero by
construction and is a control, not a result. A drop under a wrong image shows
the answer depends on the correct image; it is not evidence of robust visual
reasoning.

Priority contrast C, representation (E2). At train_40k, with both sides
evaluated on identical development rows:

| contrast | effect | 95% image-clustered | excludes zero |
|---|---|---|---|
| `e2/siglip_minus_clip_concat/train_40k` | +0.01540 | [+0.00592, +0.02481] | yes |
| `e2/siglip_minus_clip_fusion/train_40k` | +0.01475 | [+0.00601, +0.02401] | yes |
| `e2/siglip_minus_clip_product/train_40k` | +0.01356 | [+0.00403, +0.02282] | yes |
| `e2/siglip_minus_clip_question_only/train_40k` | +0.00814 | [-0.00006, +0.01586] | no |

The three multimodal gains exclude zero; the question-only gain does not,
which is consistent with the encoder swap acting mainly through the image
path. The interval is a valid evaluation-sampling interval because the rows
are identical on both sides, but the training streams are not paired: the
768-d and 512-d inputs make identical initialisation and shuffle streams
impossible, so the seed label does not pair two runs. Encoder, embedding
width and head input width all move together.

Pooled four-or-more-step deficit. Twelve deficits now carry intervals, and
all twelve exclude zero, across both frozen encoders and both scales: CLIP at
40k from +0.08090 (question_only) to +0.10502 (concat), SigLIP at 40k from
+0.08380 to +0.10241, and SigLIP at 250k from +0.06284 to +0.08380. This is
the first interval on the SigLIP arm of that claim. The deficit describes one
model's step profile against fixed v2_05b priors; it is not a contrast
between models and establishes no cause.

Efficiency. 36 rows across five timing classes: 19 END_TO_END_SERIAL, 7
CACHED_IMAGE_QUESTION_SIDE, 8 CACHED_FEATURE_HEAD_ONLY, 1
ADDITIVE_COMPONENT_SUM_SUPERSEDED and 1 COMPONENT. Every non-end-to-end row
carries `comparable_with_end_to_end: false`, and the builder refuses to emit
a frontier that mixes timing classes or nodes. Two frontiers are kept and are
never merged: E7b on otter155 with 7 members and E9 on otter159 with 11. E9's
fusion bridge control missed its pre-registered 10 per cent tolerance at
-18.69 per cent, so no adjustment factor is applied. E10's B4 and B4r have no
timing evidence of any kind; that is recorded as OPTIONAL_EVALUATION_ONLY,
not estimated, and is not a blocker.

Seed variability. 66 rows. Two are flagged for high dispersion:
`E8B/B3/train_40k`, whose three seeds are 0.53124, 0.47783 and 0.51880 with a
sample sd of 0.02795, and `v2_06/image_only/shuffled/train_40k`. A flagged
row must be shown with its seed spread wherever it is reported.

Readiness. Of 21 rows in Output H: 14 READY, 1 READY_AFTER_CPU_STATS, 3
OPTIONAL_EVALUATION_ONLY, 2 NOT_SUPPORTED (the two prohibitions) and 1
SUPERSEDED. The single READY_AFTER_CPU_STATS entry is C02, and it remains
only because the recomputation was attempted and stopped by the recorded
v2_07 mismatch, not because it was skipped.

Validation. `tests/run_all.py` exits 0 with all 17 modules passing, including
the E10 phase verifiers. `tests/run_closure.py` exits 0 with 4,190 closure
checks. All three E10 verifiers exit 0. A full re-run of the closure produces
all 224 artefacts byte-identical.

## Decisions and problems

The reproduction gate fired and the family was stopped rather than repaired.
This is the main event of the closure. The diagnosis strongly indicates a
backend tie-break on a single near-tied row rather than a defect in the
stored record, and it would have been easy to widen the tolerance or scope
the stop to the one failing arm. Both would have been outcome-conditioned
decisions made after seeing which cell failed, so neither was taken. The
mismatch, the full 40-cell diagnostic table and the tie-margin evidence are
recorded for the reviewer, who may with fresh user authorisation decide to
scope the stop more narrowly.

`tests/run_all.py` was left untouched, and the closure ships its own runner.
Wiring `test_closure` into `run_all.py` was tried first and failed correctly:
`run_all.py` is inside E10's `SOURCE_PATHS`, so the edit moved the live
source digest from `e666f101...` to `71cbaaa4...` and the three E10 phase
verifiers refused to consume what they saw as stale output. The seal was
doing its job on a closed experiment, so the edit was reverted and
`tests/run_closure.py` added instead. Adding new files under `tests/` is
safe because `SOURCE_PATHS` names individual files rather than a glob, and
`run_all.py`'s embargo source scan globs `tests/*.py`, so the closure test
sources remain covered by it.

Deterministic writers were needed for idempotence to mean anything.
`numpy.savez` stamps each member with the wall-clock time, so two runs of the
same builder would differ in bytes while being identical in content. The
closure writes npz members with fixed timestamps and sorted names, and JSON
with sorted keys, so a second run is byte-identical and the idempotence claim
is checkable rather than rhetorical.

The efficiency frontier needed node-qualified keys. The first version
addressed frontier members by bare `system_id`, and the test caught that E9
re-measured its own `fusion` and `vocab1000_product` references on otter159,
so the E7b frontier lookup silently spanned both nodes. Every efficiency row
now carries a family-qualified `row_key` and frontiers are expressed in those
keys. This was exactly the cross-node merge the table exists to prevent, and
it was found by a test rather than by reading.

Rounding order was reconciled rather than hidden. The source experiments
formed a gap by subtracting two accuracies already rounded to five decimals
and rounding again; the reconstruction computes the gap once from per-row
differences. The two can differ by 1e-05. Both forms are carried, the
comparison against the stored tables uses the source's own convention, and
the difference is documented rather than absorbed.

Two prohibitions are recorded as claims rather than left implicit. No energy
or power measurement exists anywhere in this project, so no
energy-efficiency claim is permitted; the word appears in the closure outputs
only inside that prohibition, and a test enforces it. Equally, an interval
containing zero is an absence of a detected effect and never establishes
equivalence, which is the binding reading of both E10 pretraining-effect
intervals.

The `git_dirty` provenance limitation is stated, not repaired. Five canonical
analyses recorded `git_dirty: true` because the analysis ran before its own
commit. The numbers are not in doubt, but the commit recorded inside each
artefact does not by itself reproduce those exact bytes on checkout. The
closure binds their exact bytes now by SHA-256 at a clean HEAD, which is what
can honestly be claimed. No historical metadata was rewritten or backdated.

Optional measurements were identified and left unrun. E10 B4/B4r serial
timing, a single-node re-measurement to merge the two frontiers, and E8B R1
serial timing under a batch-1 scorer are all recorded as
OPTIONAL_EVALUATION_ONLY with their protocol, cost and the claim each would
support. None is a blocker and none is recommended now; the dissertation can
rest project-wide efficiency on E7b and E9 and use E10 for capacity and
pretraining science.

The clean-test embargo is intact. `data/v2/test_clean_targets.csv` was never
opened. The only place any closure module names it is a negative assertion in
`closure_common.py` that refuses such a path, matching the pattern the rest
of the repository already uses, and a test exercises that refusal as a
known-negative.
