# E8A SmolLM2-135M core: what is in git and what is not

This directory is the only part of `results/` that git tracks. It holds the
E8A dissertation core: arms A0p, A1 and A1r at train_40k and train_250k with
seeds 0, 1 and 2, eighteen cells in total. The written report is
`docs/experiments/e8a_core_135m.md`.

## Committed

Thirty-one JSON records, about 673 KiB in total:

- `run_<arm>_<scale>_seed<n>.json` — the fifteen matrix cells, each with its
  training history, its evaluation under all four intervention conditions, its
  gates and its recorded hashes.
- `pilot.json`, `pilot_A0p.json` — the three reused seed-0 pilots at
  train_40k, A1 and A1r from Phase 1A and A0p from Phase 1B.
- `core_analysis.json`, `core_analysis_train_40k.json` — the per-scale tables,
  the three contrasts with their image-clustered intervals and directional
  outcomes, the reliance and step-deficit analyses, and the cross-scale
  comparison.
- `efficiency_resources.json` — the bounded accuracy-efficiency table, the
  Pareto fronts and the resource audit against the operative gates.
- `e8a_135m_core_frozen.json` — the frozen result object. It names every
  artefact with the SHA-256 measured at freeze time and is the authority for
  the tranche.
- `execution_manifest.json` — the pre-registered matrix and recipe.
- `ARTEFACT_MANIFEST.json` — every artefact git does not carry, with size,
  SHA-256, type, cell, whether it is replaceable and where it exists.
- The supporting gate, extraction and projection records.

Together these are enough to check every number in the report, because each
accuracy, contrast, interval and deficit is recorded alongside the hash of the
correctness vectors it came from.

## Not committed

| artefact | files | size | where it is |
| --- | --- | --- | --- |
| checkpoints, `checkpoints/*.pt` | 18 | 1.44 GiB | node-local and backed up |
| correctness vectors, `correctness_*.npz` | 18 | 768 KiB | node-local and backed up |
| frozen question stores, `data/v3_slm_tokens/*.h5` | 4 | 5.32 GiB | node-local only |

Normal git is not used for these. Git stores every version of every blob
forever, so a 1.4 GiB set of binaries would be carried by every clone of the
repository from then on, and could not be removed later without rewriting
history. The binaries also change identity whenever a model is retrained,
so committing them would grow the repository without making the numbers any
more checkable than the recorded hashes already do. Git LFS was not
introduced, because that would add a dependency and a quota outside the
approved scope.

Each omitted artefact is instead named in `ARTEFACT_MANIFEST.json` with its
SHA-256, so a copy obtained from the backup can be proved to be the file the
results were computed from.

## Where the originals live

The originals were produced on `/scratch`, which is node-local: it is an ext4
volume on the compute node itself, not a shared filesystem, and CLAUDE.md
records that the contents exist only on the node where they were created. If
that node is lost, so is anything held only there.

## Verified backup

`results/experiments/e8a_question_encoder/` is copied in full to:

```
/user/HS400/mc02623/backups/efficient-vision-language-reasoning
    /results/experiments/e8a_question_encoder/
```

That destination is an NFS export of
`isilon01-az3.surrey.ac.uk:/ifs/isilon01/az3/Personal/HS400`, on a different
device and a different filesystem type from the source, so it survives loss of
the compute node.

The copy was made with `rsync -a --partial` and no `--delete`, into a path
that did not previously exist, so nothing was removed from either side and no
earlier backup was overwritten. Afterwards every file was re-hashed
independently at the destination and compared with the source: **67 of 67
artefact files, 1,541,946,524 bytes, every path, size and SHA-256 identical**,
the two full listings sharing the digest
`e14c5c45f2253e5ef57e938c5d934d1c4628e66ea21a6c60df0fcf79b41a9488`.

This README and `ARTEFACT_MANIFEST.json` were written after that copy. The
backup was re-synchronised and re-verified over the whole directory
afterwards, so it now holds 69 files; both documents are tracked in git as
well.

`data/v3_slm_tokens/` is **not** backed up. It is 5.32 GiB of frozen question
states that the project documentation does not record as irreplaceable, and
extraction is measured deterministic, so it can be rebuilt from the pinned
SmolLM2-135M revision for about 1.45 GPU-hours. Backing it up would also have
consumed most of the remaining quota on the persistent share.

## What full verification and reproduction need

- **Reading the results**: the committed JSON records alone.
- **Recomputing the statistics** — accuracies, contrasts, intervals,
  reliance, step deficits — needs the correctness vectors as well, since those
  carry the per-row correctness the bootstrap resamples. Restore them from the
  backup and run `experiments/e8a_question_encoder/analyse_core.py`.
- **Reproducing a correctness vector from its model** needs that cell's
  checkpoint, from the backup, plus the image and question token stores.
- **Retraining from scratch** needs the token stores as well, and costs about
  4.34 GPU-hours of training plus 1.45 of extraction. It would not reproduce
  the checkpoints bit for bit: training runs under bf16 autocast with
  non-deterministic fused attention backward kernels, so a rerun from the same
  seed yields different files and different hashes. That is why the
  checkpoints are marked irreplaceable rather than reproducible.

Every number here is a development-set result. The clean test was not read.

Terminology correction (5 August 2026): the `correctness_*.npz` files were
called "prediction vectors" above and in earlier records. They store
questionIds, labels and per-condition correctness Booleans only — no
predicted label or answer string — so they are correctness vectors, and the
wording above now says so. True per-row prediction artefacts, carrying
predicted label IDs, answer strings and both the raw and the pinned
normalised exact-match metrics, were introduced by the G21 remediation under
`predictions_g21_v1/`. The key name `prediction_vectors` inside the frozen
`e8a_135m_core_frozen.json` is historical evidence and is left unchanged;
the G21 freeze uses the corrected name.
