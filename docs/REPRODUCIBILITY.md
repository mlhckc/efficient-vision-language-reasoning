# Reproducibility

The aim is that another person can clone the repository, set up the environment
and obtain the same results, and that every reported number can be traced back
to the code and settings that produced it.

## How a run is fixed

- One configuration file. All settings live in config.py and are read from
  there. A run is described by config.py plus the code; there are no hidden
  command-line flags.
- One seed. config.RANDOM_SEED (42) is the single seed. utils.set_seed() applies
  it to Python, NumPy and PyTorch (CPU and CUDA) and sets PYTHONHASHSEED. Each
  stage's main() calls set_seed() first, before any data loading or model
  creation.
- Deterministic algorithms. With config.DETERMINISTIC = True, set_seed() also
  sets cuDNN to deterministic, disables cuDNN autotuning, sets
  CUBLAS_WORKSPACE_CONFIG and calls torch.use_deterministic_algorithms(True,
  warn_only=True). warn_only keeps a run from crashing if an op lacks a
  deterministic implementation; it warns instead.
- Reproducible data loading. utils.make_generator() seeds DataLoader shuffling
  and utils.seed_worker() reseeds each worker, so loading order is fixed even
  with num_workers > 0.
- Frozen encoders. CLIP is never trained, so the encoder adds no training
  randomness. Embeddings are extracted once in Stage 2 and cached, so every
  later stage reads the same vectors.
- Fixed data subset. Stage 1 selects the GQA subset (config.N_TRAIN,
  config.N_VAL) and the answer vocabulary (config.TOP_K_ANSWERS) under the seed,
  so the splits are the same on every run. The raw GQA version used will be
  recorded in the Stage 1 report.

## Pinned environment

- requirements.txt is the portable, human-readable list of dependencies.
- requirements.lock.txt is the exact set of resolved versions
  (`pip freeze`) captured after a clean install, for reproducing the same
  environment on a compatible Linux machine with a matching CUDA GPU. Some
  entries are platform-specific CUDA wheels, so the lock file targets this class
  of machine rather than every platform.
- Python version and the GPU are recorded in docs/02_environment_setup.md and in
  each run's metadata.

## Recorded with every result

utils.run_metadata() returns the git commit, seed, library versions, device and
GPU name, and the key config values; utils.save_json() writes it next to each
result. A saved number therefore carries the exact code and settings that
produced it.

## Which status wins when records disagree

Two different questions get asked of these records and they are resolved by
different authorities. Settle the lifecycle question first, then the field
question.

### 1. Lifecycle: is a packet still open?

A closed packet's own artefacts state the lifecycle they had on the day they
were written, which is usually before the review returned. For packet status,
`results/closure/pre_f1_status_supersession_20260814.json` is highest
precedence; a status block stored inside a closed packet is second. That
record states the current lifecycle of the five closed packets — E7b
(`E7B_CANONICAL_SUPERSESSION_REVIEW_PASS`), VE-0 (`VE0_PASS`, amendment
`VE0_AMENDMENT_PASS`), VE-1 (`VE1_PASS`), VE-2 (`VE2_PASS`) and E10
(`E10_PASS`) — and supersedes the stale text, including the `status.e7b =
OPEN` and `status.f1 = BLOCKED` fields inside
`results/closure/e7b_evidence_supersession.json` and the `review_requested`
Status section of the byte-pinned `docs/experiments/ve0_evidence_contract.md`.
Those historical statements are readable as the state of each packet when it
was submitted; neither file is edited.

It controls lifecycle and wording only. It changes no accuracy, latency,
interval, parameter count, Pareto membership, figure, table or raw artefact,
reinstates no withdrawn field, and does not declare F1 ready: F1 and F2 remain
unstarted and unauthorised.

### 2. Field validity: is a number still current evidence?

Some artefacts are frozen by a closed review packet and cannot be amended in
place, so a field can be withdrawn after the file carrying it was sealed. When
two records disagree about whether a number is current evidence, resolve them
in this order:

1. `results/closure/e7b_evidence_supersession.json` controls named E7b field
   validity;
2. `results/ve0/supersession_map.json` controls unnamed artefact-group status;
3. historical stored `evidence_status` is lowest precedence.

Historical numbers remain readable only as provenance. A withdrawal is
field-level: it removes the named fields from current evidence and leaves
every other field in the same row standing. Withdrawn values are never
replaced by an estimate, an inferred figure, a back-calculation or a
measurement taken on a different node.

The same rule governs E7a, whose additive `gpu_encoder_plus_head_ms`,
`full_pipeline_ms` and `amortised_ms` columns and the Pareto fronts built on
them are superseded for any end-to-end claim, while its isolated component
latencies, peak-memory components and parameter counts stay valid. Those
superseded columns are not replaced by E7b's measured serial values:
substituting one experiment's number into another's sentence would manufacture
a comparison neither made.

## Verifying the final evidence

`tests/run_all.py` is listed in E10's frozen source set, so a new suite cannot
be imported into it without moving the sealed digest and making the E10 phase
verifiers refuse the state. Each final-evidence layer therefore ships its own
CPU-only runner. New files under `tests/` are safe, because the frozen set
names individual files, and `run_all.py`'s embargo scan globs `tests/*.py`, so
the new sources are still covered by it.

    python -B tests/run_closure.py            # statistical and efficiency closure
    python -B tests/run_ve0.py                # canonical evidence contract
    python -B tests/run_ve1.py                # figures and tables
    python -B tests/run_ve2.py                # qualitative evidence
    python -B tests/run_e7b_supersession.py   # E7b withdrawal and lifecycle records

`tests/run_all.py` itself imports `tests/test_reproduction.py`, which performs
a CUDA forward pass, so it is not part of a zero-GPU verification pass.

## Clean-test governance

The clean-test contents were never inspected or used for development, model
selection, or reporting decisions. Mechanical byte access occurred in two
documented governance incidents, on 13 and 14 August 2026:

1. 13 August 2026 — an independent reviewer ran an integrity-hash command over
   the target file, which read its bytes.
2. 14 August 2026 — during the E7b canonical-supersession implementation, an
   overly broad dependency-mapping scan walked the whole project root and read
   every `.csv` file, because the traversal was not scoped away from `data/`.

Both are classification B, mechanical byte access and nothing more. In neither
case were contents inspected: no row, label, distribution or prediction was
examined, and no information from the target entered development, model
selection or any reporting decision. These are governance incidents, not
test-informed scientific selection, and they neither invalidate any result nor
require remediation of the blinded evaluation. They are recorded because the
project record must not overstate the embargo: the claim that the clean test
was never accessed would be false.

The corrective rule, recorded so the second incident cannot recur: a
repository-wide scan must exclude `data/` before reading or traversing
candidate files, not filter it afterward. Scoping the traversal is the
control; filtering the results is not, because by then the bytes have already
been read.

The canonical machine-readable record, with both incidents identified
separately, is `results/closure/e7b_evidence_supersession.json` under
`clean_test_governance`. Some frozen artefacts — `results/ve2/VE2_MANIFEST.json`
and the `experiments/ve2/run_ve2.py` string it is rebuilt from — still carry the
wording written when only the first incident was known. Those are historical,
superseded residuals inside a closed packet; this section is what governs.

## Caveats

Fixing seeds and deterministic flags makes a run repeatable on the same
hardware and software. Exact floating-point results can still differ across GPU
architectures, driver or CUDA versions, so the environment is recorded with each
result to make any difference explainable. No result is estimated; every number
comes from a real run.
