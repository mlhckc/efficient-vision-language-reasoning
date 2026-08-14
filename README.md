# Efficient Vision-Language Reasoning with Small Language Models

MSc Artificial Intelligence dissertation, University of Surrey. Supervisor:
Prof. Miroslaw Bober.

The project tests whether Visual Question Answering can be done efficiently by
reasoning in embedding space. An image and a question are each encoded into a
fixed vector by one frozen CLIP ViT-B-32 model, and a small trainable head
classifies the answer from a fixed set of the most frequent answers, on a
subset of GQA. No large vision-language model is trained, and the encoders are
never unfrozen.

Status, current to 14 August 2026. V1 is a completed legacy prototype. V2 is
complete through global-head scaling at 40k/100k/250k, and V3 through the
latent-query reasoner and its 100k/250k scaling (E1): the reasoner overtakes
every global head at the larger scales but did not materially improve on the
much smaller fusion head at 40k, and it does not reduce the multi-step
deficit at any scale. E2 (a frozen SigLIP-B/16 encoder swap on the
global-embedding path) gains about +1.2 to +1.5 points over the CLIP
counterparts in every seed while the multi-step deficit stays in the same
range, so the compositional deficit persists across two frozen encoders. E3
(the 1000-answer vocabulary) raises coverage of the raw development
distribution from 77.1% to 98.2% and answers about 3.6 to 4.1 points more
questions correctly on that full distribution, at a roughly 2-point cost on
the shared rows.

Since then the programme has moved to frozen small language models and one
compact VLM. E8A puts a frozen SmolLM2-135M on the question side; E8B and
E10 put frozen SmolLM2-135M and SmolLM2-360M on the answer side; E9 places
two frozen SmolVLM checkpoints in context, evaluation only. The answer-side
result is consistent at both sizes: no reliable positive
pretrained-over-random advantage was detected, and the intervals that
include zero establish neither equivalence nor the absence of an effect.
Training-set size is what moves accuracy. E10 completed its frozen twelve-cell
matrix on 13 August 2026 and is closed.

On efficiency, E7b is the authoritative end-to-end evidence: it measures a
warm serial batch-1 query from raw image and raw question to answer, and it
supersedes E7a's additive `full_pipeline_ms` and `amortised_ms` fields and
their Pareto fronts for any end-to-end claim. E7a's component measurements
remain valid as components. On E7b's node the small global heads sit at about
7.6 ms per query against 9.2-20.1 ms for the larger systems, and cached or
head-only figures (0.02-0.05 ms for the global heads) are partial-pipeline
measurements that are never end-to-end costs. E7b (otter155) and E9 (otter159) were measured on
different nodes; E9's bridge control missed its pre-registered 10 per cent
tolerance by -18.7 per cent, so the two sets are kept as two frontiers, are
never merged, and no adjustment factor is applied. The E9 comparison against
the compact VLMs is contextual positioning, not a fair-protocol superiority
claim. No energy or power measurement exists, so no claim of energy
efficiency is made anywhere.

Three E7b fields are withdrawn. `peak_allocated_mib` and `peak_reserved_mib`
are INVALID: the historical measurement window did not isolate the intended
batch-1 serial inference envelope. `cold_first_query_ms` is SUPERSEDED,
because the accepted source repair standardised the cold query to
`torch.no_grad()` and the stored values measure the older path. Warm serial
latency, the accuracy column, the parameter counts and the Pareto frontier
are unaffected, no replacement value exists and none was estimated. The
canonical record for field validity is
`results/closure/e7b_evidence_supersession.json`, and it takes precedence over
the row-level `evidence_status` stored in the older closure and E7b artefacts.

E7b is CLOSED: the independent review of that withdrawal returned
`E7B_CANONICAL_SUPERSESSION_REVIEW_PASS` on 14 August 2026. Five review
packets are now closed — E7b, VE-0 (`VE0_PASS`, amendment
`VE0_AMENDMENT_PASS`), VE-1 (`VE1_PASS`), VE-2 (`VE2_PASS`) and E10
(`E10_PASS`). Their lifecycle is stated in one place,
`results/closure/pre_f1_status_supersession_20260814.json`, which supersedes
the status text written before those reviews returned, including the `OPEN`
status block inside the byte-pinned E7b record. That record controls lifecycle
and status only; field validity is still resolved by the two records named
above. The model-list freeze (F1) is unstarted and unauthorised, and the
blinded clean-test evaluation (F2) is unstarted and unauthorised.

E7a's additive `full_pipeline_ms` and `amortised_ms` columns are superseded
and are not current end-to-end evidence; they are not replaced by E7b values,
because the two experiments measured different quantities. On the valid
fields, the top-1000 product head at 250k is more accurate (0.4904 against
0.4594 raw-distribution) than the 21.1M-parameter reasoner, using 16 times
fewer parameters; for measured end-to-end latency see E7b.

The statistical and efficiency evidence closure (13 August 2026) re-verified
every hash-manifested artefact, reconstructed the row-level correctness
evidence the V2/E2/E3 families never stored, and attached image-clustered
intervals to the contrasts that carry a claim; its outputs are under
`results/closure/` and its report is
`docs/experiments/closure_statistical_efficiency.md`. The model-list freeze
and the blinded clean-test evaluation are the remaining steps. Supervisor
design feedback will be recorded when available. The clean-test embargo
remains unchanged: the clean test remains blinded and no confirmatory result
has been reported. Every result above is a development-set result.

The clean-test contents were never inspected or used for development, model
selection, or reporting decisions. Mechanical byte access occurred in two
documented governance incidents, on 13 and 14 August 2026: a reviewer's
integrity-hash command over the target file, and an overly broad
dependency-mapping scan that read every `.csv` under the project root. Neither
inspected any row, label, distribution or prediction, and neither informed
development, model selection or reporting. They are governance incidents, not
test-informed scientific selection. The disclosure is in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) and the canonical record is
`results/closure/e7b_evidence_supersession.json`.

Each closed evidence layer ships its own CPU-only test runner rather than
joining `tests/run_all.py`, which is inside E10's frozen source set and cannot
gain an import without invalidating the sealed digest:

    python -B tests/run_closure.py            # statistical and efficiency closure
    python -B tests/run_ve0.py                # canonical evidence contract
    python -B tests/run_ve1.py                # figures and tables
    python -B tests/run_ve2.py                # qualitative evidence
    python -B tests/run_e7b_supersession.py   # E7b withdrawal and lifecycle records

The current project map and audit findings are in collab/PROJECT_CONTEXT.md.
Claude-Codex planning, execution and review follow collab/PROTOCOL.md.

## V1 prototype (legacy)

V1 formulates VQA as classification over the top 100 answers, using frozen
CLIP global embeddings and small MLP heads. Four models share one head design
and one training procedure, differing only in input: image-only,
question-only, concat, and a handcrafted fusion of image, question, their
elementwise product and their absolute difference.

V1 prototype results (validation accuracy, single run):

| model              | val accuracy | trainable params |
|--------------------|--------------|------------------|
| majority reference | 0.234        | 0                |
| image-only         | 0.243        | 313,956          |
| question-only      | 0.458        | 313,956          |
| concat             | 0.525        | 576,100          |
| fusion             | 0.541        | 1,100,388        |

These are legacy prototype results, not confirmatory findings. Known
limitations: a single seed; the validation set was reused for checkpoint
selection and reporting, so its numbers are optimistically biased; concat and
fusion differ in head capacity (the fused input is twice as wide), so the
comparison is not capacity-controlled; and the efficiency measurements cover
the trainable heads only, excluding the shared frozen encoder.

## V2 evaluation protocol (Day 1, complete)

V2 replaces the V1 evaluation data with a defensible protocol:

- an image-disjoint development set (777 images, 10,004 raw questions, 7,714
  in-vocabulary) partitioned out of the GQA training images;
- a vocabulary fixed from the training pool only;
- strict nested training subsets of 40,000 / 100,000 / 250,000 questions,
  drawn from 724,074 eligible pool questions by one seeded permutation, each
  smaller manifest row-for-row a prefix of the larger;
- a clean test set (8,013 questions on 972 images) built from validation
  images never touched by the V1 validation set, split into an inputs file
  and an embargoed targets file;
- an independent verifier that re-derives the whole protocol from the raw GQA
  files (99 checks, 0 failures), plus preservation and idempotence proofs
  (all generated files byte-identical across rebuilds).

The clean-test targets (data/v2/test_clean_targets.csv) must not be read by
any training or development code before final evaluation, and no clean-test
label statistics may be computed before then. Development decisions use the
dev split only.

The tracked evidence for Day 1 is under artifacts/v2_00_protocol/ (build
summary, verifier report, vocabulary, file hashes, environment summary and a
completion summary); the full protocol description is in
docs/experiments/v2_00_protocol.md. The manifests themselves are local and
git-ignored; their sha256 hashes are recorded, and rebuilding them from the
raw GQA release reproduces them byte-for-byte.

## Requirements

- One NVIDIA GPU (developed on an RTX 4000 Ada Generation, about 20 GB VRAM).
- Python 3.12 with the `venv` module.

## Setup

Run once from the project root:

    bash setup.sh

This creates the virtual environment at `.venv`, redirects caches into a
project-local `.cache` folder, and installs the dependencies in
`requirements.txt`.

Start each session with:

    source .venv/bin/activate && source env.sh

If the machine uses node-local storage for the project directory, the venv
exists only on the node where `setup.sh` ran; `bash check_env.sh` confirms the
environment is intact and reports Python, torch and CUDA.

## Running the V1 stages (legacy)

The numbered scripts are the completed V1 stages and run in order:

    python 1_prepare_gqa.py        # V1 GQA subset and answer vocabulary
    python 2_extract_embeddings.py # run frozen CLIP once and cache the vectors
    python 3_train_baselines.py    # question-only, image-only and concat baselines
    python 4_train_latent_model.py # the V1 fusion model
    python 5_evaluate.py           # accuracy, efficiency and the trade-off plot

V2 code lives under experiments/; the Day-1 protocol build and verifier are

    python -B experiments/v2_00_protocol/build_manifests.py
    python -B experiments/v2_00_protocol/verify_protocol.py

## Configuration

All fixed settings live in `config.py` (dataset, answer-set size, subset
sizes, seed, CLIP model, training hyperparameters and paths). Edit values
there rather than passing command-line flags. The V2 protocol seeds derive
from `config.RANDOM_SEED`.

## Reproducibility

Each stage calls `utils.set_seed()` first, which seeds Python, NumPy and
PyTorch and, with `config.DETERMINISTIC`, turns on deterministic
cuDNN/cuBLAS; DataLoaders are seeded, the encoders are frozen, and
`utils.run_metadata()` records the commit, seed, library versions and GPU
with every result. The V2 protocol is additionally deterministic by
construction (seeded permutations over canonically sorted string IDs) and its
outputs are pinned by sha256 hashes in
`artifacts/v2_00_protocol/manifest_hashes_public.json`.

`requirements.txt` is the portable dependency list; `requirements.lock.txt`
records the exact resolved versions:

    python -m pip install -r requirements.lock.txt

See `docs/REPRODUCIBILITY.md` for the full account and its caveats.

## Repository layout

    config.py                 central settings, imported everywhere
    env.sh                    redirect caches into .cache (source each session)
    setup.sh                  one-time venv creation and dependency install
    requirements.txt          dependencies (portable list)
    requirements.lock.txt     exact resolved versions for reproduction
    check_env.sh              per-session environment check
    1_..5_*.py                the five V1 stage scripts (legacy, complete)
    src/                      reusable code: data, models, train, utils, efficiency
    experiments/              V2/V3/E-series experiment code, and closure/
    artifacts/                small tracked evidence files (v2_00_protocol)
    docs/                     reports, protocol documents, study guide
    data/                     GQA data and V1/V2 manifests (git-ignored)
    embeddings/               cached CLIP vectors (git-ignored)
    results/                  trained heads, metrics, figures (git-ignored,
                              except the small auditable JSON/CSV records of
                              E7b, E8A, E8B, E9, E10 and closure/)

## Reports

Each V1 stage has a report in `docs/` following `docs/REPORT_TEMPLATE.md`; the
V2 protocol is documented in `docs/experiments/v2_00_protocol.md` with its
evidence in `artifacts/v2_00_protocol/`.
