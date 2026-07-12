# Efficient Vision-Language Reasoning with Small Language Models

MSc Artificial Intelligence dissertation, University of Surrey. Supervisor:
Prof. Miroslaw Bober.

The project tests whether Visual Question Answering can be done efficiently by
reasoning in embedding space. An image and a question are each encoded into a
fixed vector by one frozen CLIP ViT-B-32 model, and a small trainable head
classifies the answer from a fixed set of the most frequent answers, on a
subset of GQA. No large vision-language model is trained, and the encoders are
never unfrozen.

Status: the V1 prototype pipeline is complete, and the V2 evaluation protocol
(Day 1) is built and verified. The next step is V2 embedding extraction for
the new manifests, followed later by multi-seed training.

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
    experiments/              V2 experiment code (v2_00_protocol: build and verify)
    artifacts/                small tracked evidence files (v2_00_protocol)
    docs/                     reports, protocol documents, study guide
    data/                     GQA data and V1/V2 manifests (git-ignored)
    embeddings/               cached CLIP vectors (git-ignored)
    results/                  trained heads, metrics, figures, reports (git-ignored)

## Reports

Each V1 stage has a report in `docs/` following `docs/REPORT_TEMPLATE.md`; the
V2 protocol is documented in `docs/experiments/v2_00_protocol.md` with its
evidence in `artifacts/v2_00_protocol/`.
