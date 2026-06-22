# Efficient Vision-Language Reasoning with Small Language Models

MSc Artificial Intelligence dissertation, University of Surrey. Supervisor:
Prof. Miroslaw Bober.

The project tests whether Visual Question Answering can be done efficiently by
reasoning in embedding space. An image and a question are each encoded into a
fixed vector by one frozen CLIP model, and a small trainable head classifies the
answer from a fixed set of the most frequent answers. Three baselines are
compared against a proposed fusion model on both accuracy and efficiency. No
large vision-language model is trained, and the encoders are never unfrozen.

## Requirements

- One NVIDIA GPU (developed on an RTX 4000 Ada Generation, about 20 GB VRAM).
- Python 3.12 with the `venv` module.

## Setup

Run once from the project root:

    bash setup.sh

This creates the virtual environment at `.venv`, redirects caches into a
project-local `.cache` folder (so downloads do not fill the home directory), and
installs the dependencies in `requirements.txt`.

Start each session with:

    source .venv/bin/activate && source env.sh

Note that `/scratch` is node-local on this cluster, so the virtual environment
exists only on the node where `setup.sh` was run. After connecting, confirm you
are on that node and that the environment is intact:

    bash check_env.sh

It reports the host, Python, torch and CUDA, and exits with a clear message if
the venv is missing (in which case re-run `bash setup.sh` on the current node)
or if any package is absent.

## Running the stages

The numbered scripts are the project stages and run in order:

    python 1_prepare_gqa.py        # prepare the GQA subset and answer vocabulary
    python 2_extract_embeddings.py # run frozen CLIP once and cache the vectors
    python 3_train_baselines.py    # question-only, image-only and concat baselines
    python 4_train_latent_model.py # the proposed fusion model
    python 5_evaluate.py           # accuracy, efficiency and the trade-off plot

They currently raise `NotImplementedError`; each stage is implemented in turn.

## Configuration

All fixed settings live in `config.py` (dataset, answer-set size, subset sizes,
seed, CLIP model, training hyperparameters and paths). Edit values there rather
than passing command-line flags.

## Repository layout

    config.py                 central settings, imported everywhere
    env.sh                    redirect caches into .cache (source each session)
    setup.sh                  one-time venv creation and dependency install
    requirements.txt          dependencies
    1_..5_*.py                the five ordered stage scripts
    src/                      reusable code: data, models, utils
    docs/                     report template and per-stage reports
    data/                     GQA subset and answer vocabulary (git-ignored)
    embeddings/               cached CLIP vectors (git-ignored)
    results/                  trained heads, metrics and figures (git-ignored)

## Reports

Each stage has a short report in `docs/`, following `docs/REPORT_TEMPLATE.md`.
