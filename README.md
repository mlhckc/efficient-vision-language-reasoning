# Efficient Vision-Language Reasoning with Small Language Models

MSc Artificial Intelligence dissertation, University of Surrey. Supervisor:
Prof. Miroslaw Bober.

## Project summary

This project asks whether Visual Question Answering can be done efficiently
by reasoning in embedding space. An image and a question are each encoded by
one frozen CLIP ViT-B-32 model, and a small trainable head classifies the
answer from a closed set of frequent answers, on a subset of GQA. No large
vision-language model is trained; the encoders are never unfrozen. Around
that core, the project measures — under multi-seed controls, clustered
statistics and a hard test-set embargo — what such lightweight systems can
and cannot do, and what one query actually costs.

## Research question

Can frozen small encoders plus a tiny trainable head deliver useful VQA
accuracy at a small measured cost, instead of running a large
autoregressive vision-language model — and where exactly are the limits of
that approach?

## Why this project matters

A closed-set answer from cached frozen embeddings costs single-digit
milliseconds measured end to end; the compact autoregressive VLMs measured
in this project took approximately 24 to 25 times higher warm serial
latency under the same-node contextual protocol. Knowing what the cheap
path extracts, what it provably does
not (multi-step composition), and which ingredients matter (features,
capacity, data, token access, pretrained language models) is useful both as
engineering guidance and as evidence about frozen multimodal
representations. The negative results are contributions: each one
eliminates a candidate explanation under controls.

## System overview

Four trainable model families over the same frozen base, plus one external
reference class (details and diagrams in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)):

- **Global fusion heads** — small MLPs over pooled CLIP vectors: concat,
  handcrafted fusion (product and absolute-difference interactions),
  parameter-matched and ablation controls.
- **Token-level latent-query reasoner** — 32 learned latents attending over
  50 image tokens and question word tokens (21.1M parameters); the central
  architectural contribution.
- **Question-side SLM interface (E8A)** — frozen SmolLM2-135M question
  states projected into the reasoner, with a random-initialised causal
  control.
- **Answer-side SLM readout (E8B, E10)** — the reasoner's latents read out
  through a frozen SmolLM2-135M/360M, against random-initialised and
  language-model-free controls.
- **Compact VLMs (E9)** — frozen SmolVLM-256M/500M, evaluation only, for
  contextual positioning.

## Dataset and evaluation setting

A verified V2 protocol over the balanced GQA release (v2_00): an
image-disjoint development partition (777 images, 10,004 raw questions,
7,714 in the top-100 vocabulary), nested training subsets of 40k/100k/250k
questions, a vocabulary computed from the training pool only, and a clean
test set (8,013 questions on 972 images) whose targets file is embargoed.
An independent verifier re-derives the protocol from the raw GQA files (99
checks, 0 failures). Primary metrics are strict raw exact match and one
pinned VQA-style normalised exact match. Every result in this repository
is a development-set result: the model-list freeze (F1) is unstarted and
the blinded clean-test evaluation (F2) is unstarted and unauthorised.

## Experimental journey

V1 prototype (legacy, selection-biased — fixed by the V2 protocol) →
five-seed baselines and the fusion decomposition (v2_02-v2_04) →
type/step analysis finding the multi-step deficit, and visual-reliance
interventions (v2_05-v2_06) → data scaling to 250k (v2_07) → token-level
latent-query reasoner, negative at 40k, ahead at 100k/250k, deficit
unchanged (v3_00-v3_03) → frozen SigLIP swap (E2) and top-1000 vocabulary
(E3) → measured component and serial efficiency (E7a/E7b) → frozen small
language models on the question side (E8A) and answer side (E8B) →
compact-VLM context (E9) → answer-side capacity at 360M (E10) →
statistical closure and frozen evidence layers (VE-0/VE-1/VE-2). The full
story: [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md).

## Key results

In-vocabulary development accuracy (7,714 rows; five seeds for V2/E2
rows, three for V3/E8/E10), 40k / 250k training questions:

| System | 40k | 250k |
|---|---|---|
| majority reference | 0.2247 | — |
| question_only | 0.4580 | 0.4977 |
| concat | 0.5240 | 0.5786 |
| fusion | 0.5384 | 0.5823 |
| reasoner (21.1M) | 0.5422 | 0.5958 |
| SigLIP fusion (E2) | 0.5532 | 0.5939 |
| E8A: A0p / A1 / A1r | 0.54045 / 0.52861 / 0.49041 | 0.59446 / 0.57316 / 0.52480 |
| E8B: B1 / B2 / B3 | 0.54507 / 0.53392 / 0.50929 | 0.59394 / 0.60086 / 0.59610 |
| E10: B4 / B4r (360M) | 0.52636 / 0.53319 | 0.59407 / 0.59645 |

On the raw distribution (all 10,004 dev questions, 250k): the top-1000
product head reaches 0.4904 against 0.4490 for the top-100 fusion head;
SmolVLM-256M scores 0.436525 and SmolVLM-500M 0.489004 under the pinned
normalised metric. Full precision, contrasts and intervals:
[docs/EXPERIMENTS_AND_RESULTS.md](docs/EXPERIMENTS_AND_RESULTS.md).

## Main scientific findings

- Language priors are strong, but the visual signal contributes
  materially and grows with data; wrong images actively mislead.
- The handcrafted fusion advantage is real, half-capacity/half-features,
  carried by either interaction term alone, and decays to noise at 250k.
- The multi-step (>=4-step) deficit of about 0.06-0.11 persists across
  every head, two frozen encoders and three scales — the project's most
  robust negative finding.
- Question-side SLM pretraining helps against a matched random control;
  answer-side SLM pretraining shows no detected advantage at 135M or
  360M. Training-set size dominates.
- Representation quality (SigLIP: +1.2 to +1.5 points) and answer-space
  design (top-1000: +3.6 to +4.1 raw-distribution points) are the levers
  that move accuracy.

Discovery-by-discovery detail:
[docs/SCIENTIFIC_FINDINGS.md](docs/SCIENTIFIC_FINDINGS.md).

## Efficiency summary

E7b is the authoritative end-to-end evidence: measured warm serial
batch-1 queries (raw image and question to answer, node otter155) cost
about 7.6 ms per query for the CLIP global heads, 9.2-20.1 ms for the
reasoner-class and SLM systems; the primary accuracy-latency frontier is
concat, fusion and the top-1000 product head. Under the same-node E9
protocol (otter159), SmolVLM-500M measured 151.380 ms on its open-readout
row (accuracy 0.489004) and 152.854 ms on its constrained row (accuracy
0.485706), against 6.199 ms for the top-1000 product head (0.49050):
approximately 24.4 and 24.7 times the warm serial latency respectively.
These are
latency and parameter statements only: no energy, power, carbon or
monetary measurement exists, so no such claim is made anywhere. On
E7a's retained fields, the top-1000 product head at 250k is more
accurate (0.4904 against 0.4594 raw-distribution) than the
21.1M-parameter reasoner, using 16 times fewer parameters. E7a's
additive end-to-end sums and three E7b fields (peak memory, cold-start)
are withdrawn; field validity resolves through
`results/closure/e7b_evidence_supersession.json`.

## Current status

The scientific evidence summarised here is current to 14 August 2026
(governance and status records are dated separately in
[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)). All experimental
packets are complete. Five
review packets are closed with independently accepted verdicts: E7b,
VE-0, VE-1, VE-2 and E10 (lifecycle record:
`results/closure/pre_f1_status_supersession_20260814.json`); E7b is CLOSED
at E7B_CANONICAL_SUPERSESSION_REVIEW_PASS. The 14
August pre-F1 evidence metadata repair — correcting nine E8B B1-bearing
checkpoint-selection rows (B1 is best-on-development; B2/B3 are fixed
epoch 22), five presentation-precision fields, the E7a locator and one
figure placement — is approved, pushed and closed
(`results/closure/pre_f1_evidence_metadata_repair_20260814.json`).
Detailed state: [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md).

The clean-test contents were never inspected or used for development,
model selection, or reporting decisions. Mechanical byte access occurred
in two documented governance incidents, on 13 and 14 August 2026. The
first was an independent reviewer's integrity-hash command over the
target file; the second was an overly broad dependency-mapping scan that
read every .json, .py, .md, .csv and .txt file under the project root
because the traversal was not scoped away from data/. Neither inspected
any row, label, distribution or prediction, and neither informed
development, model selection or reporting. They are governance
incidents, not test-informed scientific selection. The governing
disclosure is in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## What remains

The model-list freeze (F1, unstarted) and only then the blinded
clean-test evaluation (F2, unstarted and unauthorised), each requiring
explicit user authorization; supervisor design feedback, still to be
obtained and recorded; and the dissertation manuscript itself, written
from the frozen evidence base — the handoff for that is
[docs/DISSERTATION_HANDOFF.md](docs/DISSERTATION_HANDOFF.md).

## Repository structure

    config.py                 central settings, imported everywhere
    1_..5_*.py                the five V1 stage scripts (legacy, complete)
    src/                      reusable code: data, models, reasoner, utils
    experiments/              V2/V3/E-series experiment code, closure/, ve0-ve2
    docs/                     this documentation set, per-experiment reports,
                              REPRODUCIBILITY, RELATED_WORK, references.bib
    docs/experiments/         one tracked report per experiment (canonical prose)
    artifacts/                small tracked evidence (v2_00 protocol, results_export)
    results/                  git-ignored except the auditable JSON/CSV records:
      results/experiments/      per-experiment result records
      results/closure/          statistical closure + canonical successor records
      results/ve0|ve1|ve2/      frozen evidence inventory, figures/tables, gallery
    tests/                    per-layer CPU-only runners and test modules
    data/, embeddings/        GQA data, manifests, caches (git-ignored;
                              data/v2/test_clean_targets.csv is embargoed)

Where to look: "what code produced this?" → `experiments/` and `src/`;
"what evidence supports this?" → `results/experiments/`,
`results/closure/`, `artifacts/results_export/`; "which value is
canonical?" → the successor records named in
[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md); "what is historical?"
→ anything a successor record supersedes, plus `docs/STUDY_GUIDE.md` and
`docs/PROGRESS_REPORT.md` (V1-era, banners included).

## Reproducibility

Requirements: one NVIDIA GPU (developed on an RTX 4000 Ada Generation,
about 20 GB VRAM) and Python 3.12 with the venv module. One-time setup,
then per-session activation:

    bash setup.sh
    source .venv/bin/activate && source env.sh
    bash check_env.sh

All settings live in `config.py`; every stage seeds via
`utils.set_seed()`; DataLoaders are seeded; the encoders are frozen;
`utils.run_metadata()` records commit, seed, versions and GPU with every
result. `requirements.lock.txt` pins exact versions. The V2 protocol is
deterministic by construction and hash-pinned
(`artifacts/v2_00_protocol/manifest_hashes_public.json`). The legacy V1
stages run as `python 1_prepare_gqa.py` … `python 5_evaluate.py`; V2
work lives under `experiments/`. Each closed evidence layer ships a
CPU-only runner:

    python -B tests/run_closure.py
    python -B tests/run_ve0.py
    python -B tests/run_ve1.py
    python -B tests/run_ve2.py
    python -B tests/run_e7b_supersession.py
    python -B tests/run_pre_f1_evidence_metadata_repair.py

(`tests/run_all.py` is inside E10's frozen source set and needs a GPU;
it is not part of the zero-GPU verification path.) Full contract and
caveats: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Dissertation documentation

- [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) — the complete story
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — model families and diagrams
- [docs/EXPERIMENTS_AND_RESULTS.md](docs/EXPERIMENTS_AND_RESULTS.md) — every experiment, every number
- [docs/SCIENTIFIC_FINDINGS.md](docs/SCIENTIFIC_FINDINGS.md) — findings with confidence and limits
- [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) — done, closed, remaining
- [docs/DISSERTATION_HANDOFF.md](docs/DISSERTATION_HANDOFF.md) — for whoever writes the dissertation
- collab/PROJECT_CONTEXT.md and collab/PROTOCOL.md — agent-collaboration context
