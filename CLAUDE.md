# CLAUDE.md

Project instructions for Claude Code. Read this at the start of every session
before doing anything else.

## Project

MSc Artificial Intelligence dissertation, University of Surrey. Supervisor:
Prof. Miroslaw Bober. Title: "Efficient Vision-Language Reasoning with Small
Language Models".

Research question: whether Visual Question Answering can be done efficiently by
reasoning in embedding space, using frozen small encoders plus a tiny trainable
head, instead of a large autoregressive vision-language model. An image and a
question are each turned into a fixed vector by one frozen CLIP encoder, and a
small head classifies the answer from a fixed set of the most frequent answers.
Three baselines are compared against a proposed fusion model, measuring both
accuracy and efficiency.

## Current status

- V1 (the five numbered stage scripts) is a completed legacy prototype
  pipeline: 40,000 training questions, 8,000 legacy validation questions
  (data/val.csv, referred to as legacy_v1_validation) and a top-100 answer
  vocabulary. V1 results are prototype results from a single seed, with the
  validation set reused for checkpoint selection; they must not be presented
  as final confirmatory results.
- V2 Day 1 is complete and verified: an image-disjoint development partition
  drawn from the GQA training data, a vocabulary computed from the training
  pool only, strict nested question-level training subsets, and a clean test
  set built from validation images never touched by legacy_v1_validation.
  See docs/experiments/v2_00_protocol.md and artifacts/v2_00_protocol/.
- Verified Day-1 counts: dev 777 images with 10,004 raw and 7,714
  in-vocabulary questions; training pool 71,363 images with 724,074 eligible
  questions (margin 474,074 over the required 250,000); train_40k /
  train_100k / train_250k cover 27,622 / 46,158 / 61,859 unique images and
  nest row-for-row; clean test 8,013 questions on 972 images (structural
  counts only); verifier 99 checks with 0 failures, 0 missing images, 0
  duplicate questionIds; idempotence proven (13 generated files
  byte-identical across rebuilds).
- The V1 and V2 vocabularies contain the same 100 answers but 11 answers have
  different indices, so V1 label indices must never be mixed with V2
  manifests. All V2 work uses data/v2/answer_vocab_v2.json.

## V2 protocol rules (binding)

- Clean-test targets (data/v2/test_clean_targets.csv) are embargoed until the
  final model list and all development decisions are frozen. No development or
  training code may read that file.
- Do not calculate or expose clean-test label statistics before final
  evaluation: no vocabulary coverage, OOV counts, answer distribution, yes/no
  share or question-type statistics. Clean-test reporting is structural only.
- All development decisions use data/v2/dev.csv only. The clean test must not
  be used for early stopping, hyperparameter tuning, architecture selection,
  fusion selection, latent-query-count selection or depth selection.
- Status: v2_01 (embedding extraction and zero-shot floor 0.080), v2_02
  (five-seed baselines: fusion 0.5384 beats concat 0.5240 in every seed),
  v2_03 (parameter matching halves the fusion gain), v2_04 (either
  interaction term alone carries it; the terms are redundant), v2_05/v2_05b
  (gains concentrate in verify/logical/obj/rel; choose cost seed-robust;
  multi-step lift deficit about 0.08), v2_06 (fusion relies most on the
  image; excess reliance in verify/logical) and v2_07 (at 250k the feature
  advantage decays to noise, the multimodal margin grows, the multi-step
  deficit persists) are complete, as is v3_00 (token stores extracted,
  consistency-checked against the V2 globals, loaders benchmarked). The
  next stage is v3_01: the question-conditioned latent-query reasoner over
  the cached token stores.
- V3's intended central contribution is a lightweight question-conditioned
  latent-query reasoner over token-level visual features, evaluated against
  controlled global-embedding baselines.
- No large architectural change without a research question and a controlled
  comparison.

## Locked scope (do not change without asking)

- Task: discriminative VQA as answer classification. No text generation.
- Main dataset: a subset of GQA. VQA v2 is optional and only after GQA works.
- Encoders: frozen CLIP for both the image and the question, never trained. One
  model is used for both, so the two vectors share the same space.
- Trainable part: lightweight heads over frozen CLIP features. V1/V2 use the
  MLP heads; the approved V3 central contribution is the lightweight
  question-conditioned latent-query reasoner over cached token-level
  features, with cached-token training as the primary pipeline and raw-path
  equivalence and efficiency measured separately. The encoders stay frozen.
- Answer set: top 100 answers first, scaling to 1000 later as an experiment.
- All fixed settings live in config.py and are read from there, never
  hard-coded.

## Hard rules

- Do not train or fine-tune any large VLM. Do not fine-tune a 7B model.
- Do not unfreeze the encoders.
- Do not invent or estimate results. Every number must come from a real run.
- Do not add datasets, models or dependencies outside this scope without
  asking.

## Environment

- Hardware: one NVIDIA RTX 4000 Ada Generation, about 20 GB VRAM (the card
  described in the brief as "RTX A4000 Ada"). Single GPU only.
- Python virtual environment at .venv inside the project, not Conda.
- The home directory has almost no free storage; scratch has space. All caches
  are redirected into the project-local .cache folder by env.sh so downloads do
  not fill the home quota.
- /scratch is node-local, not shared like the home filesystem, so the venv
  exists only on the node where setup.sh was run. On a different node, `import
  torch` fails even though activation appears to succeed. Re-run `bash setup.sh`
  on that node to rebuild the venv there.
- Per-session startup, after the one-time `bash setup.sh`:

      source .venv/bin/activate && source env.sh

  Then `bash check_env.sh` confirms the node has the venv and that torch and
  CUDA are visible, failing with a clear message rather than a bare
  ModuleNotFoundError.

## Stage workflow (legacy V1 pipeline)

The numbered scripts are the completed V1 stages and run in order; they remain
runnable but produce prototype results only (see Current status). V2 work
lives under experiments/ and uses the data/v2 manifests.

1. `1_prepare_gqa.py` — prepare the GQA subset and the answer vocabulary.
2. `2_extract_embeddings.py` — run frozen CLIP once and cache vectors to disk.
3. `3_train_baselines.py` — train the question-only, image-only and concat
   baselines.
4. `4_train_latent_model.py` — train the proposed fusion model: concatenate
   image, question, image * question and abs(image - question), then an MLP
   head.
5. `5_evaluate.py` — evaluate accuracy and efficiency and make the trade-off
   plot.

Reusable code lives in src/: data.py (dataset and dataloaders over cached
vectors), models.py (the MLP head, the baselines and the fusion model), utils.py
(seeding, device, parameter counting, timing, saving results).

## Coding conventions

- Read every setting from config.py. Do not hard-code values that belong there.
- Call utils.set_seed() as the first line of every stage's main(), before any
  data loading or model creation, so randomness is fixed from
  config.RANDOM_SEED.
- Save utils.run_metadata() with every result (via utils.save_json) so each
  number is traceable to the code, settings and environment that produced it.
- For reproducible data loading use utils.make_generator() and
  utils.seed_worker() on the DataLoader.
- Select the device through config.DEVICE (or utils.get_device()).
- See docs/REPRODUCIBILITY.md for the full reproducibility contract; keep
  requirements.lock.txt current after any dependency change (python -m pip
  freeze > requirements.lock.txt).
- Keep functions small and focused; put shared logic in src/ rather than copying
  it between stages.
- Use clear type hints where they aid reading.
- Write large artefacts (subsets, vectors, results) to data/, embeddings/ and
  results/, which are git-ignored.
- Each stage should be runnable on its own once the earlier stages have
  produced their outputs.

## Reporting requirement

Each stage produces a short report in docs/ following docs/REPORT_TEMPLATE.md,
with sections Purpose, Method, Outputs, Results, and Decisions and problems.
Reports record what was actually run and the real numbers produced.

## Writing style for all repository text

Reports, README, comments and commit messages follow the same style:

- Plain, factual, academic English, the way a careful graduate student writes.
- No emoji anywhere.
- No hype words such as powerful, seamless, cutting-edge, leverage, delve,
  unlock.
- Do not open sections with filler such as "In this section we will".
- Do not over-format. Short paragraphs. Lists only for real lists.
- Commit messages short and imperative, for example "add config and project
  scaffold".
