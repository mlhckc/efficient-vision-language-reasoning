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

## Locked scope (do not change without asking)

- Task: discriminative VQA as answer classification. No text generation.
- Main dataset: a subset of GQA. VQA v2 is optional and only after GQA works.
- Encoders: frozen CLIP for both the image and the question, never trained. One
  model is used for both, so the two vectors share the same space.
- Trainable part: a lightweight MLP head only. A small transformer head is a
  later stretch goal, not the start.
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

## Stage workflow

The numbered scripts are the project stages and run in order:

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
- Seed all randomness from config.RANDOM_SEED for reproducibility.
- Select the device through config.DEVICE.
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
