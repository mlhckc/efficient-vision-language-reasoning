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

## Agent collaboration

Claude is the primary planner and primary reviewer. Codex is the executor and
performs a secondary self-review. The user is the final authority on scope and
scientific decisions. Read AGENTS.md, collab/PROJECT_CONTEXT.md and
collab/PROTOCOL.md before planning or reviewing a task.

When acting as planner or reviewer, Claude must not edit source files, launch
training, change permissions or inspect the embargoed clean-test targets. Use
the read-only research-planner and research-reviewer agents under
.claude/agents/. Runtime handoffs use .agent-bridge/ and the state machine in
collab/PROTOCOL.md. A review is invalid if its recorded base commit or patch
hash no longer matches the worktree.

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
- V2 is complete through v2_07: global embedding extraction, five-seed
  baselines, parameter matching, interaction ablations, type and reliance
  analyses, and 40k/100k/250k global-head scaling.
- V3 is complete through v3_03. v3_00 built the token stores; v3_01 trained
  the 40k latent-query reasoner; v3_02a added direct-linear and mean-patch
  references, repaired pooled step statistics, image-clustered uncertainty,
  attention diagnostics, CLS removal and a patches-only training probe. At
  40k, the 21.1M-parameter reasoner did not materially outperform the 1.1M
  fusion head or reduce the repaired compositional deficit.
- v3_03 (E1, completed 31 July 2026) scaled the reasoner to 100k/250k under
  the frozen v3_01 recipe: dev accuracy 0.5706 +/- 0.0002 at 100k and
  0.5958 +/- 0.0074 at 250k, beating every stored v2_07 head in every seed
  (fusion gap +0.0065 and +0.0136). The pooled >=4-step deficit persists at
  every scale and remains statistically indistinguishable from fusion's
  (both clustered bootstrap intervals include zero). Development results
  only; see docs/experiments/v3_03_scaling.md.
- E2 (completed 31 July 2026) swapped the frozen encoder to SigLIP-B/16 on
  the global path under the identical v2_02 recipe: every multimodal head at
  40k, and concat and fusion at 250k, gain about +1.2 to +1.5 points over
  their CLIP counterparts in every seed (fusion 0.5532/0.5939; the 250k
  product reference is capacity-mismatched and excluded from headline
  comparisons), question_only is nearly unchanged,
  and the strongest SigLIP global heads reach the v3_03 reasoner's accuracy
  at a fraction of the size. The pooled >=4-step deficit does not move
  (0.063-0.102 across models and scales, the same range as CLIP): the
  compositional deficit now persists across two frozen dual-encoders, three
  scales and heads from 0.1M to 21.1M parameters. Development results only;
  see docs/experiments/e2_siglip.md.
- E3 (completed 1 August 2026) grew the closed answer set to the top 1000
  under the unchanged protocol (partition reused verbatim; the first 100
  vocabulary entries gated identical to answer_vocab_v2.json): dev coverage
  rises from 0.7711 to 0.9819; the 1000-way heads lose about 2 points on the
  shared head rows to class competition but win about +3.6 to +4.2 points of
  raw-distribution accuracy (in-vocab accuracy times coverage) at 250k; the
  tail (ranks 101-1000) is data-hungry (fusion 0.1588 at 40k to 0.2504 at
  250k, seed 0). Development results only; see
  docs/experiments/e3_vocab1000.md.
- Current gate, updated 1 August 2026: the user authorized the strengthening
  programme on 30 July 2026: V3 reasoner scaling to 100k/250k (E1), a
  SigLIP-B/16 frozen-encoder-swap experiment on the global-embedding path (E2)
  and a 1000-answer vocabulary experiment (E3), in that order after the
  S1-S3 correction and P0-P3 preparation packets. All of E1-E3 are complete
  (E1 and E2 on 31 July, E3 on 1 August 2026); the strengthening programme
  is finished. The next steps are the model-list freeze (F1) and only then
  the blinded clean-test evaluation (F2), both requiring explicit user
  authorization, with supervisor design feedback still to be recorded when
  available.
  Supervisor design feedback is still to be obtained and recorded when
  available, and the final venue decision will be discussed with Prof. Bober.
  No final clean-test evaluation has occurred; the clean-test embargo is
  unchanged until the final model list is frozen; all model findings remain
  development-set results.

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
  deficit persists) are complete. V3_00, v3_01 and v3_02a are also complete;
  see Current status and their reports for the current gate.
- V3's central contribution is the lightweight question-conditioned
  latent-query reasoner over token-level visual features and its controlled
  comparison with global-embedding baselines, including the negative 40k
  result.
- No large architectural change without a research question and a controlled
  comparison.

## Locked scope (do not change without asking)

- Task: discriminative VQA as answer classification. No text generation.
- Main dataset: a subset of GQA. VQA v2 is optional and only after GQA works.
- Encoders: frozen CLIP for both the image and the question, never trained. One
  model is used for both, so the two vectors share the same space.
  A single frozen SigLIP-B-16 encoder-swap experiment on the global-embedding
  path was approved by the user on 30 July 2026 (E2); all encoders remain
  frozen.
- Trainable part: lightweight heads over frozen CLIP features. V1/V2 use the
  MLP heads; the approved V3 central contribution is the lightweight
  question-conditioned latent-query reasoner over cached token-level
  features, with cached-token training as the primary pipeline and raw-path
  equivalence and efficiency measured separately. The encoders stay frozen.
- Answer set: top 100 answers first; the 1000-answer vocabulary experiment
  (E3) was approved by the user on 30 July 2026.
- Shared and V1 settings live in config.py. Experiment-specific constants,
  such as V3 search grids, gates and bootstrap counts, must be named near the
  experiment entry point and recorded in its report and result metadata.

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
