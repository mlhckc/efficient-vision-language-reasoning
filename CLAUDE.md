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
  shared head rows (class competition and the smaller head-answer share of
  the fixed budget are confounded) but win about +3.6 to +4.1 points of
  raw-distribution accuracy (in-vocab accuracy times coverage) at 250k; the
  tail (ranks 101-1000) is data-hungry (fusion 0.1588 at 40k to 0.2504 at
  250k, seed 0). Development results only; see
  docs/experiments/e3_vocab1000.md.
- E7a (completed 1 August 2026) measured, under one protocol, the cost of
  every stored head plus both frozen encoders: the encoder dominates
  (CLIP 2.2510 ms image + 1.7309 ms text on GPU plus 2.295 ms CPU decode,
  against 0.0162-0.0469 ms for the global heads, 0.26-0.74 per cent of
  the full pipeline for image-using heads), caching image features
  across about 10 questions
  per image cuts a query from 6.35 to 2.25 ms, and on both end-to-end
  latency Pareto fronts the only optimal models are top-1000 global
  heads. The top-1000 product head at 250k is both more accurate
  (0.4904 against 0.4594 raw-distribution) and cheaper (6.35 against
  7.71 ms) than the 21.1M reasoner. Parameter count is a poor latency
  proxy (fusion is 48% slower than concat_wide at equal parameters).
  Supersedes the V1 stage-5 and src/efficiency.py-derived latencies.
  Development results only; see docs/experiments/e7a_efficiency.md.
- E9 (authorized and completed 10 August 2026) placed one frozen compact
  integrated VLM in context against the lightweight systems, evaluation only
  and with zero trainable parameters. On the 10,004-row raw development
  partition under the pinned G21 normalised metric, SmolVLM-256M-Instruct
  reaches 0.436525 with free greedy generation: above every E8B arm at
  train_40k and below every one at train_250k, all six paired contrasts
  directional. SmolVLM-500M-Instruct, a secondary normal-condition-only
  capacity point, reaches 0.489004 and exceeds every E8B arm at both scales
  (+0.026 to +0.031 over the 250k arms). Constraining generation to the
  top-1000 answer support moves the score by at most about 0.005 with every
  interval containing zero, so output format is not what limits the compact
  VLM; strict raw exact is near zero only because it emits " Yes." against
  gold "yes". The matched image-partner derangement costs it +0.16763,
  about a third more than any E8B arm's +0.084 to +0.126, so it relies on
  the image more. On one node under the frozen E7b serial protocol the
  top-1000 global head reaches the same raw-distribution accuracy at 6.199 ms
  against SmolVLM-500M's 152.854 ms, about 25 times cheaper; the E8B bridge
  measures B1 at 10.242 ms and B2/B3 R2/R3 at 38.8 to 39.7 ms. The fusion
  bridge control failed its pre-registered 10 per cent tolerance against the
  historical E7b node (-18.7 per cent), so those frontiers are not merged and
  no adjustment factor is applied. E9 is contextual positioning, not a matched
  causal comparison: training history, multimodal pretraining, answer support
  and output format all differ, and SmolVLM-256M's text backbone is the
  Instruct checkpoint where E8A and E8B use base. Development results only;
  see docs/experiments/e9_compact_vlm.md.
- E10 Phase 0 (completed 10 August 2026) implemented the registered B4/B4r
  SmolLM2-360M answer-readout family, verified the exact pinned base model and
  ran one bounded non-scientific B4r calibration. It charged 0.032714 GPU-hours
  to `random_smollm2_360m`, completed no epoch or scientific cell, produced no
  checkpoint or performance result, and passed every Gate-1 hard gate. The
  12-cell B4/B4r core remains unauthorised. `E10_TRAINING_AUTHORIZED` and both
  E10 scientific budget constants remain `None`; the one-time calibration
  authorisation is revoked. See
  docs/experiments/e10_phase0_calibration.md.
- Current gate, updated 2 August 2026 (E8A/E8B/E9 programme). The user
  authorized the E8A frozen-SLM question-encoder branch, the E8B frozen-SLM
  readout and bounded-generation branch, and the E9 evaluation-only compact-VLM
  baselines. Binding restrictions, recorded here and not only in the
  gitignored bridge packets:
  Permitted models and roles: SmolLM2-135M base and SmolLM2-360M base as E8A
  semantic encoders and E8B readouts; FLAN-T5-small as an E8A cross-family
  sensitivity point only, never as an answer readout; SmolVLM-256M-Instruct
  and SmolVLM-500M-Instruct as E9 evaluation-only baselines, never trained or
  fine-tuned; deterministic random-initialised SmolLM2-135M and SmolLM2-360M
  as non-pretrained causal controls in both branches. No other model.
  Every vision encoder, language model and compact VLM stays frozen. No LM or
  VLM fine-tuning, no LoRA or other adaptation, no unfreezing. No new dataset.
  Generation is bounded to E8B R2 (trie-constrained), E8B R3 (free, fixed
  token cap) and E9 greedy evaluation; no sampling, temperature search, prompt
  search or long-form generation.
  Stopping is condition-based, with no calendar cutoff. Numeric compute gates
  halt execution and return to the user if a pilot projects any of: one
  principal run above 8 GPU-hours (a hard operational wall-clock halt with a
  recorded failure status); one model aggregate above 40 GPU-hours (see the
  cross-E8 ceiling entry below); the worst-case remaining-core projection
  above min(180 GPU-hours, three times
  the revised expected projection); peak memory above 80 per cent of usable
  GPU memory; or storage above the project allocation. The core ceiling was
  raised from 150 to 180 GPU-hours by the user on 2 August 2026; the per-run,
  memory and storage ceilings are unchanged.
  CROSS-E8 PER-MODEL-IDENTITY CEILING, amended by the user on 8 August 2026
  and BINDING: the per-model-identity aggregate ceiling is 40 GPU-hours,
  PROGRAMME-WIDE. It supersedes the previous active 35 GPU-hour cross-E8
  ceiling from 8 August 2026 onward and is NOT E8B-only. The aggregate is
  per frozen model identity; for pretrained SmolLM2-135M it sums the E8A
  arms A1, A4 and A7c together with the final E8B B3 programme, its
  readouts and interventions, and every already-spent hour that loaded the
  pinned pretrained checkpoint. The single authoritative constant is
  config.PER_MODEL_IDENTITY_CEILING_HOURS; every executable gate imports
  it and no module may declare its own copy. Any 35 GPU-hour figure in a
  record or comment is HISTORICAL: it states the ceiling active when that
  text was written and is superseded. See
  results/cross_e8_ceiling_supersession_20260808.json. This was a
  pre-result resource-governance amendment: no final core cell had run,
  and no scientific result, model, seed, scale, intervention, denominator,
  precision rule, fixed-22 rule or execution matrix changed.
  Clarified by the user on 7 August 2026 (decision P3) and recorded here
  because the sentence above no longer states the halting rule exactly: the
  GOVERNING, halting comparison for the core ceiling is the expected-epoch
  projection against 180 GPU-hours alone. The min(180, three times expected)
  form is still computed and reported as a mandatory stress scenario, but it
  is explicitly NON-HALTING on its own. The executable gate,
  run.remaining_core_gate, implements exactly this.
  The clean test stays embargoed and no experiment code may resolve its path.
  F1 and F2 remain unauthorized and unstarted. The post-core research backlog
  is recorded but NOT authorized. Any new model, dependency or architectural
  direction requires fresh explicit approval.
  A5 (SmolLM2-360M question-only) and A8c (CLIP image concatenated with pooled
  360M question features) are unconditional core controls at 40k and 250k with
  seeds 0/1/2. Random-initialised SmolLM2-135M and SmolLM2-360M are authorized
  in both E8A and E8B, unconditionally, at the same scales and seeds.
  Scoring and comparison restrictions: no direct comparison between published
  official-GQA scores and this project's custom development split, in either
  direction, because the split, the answer support and the scorer all differ;
  and no semantic matching, synonym list, embedding similarity or
  language-model judging in any primary score. The two reported primary scores
  are strict raw exact match and one pinned VQA-style normalised exact match.
  Condition-based stopping, no calendar cutoff. Work continues while it is
  directly relevant to the P2607 questions, the core E8A/E8B/E9 programme is
  progressing under the reviewed protocol, F1/F2 and dissertation writing are
  not put at material risk, no unresolved scientific or protocol blocker
  remains, no uncontrolled architecture or hyperparameter search has been
  introduced, and each experiment has a clear hypothesis, control and expected
  contribution. Priority 1 is the core programme; priority 2 the confirmatory
  seeds, scales, controls, reliance, deficit and efficiency measurements;
  priority 3 optional extensions, only once the core is closed and only after
  presenting the question, the runtime and what it would delay. Work stops when
  the core questions are answered with adequate controls, when a further
  experiment would add breadth without changing the conclusion, when remaining
  GPU or engineering work would threaten F1/F2 or the dissertation, when a new
  model, dependency, dataset or architectural direction would be required, when
  reviewers judge the evidence sufficient, or when expected scientific value
  falls below the cost and schedule risk. Each phase ends with a closure report.
  Pair preservation, binding: A1 with A1r, A2 with A2r, B3 with B2 and B4
  with B4r are inseparable pairs. No pretrained arm runs at a scale where its
  within-size random control does not. No core arm is ever descoped
  automatically: if a compute gate fires, execution stops and returns to the
  user with pair-preserving alternatives.
  Phase 1 (E8B) HAS begun and is superseded in part; see the E8B gate
  immediately below. E8A execution has also begun (see results/).
- E8B gate, updated 7 August 2026 and BINDING. Read before any E8B work.
  1. The eight-point E8B recipe search (master protocol 7.4) is
     PERMANENTLY ABANDONED. Grid points 4-8 must never run. Grid points
     1-3 are exploratory protocol-diagnostic evidence only: they are
     never core results, never select the recipe, and support no
     superiority claim in either direction. Every statistic derived from
     them for ranking purposes is INVALIDATED and recorded as such; do
     not invent another. The honest narrative is that the exploratory
     search could not support a meaningful winner claim, so the
     pre-result default recipe was frozen outcome-independently.
  2. The final B2/B3 recipe is FIXED at lr 3e-4, warmup 0, dropout 0.1,
     retained SOLELY because it was the pre-result pilot and default
     configuration (it is also the section 7.1 inherited v3_01
     configuration). NO statistic from Grids 1-3 may justify the recipe
     choice or support a superiority claim in EITHER direction. The
     historical observation that grid point 1 also showed the highest
     BF16 maximum may be mentioned only as an exploratory,
     NON-CANONICAL, superseded fact - never as support for the choice.
  3. U4 is WITHDRAWN. No search checkpoint is promoted. All 18 core
     cells (B1/B2/B3 x train_40k/train_250k x seeds 0/1/2) are trained
     fresh.
  4. Training remains bf16 autocast on the training path. CANONICAL
     SCIENTIFIC EVALUATION IS FP32, computed directly in fp32 from the
     trainable trunk onward; a bf16-computed prefix cast to fp32 is
     FORBIDDEN. This supersedes master protocol section 20's
     pre-registered "the frozen language models are never upcast"
     clause for E8B only; the stop-and-ask that clause requires was
     raised and the user authorised it on 7 August 2026. The frozen
     model identity is unchanged: bf16 on disk promotes losslessly.
     bf16 figures are secondary deployment diagnostics only, never a
     scientific accuracy and never a selection basis.
  5. For B2/B3: exactly 22 epochs, NO patience early stopping, and the
     EPOCH-22 checkpoint is the canonical primary. Primary R1/R2/R3 and
     the B3-B2 contrast use epoch 22. Any best-of-22 result is a clearly
     labelled secondary diagnostic and never determines the primary
     comparison, the model freeze or the clean-test checkpoint. B1
     retains its frozen section 7.3 classifier recipe and is NOT forced
     onto the 22-epoch rule.
  6. Strict determinism cannot be established at startup: utils.set_seed
     unconditionally sets warn_only=True, and build_arm re-seeds because
     G13 requires it. Enforcement must be re-imposed after EVERY
     reseeding point and asserted at the point of use.
  7. Scientific core execution was AUTHORISED by the user on 8 August
     2026 at HEAD 5228f7e, on the targeted certification of that state
     (ACCEPT; 0 BLOCKER, 0 HIGH, 0 unresolved execution-critical
     MEDIUM, 946 checks, divergence 0 0, clean worktree, zero core
     artefacts, zero execution locks, active resource gate fires
     false). run.TRAINING_AUTHORIZED is now "core-matrix-approved".
     The grant covers exactly the 18 frozen core cells and their
     mandatory frozen evaluations, run in the pair-preserving order
     that run.pair_preserving_order() prints, under the frozen recipe
     of point 2 and the fixed-22 rule of point 5. It broadens nothing:
     grid points 4-8, E9, the E10 scientific core, F1, F2 and the clean test remain
     refused, the scientific design is unchanged, and the matrix is
     fixed regardless of what early results look like.
     SUPERSEDED IN PART on 10 August 2026: the user authorized the E9
     design with targeted amendments and E9 has since been executed and
     closed; separately, the user authorized and Codex completed E10 Phase 0
     implementation, provenance, bounded non-scientific calibration and Gate
     1. These narrow supersessions do not authorize the E10 scientific core,
     grid points 4-8, F1, F2 or the clean test, and nothing else in this grant
     changes.
     Performance-driven retries are forbidden, a failed execution
     grants no automatic retry, and any retry that does not fit the
     frozen ceiling and floor policy halts and returns to the user.
     Non-scientific probes are separate and are never promoted. The
     clean-test embargo is unchanged.
  Authoritative records: results/experiments/e8b_readout_generation/
  protocol_amendment_20260807_fp32.json, protocol_amendment_20260807_
  fixed22.json, superseded_evidence_20260807.json.
- Current gate, updated 1 August 2026: the user authorized the strengthening
  programme on 30 July 2026: V3 reasoner scaling to 100k/250k (E1), a
  SigLIP-B/16 frozen-encoder-swap experiment on the global-embedding path (E2)
  and a 1000-answer vocabulary experiment (E3), in that order after the
  S1-S3 correction and P0-P3 preparation packets. All of E1-E3 are complete
  (E1 and E2 on 31 July, E3 on 1 August 2026); the strengthening programme
  is finished. On 1 August 2026 the user additionally authorized E7a only
  (the efficiency Pareto and end-to-end cost analysis, an evaluation-only
  measurement over stored checkpoints, completed the same day) and
  restated that E4 (five-seed reasoner completion), E5 (parameter-matched
  CLIP versus SigLIP), F1 and F2 are not authorized and that clean-test
  labels must not be accessed. The next steps are the model-list freeze
  (F1) and only then the blinded clean-test evaluation (F2), both
  requiring explicit user authorization.
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
  Narrow exception, approved by the user on 2 August 2026 for the E8A/E8B/E9
  programme only: greedy short-answer generation is permitted in the E8B R2
  readout (constrained to the closed answer vocabulary by a prefix trie), the
  E8B R3 readout (free but bounded by a fixed maximum token count) and the E9
  compact-VLM evaluation. Sampling, temperature search, prompt search,
  long-form generation and generation on the clean test before F1/F2
  authorisation all remain forbidden. Classification remains the primary task.
- Main dataset: a subset of GQA. VQA v2 is optional and only after GQA works.
- Encoders: frozen CLIP for both the image and the question, never trained. One
  model is used for both, so the two vectors share the same space.
  A single frozen SigLIP-B-16 encoder-swap experiment on the global-embedding
  path was approved by the user on 30 July 2026 (E2); all encoders remain
  frozen.
  Narrow exception, approved by the user on 2 August 2026 for the E8A
  experiment only: E8A creates an explicit, experiment-specific exception to
  the rule that the image and question representations originate from the same
  CLIP model and share a pretrained space by construction. In E8A the image
  tokens come from frozen CLIP ViT-B/32, the question tokens come from a frozen
  small language model (SmolLM2-135M base, SmolLM2-360M base, or FLAN-T5-small),
  their original representation spaces are different, and a trainable linear
  projection maps the language-model states into the common 512-dimensional
  reasoner interface. That projected interface is a learned common width and is
  not claimed to be a naturally shared pretrained embedding space. Every
  disclosure of an E8A result must state this. The freezing guarantee is
  unchanged: every encoder and every language model stays frozen and is never
  trained or fine-tuned.
- Trainable part: lightweight heads over frozen CLIP features. V1/V2 use the
  MLP heads; the approved V3 central contribution is the lightweight
  question-conditioned latent-query reasoner over cached token-level
  features, with cached-token training as the primary pipeline and raw-path
  equivalence and efficiency measured separately. The encoders stay frozen.
  Narrow exception, approved by the user on 2 August 2026 for the E8A/E8B/E9
  programme only. What stays trainable: the existing latent-query reasoner
  under the reviewed recipe, and the existing classifier or readout head where
  applicable. What is newly trainable: one experiment-specific linear
  projection per configuration, mapping frozen language-model hidden states
  into the 512-dimensional reasoner width (E8A) or the 32 reasoner latents into
  the language model's embedding width (E8B). The projection is the only newly
  introduced trainable component in the language-model paths; it is not the
  only trainable component in the pipeline. Also authorised as lightweight,
  experiment-local heads over frozen features: question-only classifiers over
  frozen SmolLM2 or FLAN-T5-small question features (arms A4 at 135M and A5 at
  360M), and global-fusion heads over frozen CLIP image features concatenated
  with pooled frozen SmolLM2 question features (arms A7c at 135M and A8c at
  360M). The user decided on 2 August 2026 that A5 and A8c are unconditional
  core controls, run at 40k and at 250k with seeds 0/1/2; no result-dependent
  trigger governs them, and the earlier "conditionally authorised 360M" and
  result-dependent wordings no longer govern. Also authorised in both branches:
  deterministic random-initialised SmolLM2-135M and SmolLM2-360M, each with
  architecture and tokenizer configuration identical to its pretrained
  counterpart, weights created from a pinned seed, fully frozen, used as the
  non-pretrained causal controls and never tuned separately from the pretrained
  models; and one lightweight
  language-model-free readout that consumes all 32 reasoner latents through a
  pre-registered attention-pooling or lightweight nonlinear readout into the
  same fixed-vocabulary classifier, parameter-counted and efficiency-measured,
  existing to isolate the wider latent interface from pretrained-language-model
  effects. Neither clause authorises unrestricted readout architecture search.
  E9 compact VLMs are inference-only and have no trainable component.
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
- Approved by the user on 2 August 2026 for the E8A/E8B/E9 programme only:
  five pretrained checkpoints plus TWO deterministic random-initialised model
  controls, each bound to the roles listed below and no others.
  SmolLM2-135M base: E8A semantic encoder and E8B readout.
  SmolLM2-360M base: E8A semantic encoder and E8B readout.
  FLAN-T5-small: E8A cross-family semantic-encoder sensitivity only; FLAN-T5
  answer-readout experiments are NOT authorised.
  SmolVLM-256M-Instruct: E9 evaluation-only compact-VLM baseline.
  SmolVLM-500M-Instruct: E9 evaluation-only compact-VLM baseline. SmolVLM
  training or fine-tuning is NOT authorised.
  Random-initialised SmolLM2-135M and random-initialised SmolLM2-360M,
  extended by the user on 2 August 2026: authorised as non-pretrained causal
  controls in BOTH branches, as E8A question encoders and as E8B answer
  readouts, unconditionally at 40k and 250k with seeds 0/1/2. Each uses the
  identical architecture and tokenizer configuration as its pretrained
  counterpart, deterministic weights from a pinned seed, no downloaded
  pretrained weights, all parameters frozen, and no independent hyperparameter
  tuning. Neither is a downloaded checkpoint and neither has a model-card
  revision; each is pinned by its seed and by the configuration it is built
  from. The primary within-size causal comparisons are pretrained minus random
  at the SAME size; a random 135M model is never the causal control for a
  pretrained 360M model.
  Exact official model IDs, pinned revision SHAs, licences, tokenizer hashes
  and configuration hashes must be verified and recorded in the task packets
  before any model is downloaded. All are frozen; none is trained or
  fine-tuned.

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
