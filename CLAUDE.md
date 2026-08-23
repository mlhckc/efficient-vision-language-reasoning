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
  every stored head plus both frozen encoders. PARTLY SUPERSEDED, and the
  boundary is binding. Its ADDITIVE end-to-end columns
  gpu_encoder_plus_head_ms, full_pipeline_ms and amortised_ms, and every
  Pareto front derived from those sums, are SUPERSEDED: they are sums of
  stage medians timed in isolation, no serial pass was timed in E7a, and
  they must not be used as current end-to-end latency evidence, in any
  comparison, ranking or "cheaper" claim. They are NOT replaced by E7b
  values; substituting one experiment's number into another's sentence
  would manufacture a comparison neither made. For measured end-to-end
  latency see E7b. Retained and valid: the isolated component latencies
  (CLIP 2.2510 ms image + 1.7309 ms text on GPU plus 2.295 ms CPU decode,
  against 0.0162-0.0469 ms for the global heads), the peak-memory
  components, the parameter counts and the accuracy column. Bounded
  statement on those fields only: the top-1000 product head at 250k is
  more accurate (0.4904 against 0.4594 raw-distribution) than the
  21.1M-parameter reasoner, using 16 times fewer parameters. Parameter
  count is a poor latency proxy (fusion is 48% slower than concat_wide at
  equal parameters). Supersedes the V1 stage-5 and src/efficiency.py-derived
  latencies. Development results only; see docs/experiments/e7a_efficiency.md.
- E7b status, updated 14 August 2026 and BINDING. E7b measured the serial
  end-to-end batch-1 query on otter155 and is the authoritative end-to-end
  latency evidence. THREE of its stored fields are WITHDRAWN and must not
  appear in any dissertation number, figure, table, comparison, Pareto axis
  or GPU-memory footprint claim. `peak_allocated_mib` and `peak_reserved_mib`
  are INVALID_WITHDRAWN, cause class MEASUREMENT_BOUNDARY_DEFECT: the
  historical measurement window did not isolate the intended batch-1 serial
  inference envelope. `cold_first_query_ms` is SUPERSEDED_WITHDRAWN, cause
  class GRAD_MODE_STANDARDISATION: the historical cold queries ran on a
  grad-enabled path and the accepted source repair standardised the cold
  query to torch.no_grad(), so the stored values do not measure the current
  path. The two causes are distinct and must not be conflated. RETAINED and
  unchanged: warm serial median latency, the across-pass spread, the
  raw-distribution accuracy on the common denominator, the trainable,
  total-loaded and resident-frozen parameter counts, and the primary Pareto
  frontier (concat, fusion, vocab1000_product), because neither withdrawn
  field is a frontier axis. NO replacement value exists and none may be
  estimated, inferred, back-calculated or substituted from another node.
  Remeasurement is NOT authorised and has not run: the historical node was
  otter155, the current project copy is otter159, and the fusion bridge
  control failed its pre-registered 10 per cent tolerance at -18.69 per cent,
  so cross-node substitution is forbidden and no adjustment factor is
  permitted. The withdrawal is published as a field-level overlay rather than
  an edit, because every artefact carrying the withdrawn fields is byte-pinned
  by the closed VE-0 and VE-1 packets. Precedence, binding: (1)
  results/closure/e7b_evidence_supersession.json controls named E7b field
  validity; (2) results/ve0/supersession_map.json controls unnamed
  artefact-group status; (3) historical stored evidence_status is lowest
  precedence. Historical numbers remain readable only as provenance. E7b is
  CLOSED: the independent review returned
  E7B_CANONICAL_SUPERSESSION_REVIEW_PASS on 14 August 2026. The overlay's own
  status block still reads OPEN and its f1 field still reads BLOCKED, because
  it was written before that review returned and is byte-pinned; those
  lifecycle fields are superseded by
  results/closure/pre_f1_status_supersession_20260814.json, which controls
  lifecycle and status only and changes no field validity. F1 and F2 remain
  unstarted and unauthorised. See docs/experiments/e7b_serial_efficiency.md.
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
  against SmolVLM-500M's 152.854 ms: SmolVLM-500M was approximately 24.7x
  slower in warm serial latency under the same-node contextual protocol.
  That is a latency statement only; no energy, power, carbon, monetary-cost
  or general computational-cost claim follows from it. The E8B bridge
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
- E10 Phase 1 (completed 10 August 2026), after the independent Phase-0 review
  returned PASS with one execution-readiness finding. The user set the two E10
  resource constants and the retry policy; nothing scientific was authorised.
  Binding: `E10_PER_CELL_WALL_CLOCK_HOURS = 12.0` is a HARD executable
  WHOLE-CELL operational halt, not a projection field and not a training-only
  bound. The same deadline governs setup, training, the development
  evaluations, the G14 diagnostics, the final R1 evaluation and result
  publication. Every stage runs inside a guarded wrapper; a `completed`
  outcome requires every stage to have closed inside the wall; finalisation
  re-derives the deadline on every exit path, including a normal exit in which
  nothing checked; and accounting plus ledger validation independently refuse
  a `completed` cell at or above the wall. Equality halts in every layer: the
  guard stops at `now >= deadline` and both accounting backstops stop at
  `occupancy >= wall`. The guard does NOT interrupt an in-flight
  operation: the guarantee is that a crossing can never publish a successful
  scientific result, can never be charged as completed, and can never continue
  authorised scientific work under that permit. The independent Phase-1 review
  returned CHANGES_REQUIRED because the first implementation enforced the wall
  only at guarded interactions; that defect was reproduced and repaired, and
  the frozen recipe was not touched. `E10_PER_IDENTITY_CEILING_HOURS = 40.0` is E10-specific and may
  never be more permissive than the programme-wide bound: every gate uses
  min(E10_PER_IDENTITY_CEILING_HOURS, PER_MODEL_IDENTITY_CEILING_HOURS),
  computed once in e10_common.effective_identity_ceiling_hours, and no E10
  module declares its own copy of the programme-wide value. NO AUTOMATIC
  RETRY: the earlier nominal one-forced-retry behaviour is withdrawn because
  the Gate-1 stress projection leaves post-retry margins of 0.040641 and
  0.007927 GPU-hours; a failed scientific cell now requires a fresh explicit
  user or reviewer authorization record, single-use and bound to one cell,
  which no project code may write. A5/A8c keep OPEN_UNQUANTIFIED_RESERVE, no
  numeric reserve was invented, and if they use the same pinned pretrained
  SmolLM2-360M identity their future compute is charged against the same
  authoritative frozen-model identity accounting. Storage keeps its
  qualification: physical fit established, formal project allocation
  unestablished. The reviewer's finding, that the E10 recipe drops the
  inherited E8B wall_clock_halt_hours field and no runner re-imposed a wall, is
  repaired outside the frozen recipe: the wall is carried by a signed
  process-bound cell permit and enforced by ScientificCellGuard, so all twelve
  recipe digests are identical to the Phase-0 recipe contract and the frozen
  matrix, seeds, scales and pair preservation are unchanged. Setting the
  constants authorises nothing: `E10_TRAINING_AUTHORIZED` is still `None` and
  `run core-cell B4 train_40k 0` still refuses. See
  docs/experiments/e10_phase1_guardrails.md.
- E10 Phase 2 (completed 10 August 2026) implemented the scientific pipeline
  behind the reviewed Phase-1 guardrails, and executed none of it. New:
  labelled train/dev loading against pinned manifest and token-store digests,
  B4/B4r construction with the audited G13 order and the frozen parameter
  counts, the 22-epoch loop with no early stopping, the pinned cadence
  (1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 19, 22) read from the frozen contract,
  scheduler horizon 100 epochs, epoch 22 as the sole primary checkpoint and
  result, the ported G14-FP32 scorer-identity gate at both stages, the final
  canonical R1 evaluation with G9/G10/G11 and the pinned G21 scoring, atomic
  publication with fail-closed cell reconciliation, and the frozen
  post-matrix analysis that refuses anything short of the complete,
  pair-preserved, accounting-reconciled twelve-cell set. Every stage runs
  inside ScientificCellGuard and guarded_optimizer_step is the only path to
  an optimizer step. Three deliberate differences from E8B are recorded: the
  pinned cadence, no G8 overfit gate (the frozen Gate-1 per-cell budget does
  not fund it), and no resume (a leftover training state is a hard refusal).
  The independent Phase-2 review confirmed as non-blocking and
  contract-consistent that G8 is not part of the frozen E10 protocol, that
  final R1 means one normal-condition final R1 evaluation, that no executable
  E10 visual-intervention set is registered or funded, and that E10 produces
  no new visual-reliance evidence. It returned CHANGES_REQUIRED on two
  blocking defects, both repaired in revision 2: strict determinism was not in
  force before the guard's first stage, so a fresh-process cell could not open
  setup, and the difference in differences was reported at half its magnitude
  because the composite contrast averaged its two sides. A real end-to-end run
  of the production runner on synthetic inputs now reaches a reconciled
  COMPLETE cell from a non-strict process state. Disclosed and non-blocking:
  G11 repeats the final development pass, so a cell makes 14 full-development
  passes against the 13 the Gate-1 projection budgeted, about 0.75 GPU-hours
  over the matrix, with no wall or ceiling consequence. Development-set work only,
  and nothing scientific ran: `E10_TRAINING_AUTHORIZED` is still `None`,
  `run core-cell B4 train_40k 0` still refuses, zero optimizer steps, zero
  checkpoints, zero GPU-hours charged to any core cell. See
  docs/experiments/e10_phase2_scientific_pipeline.md.
- E10 Phase-2 review CLOSED on 11 August 2026. The independent reviewer
  returned PASS on the repair patch at HEAD 45986fc, discharging both blocking
  findings and separately verifying the production synthetic end-to-end run,
  production G10 enforcement, provenance and record regeneration,
  non-regression, the unset scientific authorisation, the absence of any real
  scientific execution and the intact clean-test embargo. Its conclusion:
  explicit scientific authorisation is now the only remaining closed gate
  before the frozen 12-cell E10 matrix. The closure is recorded in
  results/experiments/e10_capacity_360m/phase2_review_closure_20260810.json and
  changed no scientific source, test, configuration, recipe, resource constant,
  result implementation or frozen record. IT AUTHORISES NOTHING: the E10
  scientific core is still unauthorised, `E10_TRAINING_AUTHORIZED` is still
  `None`, and starting the twelve-cell matrix requires a separate explicit
  user decision that has not been given.
- E10 Phase 3 (completed 11 August 2026; independent review PASS, packet
  closed at ce2495b, closure recorded at
  results/experiments/e10_capacity_360m/phase3_review_closure_20260811.json)
  repaired the self-defeating authorisation gate and changed no scientific
  behaviour. The defect, reproduced at the Phase-2 closure HEAD: because
  e10_common.py is inside SOURCE_PATHS, setting `E10_TRAINING_AUTHORIZED` to
  the approval token moved the live source digest (809320d7 to 214fcc39) and
  made phase1-verify, phase2-verify and the core cell fail on stale
  provenance, so the only implemented authorisation mechanism could not be
  used without invalidating the reviewed source it was meant to open.
  BINDING FROM NOW ON: `E10_TRAINING_AUTHORIZED` is a LEGACY REFUSAL SENTINEL
  and must remain `None`; any other value is itself a refusal. Scientific
  authorisation is granted ONLY by one immutable EXTERNAL grant record in
  shared state, outside the repository and outside SOURCE_PATHS and
  CONFIG_PATHS, validated by the single canonical validator
  e10_common.validate_core_authorization_grant. It grants exactly the frozen,
  pair-preserved twelve-cell B4/B4r matrix or it refuses; there is no partial
  authorisation, no environment-variable route, no config-mutation route and
  no source-edit route. No project source may write the grant. Creating one
  provably does not move the source digest. A narrow Phase-3 amendment carries
  every approved Phase-0/1/2 record forward byte for byte. Nothing scientific
  ran: no grant exists, `run core-cell B4 train_40k 0` still refuses, zero
  cells, checkpoints, accuracies and core GPU-hours. The independent review
  took two rounds: round 1 returned CHANGES_REQUIRED on one blocking defect,
  that the pre-Phase-3 always-refuses verifier invariant made all three
  verifiers fail once a valid grant existed; the repair made the invariant
  grant-aware and round 2 returned PASS. All three verifiers now pass with no
  grant, with a valid grant and after a grant is removed. The reviewer's
  conclusion is that Phase 3 is technically READY for the creation of the
  explicit real user scientific authorisation grant. THE CLOSURE AUTHORISES
  NOTHING: creating that grant is a separate explicit user decision that has
  not been given. See docs/experiments/e10_phase3_authorization_binding.md.
- E10 core matrix EXECUTED (authorised by the user 11 August 2026, completed
  13 August 2026). The one real immutable external core-authorisation grant
  was created out of band in shared state, grant id 7ce353d7ec9d93b6, and
  provably did not move the source or config digest. All twelve frozen
  B4/B4r cells ran in the frozen pair-preserving order, first attempt, zero
  automatic and zero authorised retries, no gate halt, no wall approach: the
  longest cell used 54.3 per cent of the 12.0 hour wall. Development results
  only. B4 (pretrained frozen SmolLM2-360M) reaches 0.52636 +/- 0.00466 at
  train_40k and 0.59407 +/- 0.00831 at train_250k; B4r (architecture-matched
  random) reaches 0.53319 +/- 0.00473 and 0.59645 +/- 0.00343. Both scale
  effects are positive and directional (B4 +0.06771 [0.05800, 0.07738], B4r
  +0.06326 [0.05287, 0.07296]). Neither pretraining effect is directional
  (-0.00683 at 40k, -0.00238 at 250k, both intervals containing zero, seeds
  disagreeing in sign at both scales), and the difference in differences,
  (B4-B4r)@250k - (B4-B4r)@40k, is +0.00445 [-0.00709, 0.01641], also not
  directional. NO RELIABLE POSITIVE pretrained-over-random advantage was
  detected at either scale; both pretraining-effect intervals include zero and
  establish neither equivalence nor the absence of an effect. Training-set
  size is what moves accuracy. E10 qualitatively reproduces E8B's finding that
  this answer-side frozen-SLM interface shows no reliable positive
  pretrained-over-random advantage, now at 360M; it does NOT reproduce E8B's
  directional negative 40k effect (E8B B3-B2 at 40k was -0.02463 with interval
  [-0.03333, -0.01619] excluding zero and all three seeds negative, against
  E10's uncertain -0.00683 with mixed seed signs). The 135M-to-360M comparison
  is whole-system capacity sensitivity, not an isolated causal LM-size effect:
  trainable capacity also moves, 21,343,808 to 21,540,800 parameters, about
  0.923 per cent, because the projection width follows the hidden size. The sd
  figures above are the sample standard deviation across the three training
  seeds (ddof=1) and are not confidence intervals; the bracketed intervals are
  image-clustered bootstrap intervals that condition on the fixed trained seed
  set and carry evaluation-sampling variation, not training-seed variation.
  Cost: 51.07838121415333 core GPU-hours, 25.56327 on the pretrained and
  25.54783 on the random identity, both against the 40.0 hour ceiling, 2.72
  per cent above the Gate-1 projection of 49.72765441864283 and 1.18 per cent
  above the adjusted 50.48143730636344 once the known fourteenth development
  pass is added, a residual of 0.5969439077898893; core actual is 82.17 per
  cent of the corresponding CORE stress budget of 62.15956802330354, which is
  not the calibration-inclusive 62.19228220657687 total. KNOWN DEFECT, for the
  reviewer and REPAIRED IN R3 on 13 August 2026 under separate explicit user
  authorisation: the pre-R3 terminal state was phase1-verify PASS,
  phase2-verify FAIL, phase3-verify FAIL (phase 3 not only transitively:
  phase3.authorization_contract had its own direct stale
  assert_no_scientific_cells call at phase3.py:438) and 15 of 97 E10 tests
  failing, all lifecycle and state-coupling failures rather than scientific
  regressions, principally through mirror_governance() copying the real
  completed governance tree into temporary test trees. R3 made
  phase2.pipeline_contract and the phase3 call execution-aware behind one
  narrow helper, e10_common.core_execution_state, with three explicit
  lifecycle states (PRE_AUTHORIZATION, AUTHORIZED_INCOMPLETE,
  AUTHORIZED_COMPLETE) and terminal acceptance only on positive proof of the
  exact completed matrix via assert_exact_completed_matrix. The strict
  pre-execution invariants are unchanged and the record writers still use
  them verbatim. All three verifiers now PASS, all 100 E10 tests and all
  twelve non-E10 modules pass, and tests/run_all.py exits 0. R3 intentionally
  moved the live source digest from 857ec44e to cdc57315 because the repaired
  files are inside SOURCE_PATHS; the config digest 4b9fa43c is unchanged. The
  completed evidence is carried across that move by the immutable
  post-execution binding amendment
  results/experiments/e10_capacity_360m/post_execution_binding_amendment_20260813.json,
  the newest link in _amendment_chain, which binds the grant, the twelve cell
  records, their checkpoints and per-row arrays and the frozen analysis by
  name and by bytes. No historical record was rewritten and the new digest was
  not backdated into any of them. GRANT TERMINAL POLICY, binding: the original
  grant stays historically valid as provenance for the twelve completed cells
  and is operationally SPENT; every authorised cell now refuses with
  "operationally SPENT", unauthorised arms/scales/seeds still refuse at the
  entry gate, retries remain zero, completed cells remain immutable, and any
  further scientific work requires a fresh explicit user authorisation and a
  new valid grant. R3 changed no accuracy, contrast, interval, checkpoint,
  per-row array or result record. See docs/experiments/e10_core_matrix.md, the
  immutable addendum
  results/experiments/e10_capacity_360m_core/e10_core_execution_correction_20260813.json
  and the R3 record
  results/experiments/e10_capacity_360m_core/e10_core_r3_terminal_state_20260813.json.
  R3.1 (13 August 2026, explicit user authorisation) then repaired the one
  governance blocker the independent R3 review returned, G-R3-1, with zero
  scientific and zero execution blockers: the operationally-SPENT refusal was
  keyed only on the whole-matrix proof, so losing or altering ONE untracked
  local binary degraded that proof and re-opened every already-completed cell
  at the entry gate, letting authorize_cell_execution mint a signed permit for
  a finished cell. Reproduced before repair. The fix adds
  e10_common.cell_previously_completed and a per-cell immutability guard in
  assert_core_entry_authorized that refuses on EITHER a published result
  record OR a completed ledger charge for that exact cell, independent of the
  whole-matrix state, while keeping the full-matrix SPENT refusal. It is NOT a
  blanket incomplete-state refusal: a cell that has never run is still
  admitted, which the original execution lifecycle requires and a regression
  test pins. One documented exception: charge_core_cell_hours passes
  purpose="accounting" because accounting runs after publication; it is not a
  bypass, because accounting also requires a signed process-bound permit.
  R3.1 moved the live source digest again, cdc57315 to e666f101, config
  4b9fa43c unchanged, and added a second immutable link
  post_execution_binding_amendment_r31_20260813.json that preserves the R3
  amendment by exact bytes, giving the chain original execution binding ->
  R3 post-execution amendment -> R3.1 live-binding amendment. 104 E10 tests
  and 12 non-E10 modules pass, run_all exits 0, and no accuracy, contrast,
  interval, checkpoint or per-row hash changed. See
  results/experiments/e10_capacity_360m_core/e10_core_r31_spent_grant_20260813.json.
  E10 is CLOSED at E10_PASS, as recorded in the frozen VE-0 supersession map
  (results/ve0/supersession_map.json): "E10 is CLOSED at E10_PASS and must not
  be reopened. Its grant is operationally SPENT, its completed cells are
  immutable and its live source digest is unchanged." The earlier note that
  R3.1 was review_requested and that no final E10 PASS had been declared is
  historical lifecycle state and is superseded by
  results/closure/pre_f1_status_supersession_20260814.json. The clean test
  remains embargoed and F1/F2 remain unstarted and unauthorised.
- Pre-F1 lifecycle status, updated 14 August 2026 and BINDING. Five review
  packets are CLOSED with independently accepted verdicts: E7b
  (E7B_CANONICAL_SUPERSESSION_REVIEW_PASS), VE-0 (VE0_PASS, with the
  13 August 2026 amendment accepted at VE0_AMENDMENT_PASS), VE-1 (VE1_PASS,
  after a first round of VE1_CHANGES_REQUIRED), VE-2 (VE2_PASS) and E10
  (E10_PASS). Several closed artefacts still carry the status they had on the
  day they were written, before their reviews returned: the E7b overlay's
  status block reads OPEN with f1 BLOCKED, and the byte-pinned
  docs/experiments/ve0_evidence_contract.md still says review_requested. Those
  statements are historical lifecycle state, they are not edited, and they are
  superseded by results/closure/pre_f1_status_supersession_20260814.json,
  built by experiments/closure/build_pre_f1_status_supersession.py. Precedence,
  binding: that record is highest for LIFECYCLE and STATUS; it controls NO
  field validity, which stays with (1)
  results/closure/e7b_evidence_supersession.json for named E7b fields and (2)
  results/ve0/supersession_map.json for unnamed artefact-group status. It
  changes no accuracy, latency, interval, parameter count, Pareto membership,
  figure, table or raw artefact, reinstates no withdrawn field and authorises
  nothing. F1 is UNSTARTED and unauthorised and is NOT declared ready by that
  record; F2 is unstarted and unauthorised; the clean-test embargo is
  unchanged. Each closed evidence layer ships its own CPU-only runner
  (tests/run_closure.py, run_ve0.py, run_ve1.py, run_ve2.py,
  run_e7b_supersession.py) because tests/run_all.py is inside E10's frozen
  SOURCE_PATHS and must not gain an import.
- Pre-F1 evidence metadata, updated 14 August 2026 and awaiting independent
  review. The candidate field-level authority is
  results/closure/pre_f1_evidence_metadata_repair_20260814.json, built by
  experiments/closure/build_pre_f1_evidence_metadata_repair.py. It corrects
  exactly nine E8B B1-bearing checkpoint-selection records: five B1-only rows
  are BEST_ON_DEVELOPMENT and four B1-versus-B2/B3 contrasts are
  MIXED_WITHIN_ROW; B2/B3 remain FIXED_EPOCH_22. It also controls five
  source-proven presentation-precision fields, the E7a tracked locator and
  VE1-FIG-04-FULL appendix placement. Closed VE0/VE1 bytes and their legacy
  builders remain historical and must be read through this successor. It
  changes no other numerical field, reinstates no E7a/E7b withdrawal and
  declares no F1/F2 readiness or authorisation. See
  docs/experiments/pre_f1_evidence_metadata_repair.md.
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
  Deferral decision, 23 August 2026: the seven unexecuted E8A arms (A2,
  A2r, A4, A5, A7c, A8c and AF/FLAN-T5-small) are authorised-but-unrun
  and deferred from the current MSc execution scope by explicit user
  decision, made before F1 and F2. No canonical or accepted
  experimental result exists for any of them, and the scoped
  execution/result audit found no execution artefact for them, so the
  deferral is not based on observing their experimental outcomes. The
  A5/A8c unconditional-core sentences above are preserved as the
  historical record; only their operational execution obligation is
  superseded, for the current MSc scope. No arm is withdrawn or
  cancelled. Before F1, an explicit user authorization — including
  authorization made in response to supervisor feedback — may reopen an
  arm or required pair, preserving pair integrity (A2 with A2r). See
  docs/experiments/pre_f1_unrun_arms_disposition.md.
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
     CURRENT METADATA RESOLUTION: five B1-only evidence rows are
     BEST_ON_DEVELOPMENT; the four B1-versus-B2/B3 rows explicitly carry both
     primitive selection classes and are not checkpoint-selection matched.
     Resolve the frozen VE0/VE1 carrier fields through
     results/closure/pre_f1_evidence_metadata_repair_20260814.json. Frozen
     FIG-08 PDF/PNG renders must additionally be paired with that record's
     hash-guarded effective virtual note; the corrected text is not embedded
     in their unchanged historical bytes.
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
- Clean-test governance, updated 14 August 2026 and BINDING. Required wording,
  to be used verbatim wherever the embargo is described: "The clean-test
  contents were never inspected or used for development, model selection, or
  reporting decisions. Mechanical byte access occurred in two documented
  governance incidents, on 13 and 14 August 2026." Both incidents are
  classification B, mechanical byte access and nothing more, and both must be
  identified separately:
  13 August 2026 — an independent reviewer ran an integrity-hash command over
  the target file, which read its bytes.
  14 August 2026 — during the E7b canonical-supersession implementation, an
  overly broad dependency-mapping scan walked the whole project root and read
  every .json, .py, .md, .csv and .txt file, because the traversal was not
  scoped away from data/.
  For BOTH: contents were not inspected; no row, label, distribution or
  prediction was examined; nothing informed development; nothing informed
  model selection; nothing informed reporting. They are governance incidents,
  not test-informed scientific selection, and neither invalidates a result nor
  requires remediation of the blinded evaluation.
  PROHIBITED wording: the project must not claim the clean test was never
  accessed, because that claim would be false, and must not use any phrasing
  implying a single total access. Do not open, read, hash, stat, glob, search,
  parse or traverse the target, and do not "verify" the embargo by touching
  it.
  CORRECTIVE RULE, binding: a repository-wide scan must exclude data/ before
  reading or traversing candidate files, not filter it afterward. Scoping the
  traversal is the control; filtering the results is not, because by then the
  bytes have already been read. This is how both incidents happened.
  Canonical records: results/closure/e7b_evidence_supersession.json under
  clean_test_governance, and
  results/closure/pre_f1_status_supersession_20260814.json. Closed VE/closure
  wording remains historical and is not a current disclosure surface; the
  current prose disclosure is in docs/REPRODUCIBILITY.md and the editable
  reader-facing documents named by
  results/closure/pre_f1_evidence_metadata_repair_20260814.json.
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
  result-dependent wordings no longer govern. By a further explicit user
  decision of 23 August 2026, the unexecuted arms A4, A5, A7c and A8c
  are deferred from the current MSc execution scope,
  authorised-but-unrun and not withdrawn; see
  docs/experiments/pre_f1_unrun_arms_disposition.md. Also authorised in
  both branches:
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
