# Dissertation handoff

This document is for the person or AI assistant who helps write the
dissertation from this repository. It is self-contained: identity, scope,
evidence, claim boundaries, and where every number lives. Read it together
with [EXPERIMENTS_AND_RESULTS.md](EXPERIMENTS_AND_RESULTS.md) (detailed
results), [ARCHITECTURE.md](ARCHITECTURE.md) (model families),
[SCIENTIFIC_FINDINGS.md](SCIENTIFIC_FINDINGS.md) (findings) and
[PROJECT_STATUS.md](PROJECT_STATUS.md) (state).

# Dissertation identity

- Title: "Efficient Vision-Language Reasoning with Small Language Models".
- Degree: MSc Artificial Intelligence, University of Surrey.
- Supervisor: Prof. Miroslaw Bober.
- Repository: this one, branch v2-protocol.

# Research question

Can Visual Question Answering be done efficiently by reasoning in embedding
space — frozen small encoders plus a tiny trainable head — instead of
running a large autoregressive vision-language model?

# Aim

Measure, under controls, what lightweight systems built on frozen
multimodal representations can and cannot do on GQA closed-set VQA, in both
accuracy and measured cost, and locate where their limits come from.

# Objectives

1. Build a defensible evaluation protocol (image-disjoint dev, embargoed
   clean test, nested training scales, pool-only vocabulary). Done: v2_00.
2. Establish controlled multi-seed baselines and decompose the fusion
   advantage into capacity and features. Done: v2_02-v2_04.
3. Localise gains and weaknesses by question type and reasoning depth, and
   test visual reliance. Done: v2_05-v2_06.
4. Measure the effect of data scale. Done: v2_07.
5. Design and test a token-level latent-query reasoner against the global
   heads. Done: v3_00-v3_03.
6. Test representation-side levers: a second frozen encoder (E2) and a
   larger answer space (E3). Done.
7. Measure true end-to-end cost and build an accuracy-latency frontier.
   Done: E7a (components), E7b (serial, authoritative).
8. Test frozen small language models on the question side (E8A) and answer
   side (E8B), with random-initialised causal controls, and answer-side
   capacity sensitivity (E10). Done.
9. Place a compact integrated VLM in context, evaluation only. Done: E9.
10. Freeze the model list (F1) and run the blinded clean-test evaluation
    (F2). NOT DONE: F1 unstarted, F2 unstarted and unauthorised.

# Final scientific scope

Discriminative closed-set VQA on a GQA subset. Frozen encoders throughout;
trainable components are lightweight heads only. Narrow approved exceptions:
bounded greedy generation in E8B R2/R3 and E9; the E8A cross-space interface
with its mandatory disclosure. Locked out of scope: training or fine-tuning
any VLM or LM, unfreezing encoders, new datasets, sampling or prompt search,
semantic/embedding/LLM-judged scoring in any primary metric.

# Dataset and protocol

- GQA balanced release. Development: 777 images, 10,004 raw questions,
  7,714 in-vocabulary (top-100). Training pool: 71,363 images, 724,074
  eligible questions; nested subsets train_40k/100k/250k (row-for-row
  prefixes). Clean test: 8,013 questions on 972 images, structural counts
  only, targets embargoed.
- Vocabulary: top-100 from the training pool only
  (`data/v2/answer_vocab_v2.json`); E3 extends to top-1000 with the first
  100 entries gated identical. V1 label indices are incompatible (11 index
  differences) and must never be mixed with V2 manifests.
- Seeds: V2 family uses five seeds (0, 1, 2, 3, 42); V3/E8A/E8B/E10 use
  three (0, 1, 2). The V2 protocol RNG streams derive from seed 42.
- Primary metrics: strict raw exact match and one pinned VQA-style
  normalised exact match (G21). On the top-100 vocabulary the two provably
  coincide (zero normalised collisions).
- Denominators, three, never silently mixed: 7,714 rows = top-100
  in-vocabulary development accuracy; 9,823 rows = top-1000
  in-vocabulary development accuracy; 10,004 rows = raw-distribution
  development accuracy (with coverage-adjusted raw-distribution
  accuracy for cross-vocabulary comparisons). Top-100 and top-1000
  rows are never paired.

# System evolution

V1 prototype (legacy, biased evaluation) -> V2 protocol and controlled
global-head family -> V3 token-level latent-query reasoner -> E2/E3
representation-side levers -> E7a/E7b cost measurement -> E8A/E8B frozen
SLM interfaces -> E9 compact-VLM context -> E10 answer-side capacity ->
statistical closure and evidence freeze (VE-0/VE-1/VE-2) -> pre-F1
lifecycle and metadata records. The narrative logic is in
[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).

# Architecture families

Four internal families — global fusion, token-level reasoner, question-side
SLM, answer-side SLM — plus the external compact-VLM class. Details and
diagrams: [ARCHITECTURE.md](ARCHITECTURE.md). Do not present them as one
architecture, and do not attribute one family's results to another.

# Complete experiment map

See the master table in
[EXPERIMENTS_AND_RESULTS.md](EXPERIMENTS_AND_RESULTS.md). One-line index:
V1 (legacy baselines and fusion); v2_00 (protocol); v2_01 (embeddings,
zero-shot floor); v2_02 (five-seed baselines); v2_03 (parameter matching);
v2_04 (interaction ablation); v2_05/b/c (types, steps, clustered
intervals); v2_06 (visual reliance); v2_07 (scaling); v3_00 (token
stores); v3_01 (reasoner 40k, negative); v3_02a (references, statistics
repair, diagnostics); v3_03/E1 (reasoner scaling); E2 (SigLIP); E3
(top-1000); E7a (components, partly superseded); E7b (serial efficiency,
authoritative); E8A (question-side SLM); E8B (answer-side SLM, 135M); E9
(compact VLM context); E10 (answer-side 360M).

# Canonical result summary

In-vocabulary dev accuracy (7,714 rows) unless noted. 40k / 100k / 250k
where applicable; "—" means the cell was not run.

| System | 40k | 100k | 250k |
|---|---|---|---|
| majority reference | 0.2247 | — | — |
| zero-shot CLIP (image-only) | 0.0770-0.0795 | — | — |
| image_only | 0.2344 | — | — |
| question_only | 0.4580 | 0.4829 | 0.4977 |
| concat | 0.5240 | 0.5508 | 0.5786 |
| fusion | 0.5384 | 0.5631 | 0.5823 |
| concat_wide (param-matched) | 0.5326 | — | — |
| fusion_narrow (param-matched) | 0.5334 | — | — |
| product_576k (ablation) | 0.5335 | 0.5636 | 0.5824 |
| difference_576k (ablation) | 0.5334 | — | — |
| direct_linear | 0.4813 | — | — |
| reasoner (21.1M) | 0.5422 | 0.5706 | 0.5958 |
| SigLIP concat (E2) | 0.5394 | — | 0.5918 |
| SigLIP fusion (E2) | 0.5532 | — | 0.5939 |
| E8A A0p / A1 / A1r | 0.54045 / 0.52861 / 0.49041 | — | 0.59446 / 0.57316 / 0.52480 |
| E8B B1 / B2 / B3 | 0.54507 / 0.53392 / 0.50929 | — | 0.59394 / 0.60086 / 0.59610 |
| E10 B4 / B4r | 0.52636 / 0.53319 | — | 0.59407 / 0.59645 |

Raw-distribution accuracy at 250k (10,004 rows): top-1000 product 0.4904,
top-1000 fusion 0.4866, top-1000 concat 0.4822, reasoner 0.4594, top-100
fusion 0.4490, top-100 concat 0.4462. E9 (G21 normalised, raw
denominator): SmolVLM-256M 0.436525, SmolVLM-500M 0.489004. V1 legacy
(biased, single seed): fusion 0.541, concat 0.525, question-only 0.458.

Key contrasts (bracketed ranges are image-clustered 95 per cent bootstrap
intervals; plus-minus values are across-seed standard deviations, which are
not confidence intervals):

- fusion − concat at 40k: +0.0144 +/- 0.0014 (all five seeds positive);
  at 250k +0.0037 +/- 0.0044 (sign flips in two seeds).
- reasoner − fusion: +0.0031 +/- 0.0062 (40k, noise); +0.0065 (100k);
  +0.0136 (250k), positive in every seed at both larger scales.
- Pooled >=4-step deficit: persists for every model at every scale
  (about 0.06-0.11); reasoner-minus-fusion deficit difference includes
  zero at all three scales.
- E8A A1 − A1r: +0.03820 [0.02947, 0.04680] (40k); +0.04835 [0.03935,
  0.05803] (250k). A1 − A0p: −0.01184 [−0.01996, −0.00427]; −0.02130
  [−0.02983, −0.01253].
- E8B B3 − B2: −0.02463 [−0.03333, −0.01619] (40k); −0.00475 [−0.01208,
  +0.00297] (250k).
- E10 B4 − B4r: −0.00683 [−0.01508, +0.00114] (40k); −0.00238 [−0.01071,
  +0.00588] (250k); difference in differences +0.00445 [−0.00709,
  +0.01641].

# Statistical evidence

- Uncertainty model: the development rows share images (7,714 rows over
  768 images), so the project's canonical interval is the image-clustered
  bootstrap (2,000 draws, default_rng(0), 2.5/97.5 percentiles), paired
  within draws for contrasts. Row-level (independence-assuming) intervals
  are superseded for inference (v2_05c).
- Seed spread (sample SD, ddof=1) and clustered intervals are different
  quantities and are never mixed: SD carries training-seed variation;
  the intervals condition on the fixed seed set and carry
  evaluation-sampling variation. Never present mean +/- SD as a
  confidence interval.
- Directional-claim rule (E8-family): interval excludes zero plus seed
  sign agreement; "uncertain or mixed" otherwise. An interval containing
  zero never establishes equivalence or absence.
- Multiplicity: interval counts are disclosed and uncorrected
  (`results/closure/F_multiplicity_status.json`).
- The v2_07 family carries no clustered intervals (row-level
  reconstruction stopped on a one-row mismatch, documented in the
  closure); its stored aggregates remain canonical. The E2-minus-CLIP
  contrast at 250k cannot be formed for the same reason.

# Efficiency evidence

- Authoritative end-to-end evidence: E7b warm serial medians (otter155,
  batch 1, raw inputs to answer): question_only 1.960 ms; concat 7.610;
  fusion 7.635; vocab1000_product 7.671; reasoner 9.172; e8a_A0p 9.212;
  siglip_fusion 9.843; e8a_A1 20.146. Primary Pareto frontier: concat,
  fusion, vocab1000_product.
- E7a component figures remain valid as components (CLIP towers 2.2510 /
  1.7309 ms; CPU decode 2.295 ms; heads 0.0162-0.0469 ms; reasoner head
  1.3968 ms). E7a's additive end-to-end sums are withdrawn and are not
  replaced by E7b values.
- E7b's peak-memory fields and cold-start field are withdrawn (two
  distinct causes; no replacement exists and none may be estimated).
- E9 same-node protocol (otter159): the top-1000 head measured 6.199 ms
  at accuracy 0.49050; SmolVLM-500M measured 151.380 ms on its
  open-readout row (accuracy 0.489004) and 152.854 ms on its constrained
  row (accuracy 0.485706) — approximately 24.4 and 24.7 times the warm
  serial latency respectively, under the same-node contextual protocol.
  The comparison is contextual, never a matched causal or leaderboard
  comparison. The two nodes' frontiers are never merged (bridge control
  failed at −18.7 per cent against a 10 per cent tolerance; no adjustment
  factor).
- Cached-image and cached-feature figures are partial-pipeline
  measurements, labelled as such, never end-to-end costs.
- Parameter count is a poor latency proxy (fusion is 48 per cent slower
  than concat_wide at equal parameters). Quote measured latency.
- The retained "16 times fewer parameters" comparison (the top-1000
  product head against the reasoner) is a TRAINABLE-parameter ratio:
  about 1.30M against 21.1M trainable parameters. It must not be read
  as 16 times fewer total loaded parameters; total loaded parameters
  differ by only about 1.13 times, both systems being dominated by the
  same frozen CLIP encoder.

# Main positive findings

1. Small heads over frozen representations reach useful GQA accuracy at
   millisecond measured latency; the frontier is owned by the simplest
   systems (concat, fusion, vocab1000_product).
2. The handcrafted fusion features give a small, reproducible low-data
   advantage (either interaction term suffices).
3. Question-side pretraining helps: A1 beats A1r at both scales, growing
   with scale.
4. Representation quality (E2) and answer-space design (E3) move accuracy
   materially; the top-1000 vocabulary wins +3.6 to +4.1 raw-distribution
   points at 250k.
5. The reasoner overtakes every global head at 100k/250k (a
   configuration-level finding).

# Main negative findings

1. The multi-step (>=4-step) deficit persists across every model family,
   two frozen encoders, three scales and heads from 0.1M to 21.1M
   parameters; token-level access does not reduce it.
2. The 40k reasoner result is negative: fusion parity at 19 times the
   size.
3. Answer-side pretraining shows no positive advantage (E8B), and 360M
   capacity does not change that (E10).
4. The fusion feature advantage decays to noise at 250k.
5. The E8A SLM question path loses to the interface-matched CLIP control
   and doubles the serial cost.

These negatives are load-bearing: each one eliminates a candidate
explanation and localises the bottleneck in the frozen representations
and closed-set formulation, not in the trained heads.

# Inconclusive findings

- E8B B3 − B2 at 250k and both E10 pretraining effects: intervals include
  zero; failures to detect, not demonstrations of absence or equivalence.
- The E10 difference in differences (+0.00445, interval spans zero).
- fusion − concat_wide at 40k under clustered uncertainty (+0.00578
  [−0.00116, +0.01226]).
- Per-type fusion losses other than choose (compare, global: intervals
  include zero).

# Limitations

- Every result is a development-set result; nothing is confirmatory until
  F2.
- Question-level scaling mixes new images and more questions per image.
- The reasoner-versus-heads comparison is configuration-level (no factor
  isolation); A1 − A0p cannot isolate LM semantics from the shared-space
  property, and its recipe provenance favours A0p.
- The step-count slice is confounded with question format.
- Three seeds (V3/E8/E10 families) estimate training-seed variance
  coarsely; five seeds elsewhere.
- v2_06 row-level shuffle carries 11 self-pairs (0.14 per cent);
  derangement supersedes it where both exist.
- E9 differs in training history, answer support, output format and
  backbone variant (Instruct vs base); GQA-image exposure for SmolVLM is
  likely (Visual Genome ancestry).
- No energy or power measurement exists anywhere in the project.

# Supported claims

Phrase them exactly this cautiously:

- Lightweight heads over frozen CLIP/SigLIP representations provide useful
  GQA closed-set VQA accuracy (about 0.58-0.60 in-vocabulary at 250k;
  about 0.49 raw-distribution with the top-1000 head) without any large
  autoregressive VLM.
- Measured serially on the historical E7b node, those systems answer a
  full query in about 7.6 ms; under the same-node E9 protocol the
  top-1000 global head reaches SmolVLM-500M's raw-distribution accuracy
  at approximately 24 to 25 times lower warm serial latency (open-readout
  row 24.4x, constrained row 24.7x; latency statement only).
- Hand-designed fusion gives a small, seed-robust low-data advantage that
  narrows with data.
- Question-side pretrained SLM features beat matched random features;
  answer-side pretrained readouts show no detected advantage at 135M or
  360M.
- The multi-step deficit is robust across encoders, scales and heads, and
  is the clearest marker of what this formulation does not extract.
- The models rely materially on the image (interventions), and wrong
  images actively mislead.

# Claims that must NOT be made

- No energy-efficiency, power, carbon, or monetary-cost claim ("cheaper",
  "25x cheaper", "lower cost", "more efficient" in a cost sense). Only
  measured latency and parameter statements are supported.
- No broad VLM-superiority claim and no state-of-the-art claim. No
  comparison with published official-GQA scores in either direction (the
  split, answer support and scorer all differ).
- No claim that deep, internal or compositional reasoning has been
  demonstrated; visual reliance is not proof of reasoning.
- No causal-superiority claim for the latent reasoner (configuration-level
  only), and no claim that A1 − A0p isolates language-model semantics.
- No claim that answer-side pretraining helps, and no claim that it
  provably does nothing (intervals include zero).
- No claim that larger answer-side capacity solves the problem.
- No robust out-of-distribution or broad-generalisation claim.
- No presentation of E9 as a matched or leaderboard comparison.
- No clean-test result of any kind before F2; no claim that F1 or F2 is
  complete or ready.
- Do not resurrect withdrawn numbers: E7a additive sums ("cuts a query
  from 6.35 to 2.25 ms", "on both end-to-end latency Pareto fronts"),
  E7b peak-memory or cold-start values, the v2_07 unweighted deficit
  definition, row-level intervals for inference, or Grid 1-3 statistics
  as recipe justification.

Where a claim's strength matters, label its evidence class explicitly:
VERIFIED (hash-pinned artefact, reproduced), REPORTED (stated in a
canonical report), INFERENCE (interpretation offered as such in a
report), PROVISIONAL (open finding or unresolved ambiguity).

# Clean-test governance

Required wording, verbatim, wherever the embargo is described:

"The clean-test contents were never inspected or used for development,
model selection, or reporting decisions. Mechanical byte access occurred
in two documented governance incidents, on 13 and 14 August 2026."

The two incidents (both classification B, mechanical byte access and
nothing more, identified separately): 13 August 2026 — an independent
reviewer ran an integrity-hash command over the target file; 14 August
2026 — an overly broad dependency-mapping scan walked the project root
and read every .json, .py, .md, .csv and .txt file because the traversal
was not scoped away from data/. Neither inspected any row, label,
distribution or prediction; neither informed development, model selection
or reporting; neither invalidates a result.

Prohibited: claiming the clean test was "never accessed", "unread" or
"untouched" (false), or any phrasing implying a single total access. Do
not open, read, hash, stat, glob, search, parse or traverse the target,
and do not "verify" the embargo by touching it. Any repository-wide scan
must exclude data/ before traversal, not filter afterwards. Canonical
records: `results/closure/e7b_evidence_supersession.json`
(clean_test_governance) and
`results/closure/pre_f1_status_supersession_20260814.json`.

# Historical vs current evidence

The project deliberately never rewrites closed artefacts; it supersedes
them with overlay records. Consequences you must handle:

- A closed artefact's own status text states the lifecycle on the day it
  was written. The E7b overlay says OPEN/f1-BLOCKED; the VE-0 report says
  review_requested; the metadata-repair record says awaiting review;
  `collab/PROJECT_CONTEXT.md` predates E10 execution. All are superseded
  for lifecycle by `pre_f1_status_supersession_20260814.json` and, for
  the metadata repair, by its recorded approval (revision 2, pushed as
  db5a1bf).
- Historical numbers in superseded fields remain readable as provenance
  only. Resolve field validity through the successor records before
  quoting anything from a closed packet.
- `docs/STUDY_GUIDE.md` and `docs/PROGRESS_REPORT.md` are V1-era and
  carry historical banners.
- The E8B core report contains a stale "Tables 6 and 7 — NOT PRODUCED"
  stub that predates the completed final evaluation in the same file;
  the populated tables and the hypotheses section are the operative text.

# Superseded results and artefacts

- E7a: `gpu_encoder_plus_head_ms`, `full_pipeline_ms`, `amortised_ms` and
  every derived Pareto front — SUPERSEDED, not replaced.
- E7b: `peak_allocated_mib`, `peak_reserved_mib` — INVALID_WITHDRAWN;
  `cold_first_query_ms` — SUPERSEDED_WITHDRAWN. Distinct causes; never
  conflate them.
- v3_01's unweighted two-bucket deficit definition — superseded by the
  question-weighted pooled definition (v3_02a).
- v2_05b row-level intervals — superseded for inference by clustered
  intervals (v2_05c).
- v2_06 row-level shuffle — superseded by the imageId-level derangement
  wherever both exist.
- Nine E8B B1-bearing evidence rows — corrected selection metadata
  (five BEST_ON_DEVELOPMENT, four MIXED_WITHIN_ROW); B2/B3 rows remain
  FIXED_EPOCH_22.
- Five presentation-precision fields, the E7a locator and the
  VE1-FIG-04-FULL placement — corrected by the metadata repair.
- The 35 GPU-hour ceiling — historical; 40 GPU-hours programme-wide from
  8 August 2026.
- E8B grid points 1-3 — exploratory only; grids 4-8 never ran.
- V1 results and v3_01/v3_02a latency figures — legacy/superseded (E7a
  supersedes the old latency protocol; accuracy unaffected).
- A draft-manuscript claim of a repaired V1 11-model 100k/250k rerun
  is NOT VERIFIED and not canonically accepted: no such result exists
  in this repository. Do not use it as dissertation evidence; the
  canonical 100k/250k scaling evidence is v2_07, v3_03 and E3.

# Unexecuted E8A arms

The seven E8A arms A2, A2r, A4, A5, A7c, A8c and AF (FLAN-T5-small)
are authorised-but-unrun and deferred from the current MSc execution
scope (docs/experiments/pre_f1_unrun_arms_disposition.md). Required
wording: "No canonical or accepted experimental result exists for any
of these seven arms, and the scoped execution audit found no execution
artefact for them. The pre-F1 deferral was therefore not based on
observing their experimental outcomes." Do not use the absolute
wording "no experimental result exists anywhere".

# Recommended dissertation chapter mapping

1. **Introduction**: research question, efficiency motivation, closed-set
   formulation. Sources: PROJECT_OVERVIEW, README.
2. **Background and related work**: `docs/RELATED_WORK.md` (three areas:
   frozen dual-encoder VQA with small heads; latent-query modules and
   token-versus-global evidence; compositional limitations). The
   differentiation point — no published GQA accuracy for this exact
   frozen-dual-encoder-plus-small-head setting — is recorded there.
3. **Method / protocol**: v2_00 report; REPRODUCIBILITY; metrics and
   denominators; clean-test governance (verbatim wording).
4. **Architectures**: ARCHITECTURE.md and src/.
5. **Global-head experiments**: v2_01-v2_07 (baselines, matching,
   ablation, types, reliance, scaling).
6. **Token-level reasoner**: v3_00-v3_03 including the negative 40k
   result and diagnostics.
7. **Representation and answer-space levers**: E2, E3.
8. **Frozen SLM interfaces**: E8A, E8B, E10 (present the checkpoint
   selection rules and the causal-contrast boundaries exactly).
9. **Efficiency**: E7a components, E7b serial, E9 context. State the
   protocol scope in every sentence that quotes a latency.
10. **Discussion**: SCIENTIFIC_FINDINGS ordering works as a discussion
    skeleton; negative results are contributions.
11. **Conclusion and future work**: the pre-F1 state, F1/F2 as the
    remaining confirmatory steps, the recorded (unauthorised) backlog.

# Recommended figures and tables

The dissertation's quantitative figures and tables are already rendered,
frozen and provenance-pinned under `results/ve1/` (specifications in
`results/ve0/figure_specifications.json` and `table_specifications.json`):
main-text figures VE1-FIG-01 to VE1-FIG-09, appendix figures VE1-FIG-A1
to A5 plus VE1-FIG-04-FULL (effective placement APPENDIX; effective
counts 9 main-text, 6 appendix), main tables VE1-TAB-01 to 06, appendix
tables VE1-TAB-07 and A1-A4, captions in `results/ve1/captions.json`.
The qualitative gallery is `results/ve2/galleries/` (main 5 pages,
appendix 4 pages, 18 deterministic records). Use these renders; do not
re-derive figures from raw artefacts. The frozen VE1-FIG-08 renders must
be paired with the metadata repair's hash-guarded effective virtual note
(the corrected B1-selection wording is not embedded in their historical
bytes).

# Literature handoff

- `docs/RELATED_WORK.md`: the consolidated survey (July 2026), organised
  by the three areas above plus venue notes; every numeric claim carries
  a verified/unverified tag.
- `docs/references.bib`: 72 entries; 14 carry notes requiring venue
  re-verification at submission time. Do not invent bibliographic
  details; re-check flagged entries against the sources before final
  submission. Areas the dissertation must cover: frozen dual-encoder VQA
  and lightweight heads; connector/latent-query interfaces to frozen
  models; VQA fusion classics; compositionality benchmarks and
  frozen-representation limitations; GQA and its diagnostic variants;
  efficiency measurement practice. External paper verification is still
  required for the 14 flagged entries and for anything published after
  July 2026.

# Current F1/F2 status

F1 (model-list freeze): UNSTARTED, unauthorised, not declared ready by
any record. F2 (blinded clean-test evaluation): UNSTARTED_AND_UNAUTHORISED.
The clean-test embargo is unchanged. No confirmatory result exists.

# What remains before submission

See [PROJECT_STATUS.md](PROJECT_STATUS.md): supervisor feedback, the F1
decision, F2 under a frozen protocol, the manuscript itself, bibliography
re-verification, and final number-tracing against canonical records.

# Source-of-truth hierarchy

1. Current canonical closure and lifecycle records:
   `results/closure/pre_f1_evidence_metadata_repair_20260814.json` (its
   named fields), `results/closure/e7b_evidence_supersession.json`,
   `results/ve0/supersession_map.json`,
   `results/closure/pre_f1_status_supersession_20260814.json` (lifecycle).
2. Canonical result records and manifests: `results/experiments/`,
   `results/closure/` (registry A-H), `artifacts/results_export/`,
   `results/ve0/canonical_evidence_inventory.*`.
3. Governance and reproducibility documentation: CLAUDE.md,
   `docs/REPRODUCIBILITY.md`, the per-experiment reports under
   `docs/experiments/`.
4. Project context: `collab/PROJECT_CONTEXT.md` (partially historical).
5. Historical/superseded records — for explaining evolution only, never
   as current authority.

## If you are another AI writing this dissertation

- The repository's canonical evidence is the scientific source of truth.
  Never invent a missing result, and never silently substitute general ML
  knowledge for repository evidence. If the repository lacks a number,
  say so.
- Resolve every number through the source-of-truth hierarchy above before
  quoting it. If two surfaces disagree, identify the conflict, apply the
  precedence rules, use the current canonical value, and flag any residue
  you cannot resolve — do not guess.
- Distinguish source-derived evidence from your interpretation, and keep
  the report's own interpretation labels (observational, intervention,
  configuration-level, system comparison) attached.
- Do not claim F2 results before F2 exists; every number you may use is a
  development-set result and must be presented as such.
- Do not exaggerate negative or inconclusive results in either direction:
  a negative directional result is negative; an interval containing zero
  is a failure to detect, never equivalence.
- Do not write energy-efficiency, monetary-cost or carbon claims; latency
  and parameter statements under the measured protocol only.
- Do not present E9's contextual comparisons as matched causal
  comparisons, and always carry its caveats (training history, answer
  support, output format, Instruct-versus-base).
- Preserve limitations. They are part of the findings, not hedging to be
  trimmed.
- Use the clean-test governance wording verbatim; never write "never
  accessed".
- Respect checkpoint-selection labels: B1 is best-on-development; B2/B3
  and B4/B4r are fixed epoch 22; only selection-matched contrasts are
  causal.
- If you find an apparent scientific inconsistency, do not fix it or
  paper over it: record it explicitly and stop before publishing the
  conflicting claim.
