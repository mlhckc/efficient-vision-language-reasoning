# Supervisor feedback closure record

Compiled 5 August 2026 as part of the E8A G21 remediation. Each item below
maps a supervisor-requested question to the repository evidence. No item is
marked complete from report prose alone: every headline value was re-read
from its result artefact and every artefact hash below was recomputed at
compilation time by an independent read-only verification pass. Every result
is a development-set result under the V2 protocol; nothing here is
confirmatory and the clean test remains embargoed.

## 1. CLS-versus-patch attention

- Question: does the latent-query reasoner attend to the CLS token rather
  than the patches?
- Evidence: results/experiments/v3_02a_refs/diagnostics.json
  (sha256 a192322c877bf5b855ae6dfa47c00e572acedb143a75585f442b7c170c808743),
  keys v3_02a_diagnostics.d1.attention.*; report
  docs/experiments/v3_02a_refs.md with the S1 all-seed correction.
- Status: complete.
- Headline: per-seed mean CLS attention mass 0.3803 / 0.3340 / 0.2642
  (seeds 0/1/2), seed-by-block cells spanning 0.1805-0.5346, per-head means
  0.0530-0.8881; instrumentation gate passed with prediction agreement 1.0.
- Limitation: three seeds, dev-only, descriptive; attention weights are not
  causal evidence (stated in the artefact's own interpretation scope).

## 2. CLS masking / removal

- Question: what happens when CLS is removed at test time?
- Evidence: same diagnostics.json, keys v3_02a_diagnostics.d2.*.
- Status: complete.
- Headline: test-time CLS removal costs 0.02839-0.05017 accuracy
  (0.54641/0.54058/0.53967 falling to 0.49624/0.51219/0.49806), mean change
  -0.04006.
- Limitation: a test-time intervention; it does not show how a model trained
  without CLS would behave. Three seeds, dev-only.

## 3. Patches-only evaluation / retraining

- Question: can the reasoner work from patches alone?
- Evidence: results/experiments/v3_02a_refs/patches_probe.json
  (sha256 90513aa905d6d79b05e5e76c69e5a1955fd7bd44171ac8939d43b49b2e2a66d8).
- Status: complete as a bounded probe.
- Headline: a patches-only reasoner trained under a 1,200-second budget
  reaches dev accuracy 0.53591 (best epoch 5) without improving the
  combined >=4-step lift.
- Limitation: single seed (0), time-budgeted diagnostic, dev-only; excluded
  from the E7a efficiency comparison by design.

## 4. Mean-of-patches baseline

- Evidence: results/experiments/v3_02a_refs/refs_results.json
  (sha256 f3994a00e84999bf721af45f774c023a7cd0ab0e298668c94b9044ce5bb160e9),
  key v3_02a_refs.meanpatch_concat.
- Status: complete.
- Headline: meanpatch-concat mean dev accuracy 0.51449 over five seeds
  (std 0.00191) at 576,100 trainable parameters.
- Limitation: dev-only; its stored 0.0273 ms latency is superseded by E7a.

## 5. Direct-linear pooled-CLIP baseline

- Evidence: same refs_results.json, key v3_02a_refs.direct_linear.
- Status: complete.
- Headline: 0.48125 mean dev accuracy over five seeds (std 0.00107) with
  exactly 102,500 trainable parameters.
- Limitation: dev-only; stored latency superseded by E7a.

## 6-7. V3 reasoner scaling to 100k and 250k

- Evidence: results/experiments/v3_03_scaling/results.json
  (sha256 b618e239e051e6201b8a3b4c6833c43ceb5369a3059a07883314783012da3339),
  keys v3_03_scaling.summary and .final_runs; report
  docs/experiments/v3_03_scaling.md.
- Status: complete, with the 5 August 2026 wording correction applied: the
  configuration-level result does not causally isolate architecture,
  token access or capacity.
- Headline: 0.57061 +/- 0.0002 at 100k and 0.5958 +/- 0.0074 at 250k
  (three seeds), ahead of every stored v2_07 head in every seed; the paired
  reasoner-fusion deficit difference stays indistinguishable from zero at
  both scales.
- Limitation: three seeds; the 250k spread is driven by one seed (0.60436);
  dev-only.

## 8. Parameter-matched global controls

- Evidence: results/experiments/v2_03_param_match/results.json
  (sha256 9494f8a706b26dc82a66bb28280411acbd079991f353da8467248ac0c1947f92).
- Status: complete.
- Headline: concat-wide 0.53262 at 1,101,475 parameters; fusion-narrow
  0.53342 at 576,032 parameters; capacity explains part, not all, of the
  40k fusion gap.
- Limitation: five seeds, train_40k only, dev-only.

## 9. Image-clustered statistics

- Evidence: results/experiments/v3_02a_refs/step_statistics.json
  (sha256 1124139e70ece466c7f8a759089c5eb9700ebc4b3fcfdd1adc4e6b90a957c6a3);
  results/experiments/v3_03_scaling/results.json (bootstrap block);
  results/experiments/e8a_question_encoder/core_analysis.json and its G21
  regeneration core_analysis_g21.json; and, new in this remediation,
  results/experiments/v2_05_types/addendum_clustered.json, which replaces
  the v2_05b row-level normal-approximation intervals.
- Status: complete after correction. The v2_05b row-level intervals
  (addendum.json, ci_method "gap +/- 1.96 sd/sqrt(n)") required correction;
  the clustered regeneration reproduced every stored gap exactly and changed
  none of the ten exclusion conclusions.
- Headline: every operative interval in v3_02a, v3_03, v2_05c and E8A is an
  image-clustered bootstrap (2,000 draws, rng seed 0, represented dev
  imageIds as clusters).
- Limitation: intervals condition on the fixed trained seed set and are
  uncorrected for multiplicity, as disclosed in each artefact.

## 10. Sample counts for type / reasoning slices

- Evidence: per-slice row counts existed in v2_05 outputs (addendum.json
  "n" fields; per_type_accuracy.csv); per-slice unique-image counts existed
  only for step buckets (v3_02a step_statistics.json bucket_table). New in
  this remediation: addendum_clustered.json records n_rows and
  n_unique_images for every structural, semantic and step slice, and the
  E8A G21 analysis (core_analysis_g21.json, type_and_step_slices) records
  n_rows, n_unique_images, the metric and a clustered interval for every
  slice at every arm and scale.
- Status: complete after correction (unique-image counts were previously
  missing for type slices).
- Limitation: small slices have few image clusters; each slice entry
  carries its own counts so the reader can weigh the interval.

## 11. True end-to-end efficiency

- Evidence: results/experiments/e7a_efficiency/results.json
  (sha256 368ba5c2a149953193cf237bdf77023c17b244540ef7b9bfac55cb5a7fdc1a85).
  Verified numerically: every "GPU+head", "full pipeline" and "amortised"
  figure is an exact arithmetic sum of independently timed component
  medians (for example concat@100k 6.34318 ms = 2.25105 image tower
  + 1.73088 text tower + 2.29499 CPU decode + 0.03597 tokenise + 0.03029
  head). No stored latency is a measured serial decode-encode-head pass.
- Status: COMPLETE (5 August 2026, E7b). The serial benchmark specified
  below has run: docs/experiments/e7b_serial_efficiency.md and
  results/experiments/e7b_serial_efficiency/e7b_results.json measure
  every proposed system serially with 64/64 answer reproduction, and
  supersede E7a's additive end-to-end columns; the additive figures
  under-estimated the measured CLIP-global serial cost by about 20 per
  cent. The design below is retained as the executed specification.
- Required design for the next authorised phase (not run in this task):
  one timed pass per query executing, serially and unbroken, CPU JPEG
  decode and preprocess, host-to-device transfer, the frozen image tower,
  tokenisation, the frozen text tower and the trained head, on the E7a
  hardware, with the E7a drift protocol (three full passes over the item
  list, 50 warm-up and 300 timed iterations single-example,
  synchronisation before and after each call, clocks and temperature
  recorded per pass, deterministic mode, page-cache state stated); the
  amortised variant caches the image branch across the measured
  questions-per-image reference of 10.0; report the additive estimate
  alongside for the same items so the inter-stage overhead is measured
  rather than assumed. Models: at minimum the four E7a Pareto-front heads
  and the reasoner; both metrics of canonical section 12 for any accuracy
  restated. Estimated cost is of the E7a order (minutes of GPU time).

## Interpretation constraints preserved

A1 minus A1r remains the primary pretrained-versus-random contrast; A1
minus A0p remains a matched system comparison that does not isolate
language-model semantics; the fixed-question intervention changes both
content and length; higher overall accuracy is never read as improved
multi-step reasoning; no equivalence language is used anywhere.
