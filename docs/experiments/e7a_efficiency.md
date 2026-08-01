# Experiment E7a: efficiency Pareto and end-to-end cost accounting

## Purpose

Measure what every stored head actually costs to run, under one
protocol, and place accuracy against cost honestly. Until now no
V2/V3/E2/E3 experiment had recorded any latency or memory measurement:
the efficiency axis of a dissertation titled "Efficient Vision-Language
Reasoning" rested on parameter counts plus superseded V1 stage-5
numbers. E7a replaces that with measured evidence and answers which
models are Pareto-optimal once the frozen encoder is accounted for.

## Method

experiments/e7a_efficiency/run.py (pinned commit f8a5580 after a
four-round pre-execution review). Cost is determined by architecture and
accuracy by (architecture, scale, seed), so each of 21 architectures and
4 encoder towers was benchmarked once and joined to the stored accuracy
of every scale it was trained at, giving 39 accuracy-cost points. All
189 comparison checkpoints reproduced their stored dev accuracy within
5e-6 before inclusion; v3_01 search checkpoints and the v3_02a
patches-only probe are excluded as search artefacts and a single-seed
diagnostic.

Protocol, identical for every item: fp32, eval mode, torch.no_grad
hoisted outside the timed loop, no autocast, real cached dev features
and a real dev image and question, GPU tensors pre-placed. Single
example: 50 warm-up then 300 timed iterations with
torch.cuda.synchronize immediately before and after every call. Batch
256: 20 warm-up then 50 timed. The repetition is the outer loop (three
full passes over all 25 items), so the reported spread reflects drift
across the whole run rather than seconds-scale jitter; GPU clocks,
temperature and power were recorded per pass. Peak memory was measured
once in a separate isolated phase as a delta above the allocation
baseline captured immediately before the call. An empty-callable
overhead floor was timed identically in every pass. Image decode and
tokenisation were timed separately on CPU over 8 distinct dev images
and questions.

Two memory quantities are reported and must not be confused.
footprint_mib_head is head weights (parameters times 4 bytes) plus the
measured transient activation delta; footprint_mib_full adds the
resident frozen-encoder weights, following the serial-serving model
that matches how the latency regimes sum the same stages. Encoder
weights are charged for the whole dual-tower model even when only one
tower is used, so the memory axis does not credit question_only's
structural advantage the way the latency axis does.

Four cost regimes, never mixed: head-only; GPU encoder plus head; full
per-query pipeline (adds CPU image decode/preprocess and tokenisation);
and amortised, where image decode and the image tower are divided by a
questions-per-image reference of 10.0 (measured 10.04 on the top-100 dev
view and 12.66 on the E3 view). The common accuracy axis is
raw-distribution accuracy over the identical 10,004 raw dev questions
(in-vocabulary accuracy times coverage), which is exactly correct/10004
in both dev views; accuracies are five-seed means, three-seed for the
reasoner.

Superseded measurements: the V1 stage-5 latencies and the earlier
src/efficiency.py-derived numbers in v3_01 (1.3154 ms) and v3_02a
(0.0138 and 0.0273 ms) used random inputs, 20/200 iterations and means
without pre-call synchronisation. E7a supersedes all of them; they must
not be mixed with these numbers. Timings are conservative
deterministic-mode figures (cuDNN deterministic, benchmark off), and
decode was measured with the files in the OS page cache, so disk I/O is
excluded.

Resolvability: a latency difference counts as resolved only if it
exceeds the across-pass spread of both models compared (the artefact
stores per-pass medians and p95, so the rule is applied here on the
across-pass spread). The measured overhead floor is 0.00261 ms; the two
duplicate-architecture controls give an empirical noise floor of
0.0001 ms (question_only 0.0261 against image_only 0.0262; concat
0.0303 against meanpatch_concat 0.0304). Across-pass spreads are
0.0001-0.0014 ms for the 20 heads and 0.0625 ms for the reasoner, so
head-only differences above about 0.002 ms are resolved. The four
encoder towers are far noisier — 0.0049 (CLIP text), 0.0184 (SigLIP
text), 0.0574 (CLIP image) and 0.1279 ms (SigLIP image) — so any
end-to-end comparison must add the relevant tower spreads, and
differences of a few hundredths of a millisecond between pipelines are
not resolved. GPU clocks fell from 2340 MHz at 44 C in pass 1 to
1590 MHz at 77-78 C in passes 2 and 3 under sustained load; the
reported medians are medians across passes, and head spreads remained
at or below 0.0014 ms throughout.

Blinding: dev only; test_clean_targets.csv was never read; nothing was
trained or tuned.

## Outputs

results/experiments/e7a_efficiency/{preflight.json,
measurements_interim.json, results.json, run.log} and four figures:
accuracy_vs_params.png, accuracy_vs_latency.png,
accuracy_vs_head_latency.png, accuracy_vs_memory.png. Total wall time
6.6 minutes.

## Results

Frozen encoder towers (single example; batch-256 throughput):

| tower | single (ms) | batch-256 (ms) | throughput (ex/s) | weights (MiB) |
|---|---|---|---|---|
| CLIP image | 2.2510 | 223.4 | 1,146 | 577 |
| CLIP text | 1.7309 | 188.7 | 1,357 | 577 |
| SigLIP image | 5.4818 | 957.9 | 267 | 775 |
| SigLIP text | 2.1355 | 277.1 | 924 | 775 |

CPU terms: image decode and preprocess 2.295 ms (CLIP transform) and
2.159 ms (SigLIP); tokenisation 0.036 ms and 0.074 ms.

Representative heads (cost is scale-independent):

| head | params | head-only (ms) | throughput (ex/s) | checkpoint (MiB) |
|---|---|---|---|---|
| direct_linear | 102,500 | 0.0162 | 9,961,482 | 0.39 |
| question_only | 313,956 | 0.0261 | 6,072,180 | 1.20 |
| concat | 576,100 | 0.0303 | 4,523,007 | 2.20 |
| product_576k | 574,687 | 0.0364 | 4,634,490 | 2.19 |
| fusion | 1,100,388 | 0.0450 | 2,968,477 | 4.20 |
| concat_wide | 1,101,475 | 0.0304 | 3,386,064 | 4.20 |
| vocab1000_product | 1,299,944 | 0.0330 | 2,974,495 | 4.96 |
| siglip_fusion | 1,624,676 | 0.0446 | 2,413,956 | 6.20 |
| reasoner | 21,099,620 | 1.3968 | 5,320 | 80.53 |

Top accuracy-cost points (raw-distribution accuracy over 10,004 dev
questions; ms per query):

| model @ scale | raw acc | head-only | GPU+head | full pipeline | amortised | footprint head / full (MiB) |
|---|---|---|---|---|---|---|
| vocab1000_product@250k | 0.4904 | 0.0330 | 4.0150 | 6.3459 | 2.2545 | 4.97 / 591.9 |
| vocab1000_fusion@250k | 0.4865 | 0.0441 | 4.0260 | 6.3570 | 2.2655 | 5.97 / 592.9 |
| vocab1000_concat@250k | 0.4822 | 0.0297 | 4.0116 | 6.3426 | 2.2511 | 3.97 / 590.9 |
| siglip_product@250k | 0.4611 | 0.0338 | 7.6511 | 9.8838 | 3.0072 | 4.71 / 786.6 |
| reasoner@250k | 0.4594 | 1.3968 | 5.3788 | 7.7097 | 3.6183 | 81.40 / 667.4 |
| siglip_fusion@250k | 0.4579 | 0.0446 | 7.6619 | 9.8946 | 3.0179 | 6.21 / 788.1 |
| product_576k@250k | 0.4491 | 0.0365 | 4.0184 | 6.3493 | 2.2579 | 2.20 / 589.1 |
| fusion@250k | 0.4490 | 0.0450 | 4.0269 | 6.3579 | 2.2664 | 4.21 / 591.1 |
| concat@250k | 0.4462 | 0.0303 | 4.0122 | 6.3432 | 2.2517 | 2.21 / 589.1 |

Pareto fronts (raw-distribution accuracy against each cost axis):

- trainable parameters: direct_linear@40k, question_only@250k,
  siglip_question_only@250k, product_576k@250k, siglip_concat@250k,
  vocab1000_concat@250k, vocab1000_product@250k.
- head-only latency: direct_linear@40k, vocab1000_question_only@250k,
  vocab1000_concat@250k, vocab1000_product@250k.
- GPU encoder plus head, and full pipeline (identical membership):
  vocab1000_question_only@250k, vocab1000_concat@250k,
  vocab1000_product@250k.
- amortised: image_only@40k, vocab1000_question_only@250k,
  vocab1000_concat@250k, vocab1000_product@250k.
- head-only memory footprint: direct_linear@40k, question_only@250k,
  siglip_question_only@250k, product_576k@250k, siglip_concat@250k,
  vocab1000_concat@250k, vocab1000_product@250k.
- end-to-end memory footprint: question_only@250k,
  vocab1000_question_only@250k, product_576k@250k,
  vocab1000_concat@250k, vocab1000_product@250k.

Front membership separated by less than one seed standard deviation in
accuracy (about 0.002-0.003 raw) or 0.002 ms in latency is not
resolved; the amortised front's image_only@40k has raw accuracy 0.18073,
far below every blind floor, and is a front artefact rather than a
usable operating point.

Best under each criterion (blind floors: 0.38377 for CLIP top-100,
0.38737 for SigLIP top-100, 0.39890 for CLIP top-1000; the global blind
floor is vocab1000_question_only@250k):

| criterion | winner | raw acc | uses image |
|---|---|---|---|
| minimum trainable parameters | direct_linear@40k | 0.3711 | yes |
| lowest head-only latency | direct_linear@40k | 0.3711 | yes |
| lowest full-pipeline latency | vocab1000_question_only@250k | 0.3989 | no |
| lowest memory footprint (head) | direct_linear@40k | 0.3711 | yes |
| highest accuracy | vocab1000_product@250k | 0.4904 | yes |
| best trade-off above own-family blind floor, per parameter | product_576k@250k | 0.4491 | yes |
| best trade-off above own-family blind floor, per pipeline ms | vocab1000_product@250k | 0.4904 | yes |

The two unnormalised ratio criteria are reported in results.json but are
degenerate by construction: accuracy per parameter selects
direct_linear@40k and accuracy per pipeline millisecond selects the
blind vocab1000_question_only@250k, because accuracy does not tend to
zero as cost tends to zero. The above-blind-floor criteria are the
defensible ones.

Two of these rows are ties rather than measured separations and must be
read as such. The lowest-full-pipeline winner beats question_only@250k
by 0.00009 ms, an order of magnitude below the 0.0001 ms noise floor;
it is a latency tie resolved on accuracy by the recorded tie-break. The
best-trade-off-per-parameter winner, product_576k@250k, leads
concat@250k by 0.00012 raw accuracy against a seed standard deviation of
about 0.0025, so that ordering — and the corresponding parameter and
head-footprint front memberships — is not resolved.

## Decisions and problems

(a) The frozen encoder dominates, and the head is nearly free. A CLIP
query costs 2.2510 ms (image) plus 1.7309 ms (text) on the GPU and a
further 2.295 ms of CPU decode; the global heads add 0.0162-0.0469 ms,
which is 0.26 to 0.71 per cent of the full pipeline for image-using
heads (up to 1.46 per cent for the blind question-only head, whose
pipeline omits the image tower and decode entirely). CPU image decode
(2.295 ms, p5-p95 1.94-3.06) and GPU image encoding (2.2510 ms,
across-pass spread 0.0573) are comparable in magnitude and their
ordering is not resolved by these measurements; that decode costs about
as much as encoding is itself new to the project record. Caching image
features across the
~10 questions per image cuts a CLIP query from 6.35 ms to 2.25 ms, a
2.8-fold reduction, and is by far the most valuable engineering choice
available. The head-versus-head comparisons that occupied V2 and V3 are
real but nearly invisible at the system level.

(b) Vocabulary growth is free; the encoder swap and token-level
reasoning are not. Against fusion@250k, the top-1000 product head gains
+0.0414 raw accuracy at 1.00 times the full-pipeline cost (6.3459 against
6.3579 ms, marginally cheaper because a 1000-way output layer is
comparable to a 2048-wide fusion input); SigLIP-B/16 gains +0.0090 at
1.56 times the pipeline cost, 2.4 times the image-tower latency and 4.3
times lower batch throughput; the latent-query reasoner gains +0.0104
at 1.21 times the pipeline cost and 1.60 times the amortised cost. On
both end-to-end latency fronts (GPU encoder plus head, and full
pipeline) the only Pareto-optimal models are top-1000 global heads; the
head-only and amortised fronts additionally admit direct_linear@40k and
image_only@40k respectively, both far below every blind floor. The
reasoner and every SigLIP configuration are dominated on all four
latency fronts.
Sharpest single statement: vocab1000_product@250k is both more accurate
(0.4904 against 0.4594) and cheaper (6.35 against 7.71 ms) than the
21.1M-parameter reasoner it is compared with, using 16 times fewer
parameters.

(c) Parameter count is a poor proxy for latency. fusion (1,100,388
parameters, 0.0450 ms) is 48 per cent slower than concat_wide
(1,101,475 parameters, 0.0304 ms) despite having marginally fewer
parameters, because it constructs the elementwise product and absolute
difference and feeds a 2048-wide input. Any efficiency claim in the
dissertation should therefore quote measured latency, not parameter
counts alone. The v3_01 report's claim that the reasoner is "about 30
times slower at cached-feature inference" is corroborated under the new
protocol at 31.0 times (1.3968 against 0.0450 ms), even though the
absolute numbers differ from the superseded measurement.

(d) Memory: the frozen encoder sets the floor. End-to-end footprints are
580.1 MiB (question_only), 581.8 (vocab1000_question_only), 587.3
(direct_linear), 588.1 (image_only) and 589-593 MiB for the CLIP
multimodal global heads, 667.4 MiB for the reasoner, and 778.9 MiB
(siglip_question_only) to 785-788 MiB for the SigLIP heads. The head
therefore moves the floor by 0.40 MiB (direct_linear) to 81.40 MiB (the
reasoner, about 14 per cent of the CLIP floor); head-only footprints
span 0.40 to 81.40 MiB. Reporting only head memory would overstate how
much the head choice matters, and reporting only the floor would hide
the reasoner's 76 MiB premium over a global head.

(e) Scope and limits. Development-set results; the clean-test embargo is
untouched; nothing was trained or tuned. Batch-1 latency is
launch-latency bound, so batch-256 throughput is the compute-bound
comparison, and the two duplicate-architecture controls bound the noise
floor at 0.0001 ms. Timings are conservative deterministic-mode
figures. Decode excludes disk I/O. Peak memory was measured once, not
per pass, because allocator behaviour is deterministic for a fixed call
sequence; no memory spread is claimed. The reasoner's accuracies are
three-seed means against five-seed means elsewhere. Cached-feature sizes
per query differ in dtype by design (fp32 global stores, fp16 token
stores), so the reasoner's 64,512-byte cache entry is 15.75 times a CLIP
global head's 4,096 bytes (10.5 times SigLIP's 6,144); this is recorded
in results.json but is not folded into any latency figure. preflight.json
was written during the round-01 pin (7f89074, dirty tree) and predates
the executed pin; the identical gates, including all 189 checkpoint
reproductions, were re-run inside the f8a5580 execution and are recorded
in run.log. The raw-distribution figure for vocab1000_product@250k is
0.4904 here (computed from unrounded inputs) against 0.4905 in
docs/experiments/e3_vocab1000.md (computed from four-decimal inputs);
the E7a value is the one to quote.

(f) Consequence for the freeze (F1). On the measured evidence the
efficiency argument of the dissertation is strongest for small global
heads over a frozen dual encoder with a large answer vocabulary, which
supports considering at least one top-1000 global head for the final
model list. The
reasoner remains scientifically important as the controlled negative and
scale-reversal result, but it is not on any efficiency Pareto front. No
freeze decision is taken here.
