# Experiment E7b: measured serial end-to-end efficiency

## Purpose

Replace E7a's additive end-to-end estimates with measured serial values.
One timed pass per query executes, serially and unbroken: image file read,
CPU decode and preprocess, host-to-device transfer, frozen vision tower,
question tokenisation, frozen text or language-model tower, trainable
projection where one exists, fusion or latent reasoning, classification,
and answer decode. E7a's component measurements remain valid as
components; its additive gpu_encoder_plus_head_ms, full_pipeline_ms and
amortised_ms values and their Pareto fronts are superseded for end-to-end
claims; E7b's measured serial values are authoritative for online
inference claims.

## Method

experiments/e7b_serial_efficiency/run.py at commit b60c981, executed on
the qualified otter155 node (RTX 4000 Ada Generation, driver 580.126.09,
CUDA 13.0, Python 3.12.3, torch 2.12.1+cu130 — fingerprint-identical to
the E7a node and hard-gated), with a user-approved A1 qualification pilot
preceding the full run. Eight systems, every timed checkpoint seed 0:
question_only (a labelled non-multimodal pipeline floor, excluded from
the primary frontier), concat, fusion, vocab1000_product, reasoner
(v3_03), siglip_fusion (E2), e8a_A0p and e8a_A1 (E8A), all at train_250k.

Protocol: three pass-major passes over all systems in pinned-RNG
randomised order; a fresh subprocess per system per pass; process and
model startup, cold first query and warm latency recorded separately; 50
warm-up then 300 unsegmented timed serial queries (the warm median is the
headline), synchronisation before and after every call and a 0.0026 ms
no-op floor self-check; a separate 100-iteration segmented pass for
per-stage medians with the additive sum reported against the measured
total; cached-image and cached-feature regimes measured, not derived; a
labelled batch-64 image-branch throughput; one pinned 64-row development
sample with per-image SHA-256, identical for every system; GPU
exclusivity enforced under the NFS execution lock; atomic immutable
outputs. Preflight reproduced every system's stored seed-0 development
accuracy from the cached path within 5e-6 before any timing.

Model-cache control: the qualified online-capable Hugging Face
configuration was used (a forced-offline attempt broke open_clip's
optional-file probing and wrote nothing); all 22 real cache files were
SHA-256-hashed before and after the benchmark and are byte-identical, so
no weight, tokenizer or configuration file was downloaded, updated or
replaced; unauthenticated metadata probes were observed in the logs and
cannot alter the cache.

Reproduction gate: every system reproduced the cached-path answers on the
pinned 64 rows exactly, 64/64, in every pass — including the three
fp16-cached token systems, so the pre-registered quantisation-explanation
machinery (enumeration with logits, margins and a 0.05-logit perturbation
cap) was armed but never needed. Consecutive-duplicate queries were
bitwise identical and answers were identical across passes.

## Results

Development set only; accuracies are the frozen stored values (the 64-row
timing sample supports no accuracy claim). The common denominator is
correct answers over the identical 10,004 raw development questions; the
vocabulary-supported accuracy (each system's own support) is the
separately labelled secondary value. Multi-seed context: five seeds for
the global heads, three for reasoner/E8A.

| system | common-denom acc | vocab-supported acc | multi-seed mean +/- sd | warm serial (ms) | spread | cold (ms) | load (s) | QPS | peak MiB alloc/res | trainable | total loaded |
|---|---|---|---|---|---|---|---|---|---|---|---|
| question_only (context) | 0.38864 | 0.50402 | 0.49769 +/- 0.00439 | 1.960 | 0.013 | 75 | 2.1 | 510 | 623/690 | 313,956 | 151.6M |
| concat | 0.44462 | 0.57661 | 0.57861 +/- 0.00264 | 7.610 | 0.904 | 117 | 2.0 | 131 | 764/852 | 576,100 | 151.9M |
| fusion | 0.45062 | 0.58439 | 0.58226 +/- 0.00197 | 7.635 | 0.890 | 122 | 2.0 | 131 | 766/852 | 1,100,388 | 152.4M |
| vocab1000_product | 0.49050 | 0.49954 | 0.49946 +/- 0.00248 | 7.671 | 0.097 | 121 | 2.1 | 130 | 767/852 | 1,299,944 | 152.6M |
| siglip_fusion | 0.45792 | 0.59386 | 0.59388 +/- 0.00317 | 9.843 | 0.077 | 128 | 3.2 | 102 | 1294/1510 | 1,624,676 | 204.8M |
| reasoner | 0.45602 | 0.59139 | 0.59580 +/- 0.00741 | 9.172 | 0.921 | 118 | 2.2 | 109 | 843/950 | 21,099,620 | 172.4M |
| e8a_A0p | 0.45852 | 0.59463 | 0.59446 +/- 0.00030 | 9.212 | 0.137 | 120 | 2.2 | 109 | 844/952 | 21,362,276 | 172.6M |
| e8a_A1 | 0.44332 | 0.57493 | 0.57316 +/- 0.00154 | 20.146 | 1.330 | 238 | 2.8 | 50 | 1108/1176 | 21,395,044 | 307.2M |

The multi-seed means are the architecture-level context (vocab-supported
metric); the timed checkpoint is always the seed-0 one whose frozen
accuracy appears in its own column. Normalised exact equals raw exact on
both vocabularies (zero collisions, g21_scorer_inventory.json).

Primary multimodal Pareto frontier, common-denominator accuracy against
measured serial latency: **concat, fusion, vocab1000_product**. Every
token-level and SigLIP system is dominated: vocab1000_product delivers
the highest raw-distribution accuracy (0.4905) at global-head serial cost
(7.67 ms). E7a's headline — that top-1000 global heads are the only
end-to-end Pareto-optimal systems — survives the measured serial test.

Per-stage medians (ms): the CLIP-global pipeline is decode-bound then
tower-bound (read 0.03, decode+preprocess 3.04, image H2D 0.09, vision
tower 2.24, tokenise 0.10, text tower 2.17, head 0.03-0.09); SigLIP's
vision tower costs 4.99; the reasoner adds 1.75-1.80 of latent reasoning
on top of the token path; e8a_A1 is dominated by the frozen SmolLM2
forward at 13.46 — the first online SmolLM2 timing in the project, about
two-thirds of A1's whole serial query. The additive stage sums sit slightly ABOVE the
unsegmented measured totals on every run: measured minus additive spans
-0.4086 to -0.0105 ms across all 24 records, consistent with the
per-stage synchronisation overhead the segmented pass carries, so
inter-stage overhead is negligible within this harness and the additive
sum is, if anything, a slight over-estimate of the serial total here.

Measured against E7a's additive estimates: the CLIP-global measured
serial medians (7.61-7.67 ms) sit about 1.3 ms (about 20 per cent) above
E7a's additive full_pipeline_ms (6.34-6.35 ms); the gap traces to the
decode and text-tower stages measured in-pipeline rather than in
isolation (3.04 against 2.295 and 2.17 against 1.73). This is exactly the
error mode an additive estimate cannot see, and is why E7b supersedes
those columns.

Cached regimes (measured): cached-image question-side 1.91-1.95 ms for
the CLIP globals, 2.28 for SigLIP, 3.33-3.37 for the token systems and
13.63 for A1; cached-feature head-only 0.024-0.048 ms for the globals and
1.33-1.44 ms for the reasoner class, matching the stored E7a/E8A
head-only figures across a different harness. Batch-64 image-branch
throughput is recorded as a labelled secondary figure in the artefacts.

## Outputs

results/experiments/e7b_serial_efficiency/: benchmark_rows.json (pinned
64 rows with image hashes), preflight.json, 24 immutable per-run records,
results.json, e7b_results.json (aggregate, frontier, supersession block,
cache-integrity record), latency_memory.csv, stage_timings.csv,
pareto_serial.png, cache_hashes_before/after.txt, and the pilot evidence
(pilot_e8a_a1.json plus its three qualification per-run records under
pilot_evidence/, tracked, and superseded for measurement by the
full-run A1 records). tests/test_e7b.py (55 checks) covers the gates.

## Decisions and problems

The A1 qualification pilot's timings are evidence only; A1 was rerun
inside the randomised full schedule, where its warm median (20.15 ms,
passes 19.02/20.35/20.15) sits slightly above the standalone pilot
(18.65-19.22 ms), consistent with pass-order thermal state — which is why
the full run, not the pilot, is authoritative. One forced-offline
preflight attempt failed on open_clip optional-file probing and wrote
nothing; the qualified online-capable configuration was used with
before/after cache-hash identity as the no-mutation proof. Total
full-benchmark wall time, measured from the run log's creation to its
final write: 628 seconds (10.5 minutes) on otter155 for all 24 child
runs including every model load; the preflight took about 7 minutes and
the pilot 106 seconds, so E7b's whole execution cost is under 0.35
GPU-hours of wall occupancy.

Development results only; nothing here touches the clean test.
