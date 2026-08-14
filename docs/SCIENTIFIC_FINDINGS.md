# Scientific findings

The project's results organised by discovery rather than by experiment
number. Each finding names its supporting experiments, its strongest single
piece of evidence, the confidence it deserves, and what it does not show.
All results are development-set results under the V2 protocol; exact values
and citations are in [EXPERIMENTS_AND_RESULTS.md](EXPERIMENTS_AND_RESULTS.md).

## 1. Language priors are strong

A question-only head reaches 0.4580 at 40k and 0.4977 at 250k against a
majority floor of 0.2247 — over half of achievable accuracy with no image at
all. In every trained system, removing the question is far more damaging
than removing the image (E8B: fixed-question drops of 0.31-0.51 against
image-derangement drops of 0.08-0.13).

- Supporting: v2_02, v2_07, E8B interventions.
- Confidence: high; reproduced across five seeds, three scales and every
  model family.
- Limitation: a property of GQA's answer distribution as much as of the
  models; it says nothing about whether the visual signal is unnecessary.

## 2. Visual information still materially contributes

The multimodal margin over question-only is +0.066 at 40k and grows to
+0.081 at 250k; shuffling or deranging the image costs every multimodal
model 0.08-0.17 accuracy, and a wrong real image drives the heads below the
question-only floor: it actively misleads.

- Supporting: v2_06, v2_07, E8B Table 9, E9 derangement.
- Strongest evidence: the derangement drops are directional with clustered
  intervals excluding zero in every arm and scale measured.
- Confidence: high for dependence.
- Limitation: visual reliance is dependence, not proof of deep or
  compositional visual reasoning. The project never equates the two.

## 3. Lightweight interaction design matters in low-data settings

At 40k, adding a single multiplicative interaction term to a concat head
buys about one point at equal capacity (+0.0096, positive in every seed),
and either term (product or absolute difference) carries the whole gain;
the two are redundant.

- Supporting: v2_02, v2_03, v2_04, v2_05 (gain concentrated in
  verify/logical/obj/rel).
- Confidence: high at 40k; the sign is seed-stable.
- Limitation: the mechanism reading (an agreement signal) is
  interpretation, not a causal demonstration.

## 4. Capacity explains only part of the fusion benefit

Doubling concat's parameters lifts it +0.0086; the features add +0.0050 to
+0.0094 on top depending on the budget. The v2_02 gap of +0.0144 was
roughly half capacity, half features.

- Supporting: v2_03 (two-sided parameter matching).
- Confidence: moderate-high; the closure's clustered interval for
  fusion-minus-concat_wide at 40k (+0.00578 [-0.00116, +0.01226]) does not
  exclude zero, so the feature share at the large budget is not established
  once dev-set uncertainty is included.
- Limitation: 40k scale only.

## 5. The fusion advantage narrows as data grows

fusion-minus-concat falls from +0.0144 (positive in all five seeds) at 40k
to +0.0037 +/- 0.0044 at 250k, where the sign flips in two of five seeds.
The handcrafted interaction terms are a small-data prior that a plain
concat head learns from data.

- Supporting: v2_07.
- Confidence: high for the trend.
- Limitation: v2_07 carries no image-clustered intervals (its row-level
  reconstruction stopped on a one-row mismatch), so the 250k statement
  rests on seed spread, not clustered uncertainty.

## 6. More reasoning capacity is not automatically better

At 40k, a 21.1M-parameter latent-query reasoner matches a 1.1M fusion head
(+0.0031 +/- 0.0062, negative in two of three seeds) at about 30 times the
head latency. At 100k/250k it does pull ahead (+0.0065 / +0.0136 over
fusion in every seed) — but that is a configuration-level finding in which
architecture, token access and capacity change together.

- Supporting: v3_01 (negative), v3_03 (scaled), E7a/E7b (cost).
- Confidence: high for the 40k parity; the scaled win is deliberately not
  attributed to any single factor.
- Limitation: one architecture point, three seeds, no factor isolation.

## 7. Token-level access does not eliminate the multi-step deficit

The question-weighted pooled deficit on 4-or-more-step questions persists
for every model at every scale (about 0.06-0.11 across the project), and
the reasoner's deficit is statistically indistinguishable from fusion's at
40k, 100k and 250k (all paired intervals include zero).

- Supporting: v2_05b/c, v2_07, v3_02a, v3_03, E2, E8A.
- Strongest evidence: the closure attached clustered intervals to twelve
  pooled deficits and all twelve exclude zero — the deficit itself is real
  in every family measured.
- Confidence: high; this is the project's most robust negative finding,
  observed across two frozen encoders, three scales and heads from 0.1M to
  21.1M parameters.
- Limitation: the step-count slice is confounded with question format; the
  claim is about the multi-step axis specifically, not "reasoning" broadly.

## 8. Question-side pretraining is beneficial

A frozen pretrained SmolLM2-135M question encoder beats its
architecture-matched random-initialised twin by +0.03820 [0.02947, 0.04680]
at 40k and +0.04835 [0.03935, 0.05803] at 250k, growing with scale, with a
non-degenerate control.

- Supporting: E8A core.
- Confidence: high within its design (three seeds, clustered intervals
  excluding zero, all seeds agreeing in sign).
- Limitation: this is pretrained-versus-random at fixed size. The whole SLM
  question path still loses to the interface-matched CLIP control (A1 minus
  A0p negative at both scales) and doubles the serial query cost; the
  comparison against A0p cannot isolate language-model semantics from the
  shared-pretrained-space property, and the recipe's provenance favours
  A0p.

## 9. Answer-side pretraining is not reliably beneficial

A frozen pretrained SmolLM2-135M answer readout does not beat its random
control: B3 minus B2 is -0.02463 [-0.03333, -0.01619] at 40k (all seeds
negative) and -0.00475 [-0.01208, +0.00297] at 250k (interval spans zero).

- Supporting: E8B core matrix.
- Confidence: high for "no positive advantage detected"; the 40k negative
  is directional.
- Limitation: the 250k interval includes zero and admits small effects in
  both directions — this is a failure to detect, and establishes neither
  equivalence nor absence. B3 at 40k is also seed-unstable (SD 0.02795),
  which the fixed-epoch-22 rule surfaces honestly.

## 10. Larger answer-side capacity does not reverse that conclusion

At 360M, the pretraining effects are -0.00683 [-0.01508, +0.00114] at 40k
and -0.00238 [-0.01071, +0.00588] at 250k, seeds disagreeing in sign, and
the pretraining-by-scale difference in differences is +0.00445 [-0.00709,
+0.01641]. E10 reproduces E8B's headline at 360M; it does not reproduce
E8B's directional negative 40k effect.

- Supporting: E10 twelve-cell matrix.
- Confidence: high within its design (pre-registered, pair-preserved,
  first-attempt, zero retries).
- Limitation: whole-system capacity sensitivity, not an isolated
  language-model-size effect: trainable capacity moves with the hidden
  size (+0.923 per cent). All intervals condition on the fixed three-seed
  set.

## 11. Representation quality strongly shapes outcomes

Swapping frozen CLIP for frozen SigLIP-B/16 lifts every multimodal global
head by +1.2 to +1.5 points in every seed while question-only barely moves;
the strongest SigLIP global heads reach the 21.1M reasoner's accuracy at
13-17 times smaller. And yet the multi-step deficit range does not move
(0.063-0.102 against CLIP's 0.064-0.105).

- Supporting: E2; corroborated by E3 (answer-space design changes
  raw-distribution accuracy by +3.6 to +4.1 points at 250k).
- Confidence: high for the accuracy shift (clustered intervals at 40k
  exclude zero for every multimodal head).
- Limitation: SigLIP heads carry more parameters at equal hidden width
  (stated, not matched); the 250k SigLIP-minus-CLIP contrast has no
  clustered interval (v2_07 reconstruction stop).

## 12. Efficiency advantages are real but bounded to the measured protocol

Measured serially on one node, the small global heads answer a full query
(raw image and question to answer) in about 7.6 ms; the top-1000 product
head delivers the project's best raw-distribution accuracy (0.4905) at that
cost, and the primary accuracy-latency frontier is concat, fusion and
vocab1000_product. Under the same-node contextual warm-serial protocol,
the SmolVLM-500M open-readout row measured 151.380 ms at 0.489004 accuracy
(about 24.4 times the 6.199 ms lightweight top-1000 head), and the
constrained row 152.854 ms at 0.485706 (about 24.7 times).

- Supporting: E7b (authoritative end-to-end), E7a components, E9 bridge.
- Confidence: high within the measured protocol; the harness reproduced
  every stored accuracy and answer exactly.
- Limitation: latency and parameter statements only. No energy, power,
  carbon or monetary measurement exists anywhere in the project, so no
  "cheaper", "energy-efficient" or cost claim is permitted. E7a's additive
  end-to-end sums and E7b's peak-memory and cold-start fields are
  withdrawn; the two measurement nodes' frontiers are never merged
  (failed bridge control at -18.7 per cent).

## Reading these findings together

Findings 1-7 say the closed-set frozen-representation formulation extracts
most of what is cheaply extractable: language prior plus a real but bounded
visual contribution, with a compositional ceiling that no head-side
intervention moved. Findings 8-10 say pretrained language knowledge helps
exactly where representations are consumed (the question side) and not
where answers are emitted (the answer side) under this interface. Findings
11-12 say the representation and the answer space, not head size, are the
levers that move accuracy, and that the simplest systems own the measured
efficiency frontier. Together they support the dissertation's answer:
lightweight systems on frozen representations provide useful GQA accuracy
at millisecond measured latency, but capacity alone is not sufficient, and
the persistent multi-step deficit marks the boundary of what this
formulation extracts from current frozen representations.
