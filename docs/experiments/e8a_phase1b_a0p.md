# E8A Phase-1B: the A0p control and the three-way seed-0 pilot

A **Phase-1B pilot record**, not a completed experiment report. It covers one
new arm at one scale with one seed, and the three-way comparison that arm makes
possible. It establishes no directional claim, and no number in it is
confirmatory. Development results only; the clean test was not touched.

## Purpose

Phase 1A left one gap it could not close: `A1 - A0p`, the matched system
comparison the canonical protocol specifies, had no A0p to compare against. The
stored `v3_01` artefact is arm **A0**, which lacks the interface-matching
projection, so substituting it would reintroduce exactly the confound A0p
exists to remove. Phase 1B builds A0p, runs it at train_40k seed 0, and puts
the three seed-0 pilots side by side.

## Method

**A0p** is the unmodified latent-query reasoner plus **one trainable
`Linear(512, 512)`**, 262,656 parameters, applied token-wise to the frozen CLIP
question-token sequence:

    frozen CLIP question tokens [B, L, 512]
      -> valid-token selection and key-padding mask
      -> one trainable token-wise Linear(512, 512)
      -> [B, L, 512]
      -> the unmodified latent-query reasoner of src/reasoner.py
      -> 100-way answer classifier

21,362,276 trainable parameters, against A0's 21,099,620 and A1's 21,395,044.
Recipe 7.1 inherited unchanged, train_40k, seed 0.

The CLIP question store's on-disk layout (`ids`/`offsets`/`lengths`/`tokens`)
is structurally identical to the Phase-1A SLM stores, so A0p was implemented as
an adapter over it and **runs the same dataset, collate, gate, training,
intervention and evaluation code as A1 and A1r**. The arms differ in their
question store and in the projection's input width, and in nothing else. G13
was extended to assert trunk bitwise identity across all three arms at once;
it holds, at `10a34fa2ec649f89…`.

A0p's pinned neutral question is the CLIP token states of the fixed string
`"question"`, reproduced through the identical frozen text path `v3_00` used —
token embedding, positional embedding, transformer under its causal mask,
`ln_final`, `text_projection`, unnormalised, length taken as the EOT position
plus one. A test compares it against an independently recomputed path and finds
byte equality.

A0p writes only to its own namespace. Every stored A0 checkpoint and all four
A1/A1r artefacts were hashed before the run and re-verified byte-identical
after it.

## Outputs

- `results/experiments/e8a_question_encoder/pilot_A0p.json`, `gates_A0p.json`,
  `three_way_comparison.json`
- `checkpoints/e8a_A0p_train_40k_seed0.pt`, 81.53 MiB, SHA-256
  `ae76485bd023cafddf1dfc7e8a53e2af031c6b9a2301954d2b97a60a9240867f`
- `correctness_A0p_seed0.npz` — per-question correctness for all five
  evaluation conditions
- `experiments/e8a_question_encoder/run_a0p.py`, `compare_three_way.py`
- `tests/test_e8a.py` — 149 checks

## Results

### Gates

All A0p gates passed: G0 recipe identity; G1 answer-to-label binding on all
47,714 rows; G4; G5 (1,553 padded positions corrupted, logits moved by exactly
0); G9 bitwise-identical reload; G10; G11; G12; G13; G15; G17; G18. The
scheduler matched the canonical cosine formula exactly at seven probe steps
over the horizon `31,300 = 100 x 313`.

**G8 tiny-overfit**: loss 4.0268 to 0.0837, 0.9940 train accuracy at epoch 12,
with the projection, trunk and classifier gradient norms all finite and
non-zero — the last of these added in Phase 1B, so G7's gradient-reach leg is
now recorded for every arm including the one that has no encoder in the graph.

### The three-way pilot table, seed 0, train_40k

Compatibility was verified before any difference was computed: identical train
and dev manifests, seed, evaluation rows, answer-vocabulary SHA-256, reasoner
trunk initialisation, checkpoint-selection rule, training recipe, scoring code
and metric definition. All nine checks passed.

| | A0p | A1 | A1r |
|---|---|---|---|
| dev accuracy | **0.53578** | **0.52541** | **0.48976** |
| selected epoch | 5 | 4 | 3 |
| epochs run | 15 | 14 | 13 |
| seconds/epoch | 15.72 | 15.71 | 15.71 |
| wall clock, GPU-hours | 0.0656 | 0.0611 | 0.0568 |
| normal − fixed-image | 0.08491 | 0.07597 | 0.10319 |
| normal − fixed-question | 0.48782 | 0.29803 | 0.26070 |
| normal − shuffled image (derangement) | 0.10721 | 0.10293 | 0.11006 |
| prediction entropy, nats | 0.79681 | 0.90797 | 1.02259 |
| maximum class share | 0.22855 | 0.22880 | 0.30646 |
| distinct answers | 99 | 97 | 89 |
| trainable parameters | 21,362,276 | 21,395,044 | 21,395,044 |
| frozen question encoder | 63,428,096 | 134,515,008 | 134,515,008 |
| total loaded parameters | 84,790,372 | 155,910,052 | 155,910,052 |
| head latency, ms | 1.3816 | 1.3502 | 1.3368 |
| peak GPU memory, MiB | 1,272.5 | 1,249.2 | 1,249.2 |

The frozen-encoder counts exclude the frozen CLIP **image** tower, which is
common to all three arms.

### Contrasts

| Contrast | Value | What it is |
|---|---|---|
| `A1 − A1r` | **+0.03565** | the HA3 primary causal contrast, within size |
| `A1 − A0p` | **−0.01037** | a *system comparison*, not a causal estimate |
| `A0p − A1r` | **+0.04602** | descriptive only |

**These are one seed at one scale.** The canonical three-seed directional rule
is not applied and cannot be: it requires the three per-seed paired
differences, their mean and standard deviation, and a 95 per cent
fixed-seed-set image-clustered interval over seeds 0, 1 and 2. Seeds 1 and 2
are not authorised and have not run.

For scale, the across-seed standard deviation measured in the stored `v3_01`
runs at 40k is **0.0037**. Two of the three contrasts above are within a few
multiples of that, so nothing here is statistically established.

**Nothing in this report claims that pretraining is beneficial, that SLM
representations outperform CLIP, or that any of these differences is real.**

`A1 − A0p` carries two mandatory disclosures. The first is the canonical label,
verbatim:

> Matched system comparisons under a recipe historically selected on the
> CLIP-question-token reasoner. This selection asymmetry favours A0p and makes
> the comparison conservative with respect to the SLM arm.

The second is a confound this pilot surfaced and the protocol names in section
5: **A0p's question and image tokens both come from the same frozen CLIP and
share its pretrained space by construction, whereas A1's do not.** The single
architectural difference between the arms therefore moves two things at once —
the question semantics, and whether the two modalities share a pretrained space
— and `A1 − A0p` cannot separate them.

### What requires seeds 1 and 2

- `A1 − A1r` at 40k and at 250k — the HA3 primary causal contrast. Three seeds
  per arm per scale, with the clustered interval.
- `A1 − A0p` at 40k and at 250k — the HA1 system comparison and, through the
  difference-in-differences, HA2.
- Any statement that a difference is positive, negative or absent.

### Interventions

The fixed-question difference is **not** a pure semantic contribution and is
not reported as one. A0p's pinned neutral question is 3 valid CLIP positions
(SOT, the word, EOT) against a measured mean of **12.19** CLIP tokens on
train_40k and **12.18** on dev; A1 and A1r's is 1 SmolLM2 token against a mean
of 10.93. So the intervention confounds question content with a
sequence-length change, and the size of that confound differs across the arms
by construction, because each arm's neutral sequence comes from its own
tokenizer exactly as the protocol specifies. A0p's much larger figure (0.488
against 0.298 and 0.261) should be read in that light.

The degeneracy rule fires for no arm. A1r remains a non-degenerate control:
entropy 1.023 nats against a 0.30 threshold, class share 0.306 against 0.60,
and `normal − fixed-question` 0.261 against 0.01.

### Resource update, from measurement

| Quantity | Measured or projected |
|---|---|
| seconds/epoch at 40k, max over the three arms | **15.77** |
| one 40k run, expected epochs | **0.0657 GPU-h** |
| one 40k run, 100-epoch cap | **0.4379 GPU-h** |
| one 250k run, expected | **0.6021 GPU-h** |
| one 250k run, 100-epoch cap | **2.7370 GPU-h** |
| extraction, one complete 135M pass | **0.5835 GPU-h measured** |
| remaining 40k seed runs (6: seeds 1-2 for three arms) | 0.394 h expected, 2.628 h at the cap |
| 250k tranche (9: three arms x three seeds) | 5.419 h expected, 24.633 h at the cap |
| **complete 135M core tranche** (18 runs + 2 extraction passes) | **7.177 h expected, 8.613 h with 20 % contingency** |
| peak GPU memory | 1,272.5 MiB against the 16,014 MiB ceiling |
| storage written across Phases 1A and 1B | 0.928 GiB |

The 250k figures use the row ratio 6.25 rather than v3_03's measured 5.656,
the conservative choice, because no E8A 250k run exists.

**Gate outcomes.** The 8-hour per-run ceiling does not fire: the largest
projected run is 2.737 h, 5.263 h inside it. The 80-per-cent memory ceiling
does not fire. The storage gate does not fire; A0p writes no hidden-state store
at all, because it reads the CLIP question tokens `v3_00` already produced.

**Three gates remain unresolved and are not claimed to be resolved**: the E8B
per-epoch multiplier (still the back-solved assumption 2.931 to 8.004; no E8B
code has ever run), the 35 GPU-hour cross-E8 per-model aggregate for pretrained
SmolLM2-135M (which sums A1 + A4 + A7c + B3 and needs B3), and the
`min(180, 3 x expected)` programme gate, which the same unknown dominates.

> **Dated supersession note, 8 August 2026.** The "35 GPU-hour" cross-E8
> per-model aggregate named above was the ceiling ACTIVE WHEN THIS REPORT
> WAS WRITTEN. It was superseded programme-wide by a **40 GPU-hour**
> per-model-identity ceiling on 8 August 2026, by explicit user decision.
> The sentence above is preserved unchanged as historical evidence and is
> correct as of its date; it does not state the current ceiling. The single
> authoritative constant is `config.PER_MODEL_IDENTITY_CEILING_HOURS`. See
> `results/cross_e8_ceiling_supersession_20260808.json`. No measured hour,
> result or scientific conclusion in this report changes.

> **Dated supersession note, 7 August 2026.** The E8B statements in the
> paragraph(s) above are superseded and are retained here only as the
> historical record of what was known when this report was written. E8B
> HAS since run: search grid points 1 to 3 executed on 6-7 August 2026,
> so B3 is no longer without measurement, and the per-epoch multiplier is
> no longer an unresolved 2.931-to-8.004 assumption. It was measured at
> 9.091 and 9.121 under the then-current BF16 design - **above** the
> assumed 8.004 upper bound - and those figures are themselves now
> superseded by the amended strict-deterministic FP32 core design. The
> eight-point search was afterwards permanently abandoned and grid points
> 1 to 3 are exploratory protocol-diagnostic evidence only. For the
> current position see
> `results/experiments/e8b_readout_generation/core_resource_projection_20260807.json`,
> `protocol_amendment_20260807_fp32.json`,
> `protocol_amendment_20260807_fixed22.json` and
> `superseded_evidence_20260807.json`. Nothing above has been rewritten.

The extraction measurement is the one figure that exceeds its canonical
assumption: 0.5835 h per complete 135M pass against an assumed 0.200-0.400 h.

### The 426-versus-420 discrepancy

Recorded, unchanged from Phase 1A: the six B1 classification evaluations are
counted in the canonical total of 426 but have no printed itemised cost row in
the table that itemises 420. The stated totals already include them — the
thirteen printed rows sum to 60.822 / 122.875 against stated totals of
61.072 / 123.375, a difference of exactly `6 x 0.041667` and `6 x 0.083333` —
so **no stated GPU-hour total changes**. The frozen canonical resource table
was not modified for presentation; that would need its own reviewed amendment.

## Decisions and problems

**The G20 ambiguity was resolved by the user and applied as a narrow
amendment.** G20 is a storage round-trip fidelity gate; it does not use an
independently recomputed fp32 forward as its pass or fail reference. The
fp32-forward differences remain disclosed as compute-precision sensitivity
diagnostics. Three fresh-context reviews were run: scientific
(`changes_requested`, one HIGH — the Phase-1A report still described the matter
as unresolved), protocol/scope (`approved`, state preserved), and a narrow
re-review after the fixes (`approved`). The canonical amendment is confined to
section 18.1, its gate-table row, the header authority list and the provenance
block; every run count, resource figure, ceiling, recipe constant, threshold
and the universal directional rule are byte-identical to the pre-amendment
file.

**Two blockers were caught by review, both in the same place.** Generalising
the training path over the arm left a dangling `del lm`, and the fix for the
RMS support left a dangling `all_states`. Either would have completed the full
pilot and then died building its record. Both lived in `train_arm` and
`evaluate_arm`, which no test executes. The response was a static check over
every function in the package that flags any name used but bound nowhere in
that scope; its first version was itself unsound — it resolved names against
every binding anywhere in the file, so it would not have caught the first
blocker — and was rewritten to respect scope boundaries, with four
known-negative probes that reproduce the historical defects in their real
setting.

**A0 is not a matched comparator for A0p, and the report says so.**
Architecturally they differ in exactly the trainable `Linear(512, 512)`. But
the stored A0 was trained through a different loader whose shuffle generator
uses the default seed 42 regardless of the run seed, while A0p uses
`make_generator(seed)`. The two runs therefore also differ in training data
order. A0's 0.54641 is cited as context only.

**What this pilot does not establish.** One seed, one scale, one model size. No
HA1, HA2 or HA3 conclusion. The E8B multiplier, which dominates the programme
total, is untouched. Two of the five numeric compute gates remain unevaluable.

> **Dated supersession note, 7 August 2026 (second location).** The
> statement immediately above is superseded on the same grounds as the
> note earlier in this report, and is kept as the historical record. The
> E8B multiplier is no longer untouched: grid points 1 to 3 ran on 6-7
> August 2026 and measured it at 9.091 and 9.121 under the then-current
> BF16 design, above the assumed 8.004 upper bound; those figures are in
> turn superseded by the amended strict-deterministic FP32 design. The
> compute gates are no longer unevaluable: all five are evaluated in
> `results/experiments/e8b_readout_generation/core_resource_projection_20260807.json`,
> and the 35-hour per-model-identity ceiling is now enforced in code by
> `run.per_identity_gate` rather than checked by hand. Nothing above has
> been rewritten.
>
> **Amended 8 August 2026.** That ceiling is now **40 GPU-hours**, not 35,
> superseded programme-wide by explicit user decision and read from the
> single authoritative constant `config.PER_MODEL_IDENTITY_CEILING_HOURS`.
> `run.per_identity_gate` still enforces it; only the value changed. The
> line above states the ceiling active when this note was written. See
> `results/cross_e8_ceiling_supersession_20260808.json`.
