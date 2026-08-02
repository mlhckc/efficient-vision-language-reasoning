# E8A Phase-1A: bounded pilot of the frozen-SLM question encoder

This is a **Phase-1A pilot record**, not a completed experiment report. It
covers two runs of one arm pair at one scale with one seed. It is not the E8A
result, it establishes no directional claim, and no number in it is a
confirmatory or final result. The completed E8A experiment, if authorised,
will have its own report.

## Purpose

The user authorised Phase 1A only: implement the canonical E8A sequence
interface for SmolLM2-135M, verify it, measure what it costs, and stop. The
pilot answers three questions and no others:

1. Does the implementation satisfy the approved protocol's gates?
2. What does an E8A run actually cost in time, memory and storage?
3. Does the interface learn at all, and is the random control degenerate?

The complete E8A matrix, seeds 1 and 2, train_250k, SmolLM2-360M, E8B, E9, the
clean test, F1 and F2 are all outside this pilot and were not run.

## Method

The canonical primary interface (master protocol section 8.0, E8A plan
section 3):

    question text
      -> SmolLM2-135M tokenisation
      -> frozen last_hidden_state [B, L, 576]
      -> valid-token selection and key-padding mask
      -> one trainable token-wise Linear(576, 512)
      -> [B, L, 512]
      -> the unmodified latent-query reasoner of src/reasoner.py
      -> 100-way answer classifier

Two arms: **A1**, pretrained frozen SmolLM2-135M at revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; **A1r**, an architecture-matched
random SmolLM2-135M built with `from_config` from the same config under the
pinned seed 20260802, every parameter frozen, no pretrained weight loaded.
Both at `train_40k`, seed 0, under the fixed inherited v3_01 recipe (7.1):
AdamW, batch 128, lr 3e-4, warmup 0.0, dropout 0.1, weight decay 0.01 on
`ndim >= 2`, gradient clip 1.0, patience 10, 100-epoch configured maximum,
cosine-after-warmup `LambdaLR` over `100 x steps_per_epoch` stepped per
optimizer step, bf16 autocast on the training path only, fp32 evaluation,
best-development-accuracy checkpoint selection with the earliest epoch winning
ties.

`src/reasoner.py` was imported unmodified and its content is pinned by SHA-256
in the test module. New code is confined to
`experiments/e8a_question_encoder/` and `tests/test_e8a.py`.

**Extraction encodes one question string per forward pass.** This was decided
by measurement, not convenience: in bfloat16 the same string encoded in
batches of 1 and 64 gives hidden states differing by up to about 3 per cent
relative even with no padding present, while the same comparison in float32
agrees to about 3e-6 relative. The float32 control establishes that the
attention masking is correct and the disagreement is bfloat16 rounding.
One-per-forward makes each stored state a pure function of its own tokens and
is bitwise reproducible; the store regenerated in a separate process hashed
byte-identically to an earlier one.

Commands, with full evidence (command, UTC, HEAD, worktree, exit code, stdout,
stderr) recorded per run:

    python -B tests/test_e8a.py
    python -B tests/run_all.py
    python -B experiments/e8a_question_encoder/extract_hidden.py --preflight
    python -B experiments/e8a_question_encoder/extract_hidden.py --write
    python -B experiments/e8a_question_encoder/run_pilot.py --gates
    python -B experiments/e8a_question_encoder/run_pilot.py --train
    python -B experiments/e8a_question_encoder/stage8_projection.py

## Outputs

- `data/v3_slm_tokens/e8a_135m_A1_train40k_dev.h5` — 0.4638 GiB, SHA-256
  `0572b9a8b7db5ba7da06eb9f39e5c87dbe7c66be805050ca7d2daf24902a24a6`
- `data/v3_slm_tokens/e8a_135m_A1r_train40k_dev.h5` — 0.4638 GiB, SHA-256
  `f42e36b7dd2c27f9c1e3e695a57764a5ac49c54bde44a4ebe702d8e8afce4476`
- `results/experiments/e8a_question_encoder/extraction.json`, `gates.json`,
  `pilot.json`, `stage8_projection.json`. `extraction.json` predates the
  user's G20 decision and was deliberately not rewritten: its wording reflects
  the interpretation in force when the run executed, and its numbers are
  unaffected by the decision.
- `results/experiments/e8a_question_encoder/checkpoints/` — two selected
  checkpoints, 81.66 MiB each, hashed in `pilot.json`
- `results/experiments/e8a_question_encoder/correctness_{A1,A1r}_seed0.npz` —
  per-question correctness vectors for all five evaluation conditions
- `experiments/e8a_question_encoder/` — the implementation
- `tests/test_e8a.py` — 97 checks

## Results

Every figure below comes from a real run.

### Gates

All pre-training gates passed: G0 recipe identity against the live v3_01
constants and its stored `selected_config`; G1 answer-string to label-index
binding on all 40,000 training and 7,714 development rows, zero mismatches;
G4 one-batch forward; G5 corrupting 1,552 padded positions changed the logits
by exactly 0; G7 all 272 frozen language-model parameters had `grad is None`
and `requires_grad False` while all 119 trainable tensors received finite
non-zero gradients, for both arms; G9 reload reproduced bitwise-identical
logits; G11 repeated evaluation bitwise identical; G12 read-only stores; G13
the reasoner trunk was bitwise identical across A1, A1r and a freshly seeded
unmodified `LatentQueryReasoner`, and the projection initial weights were
identical within the pair; G15 seven artefacts pinned by path and SHA-256;
G17 no clean-test path resolvable; G18 the V2 vocabulary SHA-256 matched the
canonical pin; G20 as below.

The scheduler was verified to equal `0.5 * (1 + cos(pi * progress))` exactly
at seven probe steps over the horizon `31,300 = 100 x 313`, stepped per
optimizer step.

**G8, tiny-subset overfit** (1,000 examples, the v3_01 GATE 3 settings
inherited unchanged): A1 reached 0.9950 train accuracy at epoch 12, loss
4.0542 to 0.0679. A1r reached 0.9940 at epoch 10, loss 3.7061 to 0.1454. Both
pass; for A1r this is a recorded diagnostic rather than a halting gate. That
the random arm also overfits 1,000 examples is expected: 21.4M trainable
parameters over an intact image path can memorise that set regardless of the
question representation.

### G20 extraction precision

| Arm | relative L2 reconstruction | non-finite after cast | max abs activation |
|---|---|---|---|
| A1 | 0.0 | 0 | 46.25 |
| A1r | 0.0 | 0 | 4.78 |

The reconstruction error is exactly zero **by construction**, and this is
recorded so the pass is not misread as a validation of fp16 storage in
general: bfloat16 carries 8 mantissa bits and float16 carries 11 in the normal
range, so every finite bfloat16 value inside float16's exponent range is
exactly representable. The gate can still fail on overflow or non-finite
values, which is the failure mode it genuinely detects.

Measured separately over the same fixed sample and recorded as a
**compute-precision sensitivity diagnostic** under canonical section 18.1 as
amended: the difference between the stored values and an independently
recomputed fp32 model forward is, as a **relative L2**, **0.04462 for A1 and
0.02553 for A1r**; the corresponding maximum absolute deviations are 2.86133
and 0.185956, and the per-arm percentiles are in `extraction.json`. No
threshold binds any of them and none is a G20 pass or fail value. They are
disclosed because they are the only recorded quantities bearing on the
numerical fidelity of the extraction as a whole. See "Decisions and problems"
for the user's decision of 2 August 2026 that settled which comparison G20
gates.

**What the write-and-reload leg was for these two stores.** Canonical section
18.1 as amended states that the leg is discharged by reopening the written
store and verifying it against the values held before the write. For A1 and
A1r that verification was structural — dataset shape, questionId count, and a
256-row prefix of the state block compared against memory — and it was not
recorded in `extraction.json`. The whole-store comparison now in
`extract_hidden.py`, covering all 426,790 state rows plus the offset and length
indexes, binds subsequent extractions. Neither store was regenerated: the user
directed that valid stores not be regenerated unnecessarily, extraction is
deterministic, the A1 store already hashed byte-identically across two separate
processes, and both stores were then consumed end to end by 14 and 13 training
epochs and five evaluation conditions each.

### Token budget

`L = 32` truncates nothing under the SmolLM2 tokenizer: maximum 31 tokens over
train_40k + dev and maximum 31 over train_250k + dev, mean 10.93 and 11.33,
truncation rate 0 on both scopes. `L` is unchanged and the canonical storage
estimate needs no revision. The tokenizer adds no special token at all —
`add_bos_token` and `add_eos_token` are both false and
`add_special_tokens=True` and `False` give byte-identical ids — so every
position in the E8A question sequence is a real question token. That is
measured, not assumed, and is asserted by a test that fails if a tokenizer
ever starts adding one.

### Pilot runs

| Arm | dev accuracy | selected epoch | epochs run | s/epoch | wall clock | peak memory |
|---|---|---|---|---|---|---|
| A1 | **0.52541** | 4 | 14, early stop | 15.705 | 220.1 s = 0.0611 GPU-h | 1,249 MiB |
| A1r | **0.48976** | 3 | 13, early stop | 15.709 | 204.4 s = 0.0568 GPU-h | 1,249 MiB |

Both arms have 21,395,044 trainable parameters (21,099,620 trunk + 295,424
projection) and 155,910,052 total loaded parameters including the frozen
language model. Loader throughput was about 22,000 samples/s; head-only
latency was 1.350 ms (A1) and 1.337 ms (A1r) per single example over cached
features, which excludes both frozen encoders and is not an end-to-end query
cost.

`A1 - A1r = +0.03565` at 40k, seed 0. **This is one seed at one scale.** It is
not the HA3 causal estimate, which requires seeds 0/1/2 at 40k and 250k with
the three per-seed differences and the fixed-seed-set image-clustered
interval. No directional claim is made from it.

### Same-checkpoint interventions

| Condition | A1 | A1r |
|---|---|---|
| normal | 0.52541 | 0.48976 |
| matched fixed-image | 0.44944 | 0.38657 |
| matched fixed-question | 0.22738 | 0.22906 |
| shuffled image, imageId derangement, 0 self-pairs | 0.42248 | 0.37970 |
| shuffled image, v3_01 row rule, 11 self-pairs | 0.42313 | 0.37970 |
| normal − fixed-image | 0.07597 | 0.10319 |
| normal − fixed-question | 0.29803 | 0.26070 |
| normal − shuffled (derangement) | 0.10293 | 0.11006 |
| prediction entropy, nats | 0.90797 | 1.02259 |
| maximum class share | 0.22880 | 0.30646 |
| distinct answers predicted | 97 | 89 |
| constant-answer rate | 0.22880 | 0.30646 |

Both shuffled-image constructions are reported because the canonical names two
and states that which is used must be stated per analysis: the imageId-level
derangement the Phase-1A instruction requires (zero self-pairs, asserted at
both the map and the row level), and the row-level permutation of
`v3_01/run.py:487` that makes the drop comparable with the stored v3_01
reference (11 observed self-pairs on 7,714 rows). They agree to within 0.0007.

**The pre-registered degeneracy rule does not fire for A1r.** Neither trigger
is met: `normal − fixed-question` is 0.2607 against a threshold of 0.01, and
the collapse criterion needs entropy at or below 0.30 nats or class share at
or above 0.60, against measured 1.023 nats and 0.306. A1r is therefore a
non-degenerate random control on this evidence, which matters because the
canonical would otherwise require `A1 − A1r` to be described as "pretrained
weights versus a degenerate random representation" rather than as a
pretraining contrast. That disposition is provisional on one seed.

Two caveats attach to the fixed-question intervention. The pinned neutral
string `"question"` tokenises to a **single** token against a measured mean of
10.93 for real questions, so `normal − fixed-question` confounds question
content with an 11-to-1 sequence-length change. The construction is exactly the
one master protocol section 11.1 pins; the confound is a property of that
pinned construction. And under that intervention both arms collapse to two
distinct answers with class shares of 0.687 and 0.873 — expected, since the
question carries no information, and not a degeneracy of the trained model.

### Resource re-projection from measurement

| Quantity | Canonical | Measured or re-projected |
|---|---|---|
| E8A seconds/epoch at 40k | 15.27 | **15.75** (max), ratio 1.032 |
| E8A 40k run, expected epochs | 0.063625 h | **0.06126 h** |
| E8A 40k run, 100-epoch cap | 0.424167 h | **0.43756 h** |
| E8A 250k run, expected | 0.527756 h | **0.60164 h** |
| E8A 250k run, 100-epoch cap | 2.398889 h | **2.73472 h** |
| extraction, one full pass | 0.200–0.400 h assumed | **0.5834 h measured at 135M** |
| E8A branch total, expected basis | 20.413–21.968 h | **24.283 h** |
| E8A branch total, cap basis | 94.176–95.730 h | **107.090 h** |
| peak GPU memory | 80 % ceiling = 16,014 MiB | **1,249 MiB, 7.8 %** |
| Phase-1A storage written | — | **0.928 GiB** |
| full sequence stores | 41.05 GiB padded estimate | **14.50 GiB packed** |

The 250k figures use the row ratio 6.25 rather than v3_03's measured 5.655
ratio, the more conservative of the two, because no E8A 250k run exists and
Phase 1A forbids one.

**Gate outcomes.** The 8-hour per-run ceiling does not fire: the largest
projected E8A run is 2.735 h at the configured cap, 5.265 h inside the ceiling,
and the largest measured run was 0.061 h. The 80-per-cent memory ceiling does
not fire, with 14.8 GiB of headroom. The storage gate does not fire; the packed
layout uses about a third of the canonical padded estimate.

**Two gates cannot be recomputed from Phase-1A evidence and are not claimed
to be.** The 35-hour per-model ceiling for pretrained SmolLM2-135M aggregates
A1 + A4 + A7c + B3, and B3 is an E8B arm with no measurement anywhere in this
repository. The `min(180, 3 x expected)` total-programme gate is dominated by
the E8B per-epoch multiplier, which remains the back-solved assumption 2.931
to 8.004. Phase 1A reduces that uncertainty by nothing at all, because it ran
no E8B code.

### The 426-costed versus 420-itemised discrepancy

Resolved by arithmetic on the canonical's own tables. Section 13.1 counts
`7+72+6+88+144+12+90+6+1 = 426` evaluation passes. Section 13.2 has a cost row
for `7+156+72+88+90+6+1 = 420` of them. **The six that are counted but have no
cost row are the B1 classification evaluations.**

The stated totals are nevertheless right and no GPU-hour figure is wrong: the
thirteen printed rows sum to 60.822 / 122.875, against stated totals of
61.072 / 123.375, and the difference is exactly `6 x 0.041667` and
`6 x 0.083333` — the six B1 passes at the stated per-pass rate. The stated
expected value 92.224 is likewise the sum of **fourteen** midpoints, not the
thirteen the text claims.

Three presentational corrections follow, for a future authorised round; the
canonical was not edited:

- add the row `B1 classification evaluations | 6 x 0.041667 / 0.083333 |
  0.250 | 0.500`;
- correct "sum of the thirteen per-component midpoints" to fourteen, and the
  configured-cap row "all evaluation rows, unchanged | 14.316 | 28.786" to
  14.566 / 29.287;
- correct the margin at section 13.2d from 32.550 h to **31.950 h**; 32.550 is
  `180 − 122.875 x 1.2`, the printed-rows-only total, while the correct value
  `180 − 123.375 x 1.2` is already stated correctly at two other places.

No gate outcome changes under either figure.

## Decisions and problems

**The implementation reviewer was dispatched after GPU extraction had begun,
not before.** The Phase-1A instruction requires a clean implementation review
before any GPU extraction or training. The bounded G20 preflight and one
complete A1 store write ran first. The deviation was caught mid-run and the
background write stopped; the round-1 reviewer independently raised it as a
BLOCKER. Remedy: both stores were regenerated from corrected, committed code
in a single gated run with a complete provenance record. The regenerated A1
store hashed byte-identically to the pre-review one, so nothing was lost and
the incident additionally demonstrates cross-process extraction determinism.
Reported to the user rather than absorbed.

**Section 18.1 admitted two readings; the user has since resolved it.** As
written, "fp16 storage is compared against fp32 on the fixed sample" could
mean the storage cast against the output it was cast from, or the stored
values against an independently recomputed fp32 model forward. The first
passes at exactly 0.0; the second is 0.0446 for A1 and 0.0255 for A1r, which
would not meet 1e-3. The pilot adopted the first, recorded both numbers per
arm, and escalated rather than deciding for the protocol.

**The user's decision of 2 August 2026 makes G20 a storage round-trip fidelity
gate**, comparing the frozen model's output under the authorised compute
precision against that same output after the storage cast, write and reload,
and explicitly not using an independently recomputed full-fp32 forward as its
pass or fail reference. The storage round-trip deviations of 0.0 therefore
pass G20 and both stores remain valid. The fp32-forward differences remain
disclosed as **compute-precision sensitivity diagnostics**: not storage-cast
errors, not G20 pass or fail values, never relabelled as zero, and not by
themselves a reason to regenerate a store. Applied to canonical section 18.1
by the Phase-1B narrow amendment; the decision is recorded verbatim in the
bridge packet.

**bfloat16 makes the frozen encoder's output depend on batch shape.** Found by
a test that was meant to confirm padding invariance and instead failed at 1.25
absolute. Diagnosed rather than suppressed: the float32 control agrees to
3e-6 relative, so the masking is correct and the effect is bfloat16 rounding
made visible by the kernel's tile shape. Answered by encoding one string per
forward pass, at a measured cost of 0.127 GPU-hours per 40k store.

**A0p does not exist and cannot be reused.** The stored v3_01 artefact is arm
A0, not A0p; the two differ in exactly the interface-matching
`Linear(512, 512)` that A0p exists to provide. Substituting A0 would
reintroduce the confound A0p removes. Recorded in the packet's A0p audit.
Consequence: **no `A1 − A0p` system comparison appears in this pilot.** The
stored A0 seed-0 development accuracy of 0.54641 is a contextual figure from a
single-seed run of a different arm and is labelled as such wherever cited.

**Two bugs were found by running, after the round-1 review.** The G12
read-only scanner matched its own source line, and the G9 save-and-reload check
compared logits drawn from a shuffling loader, so the two passes saw different
rows and it failed at a deviation of 21.28. Both were real defects in the
checking code rather than in the model path; both are fixed and the failed
gate record is retained as evidence.

**`constant-answer rate` is required by the canonical and defined nowhere.**
The round-10 scientific reviewer's own suggested clause was adopted — the
fraction of rows on which the model emits its single most frequent answer —
and disclosed in the emitted JSON alongside the note that it equals maximum
class share by construction. No decision rule consumes it.

**What the pilot does not establish.** It is one seed at one scale on one
model. It supports no HA1, HA2 or HA3 conclusion. A1's 0.52541 is below the
stored A0 reference of 0.54641, but A0 is not the interface-matched control
and that comparison is not the one the protocol specifies. The E8B multiplier,
which dominates the programme total, is untouched. Two of the five numeric
compute gates remain unevaluable.
