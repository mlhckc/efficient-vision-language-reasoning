# E9: a compact integrated VLM as a contextual baseline

## Purpose

Where does an off-the-shelf compact integrated vision-language model sit on the
GQA accuracy-efficiency-visual-reliance trade-off relative to this project's
lightweight systems?

E9 is a **contextual baseline, not a clean causal architecture experiment**. No
E9-versus-E8B difference is attributed to integrated architecture alone,
because training history and multimodal pretraining differ as well. SmolVLM is
externally pretrained and instruction-tuned; SmolVLM-256M's text backbone is
SmolLM2-135M-**Instruct** while E8A and E8B use **base** checkpoints. No direct
GQA supervised-training claim is made in either direction, and the absence of
GQA from a published training-dataset list is not proof of zero image or
content overlap. GQA imagery derives from Visual Genome, which Cauldron subsets
draw on, so image-level exposure is likely and E9 is never described as
zero-shot at the image level.

E9 answers the question E8B left open. E8B established that within the
frozen-encoder family, a pretrained small language model used as a readout
gives no positive primary-accuracy advantage over a within-size random control
(40k -0.02463, CI [-0.03333, -0.01619]; 250k -0.00475, CI [-0.01208, +0.00297]),
that R1, R2 and R3 agree within about 0.0005 so decoding is not the bottleneck,
and that data scale dominates every between-arm difference. It did not
establish what that family's accuracy is worth against a system pretrained end
to end as a VLM.

## Method

Evaluation only. Zero trainable parameters, no fine-tuning, no adaptation, no
LoRA, no sampling, no prompt search, no new dataset. Two frozen models, pinned
by revision:

| model | role | revision | licence | parameters | trainable |
|---|---|---|---|---|---|
| SmolVLM-256M-Instruct | PRIMARY | `7e3e67ed…` | apache-2.0 | 256,484,928 | 0 |
| SmolVLM-500M-Instruct | SECONDARY | `a7da5b98…` | apache-2.0 | 507,482,304 | 0 |

The protocol was frozen and hashed before any GQA score was observed
(`frozen_protocol.json`, sha256 `e38c0b9e…`): repository, revision, chat-template
digest, prompt string and digest, image-preprocessing configuration, dtype
(bfloat16), attention implementation (sdpa), `do_sample=False`, one beam, stop
tokens, `max_new_tokens`, batch size and every other output-affecting
parameter. One pre-declared prompt, applied through each model's own official
chat template with the image in the user turn:

```
Answer the question using a single word or phrase.
Question: {question}
Answer:
```

Each model's own official processor handles preprocessing, unmodified. With
its default configuration this upscales a GQA image to a longest edge of 2048
and splits it into 512-pixel tiles, producing about 866 image tokens per query
(1,088 for the blank image, which tiles differently). That is the model's
documented deployment configuration and was not tuned.

Two readouts share one code path and differ only by a logits mask:

- **Open** — free greedy generation, capped at 20 new tokens. **PRIMARY.**
- **Constrained** — greedy decoding masked to a prefix trie over the frozen
  top-1000 answer support, cap derived as 7 from the deepest trie path.
  **SECONDARY DIAGNOSTIC only**, to separate task and content failure from
  output-format failure. It is not promoted to primary. Its 1,000 candidates
  are not equated with E8B's 100-way classifier task.

Three image conditions: **normal**; **deranged**, the identical image-partner
mapping E8B used (imported from `e8a_common.imageid_level_derangement`, seed 42,
777 images, zero self-pairs, sha256 `5f8f7868…`, asserted equal to E8B's
recorded provenance) — identical as an image-partner intervention, not as a
feature-space one, because preprocessing differs; and **blank**, a pinned
384x384 mid-grey image, labelled **AUXILIARY / UNMATCHED**, never equated with
E8B's `fixed_image`, which is a feature-space mean with no pixel-space
counterpart.

Scoring is the reviewed G21 scorer, imported unmodified: strict raw exact match
(separately labelled secondary) and the pinned VQA-style single-gold normalised
exact match (primary for every comparative claim). Denominators: 10,004 raw
development rows over 777 image clusters (primary), the common 7,714
top-100-supported rows over 768 clusters, and the 9,823 top-1000-supported rows
as a row subset of the same pass.

Intervals are the frozen project procedure: image-clustered paired bootstrap,
2,000 draws from a fresh `default_rng(0)`, 95 per cent percentile interval.
E9-internal contrasts are decided by conditions 1 and 2 of the universal
directional rule alone and are labelled **single-run**: condition 3 requires two
of three seed differences to agree and E9 has no training seeds. In place of
seeds, a 512-row bitwise replication ran in a separate process.

Efficiency uses the frozen E7b serial protocol, imported read-only: batch 1, 50
warm-up then 300 timed queries, three passes in pinned randomised order, a
fresh subprocess per system per pass, synchronisation before and after every
call, and the same pinned 64-row sample with per-image SHA-256 gating.

## Outputs

`results/experiments/e9_compact_vlm/`: `frozen_protocol.json`,
`delta_ledger.json`, `preflight.json`, `g17_remediation.json`, seven generation
records with paired token artefacts, `e9_determinism_replication.json`,
`e9_execution.json`, 33 per-pass timing records under `timing/`,
`e9_efficiency.json`, `e9_results.json`, `e9_results_tables.md` and
`resource_ledger.json`. Code in `experiments/e9_compact_vlm/`; 25 checks in
`tests/test_e9.py`, registered in `tests/run_all.py`.

Full tables are in `e9_results_tables.md`; the headline numbers follow.

## Results

Development set only. Clean-test contents were never opened, read, scored or
used. A pre-score implementation briefly resolved and `stat()`ed the embargoed
path, which violated the project's stricter G17 path-level rule; it was
detected and repaired before any scientific score was produced. See
`g17_remediation.json` and the corresponding entry under Decisions and
problems.

### Accuracy, raw 10,004-row denominator, G21 normalised

| system | normal | deranged | blank |
|---|---|---|---|
| SmolVLM-256M open (PRIMARY) | 0.436525 | 0.268892 | 0.261096 |
| SmolVLM-256M constrained (diagnostic) | 0.441623 | 0.272391 | n/a |
| SmolVLM-500M open (SECONDARY) | 0.489004 | n/a | n/a |
| SmolVLM-500M constrained (diagnostic) | 0.485706 | n/a | n/a |
| B1 40k / 250k | 0.420299 / 0.457983 | | |
| B2 40k / 250k | 0.411702 / 0.463315 | | |
| B3 40k / 250k | 0.392710 / 0.459649 | | |

The E8B values are recomputed here from E8B's frozen per-row vectors and
reproduce E8B Table 6 exactly.

**Strict raw exact match is essentially zero for open generation**: 0.000100
for the 256M and 0.000000 for the 500M. The model emits `" Yes."` where the
gold answer is `"yes"`. The entire raw-to-normalised gap is surface form, which
is exactly the structural conservatism the protocol records: a closed-set
classifier can only emit vocabulary strings, so raw exact coincides with
normalised exact for it and penalises a generative system for punctuation and
case alone. Every comparative claim below therefore rests on the normalised
metric.

### SmolVLM-256M sits between the 40k and 250k lightweight systems

Paired, image-clustered, raw denominator, open readout, normal condition:

| against | difference | 95% CI | outcome |
|---|---|---|---|
| B1 40k | +0.01623 | [+0.00266, +0.03020] | positive |
| B2 40k | +0.02482 | [+0.01093, +0.03956] | positive |
| B3 40k | +0.04382 | [+0.02996, +0.05807] | positive |
| B1 250k | -0.02146 | [-0.03479, -0.00660] | negative |
| B2 250k | -0.02679 | [-0.03990, -0.01250] | negative |
| B3 250k | -0.02312 | [-0.03688, -0.00880] | negative |

The primary compact VLM is above every lightweight arm trained on 40,000
questions and below every one trained on 250,000, in all six comparisons. The
constrained readout gives the same picture (+0.021 to +0.049 against the 40k
arms, -0.016 to -0.022 against the 250k arms).

### SmolVLM-500M exceeds every lightweight arm at both scales

| against | difference | 95% CI | outcome |
|---|---|---|---|
| B1 250k | +0.03102 | [+0.01723, +0.04386] | positive |
| B2 250k | +0.02569 | [+0.01237, +0.03843] | positive |
| B3 250k | +0.02935 | [+0.01599, +0.04266] | positive |

This is a secondary, normal-condition-only capacity context point. It is **not**
a clean causal 135M-to-360M language-backbone effect: the two SmolVLM sizes
differ in vision-connector capacity and training as well as in text backbone,
and E10 remains the project's controlled SLM size-sensitivity experiment.

### Output format is not what limits the compact VLM

Open minus constrained, paired, raw denominator:

| model | condition | difference | 95% CI | outcome |
|---|---|---|---|---|
| 256M | normal | -0.00510 | [-0.01273, +0.00274] | uncertain or mixed |
| 256M | deranged | -0.00350 | [-0.00901, +0.00202] | uncertain or mixed |
| 500M | normal | +0.00330 | [-0.00394, +0.01047] | uncertain or mixed |

Constraining generation to 1,000 vocabulary answers changes the normalised
score by at most about 0.005 and every interval includes zero. Under the G21
normaliser the open readout loses essentially nothing to output format or
support, even though only 4.7 to 7.6 per cent of its raw emissions are literally
vocabulary strings. This is the diagnostic's purpose and it settles the
question the raw score would otherwise leave open: the compact VLM's score
reflects task content, not formatting failure.

It also extends E8B's fourth finding. R1, R2 and R3 agreed within about 0.0005
for the frozen-encoder systems; the analogous open-versus-constrained contrast
agrees within about 0.005 for an integrated autoregressive VLM. Readout and
decoding are not the bottleneck for either family.

Emission validity: zero empty emissions everywhere, overlong at most 0.02 per
cent, and **zero trie escapes** in both constrained passes.

### The compact VLM relies on the image more than any lightweight system

Normal minus deranged, the matched image-partner intervention:

| system | drop | 95% CI | seeds |
|---|---|---|---|
| SmolVLM-256M open | +0.16763 | [+0.15252, +0.18319] | none (untrained) |
| SmolVLM-256M constrained | +0.16923 | [+0.15387, +0.18498] | none (untrained) |
| B1 40k / 250k | +0.09166 / +0.11752 | | three trained seeds |
| B2 40k / 250k | +0.08610 / +0.12598 | | three trained seeds |
| B3 40k / 250k | +0.08443 / +0.12275 | | three trained seeds |

The E8B rows reproduce E8B Table 9 exactly. The compact VLM's drop is about a
third larger than the largest lightweight drop. Its accuracy under a deranged
image is below every lightweight arm (-0.036 to -0.072), which is the same fact
seen from the other side: it has less to fall back on when the image is wrong.

The blank-image ablation (AUXILIARY / UNMATCHED) gives +0.17543
[+0.16112, +0.18990]. Under a blank image 61 per cent of emissions collapse into
the top-100 answer set, against 4.7 per cent under a correct image.

### Efficiency, same node, same harness

All conclusions use otter159-against-otter159 measurements. Accuracies are
node-independent frozen values; only latencies are node-sensitive.

| system | warm median (ms) | QPS | peak MiB | loaded params | accuracy |
|---|---|---|---|---|---|
| vocab1000_product | 6.199 | 161.3 | 656 | 152,577,257 | 0.49050 |
| fusion | 6.208 | 161.1 | 656 | 152,377,701 | 0.45062 |
| B1 (250k, seed 0) | 10.242 | 97.6 | 760 | 172,377,445 | 0.45372 |
| B3 R3 | 38.796 | 25.8 | 1,324 | 307,136,129 | 0.45925 |
| B2 R3 | 39.239 | 25.5 | 1,324 | 307,136,129 | 0.46315 |
| B3 R2 | 39.656 | 25.2 | 1,324 | 307,136,129 | 0.45942 |
| B2 R2 | 39.733 | 25.2 | 1,324 | 307,136,129 | 0.46328 |
| SmolVLM-256M open | 140.907 | 7.1 | 1,152 | 256,484,928 | 0.436525 |
| SmolVLM-256M constrained | 142.675 | 7.0 | 1,150 | 256,484,928 | 0.441623 |
| SmolVLM-500M open | 151.380 | 6.6 | 1,822 | 507,482,304 | 0.489004 |
| SmolVLM-500M constrained | 152.854 | 6.5 | 1,822 | 507,482,304 | 0.485706 |

The accuracy column pairs each timed configuration with its own frozen
development number: the seed-0 raw-denominator value for the E8B and E7b
systems, and the measured E9 value for the compact VLMs.

**The decisive comparison.** A top-1000 global CLIP head at 250k reaches
0.49050 raw-distribution accuracy at **6.199 ms**; SmolVLM-500M reaches 0.48900
at **152.854 ms**. Equal accuracy within 0.0015, at **24.7 times** the serial
cost. Against the E8B systems the compact VLM's advantage is real but expensive:
SmolVLM-500M buys +0.02569 over B2 at 250k for 3.85 times B2's R2 latency, and
SmolVLM-256M is slower than every lightweight system while beating none of them
at 250k.

**The E7b bridge control failed its pre-registered tolerance, and that is the
finding it exists to produce.** `fusion` measures 6.208 ms on otter159 against
the frozen 7.635 ms on otter155, a relative difference of -18.7 per cent
against a tolerance of plus or minus 10 per cent fixed before the measurement.
The frontiers are therefore **not merged**, no adjustment factor is applied, and
E7b's otter155 medians remain historical reference evidence only. Had the two
been merged, every E9 latency ratio would have been understated by about a
fifth. `vocab1000_product` was re-timed on the same node for exactly this
reason.

### Determinism, resources and integrity

| quantity | value |
|---|---|
| 512-row replication, fresh process | bitwise identical |
| greedy asserted, beams | yes, 1 |
| E9 branch GPU-hours charged | 2.2771 |
| pre-registered branch halt | 6.0 |
| headroom | 3.7229 |
| optional cells dropped | 0 |
| model download footprint | 1.42 GiB (0.49 + 0.97) |
| checkpoints written | none |
| pretrained-identity ledger after the bridge | 35.120 of 40.0 GPU-h |
| clean-test contents opened, read, scored or used | no |
| embargoed path resolved by a pre-score implementation | yes, repaired before any score (`g17_remediation.json`) |

## Decisions and problems

**A G17 implementation defect was found and repaired before any score existed.**
The first E9 implementation wrote the embargoed target's file name as a literal
and `stat()`ed that path for a size-and-mtime fingerprint. The file was never
opened and no label was read, but constructing the path resolves it, which G17
forbids, and writing the name into the package would have forced the project's
plain-substring embargo scan to carry an exemption for mentions in comments —
precisely the exemption E8A avoided by never writing the name at all. The
literal was removed, the forbidden token is now assembled at run time, and the
fingerprint was replaced by a source scan asserting that no E9 source contains
the token and therefore that no E9 code can resolve or open the file. This was
not clean-test content access, so it was not a stop condition; it is recorded
in `g17_remediation.json` rather than silently fixed, and the affected
preflight and execution fields are marked historical.

**The summary wording was too broad and has been corrected.** An earlier
version of this report and of the rendered tables said the clean test was
"embargoed and untouched" and recorded "clean test accessed: no". That is
accurate about contents and inaccurate about paths, because the pre-score
implementation above did resolve and `stat()` the embargoed path. The
operative wording throughout is now: clean-test contents were never opened,
read, scored or used; a pre-score implementation briefly resolved and
`stat()`ed the embargoed path, violating the project's stricter G17
path-level rule; and this was detected and repaired before any scientific
score was produced. The correction is a wording change only. No metric,
prediction, evaluation artefact or conclusion changed, which is verified by
`g17_wording_correction.json`.

**E8B's R1 readout is not timed, and no number stands in for it.** Amendment 7
asks for three pairings — E9 open against R3, E9 constrained against R2, and B1
as the language-model-free reference — and R1 is not among them. E8B's R1 scorer
forwards one incremental step per multi-token candidate over a shared prefix
cache, which is near-free at the batch size 128 that E8B evaluates with and
dominates a batch-1 serial query; timing it under the E7b protocol would have
spent roughly a sixth of the whole branch budget measuring an implementation
choice rather than a system property. The exclusion is recorded with its reason
and with no measurement attached, and the R2 and R3 figures are never presented
as R1's cost.

**The E8B bridge times the fp32-promoted language model.** E8B's canonical
scientific evaluation pins true IEEE fp32 and promotes the frozen bf16 state
losslessly, and its readouts produce fp32 prefixes and require it. Timing runs
that same configuration through E8B's own `pin_fp32_precision` and
`promote_lm_to_fp32`, so the latency attaches to the configuration that
produced E8B's reported accuracy. A bfloat16 timing would have been a
deployment diagnostic attached to no reported number.

**`vocab1000_product` was added to the same-node timing set.** It is the
accuracy-matched counterpart to the headline E9 result, and amendment 8 forbids
comparing latencies across nodes, so without a same-node figure the most
informative efficiency comparison in this experiment could not have been stated
at all. It adds no model, dataset, tuning search or scoring rule: it is one
already-frozen E7b checkpoint through the identical harness, at a cost of about
three minutes.

**E8B was not reopened.** The bridge is read-only. Every checkpoint was
verified against its frozen recorded SHA-256 before use, all new artefacts were
written under the E9 namespace, and no E8B scientific artefact was modified.

**The blank condition is more expensive than a real image.** A 384x384 blank
upscales to 2048 and tiles into 17 patches rather than 13, giving 1,088 image
tokens against 866. This is recorded because it is a property of the pinned
construction, and it is one more reason the blank drop is labelled auxiliary
and unmatched rather than compared with E8B's feature-space mean.

**What is supported.** That a frozen off-the-shelf compact integrated VLM, given
no GQA supervision, lands between this project's 40k-trained and 250k-trained
lightweight systems at 256M parameters and above all of them at 500M; that its
score is not limited by output format; that it relies on the image more than
any lightweight system here; and that at equal raw-distribution accuracy it
costs about 25 times more serial latency than a top-1000 global CLIP head on
the same node.

**What is not supported.** No causal attribution of any E9-versus-E8B
difference to integrated architecture, because training history, multimodal
pretraining, answer support and output format all differ. No claim about
SmolVLM's GQA exposure beyond what its model card documents. No 135M-to-360M
backbone effect from the 500M point. No comparison against published GQA
scores. No leaderboard claim. No clean-test result: clean-test contents were
never opened, read, scored or used, and F1, F2 and E10 remain unstarted. The
one qualification, stated rather than absorbed, is the pre-score path
resolution recorded above and in `g17_remediation.json`.
