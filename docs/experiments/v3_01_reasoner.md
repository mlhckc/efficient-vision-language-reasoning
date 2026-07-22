# Experiment v3_01: the question-conditioned latent-query reasoner at 40k

## Purpose

Train the V3 central-contribution architecture, a lightweight
question-conditioned latent-query reasoner over the cached token stores, at
the 40k scale, and test the pre-registered primary hypothesis: that
token-level reasoning shrinks the >=4-step lift deficit that v2_05b/v2_07
showed to be persistent for every global-embedding head.

## Method

Architecture (src/reasoner.py): 32 learned latent queries (std-0.02 init),
four pre-LN residual blocks, each in order cross-attention latents to
question tokens (key_padding_mask, True = padded, the convention proven in
the v3_00 mask test), cross-attention latents to all 50 image tokens,
latent self-attention, and a 4x GELU FFN; 8 heads and d_model 512
throughout (the stores are joint-space, so there are no input projections);
readout mean-pools the latents through LayerNorm into Linear(512, 100).
Exact parameter count: 21,099,620, all trainable. fp32 master weights;
bf16 autocast in training only; fp32 evaluation.

Training (experiments/v3_01_reasoner/run.py), fixed and not searched:
AdamW with weight decay 1e-2 on non-bias/non-LayerNorm parameters only,
batch 128, gradient clipping 1.0, cosine schedule after warmup, early
stopping on dev accuracy with patience 10, best-on-dev checkpointing,
plain cross-entropy. Cached-token training only (A1); dev selection only;
test_clean_targets.csv never read.

Pre-registered search (grid written to search.json before the search, its
results before the finals; seed 0, max 60 epochs): lr {3e-4, 1e-3} x
warmup {0, 3% of steps} x dropout {0.1, 0.3}. No other knob was tuned and
no deviation occurred.

Gates, all passed and printed: one-batch forward (finite (128, 100)
logits); model-level mask corruption (max logit change 0.00e+00 under fp32
eval); 1,000-example overfit (99.7% train accuracy at epoch 9);
memory/throughput pilot (peak 1,143 MiB allocated / 1,216 MiB reserved,
projected plan well under the 12 h gate); gradient hygiene (117/117
parameters receive gradients) and read-only store access (verified by
source inspection and unchanged file stats). One determinism note: the
memory-efficient attention backward is non-deterministic under
warn_only=True, so training runs are seed-controlled but not bitwise
reproducible; the three-seed protocol covers this.

## Outputs

Under results/experiments/v3_01_reasoner/ (git-ignored): search.json
(pre-registration and results), results.json (gates, final runs, gaps,
lift tables, reliance, efficiency, run metadata) and four checkpoints
(8 search checkpoints plus reasoner_seed{0,1,2}).

## Results

Search (dev accuracy, seed 0): 3e-4/no-warmup/0.1 reached 0.5465; the
3e-4 configs (0.5419-0.5465) all beat the 1e-3 configs (0.5219-0.5347);
warmup and heavier dropout did not help at this scale. Selected config:
lr 3e-4, no warmup, dropout 0.1.

Final, seeds {0, 1, 2}: dev accuracy 0.5422 +/- 0.0037 (per-seed 0.5464,
0.5406, 0.5397). Paired same-seed gaps:

| gap                          | mean    | std    | min     | per-seed |
|------------------------------|---------|--------|---------|----------|
| reasoner - concat            | +0.0168 | 0.0064 | +0.0124 | +0.0241, +0.0139, +0.0125 |
| reasoner - product_576k      | +0.0086 | 0.0067 | +0.0025 | +0.0157, +0.0078, +0.0025 |
| reasoner - fusion            | +0.0031 | 0.0062 | -0.0008 | +0.0102, -0.0003, -0.0008 |

Primary readout, the step-deficit test (lift per n_steps bucket, v2_05b
priors, seeds 0/1/2 averaged for every model, baselines re-evaluated from
their stored checkpoints on the same seeds):

| model        | <=2    | 3      | 4      | >=5    | ge4 lift | deficit |
|--------------|--------|--------|--------|--------|----------|---------|
| concat       | 0.3180 | 0.3635 | 0.2322 | 0.0676 | 0.1499   | 0.0954  |
| product_576k | 0.3300 | 0.3619 | 0.2422 | 0.0954 | 0.1688   | 0.0886  |
| fusion       | 0.3419 | 0.3582 | 0.2605 | 0.1050 | 0.1828   | 0.0836  |
| reasoner     | 0.3503 | 0.3617 | 0.2525 | 0.1041 | 0.1783   | 0.0888  |

Reasoner deficit per seed: 0.0967, 0.0908, 0.0789 (fusion: 0.0852,
0.0838, 0.0820).

Secondary, visual-reliance spot check (best seed 0, permutation seed 42):
normal 0.5464, shuffled 0.4364, drop 0.1101. Fusion's five-seed mean drop
in v2_06 was 0.1427.

Efficiency: 21,099,620 trainable parameters; 15.3 s per epoch; time to
best 46-77 s; peak training memory 1,143 MiB allocated; single-example
latency on cached tokens 1.32 +/- 0.02 ms (200 CUDA-synchronised passes).

## The step-deficit verdict, unsmoothed

The deficit persists. The reasoner's >=4-step lift deficit (0.0888) is
statistically indistinguishable from fusion's (0.0836) and product's
(0.0886), and only concat (0.0954) is clearly worse; the per-seed ranges
overlap. Token-level latent-query reasoning at this scale and size did not
close any of the compositional gap: the reasoner's accuracy profile is
essentially fusion's, achieved with 19x the parameters (21.1M against
1.1M), 30x the head latency (1.32 ms against 0.042 ms) and roughly 60x
the per-epoch training cost. The overall gap to fusion (+0.0031 +/-
0.0062, negative in two of three seeds) is noise. The secondary check
also contradicts the expectation: the reasoner relies on the image less
than fusion by the shuffle measure (drop 0.110 against 0.143), not more,
so there is no evidence that token-level structure is being exploited
beyond what the global agreement signal already provides.

## Limitations

40k training scale only; three seeds; a single architecture point (32
latents, 4 blocks, d_model 512) with a small pre-registered search over
optimisation knobs only. None of these numbers is a clean-test result.

## The next-step decision this sets up

v2_07 showed the multimodal margin grows with data while handcrafted
feature advantages decay; the matching question for V3 is whether the
reasoner's capacity starts to pay at 100k/250k where the small heads
saturate their input representation, or whether the deficit is a property
of the frozen CLIP features themselves, which no head on top can remove.
v3_02 should therefore run the identical frozen recipe at 100k and 250k
(the token stores already cover the full union) before any architecture
iteration is considered; if the deficit and the fusion parity persist at
250k, the honest conclusion is that the compositional bottleneck lives in
the representation, not the head, and the dissertation's contribution
becomes that negative result plus the efficiency frontier, which the
protocol is equipped to defend.
