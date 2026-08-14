# Project overview

MSc Artificial Intelligence dissertation, University of Surrey. Supervisor:
Prof. Miroslaw Bober. Title: "Efficient Vision-Language Reasoning with Small
Language Models".

This document tells the project's story from its starting question to its
current pre-F1 state, and explains why each experiment followed from the one
before it. The detailed numbers are in
[EXPERIMENTS_AND_RESULTS.md](EXPERIMENTS_AND_RESULTS.md); the model families
are in [ARCHITECTURE.md](ARCHITECTURE.md); what is finished and what remains
is in [PROJECT_STATUS.md](PROJECT_STATUS.md).

## The research question

Can Visual Question Answering be done efficiently by reasoning in embedding
space — frozen small encoders plus a tiny trainable head — instead of running
a large autoregressive vision-language model? An image and a question are
each turned into fixed representations by one frozen CLIP encoder, and a
small head classifies the answer from a closed set of frequent answers.

The aim is not to chase a leaderboard. It is to measure, under controls,
what a lightweight frozen-representation system can and cannot do on GQA,
what each design ingredient (interaction features, capacity, data scale,
token-level access, pretrained language models) actually contributes, and
what one query costs.

## Why GQA

GQA provides compositional questions with structural metadata (question
types and program-step counts), which lets the project slice performance by
reasoning demand rather than reporting a single number. That metadata is
what later made the central negative finding — the persistent multi-step
deficit — visible and quantifiable.

## Why frozen representations and lightweight heads

Frozen encoders make the expensive part of the pipeline a one-off cache:
embeddings are extracted once, and everything after that trains in seconds
to minutes on one GPU. Scientifically, freezing turns every experiment into
a controlled probe of the representation: if a small head cannot extract
something from frozen features, that is a fact about the features, not about
a co-adapted encoder. Practically, it is the efficiency hypothesis itself:
if this works, VQA answers cost milliseconds, not the serial cost of an
autoregressive VLM.

## The story, experiment by experiment

**V1 (legacy prototype).** A first pipeline established that the task is
learnable from frozen CLIP globals and that a handcrafted fusion input
(image, question, product, absolute difference) beats plain concatenation.
But V1's evaluation was single-seed and selection-biased, so nothing from
it is confirmatory. Its real product was the list of protocol defects to
fix.

**v2_00 (the protocol).** V2 rebuilt the evaluation defensibly: an
image-disjoint development partition, a vocabulary computed from the
training pool only, strictly nested 40k/100k/250k training subsets, a clean
test built from images no legacy experiment ever touched, an independent
verifier (99 checks, 0 failures), and a hard embargo on the clean-test
targets that still holds today.

**v2_01 – v2_04 (what the fusion gain is).** Re-run under the clean protocol
with five seeds, the fusion gain survived but shrank under scrutiny:
parameter matching showed it is roughly half capacity and half features
(v2_03), and the ablation showed either interaction term alone carries the
whole feature gain, the two being redundant (v2_04). The gain is real,
reproducible and modest.

**v2_05 – v2_06 (where the gain lives and whether the image matters).**
Type-sliced analysis located the gain in verify/logical/obj/rel questions —
exactly where judging agreement between image content and a proposition
helps — and exposed the project's central negative observation: on questions
needing four or more reasoning steps, every model's lift falls about 0.08
short of its own average. Intervention experiments showed the heads really
use the image (fusion most), and that a wrong image actively misleads. But
reliance is dependence, not reasoning quality.

**v2_07 (scale).** With 6.25 times more data, every model improves, the
multimodal margin over the question-only prior grows, and the
interaction-feature advantage decays to noise: the handcrafted terms are a
small-data prior that a plain concat head learns from data. The multi-step
deficit barely moves. At this point scale, capacity and handcrafted
interactions were exhausted as explanations; what remained was
architecture.

**V3 (token-level reasoning).** The central architectural contribution: a
question-conditioned latent-query reasoner over cached token-level CLIP
features (21.1M parameters against fusion's 1.1M). At 40k it delivered a
clear negative result — fusion parity at 19 times the size, with
diagnostics showing it mostly re-reads the pooled global signal. Scaled to
100k/250k under the frozen recipe (v3_03), it beats every global head in
every seed, a configuration-level accuracy win. The compositional deficit,
however, is statistically indistinguishable from fusion's at every scale.
Token access buys general accuracy, not multi-step reasoning.

**E2 (a second encoder).** Swapping frozen CLIP for frozen SigLIP-B/16 on
the global path lifts every multimodal head by +1.2 to +1.5 points — and
leaves the deficit range unchanged. The bottleneck looks like a property of
the frozen contrastive dual-encoder family, not of one encoder.

**E3 (a larger answer space).** Growing the closed vocabulary to 1000
answers costs about 2 points on the shared rows but wins +3.6 to +4.1
points of raw-distribution accuracy, because coverage jumps from 77 to 98
per cent. The efficiency claims are stated on this raw-distribution basis.

**E7a / E7b (what a query costs).** E7a measured components; its additive
end-to-end sums were later found to understate the real serial cost by
about 20 per cent and are withdrawn. E7b measured the true serial batch-1
query: the global heads answer in about 7.6 ms, the reasoner-class systems
in 9.2-20.1 ms, and the primary accuracy-latency frontier is concat, fusion
and the top-1000 product head. Three E7b fields (peak memory, cold-start)
were later withdrawn on measurement-boundary and grad-mode grounds; warm
serial latency and the frontier stand.

**E8A / E8B (small language models, both sides).** With the head design
settled, the programme asked whether frozen small language models help.
On the question side (E8A), pretrained SmolLM2-135M clearly beats its
random-initialised twin — pretraining is doing real work — but the whole
SLM path still loses to an interface-matched CLIP control and doubles the
serial cost. On the answer side (E8B), pretraining shows no positive
advantage at all: directionally negative at 40k, parity at 250k. Data
scale dominates everything.

**E9 (context).** Two frozen compact VLMs (SmolVLM-256M/500M, evaluation
only) place the lightweight systems in context: 256M lands between the
40k- and 250k-trained systems, 500M above them. Under the same-node
contextual warm-serial protocol, the SmolVLM-500M open-readout row
measured 151.380 ms at 0.489004 accuracy (about 24.4 times the 6.199 ms
lightweight top-1000 head), and the constrained row 152.854 ms at
0.485706 (about 24.7 times). This is contextual positioning, not a
matched comparison: training data, answer support and output format all
differ.

**E10 (capacity, answer side).** Does a bigger readout change the E8B
answer? Growing the frozen readout to SmolLM2-360M under a fully
pre-registered twelve-cell design reproduced the same picture: no reliable
pretrained-over-random advantage at either scale, a null
difference-in-differences, and scale effects that dwarf every pretraining
effect. Capacity does not rescue answer-side pretraining under this
interface.

## What the sequence taught us

Each experiment removed one candidate explanation. Capacity (v2_03),
feature engineering (v2_04, v2_07), data volume (v2_07, v3_03), token-level
access (V3), encoder choice (E2), answer-space size (E3), question-side
language knowledge (E8A), answer-side language knowledge (E8B), and
answer-side capacity (E10) were each tested under controls. The positive
result is that very small heads over frozen representations reach useful
GQA accuracy at millisecond latency, and that the accuracy-latency frontier
is occupied by the simplest systems. The negative result — as informative —
is that the multi-step compositional deficit survived every one of those
interventions, which localises it in the frozen representations and the
closed-set formulation rather than in any head this project trained.

## Where the project stands

All experimental packets are complete and five review packets are closed
(E7b, VE-0, VE-1, VE-2, E10); the evidence base is hash-pinned, its
statistics are image-clustered, and its figures and tables are rendered
from a frozen contract. The clean test remains embargoed under the
governance wording in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The two
remaining scientific steps — the model-list freeze (F1) and the blinded
clean-test evaluation (F2) — are unstarted, and F2 is unauthorised. Every
result in this repository is a development-set result.
