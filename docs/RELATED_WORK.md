# Related work and positioning

Consolidated from three literature surveys run on 30-31 July 2026 across
arXiv, the ACL Anthology, CVF Open Access and Semantic Scholar, covering
(A) frozen-CLIP VQA classification heads, (B) latent-query modules and
token-versus-global comparisons, and (C) compositional limitations of
CLIP-style encoders and bottleneck localization. The survey tables the
consolidation was built from are archived in the p2-related-work task
packet.

Conventions. Overlap is rated against this project's claims as none,
partial or strong; every partial or strong entry carries a one-line
differentiation, and the papers that most constrain or most support our
claims get a fuller "Differentiation" note. Every numeric claim carries a
verification tag: [verified: source] when the number was read from the
cited source during the survey, [unverified] when it comes from survey
memory and must be checked against the paper before it may be cited in
the dissertation. Publication status is as of 31 July 2026; arXiv-only
papers are marked, since their venue status can change before our
submission.

## 1. Frozen dual-encoder VQA with small trainable heads

**FrEVL** (Bourigault, 2025, arXiv:2508.04469, arXiv-only). Frozen CLIP
global embeddings feed a 68.4M-parameter trainable fusion network whose
fused vector concatenates the transformed features, their elementwise
product and their absolute difference — the same feature template as our
fusion head. VQAv2 71.4%, SNLI-VE 87.9%; no GQA; ablations are not
parameter-matched and use a single training scale; frozen embeddings are
reported to fail on counting (34.2%) and spatial reasoning (41.3%)
[verified: arXiv:2508.04469]. Overlap: strong.
*Differentiation:* FrEVL reaches the same ceiling thesis from the
global-only side with a head about sixty times larger than ours, on VQAv2
rather than GQA, without capacity control, seed replication, data-scale
analysis or compositional stratification. Our contribution is exactly the
controlled part: parameter matching halves the fusion-feature gain, the
two interaction terms are mutually redundant, the feature advantage
decays with training scale, and the multi-step deficit is head-invariant.
FrEVL must be cited and differentiated in any submission; it is recent
and arXiv-only, so its venue status needs re-checking at submission time.

**Less Is More** (Deuser, 2022, CVPR VizWiz workshop, arXiv:2206.05281).
Frozen CLIP embeddings concatenated into a linear classifier with an
answer-type gate; VizWiz 2022 60.15% answer accuracy [verified:
arXiv:2206.05281]. Overlap: strong. *Differentiation:* same recipe family
(frozen CLIP, tiny head, closed answer set) but VizWiz, concat-only, no
fusion analysis, no GQA, no controls.

**Zhou and Xie** (2026, arXiv:2606.01207, arXiv-only). Concat (296K
parameters) versus cross-attention (757K) over frozen CLIP ViT-B/32
features at four data scales on binary Flickr8k image-text matching;
concatenation wins by 4.1-5.1 points at every scale, with a
sample-complexity argument; VQA is named as future work [verified:
arXiv:2606.01207, read in full]. Overlap: strong on the scientific
question, none on task. *Differentiation:* they control scale but not our
feature set, are not parameter-matched, and do not touch VQA or
compositional strata; our work answers the VQA case they leave open. Race
risk: could publish first; cite and re-check status at submission.

**EMAP** (Hessel and Lee, 2020, EMNLP, arXiv:2010.06572). Projects a
trained multimodal model onto its best additive approximation to test
whether cross-modal interactions matter; across seven tasks the
interactions often cost little [verified: arXiv:2010.06572]. Overlap:
strong methodological. *Differentiation:* EMAP is the diagnostic
projection; our v2_03/v2_04 are the constructive analogue for explicit
handcrafted features over frozen CLIP on VQA, with capacity matching and
scale dependence EMAP does not consider.

**TAP-C** (Song, 2022, ACL, arXiv:2203.07190). Zero/few-shot VQA with
fully frozen CLIP by answer-infilled prompt matching; zero-shot VQAv2
38.7 [verified: ACL Anthology 2022.acl-long.421]; third-party tables list
TAP-C at 36.3 on GQA [verified: arXiv:2307.00862 (UniFine)]. Overlap:
partial — frozen CLIP on GQA but no trained head.

**CLIP-ViL** (Shen, 2022, ICLR, arXiv:2107.06383). CLIP as the visual
encoder inside fully fine-tuned V&L architectures; VQAv2 76.70 test-std;
GQA 61.42 test-dev with fine-tuning [verified: arXiv:2107.06383].
Overlap: partial — motivates CLIP features; opposite (fully trained)
regime.

**Hate-CLIPper** (Kumar, 2022, EMNLP NLP4PI workshop, arXiv:2210.05916).
Frozen CLIP embeddings fused by an outer-product interaction matrix into
a small MLP for meme classification [verified: arXiv:2210.05916].
Overlap: partial — interaction features over frozen CLIP, different task.

**eP-ALM** (Shukor, 2023, ICCV, arXiv:2303.11403). Frozen OPT plus frozen
ViT with only a linear projection and one soft token trained (about 4M
parameters); GQA 40.4 (OPT-2.7B, ViT-B/16) and 42.7 (OPT-6.7B, ViT-L)
[verified: arXiv:2303.11403]. Overlap: strong on the "the pooled/CLS
signal carries most of what a small trainable bridge can use" claim.
*Differentiation:* generative LLM pipeline; our result is the
discriminative, capacity-controlled version with compositional strata.

**MAPL** (Mañas, 2023, EACL, arXiv:2210.07179). A 3.4M-parameter mapping
network with learnable query tokens from frozen CLIP tokens into a frozen
GPT-J; few-shot VQAv2/OK-VQA/TextVQA, no GQA [verified:
arXiv:2210.07179]. Overlap: strong on module design, none on controlled
evaluation. *Differentiation:* closest small latent-query module over
frozen CLIP tokens, but generative and never compared against a tiny
global-feature head under matched data.

**LiMBeR** (Merullo, 2023, ICLR, arXiv:2209.15162). A single trained
linear map from frozen image encoders into frozen GPT-J; VQA capability
tracks the encoder's pretraining [verified: arXiv:2209.15162]. Overlap:
strong on localization logic. *Differentiation:* varies the encoder under
a minimal head where we vary the head under a fixed encoder — the two
designs are complementary halves of the representation-bottleneck
argument.

**LeVLJEPA** (Kuhn, 2026, arXiv:2607.00784, arXiv-only). Non-contrastive
VL pretraining; frozen vision tokens through a lightweight MLP bridge
into a frozen Llama reach GQA 44.6 at Datacomp-L scale [unverified].
Overlap: partial — frozen-encoder GQA with a small bridge, but generative
and pretraining-objective focused.

**Mod-Zero-VQA** (2023, Findings of ACL, arXiv:2305.17369). Decomposed
zero-shot GQA with frozen models; cites plain CLIP zero-shot VQA near
35.6% [unverified]. Overlap: partial.

Classic fusion and simple-baseline lineage (all trained, non-CLIP
encoders; positioning only): **Antol et al.** (2015, ICCV) fuse projected
image and question features by elementwise product into an MLP over a
top-1000 answer set, 58.16% open-ended test-std [verified: CVF
iccv_2015 Antol]; **iBOWIMG** (Zhou, 2015, arXiv:1512.02167) concatenates
bag-of-words and GoogLeNet features, 55.89 test-std [verified:
arXiv:1512.02167]; **Jabri et al.** (2016, ECCV, arXiv:1606.08390) show
MLPs over concatenated features match attention models on multiple-choice
VQA; **MCB** (Fukui, 2016, EMNLP), **MLB** (Kim, 2017, ICLR), **MUTAN**
(Ben-Younes, 2017, ICCV) and **BLOCK** (Ben-Younes, 2019, AAAI) are the
bilinear-fusion family, with MUTAN's parameter-versus-accuracy framing
the closest precedent for our capacity argument; **InferSent** (Conneau,
2017, EMNLP) is the direct source of the [u; v; |u-v|; u*v] template,
applied within one modality for NLI. Overlap: partial throughout — none
studies frozen dual-encoder features, capacity matching or term
redundancy for VQA.

**PID** (Liang, 2023, NeurIPS, arXiv:2302.12247). Information-theoretic
redundancy/uniqueness/synergy quantification for multimodal data; no
capacity-matched head comparisons [verified: arXiv:2302.12247]. Overlap:
partial (formal vocabulary for our redundancy finding).

**Axis summary.** No published paper reports GQA accuracy for frozen
dual-encoder global embeddings with a sub-1.1M head over a top-k
vocabulary; the capacity-halving result, the product/absolute-difference
mutual redundancy, and the 40k-to-250k decay of the handcrafted-feature
advantage appear unpublished. FrEVL and Zhou and Xie are the two works to
watch and differentiate.

## 2. Latent-query modules and token-versus-global evidence

**Li, Koh and Du** (2025, ACL, arXiv:2411.05195). Same frozen CLIP
ViT-L/14-336 weights: CLIP's projected CLS embedding versus an MLLM using
all 576 patch tokens; on What'sUp Left/Right, CLIP scores 1.9% pair
accuracy against LLaVA-1.5's 93.2%, and a CLS-only LLaVA ablation drops
spatial accuracy from 84.5 to 44.2 [verified: arXiv:2411.05195].
Overlap: strong — the same patch-versus-CLS question with the opposite
conclusion. *Differentiation:* their extraction head is a 7B generative
LM trained on massive data with no capacity or data control, and their
encoder is ViT-L/14-336 on spatial probes rather than ViT-B/32 on
discriminative GQA. Our controlled small-scale negative directly
qualifies their claim: token-level information may exist, but a 21.1M
trained-from-scratch head on 40k examples cannot extract more than the
pooled embedding already provides. Both results can be true; the regime
matters, and our claims are scoped accordingly.

**Efficient Probing** (Psomas, 2026, ICLR, arXiv:2506.10178). First
systematic study of attentive probing over frozen backbones; multi-query
cross-attention probes (for example 850K parameters) beat linear probes
especially for backbones without a strong global token [verified:
arXiv:2506.10178]. Overlap: strong methodological. *Differentiation:*
classification-only; our comparison adds VQA, question conditioning,
capacity matching against a fusion MLP and compositional stratification.
Their finding that contrastively trained models gain least from
attentive probing is consistent with our negative result.

**DePALM** (Vallaeys, 2024, arXiv:2403.13499). Controlled comparison of
seven connector families over frozen encoders and frozen LLMs; local
resampler variants lag a simpler global query-pooling mapper (about
17.9M parameters), especially in low-data regimes [verified:
arXiv:2403.13499]. Overlap: strong. *Differentiation:* generative,
not capacity-matched to a ~1M global-feature head, no fixed
training-budget nesting, no GQA compositional analysis — but the
direction (resamplers do not beat simple global mappers in low data)
matches our result generatively.

**Flamingo** (Alayrac, 2022, NeurIPS, arXiv:2204.14198) and **BLIP-2**
(Li, 2023, ICML, arXiv:2301.12597). The canonical latent-query modules
over frozen vision: the Perceiver Resampler (64 latents) and the Q-Former
(188M parameters, 32 learned queries — the same query count as our
reasoner). BLIP-2 zero-shot VQAv2 65.0 (FlanT5-XXL) [verified:
arXiv:2301.12597]; zero-shot GQA around 44, InstructBLIP (Dai, 2023,
NeurIPS, arXiv:2305.06500) around 49 [unverified]. Overlap: strong on
module design, none on controlled evaluation. *Differentiation:* both
depend on web-scale pretraining of the connector; our reasoner is the
same architectural idea trained only on the task data, which is exactly
the regime where it fails to beat the global head.

**Frozen** (Tsimpoukelli, 2021, NeurIPS, arXiv:2106.13884; the vision
encoder is trained, the LM frozen; zero-shot VQAv2 about 29
[unverified]), **Perceiver/Perceiver IO** (Jaegle, 2021/2022, ICML/ICLR;
architectural ancestry), **ClipCap** (Mokady, 2021, arXiv:2111.09734),
**EVL** (Lin, 2022, ECCV, arXiv:2208.03550; a positive
decoder-over-frozen-CLIP-tokens result in video classification),
**Qwen-VL** (Bai, 2023, arXiv:2308.12966) and **MQT-LLaVA** (Hu, 2024,
NeurIPS, arXiv:2405.19315; dropping to 2 visual tokens costs only 3-6%
on some benchmarks [verified: arXiv:2405.19315]). Overlap: partial —
context and supporting evidence that few-token or pooled access often
suffices.

**Connector null results at scale.** **MM1** (McKinzie, 2024, ECCV,
arXiv:2403.09611) finds connector design of "comparatively negligible
importance" next to resolution and token count; **Honeybee** (Cha, 2024,
CVPR, arXiv:2312.06742) shows resampler-style abstractors sacrifice
local context versus convolutional pooling; **DeCo** (Yao, 2024,
arXiv:2405.20985) shows 2D average pooling beats the Q-Former;
**LLaVA-1.5** (Liu, 2024, CVPR) reaches GQA 62.0 with a plain 2-layer
MLP over frozen CLIP-L tokens [verified: arXiv:2310.03744]. Overlap:
partial-strong, all consistent with our finding that latent-query
sophistication is not where the gains are. Counterpoint: **Idefics2**
(Laurençon, 2024, NeurIPS, arXiv:2405.02246) reports a 64-latent
perceiver pooling raising average performance by 8.5 points over no
pooling [verified: arXiv:2405.02246] — the main published positive
ablation for learned latent pooling, which is why our claims stay
regime-scoped rather than universal.

**Attention-sink and register evidence.** **Vision Transformers Need
Registers** (Darcet, 2024, ICLR, arXiv:2309.16588) establishes that
CLIP-family ViTs manufacture high-norm global-aggregate tokens; **See
What You Are Told** (Kang, 2025, ICLR, arXiv:2503.03321) audits visual
attention sinks inside LMM decoders and finds sink-token removal
harmless — an instructive contrast with our CLS removal costing 3-5
points, which indicates our head's CLS mass is informative rather than a
pure sink. Related: test-time registers (arXiv:2506.08010), EDIT
(arXiv:2504.06738), structured-approximation analyses
(arXiv:2507.16018). Overlap: strong on diagnostic style, none on
VQA-head audits. *Differentiation:* no published work audits where a
learned-query cross-attention head places its visual attention mass,
quantifies test-time CLS deletion, and retrains patches-only to separate
representation content from attention routing, on VQA.

**Probing-protocol context.** **Scaling ViTs** (Zhai, 2022, CVPR,
arXiv:2106.04560) found CLS/GAP/MAP pooling near-equivalent for
classification; **V-JEPA** (arXiv:2404.08471) and **AIM**
(arXiv:2401.08541) adopt attentive probes as standard for
non-contrastive backbones. Overlap: partial, supports the reading that
contrastive global embeddings already expose most linearly usable signal.

**Axis summary.** No published capacity/data-controlled comparison of a
latent-query module against a tiny global-feature MLP over the same
frozen CLIP for VQA, and no published CLS-dominance plus CLS-removal
plus patches-only-retraining diagnostic package for such a module. Both
appear to be open contributions; claims must remain scoped to small
heads, small data, ViT-B/32 and discriminative GQA.

## 3. Compositional limitations and bottleneck localization

**Koishigarina** (2026, ICLR, arXiv:2502.03566). Linear probes recover
attribute-object binding inside CLIP's uni-modal embeddings; the
cross-modal alignment, not the encoder, discards much of it [verified:
arXiv:2502.03566]. Overlap: strong and partly adversarial.
*Differentiation:* this is the paper our claim must be phrased against —
binding information that is linearly present can still be insufficient
for multi-step GQA under any head we trained. We therefore claim
"insufficient for this task at these scales", never "absent from the
representation". Our patches-only probe and 21.1M-head result are the
task-level complement to their probe-level finding.

**II-MMR** (Kil, 2024, Findings of ACL, arXiv:2402.11058). GQA stratified
by reasoning hops; BLIP-2 accuracy falls 49.6 to 42.4 to 7.7 as hops
increase [verified: arXiv:2402.11058]. Overlap: strong on the phenomenon.
*Differentiation:* documents the multi-hop deficit at MLLM scale without
localization; our contribution is holding the encoder fixed and showing
the deficit is invariant to head capacity (0.1M to 21.1M), architecture
and training scale (40k to 250k), with prior-adjusted, image-clustered
statistics.

**MMVP / Eyes Wide Shut** (Tong, 2024, CVPR, arXiv:2401.06209). CLIP-blind
pairs propagate into MLLM failures regardless of the language head
[verified: arXiv:2401.06209]. Overlap: strong. *Differentiation:* the
flagship representation-bottleneck result at MLLM scale via encoder
contrast; ours is the small-head, single-encoder, per-step-stratified
version with controlled head sweeps. **Cambrian-1** (Tong, 2024,
NeurIPS, arXiv:2406.16860) and **Law of Vision Representation** (Yang,
2024, arXiv:2408.16357, arXiv-only) generalize the localization across
encoders; **Prismatic VLMs** (Karamcheti, 2024, ICML, arXiv:2402.07865)
is the methodological kin for controlled design-axis comparisons;
**BRAVE** (Kar, 2024, ECCV, arXiv:2404.07204) shows no single encoder
wins everywhere. Overlap: partial-strong (method and direction).

**Counter-attribution at scale.** **Hidden in Plain Sight** (Fu, 2025,
COLM, arXiv:2506.08008) finds VLMs perform worse than direct readouts of
their own encoders on vision-centric tasks, locating the bottleneck in
the LM's use of features; **Takishita** (2025, Findings of EMNLP,
arXiv:2506.05439) shows large decoders compensate for degraded visual
contextualization. Overlap: strong, opposite attribution.
*Differentiation:* both operate with billion-parameter language heads
that can re-extract or compensate; our tiny heads cannot, which is
precisely why the head-invariant deficit localizes the bottleneck in the
representation at our scale. The two attributions are compatible and
jointly argue that attribution is scale-dependent — a point the
dissertation should make explicitly.

**Bag-of-words and benchmark line.** **ARO** (Yuksekgonul, 2023, ICLR,
arXiv:2210.01936), **Winoground** (Thrush, 2022, CVPR,
arXiv:2204.03162; and Diwan, 2022, EMNLP on why it is hard), **CREPE**
(Ma, 2023, CVPR), **VL-CheckList** (Zhao, 2022, EMNLP,
arXiv:2207.00221), **SugarCrepe** (Hsieh, 2023, NeurIPS D&B,
arXiv:2306.14610), **SugarCrepe++** (Dumpala, 2024, NeurIPS,
arXiv:2406.11171), **ConMe** (2024, NeurIPS D&B, arXiv:2406.08164),
**The Hard Positive Truth** (Kamath, 2024, ECCV, arXiv:2409.17958) and
**A Good CREPE needs more than just Sugar** (Udandarao, 2025,
arXiv:2506.08227, arXiv-only) collectively establish and stress-test
CLIP's relational/compositional weaknesses; the last also supports our
prior-adjusted methodology by showing blind heuristics can match CLIP on
badly constructed benchmarks. Overlap: partial throughout (motivation,
not our specific claim).

**Probing and pathology-to-VQA links.** **What'sUp** (Kamath, 2023,
EMNLP), **CountBench** (Paiss, 2023, ICCV, arXiv:2302.12066), **Does
CLIP Bind Concepts?** (Lewis, 2024, Findings of EACL, arXiv:2212.10537;
relational binding collapses in frozen CLIP embeddings — strong,
probe-level, no VQA, no capacity sweep), **When are Lemons Purple?**
(Yamada, 2023, EMNLP; concept-association-bias strength predicts VQA
performance — strong, the clearest published CLIP-embedding-to-VQA-error
link), **Is CLIP ideal?** (Kang, 2025, arXiv:2503.08723, arXiv-only;
argues a single global similarity space cannot represent compositional
structure — strong theoretical backing), **GQA** (Hudson, 2019, CVPR,
arXiv:1902.09506; defines the program metadata we stratify on; LSTM
language-only 41.07, CNN+LSTM 46.55, BottomUp 49.74, MAC 54.06 on test
[verified: arXiv:1902.09506]), **GQA-OOD** (Kervadec, 2021, CVPR;
language priors dominate GQA accuracy — precedent for our prior
adjustment), **Kervadec** (2021, NeurIPS; models guess from statistics
rather than reason). **Spatial Blindspot** (Alam, 2026, arXiv:2601.09954,
arXiv-only) and **VLMClassifier** (Zhang, 2024, NeurIPS,
arXiv:2405.18415) supply encoder-level attribution context. Overlap as
annotated.

**Axis summary.** No published work combines a program-length-stratified
GQA deficit over one fixed frozen encoder with invariance to head
capacity, architecture and training scale, localized via controlled
comparisons. The mandatory engagements are Koishigarina (phrasing), Fu
and Takishita (scale-dependent attribution), and II-MMR/MMVP (nearest
neighbours on phenomenon and localization respectively).

**Additional surveyed context (overlap none or background-partial).**
Also examined and set aside as background: TinyVQA (arXiv:2404.03574,
domain-specific edge VQA); VL-Adapter (CVPR 2022, arXiv:2112.06825,
adapters inside a generative LM); MCAN (CVPR 2019, large co-attention
heads over cached region features); UniFine (arXiv:2307.00862, source of
the tabulated TAP-C GQA number); Eagle (arXiv:2408.15998), MoVA
(arXiv:2404.13046) and related mixture-of-encoder lines; Maniparambil
(CVPR 2025, arXiv:2409.19425, MLP projectors between frozen unimodal
encoders, no VQA); an RSVQA efficiency framework (arXiv:2606.19277);
VL-JEPA (arXiv:2512.10942, embedding-space prediction during
pretraining); Beyond the Final Layer (arXiv:2601.09322, multi-layer
attentive fusion — a depth dimension we did not vary, future work);
register/sink follow-ups (arXiv:2506.08010, 2603.25803, 2504.06738,
2507.16018, 2604.10098); Do VLMs Have Bad Eyes (arXiv:2508.16652); On
the Brittleness of CLIP Text Encoders (arXiv:2511.04247); Why is
Winoground Hard (EMNLP 2022, cited above); a controlled fusion analysis
in depression detection (Brain Sciences 16:366, 2026, noting
capacity-matched fusion comparisons are rare); and one unpublished MPhil
thesis (Anderson, Cambridge, 2022, Frozen-style few-shot VQA — no
overlap threat). None of these alters the axis summaries above.

## 4. Venue notes (as surveyed, July 2026)

TMLR: rolling submissions, claims-supported-by-evidence acceptance
criterion, decisions typically about 2-3 months [verified:
jmlr.org/tmlr]; annual author submission quotas announced June 2026
[verified: TMLR announcement]. WACV 2027 Round 2: submission 28 August
2026, decisions 9 October 2026 [verified: WACV 2027 CFP]. Insights from
Negative Results in NLP, 7th edition at EMNLP 2026 (Budapest, 22-29
October 2026); about 64% acceptance in 2025; deadline expected late
July-August 2026 [verified: insights-workshop.github.io; deadline
unposted at survey time — re-check immediately]. BMVC 2026 closed 29 May
2026; BMVC 2027 is the next cycle [verified: BMVC 2026 CFP]. ARR cycles
run continually; EMNLP 2026's cycle closed 25 May 2026 [verified:
aclrollingreview.org/dates]. Reviewer expectations for this genre:
released code and exact configs, confirmatory (not dev-only) numbers,
multi-seed uncertainty; a single dataset is defensible at TMLR and
workshops when claims are scoped; one frozen-encoder swap is the
cheapest generality insurance. The final venue decision will be
discussed with Prof. Bober.

## 5. What this project claims, and what it does not

Claims are scoped to: one frozen OpenCLIP ViT-B/32 (laion2b_s34b_b79k)
encoder used for both modalities; discriminative classification over the
top-100 GQA answers (extension to 1000 authorized as E3); training
budgets up to 250k questions; trainable heads up to 21.1M parameters;
development-set evaluation under the V2 protocol until the model list is
frozen and the blinded clean test is run.

Within that regime the evidence supports: multimodal heads clearly beat
single-modality heads; the handcrafted fusion features carry a real but
small gain at matched capacity that decays with training scale; the two
interaction terms are redundant; a latent-query reasoner over token-level
features only matches the far smaller global fusion head and does not
reduce the multi-step deficit; the deficit is invariant to head capacity
and architecture; the trained reasoner routes most visual attention to
the pooled CLS token, and removing CLS at test time is costly while
patches-only retraining recovers most accuracy without improving
compositional lift.

The project does not claim: that compositional information is absent
from CLIP representations (Koishigarina shows some is linearly present;
we claim it is insufficient for this task at these scales); that
token-level access is useless in general (Li, Koh and Du and Idefics2
show large trained heads extract more; attribution is scale-dependent
per Fu and Takishita); that any result holds for other encoders,
datasets or generative VQA (the E2 SigLIP swap will provide exactly one
controlled encoder contrast); or anything about clean-test performance
before the embargoed evaluation.
