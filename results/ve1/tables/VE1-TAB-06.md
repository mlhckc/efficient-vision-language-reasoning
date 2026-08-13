# VE1-TAB-06: Measured efficiency, by node and timing class

What does each system cost per query, and under exactly which measurement conditions?

## Main text, node otter155 (E7b): authoritative END_TO_END_SERIAL evidence

| system | experiment family | timing class | node | precision | batch size | warm median serial latency (ms) | comparable with end-to-end | evidence status | accuracy pairing | evidence id |
|---|---|---|---|---|---|---|---|---|---|---|
| concat | E7b | END_TO_END_SERIAL | otter155.eps.surrey.ac.uk | fp32 inference | 1 | 7.6104 | true | VERIFIED | RESOLVED | EV-EFF-E7b.concat |
| e8a_a0p | E7b | END_TO_END_SERIAL | otter155.eps.surrey.ac.uk | fp32 inference | 1 | 9.2117 | true | VERIFIED | RESOLVED | EV-EFF-E7b.e8a_a0p |
| e8a_a1 | E7b | END_TO_END_SERIAL | otter155.eps.surrey.ac.uk | fp32 inference | 1 | 20.1460 | true | VERIFIED | RESOLVED | EV-EFF-E7b.e8a_a1 |
| fusion | E7b | END_TO_END_SERIAL | otter155.eps.surrey.ac.uk | fp32 inference | 1 | 7.6351 | true | VERIFIED | RESOLVED | EV-EFF-E7b.fusion |
| question_only | E7b | END_TO_END_SERIAL | otter155.eps.surrey.ac.uk | fp32 inference | 1 | 1.9603 | true | VERIFIED | RESOLVED | EV-EFF-E7b.question_only |
| reasoner | E7b | END_TO_END_SERIAL | otter155.eps.surrey.ac.uk | fp32 inference | 1 | 9.1719 | true | VERIFIED | RESOLVED | EV-EFF-E7b.reasoner |
| siglip_fusion | E7b | END_TO_END_SERIAL | otter155.eps.surrey.ac.uk | fp32 inference | 1 | 9.8433 | true | VERIFIED | RESOLVED | EV-EFF-E7b.siglip_fusion |
| vocab1000_product | E7b | END_TO_END_SERIAL | otter155.eps.surrey.ac.uk | fp32 inference | 1 | 7.6708 | true | VERIFIED | RESOLVED | EV-EFF-E7b.vocab1000_product |

## Main text, node otter159 (E9): authoritative END_TO_END_SERIAL evidence. Never merged with otter155

| system | experiment family | timing class | node | precision | batch size | warm median serial latency (ms) | comparable with end-to-end | evidence status | accuracy pairing | evidence id |
|---|---|---|---|---|---|---|---|---|---|---|
| e8b_B1 | E9 | END_TO_END_SERIAL | otter159.eps.surrey.ac.uk | fp32 inference | 1 | 10.2420 | true | VERIFIED | PAIRING_UNRESOLVED | EV-EFF-E9.e8b_B1 |
| e8b_B2_R2 | E9 | END_TO_END_SERIAL | otter159.eps.surrey.ac.uk | fp32 inference | 1 | 39.7326 | true | VERIFIED | PAIRING_UNRESOLVED | EV-EFF-E9.e8b_B2_R2 |
| e8b_B2_R3 | E9 | END_TO_END_SERIAL | otter159.eps.surrey.ac.uk | fp32 inference | 1 | 39.2394 | true | VERIFIED | PAIRING_UNRESOLVED | EV-EFF-E9.e8b_B2_R3 |
| e8b_B3_R2 | E9 | END_TO_END_SERIAL | otter159.eps.surrey.ac.uk | fp32 inference | 1 | 39.6561 | true | VERIFIED | PAIRING_UNRESOLVED | EV-EFF-E9.e8b_B3_R2 |
| e8b_B3_R3 | E9 | END_TO_END_SERIAL | otter159.eps.surrey.ac.uk | fp32 inference | 1 | 38.7963 | true | VERIFIED | PAIRING_UNRESOLVED | EV-EFF-E9.e8b_B3_R3 |
| e9_256m_constrained | E9 | END_TO_END_SERIAL | otter159.eps.surrey.ac.uk | fp32 inference | 1 | 142.6747 | true | VERIFIED | RESOLVED | EV-EFF-E9.e9_256m_constrained |
| e9_256m_open | E9 | END_TO_END_SERIAL | otter159.eps.surrey.ac.uk | fp32 inference | 1 | 140.9070 | true | VERIFIED | RESOLVED | EV-EFF-E9.e9_256m_open |
| e9_500m_constrained | E9 | END_TO_END_SERIAL | otter159.eps.surrey.ac.uk | fp32 inference | 1 | 152.8540 | true | VERIFIED | RESOLVED | EV-EFF-E9.e9_500m_constrained |
| e9_500m_open | E9 | END_TO_END_SERIAL | otter159.eps.surrey.ac.uk | fp32 inference | 1 | 151.3801 | true | VERIFIED | RESOLVED | EV-EFF-E9.e9_500m_open |
| fusion | E9 | END_TO_END_SERIAL | otter159.eps.surrey.ac.uk | fp32 inference | 1 | 6.2081 | true | VERIFIED | PAIRING_UNRESOLVED | EV-EFF-E9.fusion |
| vocab1000_product | E9 | END_TO_END_SERIAL | otter159.eps.surrey.ac.uk | fp32 inference | 1 | 6.1990 | true | VERIFIED | PAIRING_UNRESOLVED | EV-EFF-E9.vocab1000_product |

## Retained in full: cached, component and superseded rows, which support no end-to-end latency claim

| system | experiment family | timing class | node | precision | batch size | warm median serial latency (ms) | comparable with end-to-end | evidence status | accuracy pairing | evidence id |
|---|---|---|---|---|---|---|---|---|---|---|
| concat::cached_feature_head_only | E7b | CACHED_FEATURE_HEAD_ONLY | otter155.eps.surrey.ac.uk | fp32 inference | 1 | not bound in VE-0 | false | VERIFIED | NOT_APPLICABLE | EV-EFF-E7b.concat.cached_feature_head_only |
| concat::cached_image_question_side | E7b | CACHED_IMAGE_QUESTION_SIDE | otter155.eps.surrey.ac.uk | fp32 inference | 1 | not bound in VE-0 | false | VERIFIED | NOT_APPLICABLE | EV-EFF-E7b.concat.cached_image_question_side |
| e7a_additive_pipeline_sums | E7a | ADDITIVE_COMPONENT_SUM_SUPERSEDED | otter (E7a session) | fp32 inference | 1 | not bound in VE-0 | false | SUPERSEDED_FOR_END_TO_END_CLAIMS | NOT_APPLICABLE | EV-EFF-E7a.e7a_additive_pipeline_sums |
| e7a_component_measurements | E7a | COMPONENT | otter (E7a session) | fp32 inference | 1 | not bound in VE-0 | false | VERIFIED_AS_COMPONENTS | NOT_APPLICABLE | EV-EFF-E7a.e7a_component_measurements |
| e8a_a0p::cached_feature_head_only | E7b | CACHED_FEATURE_HEAD_ONLY | otter155.eps.surrey.ac.uk | fp32 inference | 1 | not bound in VE-0 | false | VERIFIED | NOT_APPLICABLE | EV-EFF-E7b.e8a_a0p.cached_feature_head_only |
| e8a_a0p::cached_image_question_side | E7b | CACHED_IMAGE_QUESTION_SIDE | otter155.eps.surrey.ac.uk | fp32 inference | 1 | not bound in VE-0 | false | VERIFIED | NOT_APPLICABLE | EV-EFF-E7b.e8a_a0p.cached_image_question_side |
| e8a_a1::cached_feature_head_only | E7b | CACHED_FEATURE_HEAD_ONLY | otter155.eps.surrey.ac.uk | fp32 inference | 1 | not bound in VE-0 | false | VERIFIED | NOT_APPLICABLE | EV-EFF-E7b.e8a_a1.cached_feature_head_only |
| e8a_a1::cached_image_question_side | E7b | CACHED_IMAGE_QUESTION_SIDE | otter155.eps.surrey.ac.uk | fp32 inference | 1 | not bound in VE-0 | false | VERIFIED | NOT_APPLICABLE | EV-EFF-E7b.e8a_a1.cached_image_question_side |
| fusion::cached_feature_head_only | E7b | CACHED_FEATURE_HEAD_ONLY | otter155.eps.surrey.ac.uk | fp32 inference | 1 | not bound in VE-0 | false | VERIFIED | NOT_APPLICABLE | EV-EFF-E7b.fusion.cached_feature_head_only |
| fusion::cached_image_question_side | E7b | CACHED_IMAGE_QUESTION_SIDE | otter155.eps.surrey.ac.uk | fp32 inference | 1 | not bound in VE-0 | false | VERIFIED | NOT_APPLICABLE | EV-EFF-E7b.fusion.cached_image_question_side |
| question_only::cached_feature_head_only | E7b | CACHED_FEATURE_HEAD_ONLY | otter155.eps.surrey.ac.uk | fp32 inference | 1 | not bound in VE-0 | false | VERIFIED | NOT_APPLICABLE | EV-EFF-E7b.question_only.cached_feature_head_only |
| reasoner::cached_feature_head_only | E7b | CACHED_FEATURE_HEAD_ONLY | otter155.eps.surrey.ac.uk | fp32 inference | 1 | not bound in VE-0 | false | VERIFIED | NOT_APPLICABLE | EV-EFF-E7b.reasoner.cached_feature_head_only |
| reasoner::cached_image_question_side | E7b | CACHED_IMAGE_QUESTION_SIDE | otter155.eps.surrey.ac.uk | fp32 inference | 1 | not bound in VE-0 | false | VERIFIED | NOT_APPLICABLE | EV-EFF-E7b.reasoner.cached_image_question_side |
| siglip_fusion::cached_feature_head_only | E7b | CACHED_FEATURE_HEAD_ONLY | otter155.eps.surrey.ac.uk | fp32 inference | 1 | not bound in VE-0 | false | VERIFIED | NOT_APPLICABLE | EV-EFF-E7b.siglip_fusion.cached_feature_head_only |
| siglip_fusion::cached_image_question_side | E7b | CACHED_IMAGE_QUESTION_SIDE | otter155.eps.surrey.ac.uk | fp32 inference | 1 | not bound in VE-0 | false | VERIFIED | NOT_APPLICABLE | EV-EFF-E7b.siglip_fusion.cached_image_question_side |
| vocab1000_product::cached_feature_head_only | E7b | CACHED_FEATURE_HEAD_ONLY | otter155.eps.surrey.ac.uk | fp32 inference | 1 | not bound in VE-0 | false | VERIFIED | NOT_APPLICABLE | EV-EFF-E7b.vocab1000_product.cached_feature_head_only |
| vocab1000_product::cached_image_question_side | E7b | CACHED_IMAGE_QUESTION_SIDE | otter155.eps.surrey.ac.uk | fp32 inference | 1 | not bound in VE-0 | false | VERIFIED | NOT_APPLICABLE | EV-EFF-E7b.vocab1000_product.cached_image_question_side |

**Uncertainty.** no interval; a warm median. The across-pass spread is stated in the caption

**Caveat.** Development-set result on one GQA subset. The clean test is embargoed and unread, so no held-out generalisation is claimed. Component measurements only. The additive gpu_encoder_plus_head_ms, full_pipeline_ms and amortised_ms fields and their Pareto fronts are SUPERSEDED for any end-to-end claim by the measured serial values of E7b. timing_kind=ADDITIVE_COMPONENT_SUM_SUPERSEDED; node=otter (E7a session); precision=fp32 inference; batch_size=1; comparable_with_end_to_end=false. timing_kind=COMPONENT; node=otter (E7a session); precision=fp32 inference; batch_size=1; comparable_with_end_to_end=false. Latency efficiency only, measured on one node. No power or energy measurement exists anywhere in this project. E7b (otter155) and E9 (otter159) are separate frontiers and are never merged: the fusion bridge control failed its pre-registered 10 per cent tolerance at -18.69 per cent and no adjustment factor is applied. timing_kind=END_TO_END_SERIAL; node=otter155.eps.surrey.ac.uk; precision=fp32 inference; batch_size=1; comparable_with_end_to_end=true. timing_kind=CACHED_FEATURE_HEAD_ONLY; node=otter155.eps.surrey.ac.uk; precision=fp32 inference; batch_size=1; comparable_with_end_to_end=false. timing_kind=CACHED_IMAGE_QUESTION_SIDE; node=otter155.eps.surrey.ac.uk; precision=fp32 inference; batch_size=1; comparable_with_end_to_end=false. Contextual positioning only, never a leaderboard or fair-protocol superiority claim. Training history, multimodal pretraining, answer support and output format all differ, and SmolVLM-256M's text backbone is the Instruct checkpoint where E8A and E8B use base. No comparison is made against published official GQA scores in either direction. timing_kind=END_TO_END_SERIAL; node=otter159.eps.surrey.ac.uk; precision=fp32 inference; batch_size=1; comparable_with_end_to_end=true.

**Notes.**

1. COMPUTATIONAL efficiency only. No power or energy measurement exists anywhere in this project.
2. E7b (otter155) and E9 (otter159) are separate measurement groups and are never merged: the fusion bridge control failed its pre-registered 10 per cent tolerance at -18.69 per cent and no adjustment factor is applied.
3. Only END_TO_END_SERIAL rows support an end-to-end latency claim. E7a's additive sums are superseded and appear here only as a recorded supersession.
4. E8B's canonical R1 readout has no serial latency by design; R2 and R3 are never presented as R1's cost. E10's B4 and B4r have no timing evidence of any kind.

**Bolding.** no bolding of the fastest system. The table spans two nodes and three timing classes, and a bold minimum would invite exactly the cross-node reading that is prohibited

Evidence: 36 VE-0 evidence identifiers, listed in `VE1-TAB-06.json`. Specification VE0-TAB-06. Development-set evidence only.
