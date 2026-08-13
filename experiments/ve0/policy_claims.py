"""The final claim ledger and the research-question evidence matrix.

The ledger keeps the seventeen claim identifiers C01 to C17 from the
independently approved statistical and efficiency closure, with their wording
and their limitations preserved, so a reader can follow a claim from that
review into this contract without a rename. VE-0 adds four claims the
dissertation needs and the closure did not carry: the thesis-level claim, the
cross-experiment interface interpretation, and two explicit non-claims about
held-out and out-of-distribution generalisation.

Status vocabulary:

  ESTABLISHED    a controlled contrast with an interval excluding zero, or a
                 self-limiting statement that cannot be wrong
  SUPPORTED      the evidence backs the claim as worded, with a named confound
                 or a named absence-of-detection reading
  TENTATIVE      part of the supporting evidence is missing or weaker than the
                 rest; the claim must be worded with that asymmetry visible
  NOT_SUPPORTED  the evidence does not support it and no permitted work here
                 would change that. These rows are kept, not deleted: several
                 of them are prohibitions.

The Abstract, Discussion and Conclusion may not state a stronger version of
any row here. That is what the forbidden_stronger_version field is for.
"""

from __future__ import annotations

CLAIM_STATUSES = {
    "ESTABLISHED": "a controlled contrast whose interval excludes zero, or a "
                   "self-limiting statement",
    "SUPPORTED": "backed as worded, with a named confound or a named "
                 "absence-of-detection reading",
    "TENTATIVE": "part of the supporting evidence is missing or weaker; the "
                 "wording must show the asymmetry",
    "NOT_SUPPORTED": "not supported, or an explicit prohibition",
}

CLAIMS = [
    {
        "claim_id": "C01",
        "claim_text": "At train_40k the fusion features give a real accuracy "
                      "gain over concatenation, but capacity explains part of "
                      "it: at matched trainable-parameter budgets the gain "
                      "shrinks and one of the two matched contrasts no longer "
                      "excludes zero.",
        "status": "ESTABLISHED",
        "status_reason": "four capacity-matched paired contrasts with "
                         "image-clustered intervals, three excluding zero, on "
                         "identical development rows.",
        "evidence_ids": [],
        "selectors": [{"evidence_class": "CON",
                       "experiment_family": ["v2_02", "v2_03"]}],
        "mandatory_caveat": "The unmatched v2_02 contrast is "
                            "capacity-confounded (1,100,388 against 576,100 "
                            "trainable parameters). Only the v2_03 matched "
                            "pairs isolate the feature effect, and fusion "
                            "minus concat_wide includes zero. Development "
                            "set, train_40k, five training seeds.",
        "forbidden_stronger_version": "that the fusion gain is purely "
                                      "architectural, or that interaction "
                                      "features are shown to be better than "
                                      "capacity.",
        "consumed_by": ["VE0-FIG-02", "VE0-TAB-01", "VE0-TAB-02"],
        "dissertation_sections": ["R1_BASELINES_AND_FUSION",
                                  "R2_CAPACITY_AND_FEATURES"],
        "source": "closure C01, preserved",
    },
    {
        "claim_id": "C02",
        "claim_text": "The fusion advantage over concatenation is largest in "
                      "the lower-data regime and narrows as the training set "
                      "grows.",
        "status": "TENTATIVE",
        "status_reason": "the 40k side carries an image-clustered interval; "
                         "the 250k side does not, because the v2_07 "
                         "row-level reconstruction was attempted and stopped.",
        "evidence_ids": [],
        "selectors": [{"evidence_class": "SEED",
                       "experiment_family": ["v2_07"]},
                      {"evidence_class": "CON",
                       "experiment_family": ["v2_02"]}],
        "mandatory_caveat": "v2_07 ACCEPTED DOCUMENTED LIMITATION. The "
                            "narrowing at 250k rests on stored per-seed means "
                            "and their across-training-seed sample sd alone, "
                            "with no evaluation-sampling interval, and one of "
                            "forty cells differs from reconstruction by one "
                            "numerically tied row. The claim must be written "
                            "with that asymmetry stated, or restricted to the "
                            "40k evidence.",
        "forbidden_stronger_version": "quoting a confidence interval for the "
                                      "250k fusion-minus-concat gap, or "
                                      "presenting the narrowing as "
                                      "statistically demonstrated at 250k.",
        "consumed_by": ["VE0-FIG-03", "VE0-TAB-01"],
        "dissertation_sections": ["R3_SCALING_BEHAVIOUR"],
        "source": "closure C02, preserved; status downgraded from READY to "
                  "TENTATIVE here because the missing interval is a wording "
                  "constraint on the claim, not only a readiness note.",
    },
    {
        "claim_id": "C03",
        "claim_text": "Either interaction term alone recovers most of the "
                      "fusion gain at an equal parameter budget; the product "
                      "and absolute difference terms are largely redundant "
                      "with each other.",
        "status": "SUPPORTED",
        "status_reason": "the complete nine-contrast grid, with the two "
                         "term-versus-term contrasts centred on zero.",
        "evidence_ids": [],
        "selectors": [{"evidence_class": "CON",
                       "experiment_family": ["v2_04"]}],
        "mandatory_caveat": "Redundancy is inferred from the two terms "
                            "performing alike at matched capacity, not from a "
                            "mechanism. Development set, train_40k only.",
        "forbidden_stronger_version": "that the two terms compute the same "
                                      "function, or that one term is "
                                      "unnecessary in general.",
        "consumed_by": ["VE0-FIG-A1", "VE0-TAB-02"],
        "dissertation_sections": ["R2_CAPACITY_AND_FEATURES"],
        "source": "closure C03, preserved",
    },
    {
        "claim_id": "C04",
        "claim_text": "The multimodal heads depend on the correct image: "
                      "replacing it with another development image or with "
                      "zeros costs accuracy, and the question-only control is "
                      "provably unaffected.",
        "status": "ESTABLISHED",
        "status_reason": "ten paired intervention contrasts on identical "
                         "rows with a zero-by-construction control.",
        "evidence_ids": [],
        "selectors": [{"evidence_class": "CON",
                       "experiment_family": ["v2_06"]}],
        "mandatory_caveat": "A drop under a wrong or removed image "
                            "demonstrates reliance on the image. It is NOT "
                            "proof of robust visual reasoning, of grounding "
                            "or of compositional understanding. The "
                            "question-only drop is exactly zero by "
                            "construction and is a control, not a result.",
        "forbidden_stronger_version": "that the systems reason about the "
                                      "image, ground language in the image, "
                                      "or understand the scene "
                                      "compositionally.",
        "consumed_by": ["VE0-FIG-05"],
        "dissertation_sections": ["R4_DEPTH_AND_RELIANCE"],
        "source": "closure C04, preserved",
    },
    {
        "claim_id": "C05",
        "claim_text": "Under the tested train_40k controlled comparison, with "
                      "both sides evaluated on identical development rows, "
                      "the stronger frozen representation improves the "
                      "multimodal systems, supporting representation quality "
                      "as an important factor acting mainly through the image "
                      "path.",
        "status": "SUPPORTED",
        "status_reason": "three multimodal gains exclude zero while the "
                         "question-only gain does not, on identical rows.",
        "evidence_ids": [],
        "selectors": [{"evidence_class": "CON", "experiment_family": ["e2"],
                       "source_key_contains": ["siglip_minus_clip"]}],
        "mandatory_caveat": "Representation quality is NOT isolated: encoder, "
                            "embedding width (512 against 768) and head input "
                            "width all move together. The training streams "
                            "are not paired and the seed label does not pair "
                            "two runs. At train_250k the direction rests on "
                            "stored per-seed means with no clustered "
                            "interval.",
        "forbidden_stronger_version": "PROHIBITED WORDING: that SigLIP proves "
                                      "representation quality is the sole or "
                                      "the main bottleneck. Any 'sole "
                                      "bottleneck' or 'main bottleneck' "
                                      "phrasing is forbidden.",
        "consumed_by": ["VE0-FIG-07", "VE0-TAB-A3"],
        "dissertation_sections": ["R5_LATENT_REASONING_AND_REPRESENTATION"],
        "source": "closure C05, preserved with the reviewer's exact supported "
                  "boundary wording",
    },
    {
        "claim_id": "C06",
        "claim_text": "A pooled deficit at four or more reasoning steps "
                      "persists across both frozen encoders, across training "
                      "scales and across head families.",
        "status": "ESTABLISHED",
        "status_reason": "twelve pooled deficits, every interval excluding "
                         "zero, across two encoders, two scales and four head "
                         "families.",
        "evidence_ids": [],
        "selectors": [{"evidence_class": "DEF"}],
        "mandatory_caveat": "The deficit is measured against fixed v2_05b "
                            "per-bucket priors and describes one system's "
                            "step profile; it is not a contrast between "
                            "systems and establishes no cause. The "
                            "reasoner-minus-fusion paired deficit differences "
                            "all include zero, so no system in the set has "
                            "been SHOWN to reduce the deficit, which is not "
                            "the same as showing the systems equivalent. The "
                            "v2_07 250k CLIP arm of this claim has no "
                            "interval. Step counts are GQA semantic program "
                            "lengths, a proxy for reasoning depth.",
        "forbidden_stronger_version": "that the systems cannot compose, that "
                                      "the deficit measures compositional "
                                      "reasoning ability, or that any system "
                                      "reduces it.",
        "consumed_by": ["VE0-FIG-04", "VE0-FIG-A4"],
        "dissertation_sections": ["R4_DEPTH_AND_RELIANCE"],
        "source": "closure C06, preserved",
    },
    {
        "claim_id": "C07",
        "claim_text": "The lightweight question-conditioned latent-query "
                      "reasoner over token-level visual features does not "
                      "materially outperform the far smaller global-embedding "
                      "fusion head at train_40k, and does not reduce the "
                      "multi-step deficit at any tested scale.",
        "status": "SUPPORTED",
        "status_reason": "three paired deficit-difference contrasts all "
                         "including zero, plus the v3_01 same-seed 40k gap of "
                         "+0.00307 with seeds disagreeing in sign.",
        "evidence_ids": [],
        "selectors": [{"evidence_class": "CON",
                       "experiment_family": ["v3_02a", "v3_03"]},
                      {"evidence_class": "DSC",
                       "experiment_family": ["v3_01"]}],
        "mandatory_caveat": "A system-level comparison. It does not isolate "
                            "token access causally: the reasoner differs from "
                            "the fusion head in input granularity, "
                            "architecture and parameter count at once. All "
                            "three deficit-difference intervals include zero, "
                            "which is an absence of a detected difference, "
                            "not evidence of equivalence. The 40k accuracy "
                            "comparison carries a training-seed spread only, "
                            "with no clustered interval.",
        "forbidden_stronger_version": "that token-level visual access does "
                                      "not help, as a general statement, or "
                                      "that the reasoner and the fusion head "
                                      "are equivalent.",
        "consumed_by": ["VE0-FIG-06"],
        "dissertation_sections": ["R5_LATENT_REASONING_AND_REPRESENTATION"],
        "source": "closure C07, preserved",
    },
    {
        "claim_id": "C08",
        "claim_text": "On the question side, a frozen pretrained "
                      "SmolLM2-135M encoder beats its architecture-matched "
                      "random control at both tested training scales.",
        "status": "ESTABLISHED",
        "status_reason": "two within-size paired contrasts, both intervals "
                         "excluding zero, +0.03820 at 40k and +0.04835 at "
                         "250k.",
        "evidence_ids": [],
        "selectors": [{"evidence_class": "CON", "experiment_family": ["E8A"],
                       "source_key_contains": ["A1_minus_A1r"]}],
        "mandatory_caveat": "A1 minus A1r is the only clean within-size "
                            "question-side pretraining contrast. It says "
                            "nothing about the answer side, and nothing about "
                            "whether the pretrained encoder beats the CLIP "
                            "text tower. E8A's projected interface is a "
                            "learned common width, not a naturally shared "
                            "pretrained embedding space.",
        "forbidden_stronger_version": "that pretrained language models are "
                                      "better question encoders in general, "
                                      "or that this transfers to the answer "
                                      "side.",
        "consumed_by": ["VE0-FIG-08", "VE0-TAB-04", "VE0-TAB-05"],
        "dissertation_sections": ["R6_QUESTION_SIDE_SLM"],
        "source": "closure C08, preserved",
    },
    {
        "claim_id": "C09",
        "claim_text": "The frozen CLIP text tower (A0p) outperforms the "
                      "frozen SmolLM2-135M question encoder (A1) on this "
                      "task.",
        "status": "SUPPORTED",
        "status_reason": "two paired contrasts with intervals excluding zero, "
                         "but the two sides differ in more than one factor.",
        "evidence_ids": [],
        "selectors": [{"evidence_class": "CON", "experiment_family": ["E8A"],
                       "source_key_contains": ["A1_minus_A0p"]}],
        "mandatory_caveat": "This is a system comparison, NOT an isolated "
                            "causal semantics or alignment effect. The two "
                            "sides differ in representation space, in "
                            "interface and in whether the text representation "
                            "shares a pretrained space with the image "
                            "representation.",
        "forbidden_stronger_version": "that shared pretrained space causes "
                                      "the advantage, or that language-model "
                                      "semantics are worse than CLIP text "
                                      "semantics.",
        "consumed_by": ["VE0-TAB-05"],
        "dissertation_sections": ["R6_QUESTION_SIDE_SLM"],
        "source": "closure C09, preserved",
    },
    {
        "claim_id": "C10",
        "claim_text": "On the answer side, this frozen-SLM readout interface "
                      "shows no reliable positive pretrained-over-random "
                      "advantage, at 135M (E8B) or at 360M (E10).",
        "status": "SUPPORTED",
        "status_reason": "an absence-of-detection claim, not a positive one: "
                         "the E10 intervals contain zero with seeds "
                         "disagreeing in sign, and E8B's 250k interval also "
                         "contains zero.",
        "evidence_ids": [],
        "selectors": [{"evidence_class": "CON", "experiment_family": ["E8B"],
                       "source_key_contains": ["B3 - B2"]},
                      {"evidence_class": "CON", "experiment_family": ["E10"],
                       "source_key_contains": ["pretraining_effect"]}],
        "mandatory_caveat": "Both E10 intervals include zero and the seeds "
                            "disagree in sign; they establish NEITHER "
                            "equivalence NOR the absence of an effect. E10 "
                            "does NOT reproduce E8B's directional negative "
                            "40k result: E8B's B3 minus B2 at 40k is -0.02463 "
                            "with an interval excluding zero and all three "
                            "seeds negative, against E10's uncertain -0.00683 "
                            "with mixed seed signs. E8B B3 at train_40k has a "
                            "seed sd of 0.02795 and must always be shown with "
                            "its seed spread.",
        "forbidden_stronger_version": "PROHIBITED: 'pretraining has no "
                                      "effect', 'the two are equivalent', or "
                                      "'we prove the absence of an effect'. "
                                      "Also prohibited: generalising to all "
                                      "small language models or all "
                                      "multimodal interfaces.",
        "consumed_by": ["VE0-FIG-08", "VE0-TAB-05"],
        "dissertation_sections": ["R7_ANSWER_SIDE_AND_CAPACITY"],
        "source": "closure C10, preserved",
    },
    {
        "claim_id": "C11",
        "claim_text": "Moving the answer-side frozen language model from "
                      "135M to 360M does not change the pretraining "
                      "conclusion, and training-set size is what moves "
                      "accuracy.",
        "status": "SUPPORTED",
        "status_reason": "two scale effects with intervals excluding zero "
                         "against a difference in differences that does not.",
        "evidence_ids": [],
        "selectors": [{"evidence_class": "CON", "experiment_family": ["E10"]}],
        "mandatory_caveat": "135M to 360M is whole-system capacity "
                            "sensitivity, not an isolated causal "
                            "language-model-size effect: trainable capacity "
                            "also moves, 21,343,808 to 21,540,800 parameters, "
                            "because the projection width follows the hidden "
                            "size. The difference in differences is not "
                            "directional.",
        "forbidden_stronger_version": "that language-model size does not "
                                      "matter, or that the difference in "
                                      "differences shows the pretraining "
                                      "contribution is flat in scale.",
        "consumed_by": ["VE0-FIG-08", "VE0-TAB-03", "VE0-TAB-05"],
        "dissertation_sections": ["R7_ANSWER_SIDE_AND_CAPACITY"],
        "source": "closure C11, preserved",
    },
    {
        "claim_id": "C12",
        "claim_text": "Growing the closed answer set from 100 to 1000 raises "
                      "development coverage and raises raw-distribution "
                      "accuracy, while costing accuracy on the shared head "
                      "rows.",
        "status": "SUPPORTED",
        "status_reason": "coverage read from the frozen protocol artefacts; "
                         "the accuracy movements are within-view contrasts.",
        "evidence_ids": [],
        "selectors": [{"evidence_class": "CON", "experiment_family": ["e3"]},
                      {"evidence_class": "DSC",
                       "source_key_in": ["dev.coverage_top100",
                                         "dev.coverage_top1000"]}],
        "mandatory_caveat": "Top-100 and top-1000 results are NEVER paired: "
                            "they use different development rows (7,714 "
                            "against 9,823), a different class count and "
                            "different training rows. Class competition and "
                            "the smaller head-answer share of a fixed budget "
                            "are confounded in the head-row loss. The "
                            "side-by-side comparison is context only.",
        "forbidden_stronger_version": "differencing a top-100 accuracy "
                                      "against a top-1000 accuracy, or "
                                      "presenting the two as one scaling "
                                      "curve.",
        "consumed_by": ["VE0-FIG-A3"],
        "dissertation_sections": ["R_APPENDIX_ANSWER_SET"],
        "source": "closure C12, preserved",
    },
    {
        "claim_id": "C13",
        "claim_text": "A small global-embedding head reaches comparable "
                      "raw-distribution accuracy to far larger systems at a "
                      "small fraction of the measured end-to-end serial "
                      "latency.",
        "status": "SUPPORTED",
        "status_reason": "directly measured on one node under one protocol; "
                         "the accuracy side is a cross-family comparison "
                         "rather than a matched contrast.",
        "evidence_ids": [],
        "selectors": [{"evidence_class": "EFF",
                       "timing_kind": ["END_TO_END_SERIAL"]}],
        "mandatory_caveat": "Latency efficiency only, from measured warm "
                            "serial batch-1 queries on ONE node per frontier. "
                            "E7b (otter155) and E9 (otter159) are two "
                            "separate frontiers and are never merged: E9's "
                            "fusion bridge control failed its pre-registered "
                            "10 per cent tolerance at -18.69 per cent and no "
                            "adjustment factor is applied. Cached and "
                            "head-only figures are never quoted as "
                            "end-to-end. No serial latency exists for E8B's "
                            "canonical R1 readout, and R2/R3 are never "
                            "presented as R1's cost. E10 has no timing "
                            "evidence at all.",
        "forbidden_stronger_version": "any energy or power claim, any merged "
                                      "cross-node frontier, any use of E7a's "
                                      "additive full_pipeline_ms or "
                                      "amortised_ms as current end-to-end "
                                      "evidence.",
        "consumed_by": ["VE0-FIG-09", "VE0-FIG-A6", "VE0-TAB-06"],
        "dissertation_sections": ["R8_COMPUTATIONAL_EFFICIENCY"],
        "source": "closure C13, preserved",
    },
    {
        "claim_id": "C14",
        "claim_text": "A frozen compact integrated VLM can be placed in "
                      "context against these lightweight systems.",
        "status": "SUPPORTED",
        "status_reason": "measured on the E9 node under the imported "
                         "protocol; contextual by construction.",
        "evidence_ids": [],
        "selectors": [{"evidence_class": "EFF", "experiment_family": ["E9"]}],
        "mandatory_caveat": "CONTEXTUAL POSITIONING ONLY, never a leaderboard "
                            "or fair-protocol superiority claim. Training "
                            "history, multimodal pretraining, answer support "
                            "and output format all differ, and SmolVLM-256M's "
                            "text backbone is the Instruct checkpoint where "
                            "E8A and E8B use base. No comparison is made "
                            "against published official GQA scores in either "
                            "direction, because the split, the answer support "
                            "and the scorer all differ.",
        "forbidden_stronger_version": "that this project beats or is beaten "
                                      "by SmolVLM, or any comparison against "
                                      "a published GQA leaderboard number.",
        "consumed_by": ["VE0-FIG-A5", "VE0-TAB-06"],
        "dissertation_sections": ["R8_COMPUTATIONAL_EFFICIENCY"],
        "source": "closure C14, preserved; the empty supporting_artefacts "
                  "list the closure review flagged as a LOW observation is "
                  "repaired here by binding the E9 efficiency evidence "
                  "explicitly.",
    },
    {
        "claim_id": "C15",
        "claim_text": "This project makes no claim of general "
                      "vision-language reasoning.",
        "status": "ESTABLISHED",
        "status_reason": "a self-limiting statement about scope.",
        "evidence_ids": [],
        "selectors": [],
        "evidence_free_scope_statement": True,
        "mandatory_caveat": "The task studied is controlled GQA answer "
                            "classification over a closed answer set, with "
                            "reasoning-related analyses layered on it. Every "
                            "result is a development-set result on one "
                            "dataset subset; the clean test remains embargoed "
                            "and unread.",
        "forbidden_stronger_version": "NOT_APPLICABLE: this claim only "
                                      "narrows scope.",
        "consumed_by": ["VE0-TAB-07"],
        "dissertation_sections": ["R_LIMITATIONS"],
        "source": "closure C15, preserved",
    },
    {
        "claim_id": "C16",
        "claim_text": "PROHIBITED: no claim that any system here is "
                      "energy-efficient.",
        "status": "NOT_SUPPORTED",
        "status_reason": "no direct power or energy measurement exists "
                         "anywhere in this project.",
        "evidence_ids": [],
        "selectors": [],
        "evidence_free_scope_statement": True,
        "mandatory_caveat": "Permitted wording is 'computationally "
                            "efficient', 'latency-efficient', or the directly "
                            "measured term. This row exists to record the "
                            "prohibition.",
        "forbidden_stronger_version": "any use of the words energy, power, "
                                      "watt, joule or carbon as a property of "
                                      "a system measured here.",
        "consumed_by": ["VE0-TAB-07"],
        "dissertation_sections": ["R_LIMITATIONS"],
        "source": "closure C16, preserved",
    },
    {
        "claim_id": "C17",
        "claim_text": "PROHIBITED: no merged cross-node efficiency frontier, "
                      "and no equivalence claim from an interval that "
                      "includes zero.",
        "status": "NOT_SUPPORTED",
        "status_reason": "the two nodes disagree by -18.69 per cent on the "
                         "same control system, and an interval containing "
                         "zero is an absence of detection.",
        "evidence_ids": [],
        "selectors": [],
        "evidence_free_scope_statement": True,
        "mandatory_caveat": "The E7b and E9 measurements are on different "
                            "nodes and the bridge control failed its "
                            "tolerance, so they are never merged. An interval "
                            "containing zero is an absence of a detected "
                            "effect and never establishes equivalence or the "
                            "absence of an effect.",
        "forbidden_stronger_version": "NOT_APPLICABLE: this row is itself the "
                                      "prohibition.",
        "consumed_by": ["VE0-FIG-09", "VE0-TAB-06", "VE0-TAB-07"],
        "dissertation_sections": ["R_LIMITATIONS"],
        "source": "closure C17, preserved",
    },
    # ------------------------------------------------------------- VE-0 added
    {
        "claim_id": "C18",
        "claim_text": "Closed-set visual question answering on this GQA "
                      "subset can be performed by a small trainable head over "
                      "frozen encoder features, reaching accuracy comparable "
                      "to much larger systems at a small fraction of the "
                      "measured per-query latency.",
        "status": "SUPPORTED",
        "status_reason": "the thesis-level statement, assembled from the "
                         "accuracy evidence and the measured latency "
                         "evidence. It is an assembly of two separately "
                         "measured axes, not a single controlled contrast.",
        "evidence_ids": [],
        "selectors": [{"evidence_class": "ARM",
                       "experiment_family": ["v2_02"]},
                      {"evidence_class": "EFF",
                       "timing_kind": ["END_TO_END_SERIAL"]}],
        "mandatory_caveat": "Closed-set classification over a fixed answer "
                            "vocabulary, on one image-disjoint development "
                            "partition of GQA, with the clean test embargoed "
                            "and unread. 'Comparable accuracy' and 'a "
                            "fraction of the latency' are measured on "
                            "different axes and, for the compact VLM "
                            "comparison, under a contextual rather than a "
                            "matched protocol. Computational efficiency only; "
                            "no energy was measured.",
        "forbidden_stronger_version": "that embedding-space reasoning "
                                      "replaces autoregressive "
                                      "vision-language models, that the "
                                      "result holds for open-ended VQA, or "
                                      "that it generalises beyond this "
                                      "dataset subset and answer set.",
        "consumed_by": ["VE0-FIG-02", "VE0-FIG-09", "VE0-TAB-06"],
        "dissertation_sections": ["R1_BASELINES_AND_FUSION",
                                  "R8_COMPUTATIONAL_EFFICIENCY",
                                  "R_DISCUSSION"],
        "source": "VE-0, thesis-level claim. Not present in the closure "
                  "ledger, which scoped itself to per-experiment claims.",
    },
    {
        "claim_id": "C19",
        "claim_text": "Taken together, the frozen small-language-model "
                      "results are interface-dependent: pretraining is useful "
                      "on the question side in the tested setting, while no "
                      "reliable positive advantage is detected for either "
                      "tested answer-side interface.",
        "status": "SUPPORTED",
        "status_reason": "the question-side contrasts exclude zero at both "
                         "scales while all four answer-side contrasts at two "
                         "model sizes fail to establish a positive effect.",
        "evidence_ids": [],
        "selectors": [{"evidence_class": "CON", "experiment_family": ["E8A"],
                       "source_key_contains": ["A1_minus_A1r"]},
                      {"evidence_class": "CON", "experiment_family": ["E8B"],
                       "source_key_contains": ["B3 - B2"]},
                      {"evidence_class": "CON", "experiment_family": ["E10"],
                       "source_key_contains": ["pretraining_effect"]}],
        "mandatory_caveat": "'Interface-dependent' is an INTERPRETATION "
                            "across three experiments, not a measured factor: "
                            "no single controlled experiment varied the "
                            "interface side while holding everything else "
                            "fixed. The question-side and answer-side arms "
                            "differ in architecture, in what the projection "
                            "maps, in readout and in checkpoint-selection "
                            "rule. The answer-side half is an "
                            "absence-of-detection result that establishes "
                            "neither equivalence nor absence.",
        "forbidden_stronger_version": "PROHIBITED: generalising to all small "
                                      "language models, all vision-language "
                                      "models, or all multimodal interfaces; "
                                      "or stating that answer-side "
                                      "pretraining is useless.",
        "consumed_by": ["VE0-FIG-08", "VE0-TAB-04", "VE0-TAB-05"],
        "dissertation_sections": ["R6_QUESTION_SIDE_SLM",
                                  "R7_ANSWER_SIDE_AND_CAPACITY",
                                  "R_DISCUSSION"],
        "source": "VE-0, cross-experiment synthesis claim.",
    },
    {
        "claim_id": "C20",
        "claim_text": "PLACEHOLDER, NOT YET CLAIMABLE: performance on the "
                      "held-out clean test.",
        "status": "NOT_SUPPORTED",
        "status_reason": "the clean test is embargoed and unread. F1 (model "
                         "freeze) and F2 (blinded clean-test evaluation) are "
                         "unauthorised and unstarted.",
        "evidence_ids": [],
        "selectors": [],
        "evidence_free_scope_statement": True,
        "mandatory_caveat": "No clean-test number exists anywhere in this "
                            "project. Section R10 stays a placeholder until "
                            "F2 has been authorised and executed. Nothing in "
                            "VE-0, VE-1 or VE-2 may contain a clean-test "
                            "value, statistic, coverage figure or answer "
                            "distribution.",
        "forbidden_stronger_version": "any statement, estimate, projection or "
                                      "expectation about clean-test "
                                      "performance.",
        "consumed_by": ["VE0-TAB-07"],
        "dissertation_sections": ["R10_HELD_OUT_EVALUATION_PLACEHOLDER"],
        "source": "VE-0, embargo firewall claim.",
    },
    {
        "claim_id": "C21",
        "claim_text": "PROHIBITED: no claim of out-of-distribution "
                      "compositional generalisation.",
        "status": "NOT_SUPPORTED",
        "status_reason": "no out-of-distribution split was constructed and no "
                         "compositional generalisation test was run.",
        "evidence_ids": [],
        "selectors": [],
        "evidence_free_scope_statement": True,
        "mandatory_caveat": "Three concepts must be kept distinct and never "
                            "substituted for one another: (1) held-out GQA "
                            "evaluation, which is F2 and has not happened; "
                            "(2) reasoning-depth analysis, which is the "
                            "step-bucket evidence and is in-distribution; and "
                            "(3) true out-of-distribution compositional "
                            "generalisation, which this project never tested. "
                            "The step-bucket deficit is a within-distribution "
                            "profile against fixed priors.",
        "forbidden_stronger_version": "presenting the step-bucket deficit as "
                                      "evidence about compositional "
                                      "generalisation, or the clean test as "
                                      "an out-of-distribution test.",
        "consumed_by": ["VE0-FIG-04", "VE0-TAB-07"],
        "dissertation_sections": ["R_LIMITATIONS"],
        "source": "VE-0, concept-separation claim.",
    },
]

# --------------------------------------------------------------------------
# research questions
# --------------------------------------------------------------------------
# The dissertation does not yet carry a numbered research-question list in a
# tracked artefact. These are VE-0's working formulations, derived from the
# title and from the project description in CLAUDE.md, and they are marked as
# such: they are a mapping device, and the final wording is a supervisor
# decision.

RQ_PROVENANCE = (
    "VE-0 working formulations, derived from the dissertation title and from "
    "the project description in CLAUDE.md. No tracked artefact carries a "
    "numbered research-question list, so these are proposed rather than "
    "quoted, and the final wording is to be confirmed with the supervisor. "
    "The evidence mapping does not depend on the exact wording.")

ANSWER_STATUSES = {
    "ANSWERED": "the evidence answers the question within its stated bounds",
    "PARTIALLY_ANSWERED": "answered on one side, one scale or one axis only",
    "NOT_ANSWERED": "not answered; the work that would answer it was not done "
                    "or is not authorised",
}

RESEARCH_QUESTIONS = [
    {
        "rq_id": "RQ0",
        "kind": "MAIN",
        "question": "Can visual question answering be performed efficiently "
                    "by reasoning in embedding space, using frozen encoders "
                    "and a small trainable head, instead of a large "
                    "autoregressive vision-language model?",
        "answer_status": "PARTIALLY_ANSWERED",
        "supported_answer": "Yes for closed-set answer classification on this "
                            "GQA development partition, and at a small "
                            "fraction of the measured per-query latency of "
                            "the compact integrated VLMs measured on the same "
                            "node. The comparison against those VLMs is "
                            "contextual, not matched, and no held-out "
                            "evaluation has been run.",
        "claim_ids": ["C18", "C13", "C14", "C15"],
        "limitation": "Development set only; closed answer set; contextual "
                      "rather than matched VLM comparison; computational "
                      "efficiency only, with no energy measurement.",
        "dissertation_section": "R_DISCUSSION",
    },
    {
        "rq_id": "RQ1",
        "kind": "SECONDARY",
        "question": "Does explicit multimodal interaction beat plain "
                    "concatenation of frozen global features?",
        "answer_status": "ANSWERED",
        "supported_answer": "Yes at train_40k, and part of the advantage is "
                            "capacity rather than the interaction features "
                            "themselves.",
        "claim_ids": ["C01", "C03"],
        "limitation": "train_40k only for the matched controls; the "
                      "unmatched contrast is capacity-confounded.",
        "dissertation_section": "R1_BASELINES_AND_FUSION",
    },
    {
        "rq_id": "RQ2",
        "kind": "SECONDARY",
        "question": "How much of any fusion advantage is explained by "
                    "trainable capacity rather than by the interaction "
                    "features?",
        "answer_status": "ANSWERED",
        "supported_answer": "A substantial part. At the 576k budget the "
                            "feature effect survives; at the 1.1M budget it "
                            "no longer excludes zero.",
        "claim_ids": ["C01", "C03"],
        "limitation": "Two matched budgets only, one training scale, five "
                      "seeds.",
        "dissertation_section": "R2_CAPACITY_AND_FEATURES",
    },
    {
        "rq_id": "RQ3",
        "kind": "SECONDARY",
        "question": "How do these advantages change as the labelled training "
                    "set grows?",
        "answer_status": "PARTIALLY_ANSWERED",
        "supported_answer": "Accuracy rises at every scale and the "
                            "fusion-over-concatenation advantage narrows, but "
                            "only the 40k end carries an evaluation-sampling "
                            "interval.",
        "claim_ids": ["C02", "C11"],
        "limitation": "v2_07 ACCEPTED DOCUMENTED LIMITATION: no clustered "
                      "interval at 100k or 250k for the global heads.",
        "dissertation_section": "R3_SCALING_BEHAVIOUR",
    },
    {
        "rq_id": "RQ4",
        "kind": "SECONDARY",
        "question": "Where does accuracy concentrate by question type and "
                    "reasoning depth, and does a multi-step deficit persist?",
        "answer_status": "ANSWERED",
        "supported_answer": "A pooled four-or-more-step deficit persists "
                            "across both frozen encoders, both scales and "
                            "every head family tested, and no system in the "
                            "set was shown to reduce it.",
        "claim_ids": ["C06", "C07"],
        "limitation": "Descriptive against fixed priors; establishes no "
                      "cause; step count is a program-length proxy; "
                      "in-distribution only.",
        "dissertation_section": "R4_DEPTH_AND_RELIANCE",
    },
    {
        "rq_id": "RQ5",
        "kind": "SECONDARY",
        "question": "Do these systems actually use the image?",
        "answer_status": "ANSWERED",
        "supported_answer": "Yes. Every multimodal head loses accuracy under "
                            "a wrong or removed image, and the question-only "
                            "control is exactly unaffected.",
        "claim_ids": ["C04"],
        "limitation": "Reliance, not grounding, reasoning or compositional "
                      "understanding.",
        "dissertation_section": "R4_DEPTH_AND_RELIANCE",
    },
    {
        "rq_id": "RQ6",
        "kind": "SECONDARY",
        "question": "Does reasoning over token-level visual features with a "
                    "latent-query reasoner beat the global-embedding heads?",
        "answer_status": "ANSWERED",
        "supported_answer": "Not materially, at any tested scale, and not on "
                            "the multi-step deficit.",
        "claim_ids": ["C07"],
        "limitation": "System-level comparison; token access is not isolated; "
                      "the intervals show absence of detection, not "
                      "equivalence.",
        "dissertation_section": "R5_LATENT_REASONING_AND_REPRESENTATION",
    },
    {
        "rq_id": "RQ7",
        "kind": "SECONDARY",
        "question": "How sensitive are the results to the quality of the "
                    "frozen representation?",
        "answer_status": "PARTIALLY_ANSWERED",
        "supported_answer": "Sensitive at train_40k, with the effect "
                            "concentrated on the image path; at train_250k "
                            "only the per-seed direction is available.",
        "claim_ids": ["C05"],
        "limitation": "Representation quality is not isolated and is not "
                      "shown to be the sole or main bottleneck; no clustered "
                      "contrast at 250k.",
        "dissertation_section": "R5_LATENT_REASONING_AND_REPRESENTATION",
    },
    {
        "rq_id": "RQ8",
        "kind": "SECONDARY",
        "question": "Does frozen small-language-model pretraining help, and "
                    "does the answer depend on which side of the interface "
                    "the model sits?",
        "answer_status": "ANSWERED",
        "supported_answer": "On the question side, yes at both scales. On the "
                            "answer side, no reliable positive advantage is "
                            "detected at 135M or 360M. The combined reading "
                            "is interface-dependent.",
        "claim_ids": ["C08", "C09", "C10", "C11", "C19"],
        "limitation": "The interface-side reading is a cross-experiment "
                      "interpretation, not a controlled factor; the "
                      "answer-side half is an absence of detection.",
        "dissertation_section": ["R6_QUESTION_SIDE_SLM",
                                 "R7_ANSWER_SIDE_AND_CAPACITY"],
    },
    {
        "rq_id": "RQ9",
        "kind": "SECONDARY",
        "question": "What is the accuracy against computational-cost "
                    "trade-off for these systems?",
        "answer_status": "ANSWERED",
        "supported_answer": "Measured warm serial batch-1 latency per node "
                            "places the small global-embedding heads far "
                            "cheaper than the reasoner, the SLM readouts and "
                            "the compact VLMs, at comparable accuracy.",
        "claim_ids": ["C13", "C14"],
        "limitation": "Latency only; two nodes kept separate; E8B R1 and E10 "
                      "have no serial latency evidence.",
        "dissertation_section": "R8_COMPUTATIONAL_EFFICIENCY",
    },
    {
        "rq_id": "RQ10",
        "kind": "SECONDARY",
        "question": "Do the development findings hold on a held-out clean "
                    "test drawn from images never used in development?",
        "answer_status": "NOT_ANSWERED",
        "supported_answer": "NOT_APPLICABLE. The clean test is embargoed and "
                            "unread. F1 and F2 are unauthorised and "
                            "unstarted.",
        "claim_ids": ["C20"],
        "limitation": "Section R10 remains a placeholder. No clean-test value "
                      "of any kind may appear before F2.",
        "dissertation_section": "R10_HELD_OUT_EVALUATION_PLACEHOLDER",
    },
    {
        "rq_id": "RQ11",
        "kind": "SECONDARY",
        "question": "Do these systems generalise compositionally out of "
                    "distribution?",
        "answer_status": "NOT_ANSWERED",
        "supported_answer": "NOT_APPLICABLE. No out-of-distribution "
                            "compositional split was constructed and no such "
                            "test was run. The step-bucket analysis is "
                            "within-distribution and is not a substitute.",
        "claim_ids": ["C21", "C06"],
        "limitation": "Held-out GQA evaluation, reasoning-depth analysis and "
                      "true out-of-distribution compositional generalisation "
                      "are three distinct concepts and are never "
                      "interchanged.",
        "dissertation_section": "R_LIMITATIONS",
    },
]
