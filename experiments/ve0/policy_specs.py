"""Authored figure and table specifications, frozen here for VE-1.

VE-0 renders nothing. It fixes what each figure and table is for, which
evidence rows it may consume, what uncertainty it must show, what its caption
must claim and what caveat that caption must carry.

Evidence is named either explicitly, by evidence_id, or by a selector that
build_specs resolves against the inventory. The written specification always
carries the resolved explicit list, so "exact evidence IDs consumed" is
satisfied and the validation suite can prove every one of them resolves.

Two decisions worth stating plainly, because they shape the plan:

- the candidate list was evaluated, not accepted. The proposed "latent
  reasoner plus representation quality" figure was split in two, because a
  single panel pairing V3 with E2 would suggest a shared axis between a
  system-level comparison and a representation-confounded one. The proposed
  interaction-feature figure moved to the appendix, because its message is a
  refinement of the capacity figure and a second 40k bar chart in the main
  text would be redundant; the nine ablation gaps are carried by a main-text
  table and an appendix forest plot instead.
- no figure mixes the V2-era closed-vocabulary scorer with the pinned G21
  normalised scorer on one quantitative axis. Where both appear in one
  narrative, they appear as separate panels with the metric named on each.
"""

from __future__ import annotations

PLACEMENTS = ("MAIN_TEXT", "APPENDIX")

UNCERTAINTY_KINDS = {
    "CLUSTERED_CI95": "image-clustered 95 per cent evaluation-sampling "
                      "interval; the cluster unit and the fixed trained seed "
                      "set must be named",
    "SEED_SD": "sample standard deviation across independent training seeds "
               "(ddof=1), labelled SD, never labelled CI",
    "SEED_POINTS": "every per-seed value plotted individually, no summary "
                   "bar hiding the spread",
    "NONE_STRUCTURAL": "no uncertainty: a counted quantity",
    "NONE_MEASURED_MEDIAN": "no interval: a warm median over repeated "
                            "passes; the across-pass spread is stated in the "
                            "caption",
    "NOT_APPLICABLE": "a schematic; no data",
}

# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

FIGURES = [
    {
        "figure_id": "VE0-FIG-01",
        "working_title": "System family and evidence chain",
        "scientific_question": "Which systems were built, which frozen parts "
                               "they share, and which experiment family each "
                               "branch belongs to.",
        "evidence_free_schematic": True,
        "evidence_ids": [],
        "selectors": [],
        "chart_type": "block schematic with an experiment-family legend",
        "x_axis": "NOT_APPLICABLE",
        "y_axis": "NOT_APPLICABLE",
        "grouping": "by experiment family, with frozen and trainable parts "
                    "visually distinguished",
        "uncertainty_shown": "NOT_APPLICABLE",
        "caption_claim": "The systems compared in this dissertation share a "
                         "frozen encoder and differ in how the question "
                         "reaches the answer.",
        "mandatory_caption_caveat": "The branches are alternative "
                                    "experimental configurations, not one "
                                    "deployed architecture. Only the parts "
                                    "drawn as trainable were ever trained; "
                                    "every encoder, language model and "
                                    "compact VLM stayed frozen throughout.",
        "placement": "MAIN_TEXT",
        "dissertation_section": "R0_METHOD_OVERVIEW",
        "supervisor_slide": "SLIDE-02-WHAT-WAS-BUILT",
        "likely_viva_question": "Is this one system or several?",
        "viva_answer": "Several. Each branch is a separate controlled "
                       "configuration sharing the frozen CLIP image path; the "
                       "diagram exists so a reader does not read them as one "
                       "pipeline.",
        "decision_note": "Kept in main text. Without it a reader cannot tell "
                         "which experiment a later number belongs to.",
    },
    {
        "figure_id": "VE0-FIG-02",
        "working_title": "Low-data accuracy ladder with capacity-matched "
                         "controls",
        "scientific_question": "Does explicit multimodal interaction beat "
                               "concatenation at 40k, and how much of that "
                               "survives at a matched parameter budget?",
        "evidence_free_schematic": False,
        "evidence_ids": [
            "EV-CON-v2_02.fusion_minus_concat.train_40k",
            "EV-CON-v2_02.concat_minus_question_only.train_40k",
            "EV-CON-v2_03.concat_wide_minus_concat.train_40k",
            "EV-CON-v2_03.fusion_narrow_minus_concat.train_40k",
            "EV-CON-v2_03.fusion_minus_concat_wide.train_40k",
            "EV-CON-v2_03.fusion_minus_fusion_narrow.train_40k",
        ],
        "selectors": [
            {"evidence_class": "ARM", "experiment_family": ["v2_02", "v2_03"],
             "training_scale": ["train_40k"]},
        ],
        "chart_type": "two panels: (a) accuracy per system with clustered "
                      "intervals, (b) contrast forest of the four "
                      "capacity-matched effects plus the unmatched one",
        "x_axis": "(a) development accuracy; (b) effect in accuracy points, "
                  "zero line drawn",
        "y_axis": "system, ordered question_only, image_only, concat, "
                  "concat_wide, fusion_narrow, fusion",
        "grouping": "panel (b) marks each contrast CLEAN_PAIRED_CONTROL or "
                    "CAPACITY_CONFOUNDED",
        "uncertainty_shown": "CLUSTERED_CI95",
        "caption_claim": "At 40k the fusion features give a real gain over "
                         "concatenation, and capacity explains part of it: at "
                         "matched budgets the gain shrinks and one of the two "
                         "matched contrasts no longer excludes zero.",
        "mandatory_caption_caveat": "The unmatched v2_02 contrast is "
                                    "capacity-confounded (1,100,388 against "
                                    "576,100 trainable parameters). Only the "
                                    "v2_03 matched pairs isolate the feature "
                                    "effect. Development set, train_40k, five "
                                    "training seeds. Intervals are "
                                    "image-clustered evaluation-sampling "
                                    "intervals conditioning on the fixed "
                                    "trained seed set.",
        "placement": "MAIN_TEXT",
        "dissertation_section": "R1_BASELINES_AND_FUSION",
        "supervisor_slide": "SLIDE-04-FUSION-HELPS-BUT",
        "likely_viva_question": "Is the fusion gain architectural or just "
                                "more parameters?",
        "viva_answer": "Partly architectural at most. At the 576k budget the "
                       "feature effect survives, +0.00944 excluding zero; at "
                       "the 1.1M budget fusion minus concat_wide is +0.00578 "
                       "and includes zero. The evidence supports 'capacity "
                       "explains part, not necessarily all' and nothing "
                       "stronger.",
        "decision_note": "Main text. This is the figure that both makes and "
                         "bounds the headline low-data claim.",
    },
    {
        "figure_id": "VE0-FIG-03",
        "working_title": "Training-scale behaviour of the global heads",
        "scientific_question": "How do the baseline, concatenation and fusion "
                               "heads move from 40k to 100k to 250k labelled "
                               "questions?",
        "evidence_free_schematic": False,
        "evidence_ids": [],
        "selectors": [
            {"evidence_class": "SEED", "experiment_family": ["v2_07"]},
            {"evidence_class": "DSC",
             "source_key_in": ["train_40k.n_questions",
                               "train_100k.n_questions",
                               "train_250k.n_questions"]},
        ],
        "chart_type": "line plot, one line per system, per-seed points "
                      "overplotted",
        "x_axis": "labelled training questions (40k, 100k, 250k), stated as "
                  "a categorical axis, not a log scale, because there are "
                  "three points",
        "y_axis": "development accuracy",
        "grouping": "one line per system: question_only, concat, "
                    "product_576k, fusion",
        "uncertainty_shown": "SEED_SD",
        "caption_claim": "Accuracy rises with labelled training scale for "
                         "every head, and the gap between fusion and "
                         "concatenation narrows as the training set grows.",
        "mandatory_caption_caveat": "v2_07 ACCEPTED DOCUMENTED LIMITATION. "
                                    "The error bars are the sample standard "
                                    "deviation across five training seeds "
                                    "(ddof=1); they are NOT confidence "
                                    "intervals. No image-clustered "
                                    "evaluation interval exists for this "
                                    "family: row-level reconstruction was "
                                    "attempted and stopped by a "
                                    "point-estimate reproduction mismatch, "
                                    "and one of forty cells differs from "
                                    "reconstruction by one numerically tied "
                                    "row. The 40k end of the narrowing is "
                                    "supported by the clustered evidence in "
                                    "Figure VE0-FIG-02; the 250k end is not.",
        "placement": "MAIN_TEXT",
        "dissertation_section": "R3_SCALING_BEHAVIOUR",
        "supervisor_slide": "SLIDE-05-SCALE-CLOSES-THE-GAP",
        "likely_viva_question": "Why are there no confidence intervals here?",
        "viva_answer": "Because the family's row-level reconstruction was "
                       "stopped by a one-row float32 argmax tie, and rescuing "
                       "the 39 cells that passed after seeing which one "
                       "failed would be outcome-conditioned selection. The "
                       "seed spread is shown instead and labelled as such.",
        "decision_note": "Main text, with the limitation in the caption "
                         "rather than a footnote, because the missing "
                         "interval is the single most likely thing an "
                         "examiner will ask about.",
    },
    {
        "figure_id": "VE0-FIG-04",
        "working_title": "Accuracy by reasoning depth and the pooled "
                         "four-or-more-step deficit",
        "scientific_question": "Where does accuracy fall as the number of "
                               "reasoning steps grows, and does any system in "
                               "the set reduce the deficit?",
        "evidence_free_schematic": False,
        "evidence_ids": [],
        "selectors": [
            {"evidence_class": "SLC", "experiment_family": ["v3_02a"],
             "source_key_contains": ["steps:"]},
            {"evidence_class": "DEF"},
        ],
        "chart_type": "two panels: (a) accuracy per step bucket per system, "
                      "(b) pooled four-or-more-step deficit per system and "
                      "scale with clustered intervals",
        "x_axis": "(a) step bucket, ordered <=2, 3, 4, >=5; (b) deficit in "
                  "accuracy points",
        "y_axis": "(a) development accuracy; (b) system and scale",
        "grouping": "by system; panel (b) groups by encoder and scale",
        "uncertainty_shown": "CLUSTERED_CI95",
        "caption_claim": "A pooled deficit at four or more reasoning steps "
                         "persists across both frozen encoders, across "
                         "training scales and across head families.",
        "mandatory_caption_caveat": "The deficit is measured against fixed "
                                    "v2_05b per-bucket priors and describes "
                                    "one system's step profile; it is not a "
                                    "contrast between systems and establishes "
                                    "no cause. Buckets are unequal and "
                                    "dependent, so every interval is "
                                    "clustered on the bucket's own "
                                    "represented development images; the "
                                    "per-bucket question and image counts "
                                    "must be printed. Step counts come from "
                                    "the GQA semantic program length, which "
                                    "is a proxy for reasoning depth, not a "
                                    "measure of reasoning.",
        "placement": "MAIN_TEXT",
        "dissertation_section": "R4_DEPTH_AND_RELIANCE",
        "supervisor_slide": "SLIDE-06-MULTI-STEP-IS-THE-WALL",
        "likely_viva_question": "Does this show the models cannot compose?",
        "viva_answer": "No. It shows accuracy degrades with program length "
                       "relative to fixed priors, on this split, under this "
                       "closed answer set. Compositional generalisation was "
                       "never tested out of distribution.",
        "decision_note": "Main text. It is the project's clearest negative "
                         "finding and negative findings are shown, not "
                         "omitted.",
    },
    {
        "figure_id": "VE0-FIG-05",
        "working_title": "Visual reliance under wrong and removed images",
        "scientific_question": "How much accuracy does each system lose when "
                               "the image is replaced by another development "
                               "image or by zeros?",
        "evidence_free_schematic": False,
        "evidence_ids": [],
        "selectors": [
            {"evidence_class": "CON", "experiment_family": ["v2_06"]},
        ],
        "chart_type": "grouped horizontal bars, one pair per system "
                      "(shuffled, zeroed), with clustered intervals",
        "x_axis": "accuracy drop in points, zero line drawn",
        "y_axis": "system, with question_only drawn last and labelled a "
                  "control",
        "grouping": "by intervention",
        "uncertainty_shown": "CLUSTERED_CI95",
        "caption_claim": "The multimodal heads depend on the correct image: "
                         "replacing it with another development image or with "
                         "zeros costs accuracy, and the question-only control "
                         "is provably unaffected.",
        "mandatory_caption_caveat": "A drop under a wrong or removed image "
                                    "demonstrates reliance on the image. It "
                                    "is NOT proof of robust visual reasoning, "
                                    "of grounding or of compositional "
                                    "understanding. The question-only drop is "
                                    "exactly zero by construction and is a "
                                    "control, not a result.",
        "placement": "MAIN_TEXT",
        "dissertation_section": "R4_DEPTH_AND_RELIANCE",
        "supervisor_slide": "SLIDE-07-THE-IMAGE-MATTERS",
        "likely_viva_question": "Does a large drop mean the model is "
                                "genuinely grounded?",
        "viva_answer": "No. It means the prediction changes when the image "
                       "changes. Reliance is necessary for grounding but not "
                       "sufficient, and this experiment cannot distinguish "
                       "grounding from image-conditioned shortcut use.",
        "decision_note": "Main text. The zero-by-construction control is what "
                         "makes the other bars interpretable and must be "
                         "shown with them.",
    },
    {
        "figure_id": "VE0-FIG-06",
        "working_title": "Token-level latent reasoning against the global "
                         "heads",
        "scientific_question": "Does a question-conditioned latent-query "
                               "reasoner over token-level visual features beat "
                               "the far smaller global-embedding fusion head, "
                               "and does it reduce the multi-step deficit?",
        "evidence_free_schematic": False,
        "evidence_ids": [
            "EV-CON-v3_02a.reasoner_minus_fusion_deficit.train_40k",
            "EV-CON-v3_03.reasoner_minus_fusion_deficit.train_100k",
            "EV-CON-v3_03.reasoner_minus_fusion_deficit.train_250k",
            "EV-DSC-v3_01.reasoner.train_40k.seed_mean",
            "EV-DSC-v3_01.reasoner.train_40k.seed_sd",
            "EV-DSC-v3_01.reasoner_minus_fusion.train_40k.seed_mean",
        ],
        "selectors": [
            {"evidence_class": "SEED", "experiment_family": ["v3_03"]},
        ],
        "chart_type": "two panels: (a) reasoner and fusion accuracy against "
                      "training scale, (b) forest of the three paired "
                      "reasoner-minus-fusion deficit differences",
        "x_axis": "(a) labelled training questions; (b) deficit difference "
                  "in accuracy points, zero line drawn",
        "y_axis": "(a) development accuracy; (b) training scale",
        "grouping": "panel (a) annotates each system's trainable parameter "
                    "count, because the size difference is the point",
        "uncertainty_shown": "CLUSTERED_CI95",
        "caption_claim": "The 21.1M-parameter latent-query reasoner does not "
                         "materially outperform the 1.1M global fusion head "
                         "at 40k, and no scale shows it reducing the "
                         "four-or-more-step deficit.",
        "mandatory_caption_caveat": "A system-level comparison: input "
                                    "granularity, architecture and parameter "
                                    "count change together, so token-level "
                                    "access is not isolated as a cause. All "
                                    "three deficit-difference intervals "
                                    "include zero, which is an absence of a "
                                    "detected difference, not evidence of "
                                    "equivalence. The 40k accuracy point and "
                                    "its gap come from v3_01 and carry a seed "
                                    "spread only, with no clustered interval. "
                                    "The fusion side at 100k and 250k comes "
                                    "from v2_07 and carries its documented "
                                    "limitation.",
        "placement": "MAIN_TEXT",
        "dissertation_section": "R5_LATENT_REASONING_AND_REPRESENTATION",
        "supervisor_slide": "SLIDE-08-BIGGER-REASONER-DID-NOT-PAY",
        "likely_viva_question": "So token-level access does not help?",
        "viva_answer": "Not in this configuration. Three things changed at "
                       "once, so the honest statement is about the systems as "
                       "built, not about token access in isolation.",
        "decision_note": "Main text, and deliberately separated from the "
                         "representation-quality figure. Pairing V3 with E2 "
                         "in one panel would imply a shared axis between a "
                         "system-level comparison and a "
                         "representation-confounded one.",
    },
    {
        "figure_id": "VE0-FIG-07",
        "working_title": "Representation-quality sensitivity: SigLIP against "
                         "CLIP at 40k",
        "scientific_question": "What does a stronger frozen dual encoder "
                               "change, and does it change the image path or "
                               "the question path?",
        "evidence_free_schematic": False,
        "evidence_ids": [
            "EV-CON-e2.siglip_minus_clip_concat.train_40k",
            "EV-CON-e2.siglip_minus_clip_fusion.train_40k",
            "EV-CON-e2.siglip_minus_clip_product.train_40k",
            "EV-CON-e2.siglip_minus_clip_question_only.train_40k",
        ],
        "selectors": [],
        "chart_type": "forest plot, one row per head, ordered so the "
                      "question_only row is visually last",
        "x_axis": "SigLIP minus CLIP accuracy difference, zero line drawn",
        "y_axis": "head",
        "grouping": "multimodal heads grouped and separated from the "
                    "question-only head",
        "uncertainty_shown": "CLUSTERED_CI95",
        "caption_claim": "Under the tested 40k controlled comparison, the "
                         "stronger frozen representation improves the "
                         "multimodal systems: the three multimodal gains "
                         "exclude zero while the question-only gain does not, "
                         "supporting representation quality as an important "
                         "factor acting mainly through the image path.",
        "mandatory_caption_caveat": "Representation quality is NOT isolated: "
                                    "encoder, embedding width (512 against "
                                    "768) and head input width move together. "
                                    "This does not show representation "
                                    "quality is the sole or even the main "
                                    "bottleneck. The development rows are "
                                    "identical on both sides so the interval "
                                    "is a valid evaluation-sampling interval, "
                                    "but the training streams are not paired "
                                    "and the seed label does not pair two "
                                    "runs. No clustered contrast exists at "
                                    "250k, because the CLIP partner is the "
                                    "stopped v2_07 family; the 250k direction "
                                    "is reported from stored per-seed means "
                                    "in Table VE0-TAB-A3.",
        "placement": "MAIN_TEXT",
        "dissertation_section": "R5_LATENT_REASONING_AND_REPRESENTATION",
        "supervisor_slide": "SLIDE-09-REPRESENTATION-QUALITY",
        "likely_viva_question": "Is the frozen representation the "
                                "bottleneck?",
        "viva_answer": "It is an important factor, and the effect is "
                       "concentrated on the image path. It is not shown to be "
                       "the sole or main bottleneck, because the encoder and "
                       "two widths moved together in the same swap.",
        "decision_note": "Main text as its own figure. The question-only row "
                         "is what licenses 'mainly through the image path' "
                         "and must be visible.",
    },
    {
        "figure_id": "VE0-FIG-08",
        "working_title": "Frozen small-language-model pretraining, question "
                         "side and answer side",
        "scientific_question": "Does frozen pretraining help, and does the "
                               "answer depend on which side of the interface "
                               "the language model sits?",
        "evidence_free_schematic": False,
        "evidence_ids": [
            "EV-CON-E8A.A1_minus_A1r.train_40k",
            "EV-CON-E8A.A1_minus_A1r.train_250k",
            "EV-CON-E8B.B3_-_B2.train_40k",
            "EV-CON-E8B.B3_-_B2.train_250k",
            "EV-CON-E10.pretraining_effect.train_40k",
            "EV-CON-E10.pretraining_effect.train_250k",
            "EV-CON-E10.difference_in_differences",
        ],
        "selectors": [],
        "chart_type": "forest plot in three labelled blocks: question side "
                      "135M, answer side 135M, answer side 360M",
        "x_axis": "pretrained minus architecture-matched random, in accuracy "
                  "points, zero line drawn",
        "y_axis": "block and training scale",
        "grouping": "by interface side and language-model size; every row "
                    "states its reference condition and its comparison class",
        "uncertainty_shown": "CLUSTERED_CI95",
        "caption_claim": "Frozen pretraining helps on the question side at "
                         "both scales, while no reliable positive "
                         "pretrained-over-random advantage is detected for "
                         "either tested answer-side readout.",
        "mandatory_caption_caveat": "Every row is pretrained minus its "
                                    "architecture-matched random control at "
                                    "the same model size; a random 135M model "
                                    "is never the control for a pretrained "
                                    "360M model. An interval that contains "
                                    "zero is an absence of a detected effect "
                                    "and establishes NEITHER equivalence NOR "
                                    "the absence of an effect. E10 does not "
                                    "reproduce E8B's directional negative 40k "
                                    "result. The 135M to 360M step is "
                                    "whole-system capacity sensitivity, not "
                                    "an isolated causal language-model-size "
                                    "effect: trainable capacity moves too, "
                                    "21,343,808 to 21,540,800 parameters. "
                                    "E8B B3 at 40k has a seed sd of 0.02795 "
                                    "and its per-seed values must be shown "
                                    "beside this figure. Checkpoint selection "
                                    "differs: E8A selects the best "
                                    "development epoch, E8B and E10 use the "
                                    "frozen fixed-22 rule.",
        "placement": "MAIN_TEXT",
        "dissertation_section": ["R6_QUESTION_SIDE_SLM",
                                 "R7_ANSWER_SIDE_AND_CAPACITY"],
        "supervisor_slide": "SLIDE-10-PRETRAINING-IS-INTERFACE-DEPENDENT",
        "likely_viva_question": "Does this mean pretraining does not matter "
                                "for the answer side?",
        "viva_answer": "No. It means no reliable positive advantage was "
                       "detected for these two answer-side interfaces at "
                       "these scales. The intervals contain zero, which does "
                       "not establish equivalence or absence, and the finding "
                       "does not generalise to other interfaces.",
        "decision_note": "Main text. Three families on one forest is "
                         "justified because all three share the pinned G21 "
                         "metric, the same development rows and the same "
                         "effect orientation.",
    },
    {
        "figure_id": "VE0-FIG-09",
        "working_title": "Accuracy against measured end-to-end serial "
                         "latency, by measurement node",
        "scientific_question": "What does each system cost per query, and "
                               "which systems are not dominated?",
        "evidence_free_schematic": False,
        "evidence_ids": [],
        "selectors": [
            {"evidence_class": "EFF", "timing_kind": ["END_TO_END_SERIAL"]},
        ],
        "chart_type": "two side-by-side scatter panels, one per measurement "
                      "node, never one merged frontier",
        "x_axis": "warm median serial latency, milliseconds, batch 1, log "
                  "scale permitted and labelled",
        "y_axis": "accuracy on the denominator stated per point",
        "grouping": "panel (a) E7b on otter155, panel (b) E9 on otter159; "
                    "each panel carries its own node label and its own "
                    "frontier",
        "uncertainty_shown": "NONE_MEASURED_MEDIAN",
        "caption_claim": "A small global-embedding head reaches comparable "
                         "accuracy to much larger systems at a small fraction "
                         "of the measured end-to-end serial latency.",
        "mandatory_caption_caveat": "COMPUTATIONAL efficiency only, measured "
                                    "as latency. No power or energy "
                                    "measurement exists anywhere in this "
                                    "project, so no energy-efficiency claim "
                                    "is made. The two panels are two separate "
                                    "frontiers on two different nodes and are "
                                    "NEVER merged: the fusion bridge control "
                                    "failed its pre-registered 10 per cent "
                                    "tolerance at -18.69 per cent and no "
                                    "adjustment factor is applied. Only "
                                    "END_TO_END_SERIAL rows appear; cached "
                                    "and head-only figures are never plotted "
                                    "here. E7a's additive sums and the Pareto "
                                    "fronts built on them are superseded. E9 "
                                    "is contextual positioning, not a "
                                    "fair-protocol superiority claim, and "
                                    "E10's B4/B4r have no timing evidence of "
                                    "any kind.",
        "placement": "MAIN_TEXT",
        "dissertation_section": "R8_COMPUTATIONAL_EFFICIENCY",
        "supervisor_slide": "SLIDE-11-COST-PER-QUERY",
        "likely_viva_question": "Why not one Pareto plot?",
        "viva_answer": "Because the two measurement nodes disagree by 18.7 "
                       "per cent on the same control system, which is a "
                       "hardware difference, not a system difference. Merging "
                       "them would attribute a node effect to a model.",
        "decision_note": "Main text, two panels. A single merged frontier is "
                         "prohibited by claim C17.",
    },
    {
        "figure_id": "VE0-FIG-10",
        "working_title": "Qualitative development examples",
        "scientific_question": "What do agreements and disagreements between "
                               "these systems look like on individual "
                               "development questions?",
        "evidence_free_schematic": False,
        "evidence_ids": [],
        "selectors": [
            {"evidence_class": "DSC",
             "source_key_in": ["dev.n_questions_in_vocabulary",
                               "dev.n_images"]},
        ],
        "chart_type": "image gallery: image, question, gold answer, each "
                      "compared system's prediction, correctness marks",
        "x_axis": "NOT_APPLICABLE",
        "y_axis": "NOT_APPLICABLE",
        "grouping": "one row per selected example, grouped by selection "
                    "category",
        "uncertainty_shown": "NOT_APPLICABLE",
        "caption_claim": "Examples selected by a frozen deterministic rule, "
                         "not chosen for appearance.",
        "mandatory_caption_caveat": "Development examples only; the clean "
                                    "test is embargoed and unread. Every "
                                    "example is drawn by the deterministic "
                                    "rank defined in the VE-0 qualitative "
                                    "protocol, whose salt was frozen before "
                                    "any example was inspected. Illustrative "
                                    "only: n examples cannot support a "
                                    "quantitative claim, and no scene graph "
                                    "or reasoning chain is shown that is not "
                                    "present in the canonical evidence.",
        "placement": "MAIN_TEXT",
        "dissertation_section": "R9_QUALITATIVE_AND_ERROR_ANALYSIS",
        "supervisor_slide": "SLIDE-12-WHAT-FAILURE-LOOKS-LIKE",
        "likely_viva_question": "How did you choose these examples?",
        "viva_answer": "By a SHA-256 rank over a salt frozen before any "
                       "example was seen, taking the lowest ranks in each "
                       "pre-declared category. Skips are recorded with a "
                       "reason; nothing was substituted for looking better.",
        "decision_note": "Main text but small. Populated in VE-2, not in "
                         "VE-0.",
        "populated_by": "VE-2",
    },
    # ---------------------------------------------------------------- appendix
    {
        "figure_id": "VE0-FIG-A1",
        "working_title": "Interaction-feature decomposition at matched "
                         "capacity",
        "scientific_question": "Which interaction term carries the fusion "
                               "gain, and are the product and absolute "
                               "difference terms redundant?",
        "evidence_free_schematic": False,
        "evidence_ids": [],
        "selectors": [
            {"evidence_class": "CON", "experiment_family": ["v2_04"]},
        ],
        "chart_type": "forest plot of all nine published ablation gaps",
        "x_axis": "effect in accuracy points, zero line drawn",
        "y_axis": "contrast",
        "grouping": "matched-budget contrasts separated from natural-width "
                    "contrasts",
        "uncertainty_shown": "CLUSTERED_CI95",
        "caption_claim": "Either interaction term alone recovers most of the "
                         "fusion gain at an equal parameter budget, and the "
                         "two terms behave alike.",
        "mandatory_caption_caveat": "Redundancy is inferred from the two "
                                    "terms performing alike at matched "
                                    "capacity, not from a mechanism. The "
                                    "natural-width contrasts are "
                                    "capacity-confounded. Development set, "
                                    "train_40k, five training seeds. The grid "
                                    "is complete: all nine published gaps are "
                                    "shown, including the four whose "
                                    "intervals include zero.",
        "placement": "APPENDIX",
        "dissertation_section": "R2_CAPACITY_AND_FEATURES",
        "supervisor_slide": "NOT_PRESENTED",
        "likely_viva_question": "Did you pick the ablation that worked?",
        "viva_answer": "No. The grid is complete and all nine gaps are "
                       "reported, four of them including zero.",
        "decision_note": "Moved out of the main text: its message refines "
                         "VE0-FIG-02, and a second 40k bar chart in the main "
                         "text would be redundant. The main-text carrier is "
                         "Table VE0-TAB-02.",
    },
    {
        "figure_id": "VE0-FIG-A2",
        "working_title": "Per-seed dispersion across every trained family",
        "scientific_question": "How much does a result move when only the "
                               "training seed changes?",
        "evidence_free_schematic": False,
        "evidence_ids": [],
        "selectors": [
            {"evidence_class": "SEED"},
        ],
        "chart_type": "dot plot, one dot per training seed, mean marked",
        "x_axis": "development accuracy",
        "y_axis": "family, system and scale",
        "grouping": "by family; the two high-dispersion rows are highlighted "
                    "and named",
        "uncertainty_shown": "SEED_POINTS",
        "caption_claim": "Most systems are seed-stable; two rows are not, and "
                         "are named.",
        "mandatory_caption_caveat": "The spread here is across independent "
                                    "TRAINING seeds. It is a different "
                                    "quantity from the image-clustered "
                                    "evaluation-sampling intervals used "
                                    "elsewhere and the two are never "
                                    "interchanged. Seed sets differ by "
                                    "family (five seeds for V2, E2 and E3; "
                                    "three for V3, E8A, E8B and E10) and are "
                                    "never pooled across families. Two "
                                    "metrics appear in this figure and are "
                                    "shown as separate blocks, never on one "
                                    "shared scale.",
        "placement": "APPENDIX",
        "dissertation_section": "R_APPENDIX_STATISTICS",
        "supervisor_slide": "SLIDE-BACKUP-SEED-STABILITY",
        "likely_viva_question": "How stable are these numbers?",
        "viva_answer": "Stable except for two rows, E8B B3 at 40k with a seed "
                       "sd of 0.02795 and v2_06 image_only shuffled. Both are "
                       "flagged and always shown with their spread.",
        "decision_note": "Appendix, with a backup slide. It is the honest "
                         "answer to the most common examiner challenge about "
                         "small differences.",
        "mixed_metric_justification": "Two metrics appear because the figure "
            "is about seed dispersion, not about accuracy level. They are "
            "drawn as separate blocks with the metric named on each, and no "
            "value is differenced or ranked across the blocks.",
    },
    {
        "figure_id": "VE0-FIG-A3",
        "working_title": "Answer-set size: coverage against accuracy",
        "scientific_question": "What does growing the closed answer set from "
                               "100 to 1000 buy and cost?",
        "evidence_free_schematic": False,
        "evidence_ids": [
            "EV-DSC-dev.coverage_top100",
            "EV-DSC-dev.coverage_top1000",
        ],
        "selectors": [
            {"evidence_class": "ARM", "experiment_family": ["e3"]},
            {"evidence_class": "CON", "experiment_family": ["e3"]},
        ],
        "chart_type": "two panels: coverage bars, and E3-view accuracies by "
                      "system and scale",
        "x_axis": "(a) coverage share; (b) development accuracy on the E3 "
                  "view",
        "y_axis": "answer-set size and system",
        "grouping": "by answer-set size, drawn as two clearly separated "
                    "blocks",
        "uncertainty_shown": "CLUSTERED_CI95",
        "caption_claim": "Growing the closed answer set from 100 to 1000 "
                         "raises development coverage substantially.",
        "mandatory_caption_caveat": "Top-100 and top-1000 results are NEVER "
                                    "paired: they use different development "
                                    "rows (7,714 against 9,823), a different "
                                    "class count and different training rows. "
                                    "No accuracy is differenced across the two "
                                    "blocks. Class competition and the smaller "
                                    "head-answer share of a fixed budget are "
                                    "confounded in any head-row comparison.",
        "placement": "APPENDIX",
        "dissertation_section": "R_APPENDIX_ANSWER_SET",
        "supervisor_slide": "NOT_PRESENTED",
        "likely_viva_question": "Why not use the 1000-answer set "
                                "throughout?",
        "viva_answer": "Because the whole programme's controls, capacity "
                       "matching and reliance analyses were run on the "
                       "top-100 view, and the two views are not row-"
                       "comparable. E3 answers an answer-set-design question "
                       "and is reported separately.",
        "decision_note": "Appendix. E3 is deliberately outside the main "
                         "sequence: it answers an answer-set question rather "
                         "than a reasoning or efficiency question.",
    },
    {
        "figure_id": "VE0-FIG-A4",
        "working_title": "Per-type accuracy and the fusion-minus-concat gap "
                         "by question type",
        "scientific_question": "Which question types carry the fusion "
                               "advantage?",
        "evidence_free_schematic": False,
        "evidence_ids": [],
        "selectors": [
            {"evidence_class": "SLC", "experiment_family": ["v2_02", "v2_05c"]},
        ],
        "chart_type": "two panels: per-type accuracy, and per-type gap forest",
        "x_axis": "accuracy, then gap in accuracy points",
        "y_axis": "question type, structural then semantic",
        "grouping": "structural and semantic types kept in separate blocks",
        "uncertainty_shown": "CLUSTERED_CI95",
        "caption_claim": "The fusion advantage concentrates in a subset of "
                         "question types rather than lifting all types "
                         "equally.",
        "mandatory_caption_caveat": "Slices are unequal and dependent; every "
                                    "interval is clustered on the slice's own "
                                    "represented development images and the "
                                    "per-slice question and image counts must "
                                    "be printed. The v2_05b row-level "
                                    "normal-approximation intervals are "
                                    "superseded by these clustered intervals "
                                    "and are never re-quoted. Both the "
                                    "five-seed and the seed-42 conventions "
                                    "exist in the source; only one is plotted "
                                    "and the caption names which.",
        "placement": "APPENDIX",
        "dissertation_section": "R_APPENDIX_SLICES",
        "supervisor_slide": "NOT_PRESENTED",
        "likely_viva_question": "Are these per-type differences significant "
                                "after multiple comparisons?",
        "viva_answer": "They are nominal 95 per cent intervals, uncorrected "
                       "for multiplicity, and the report says so. With "
                       "fourteen slices, some intervals excluding zero by "
                       "chance is expected, which is why per-type findings "
                       "are secondary.",
        "decision_note": "Appendix. The main text carries the pooled depth "
                         "story; the full type breakdown belongs with the "
                         "supporting statistics.",
    },
    {
        "figure_id": "VE0-FIG-A5",
        "working_title": "Compact VLM in context",
        "scientific_question": "Where does a frozen compact integrated VLM "
                               "sit relative to these lightweight systems?",
        "evidence_free_schematic": False,
        "evidence_ids": [],
        "selectors": [
            {"evidence_class": "EFF", "experiment_family": ["E9"]},
        ],
        "chart_type": "annotated scatter on the E9 node only, accuracy "
                      "against latency, compact VLMs marked distinctly",
        "x_axis": "warm median serial latency on otter159, milliseconds",
        "y_axis": "normalised accuracy on the raw development partition",
        "grouping": "compact VLMs separated from the lightweight systems",
        "uncertainty_shown": "NONE_MEASURED_MEDIAN",
        "caption_claim": "A frozen compact integrated VLM can be placed in "
                         "context against these lightweight systems on one "
                         "node.",
        "mandatory_caption_caveat": "CONTEXTUAL POSITIONING ONLY, never a "
                                    "leaderboard or fair-protocol superiority "
                                    "claim. Training history, multimodal "
                                    "pretraining, answer support and output "
                                    "format all differ, and SmolVLM-256M's "
                                    "text backbone is the Instruct checkpoint "
                                    "where E8A and E8B use base. No "
                                    "comparison is made against published "
                                    "official GQA scores in either direction, "
                                    "because the split, the answer support "
                                    "and the scorer all differ. Single node "
                                    "otter159; these points are never merged "
                                    "with the otter155 frontier.",
        "placement": "APPENDIX",
        "dissertation_section": "R8_COMPUTATIONAL_EFFICIENCY",
        "supervisor_slide": "SLIDE-BACKUP-COMPACT-VLM",
        "likely_viva_question": "Did you beat SmolVLM?",
        "viva_answer": "That is not a question this evidence can answer. The "
                       "systems differ in training history, answer support "
                       "and output format, so the comparison is contextual "
                       "positioning, not a controlled contest.",
        "decision_note": "Appendix with a backup slide. Main-text prominence "
                         "would invite exactly the leaderboard reading claim "
                         "C14 forbids.",
    },
    {
        "figure_id": "VE0-FIG-A6",
        "working_title": "Where the time goes: cached and component costs "
                         "against measured end-to-end",
        "scientific_question": "Which stage of the pipeline dominates a "
                               "query, and how much does feature caching "
                               "remove?",
        "evidence_free_schematic": False,
        "evidence_ids": [],
        "selectors": [
            # deliberately excludes ADDITIVE_COMPONENT_SUM_SUPERSEDED. That
            # row is a recorded supersession, not a measurement, and plotting
            # it as a bar beside measured values is exactly the error the
            # supersession exists to prevent.
            {"evidence_class": "EFF", "experiment_family": ["E7a", "E7b"],
             "timing_kind": ["END_TO_END_SERIAL", "CACHED_FEATURE_HEAD_ONLY",
                             "CACHED_IMAGE_QUESTION_SIDE", "COMPONENT"]},
        ],
        "chart_type": "grouped bars per system: end-to-end serial, "
                      "cached-image question side, cached-feature head only, "
                      "each block labelled with its timing class",
        "x_axis": "milliseconds",
        "y_axis": "system and timing class",
        "grouping": "by timing class, with the class name printed on every "
                    "bar",
        "uncertainty_shown": "NONE_MEASURED_MEDIAN",
        "caption_claim": "The frozen encoder dominates a raw query; the "
                         "trainable head is a small fraction of it.",
        "mandatory_caption_caveat": "Only the END_TO_END_SERIAL bars are "
                                    "end-to-end query latencies. The cached "
                                    "and head-only bars are partial pipelines "
                                    "and are never quoted as a query cost. "
                                    "E7a's additive gpu_encoder_plus_head_ms, "
                                    "full_pipeline_ms and amortised_ms fields "
                                    "and every Pareto front built on them are "
                                    "SUPERSEDED by E7b's measured serial "
                                    "values and must not appear. Single node "
                                    "otter155.",
        "placement": "APPENDIX",
        "dissertation_section": "R8_COMPUTATIONAL_EFFICIENCY",
        "supervisor_slide": "NOT_PRESENTED",
        "likely_viva_question": "Could caching make this much faster in "
                                "deployment?",
        "viva_answer": "Caching image features removes the image-encoder "
                       "stage for repeated questions about one image, and the "
                       "cached bars measure that. It is a deployment "
                       "observation, not the end-to-end cost, and the older "
                       "amortised figure derived from additive sums is "
                       "superseded.",
        "decision_note": "Appendix. It exists so the main efficiency figure "
                         "can stay strictly end-to-end without losing the "
                         "component story.",
    },
]

# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------

TABLES = [
    {
        "table_id": "VE0-TAB-01",
        "working_title": "Main experimental progression: global-embedding "
                         "heads",
        "scientific_question": "What accuracy does each global-embedding head "
                               "reach at each training scale?",
        "evidence_ids": [],
        "selectors": [
            # v2_04's ablation arms are deliberately excluded: they belong to
            # the feature decomposition in VE0-TAB-02, and listing them here
            # would make the progression table read as a flat catalogue.
            {"evidence_class": "ARM", "experiment_family": ["v2_02", "v2_03"]},
            {"evidence_class": "SEED", "experiment_family": ["v2_07"]},
        ],
        "columns": ["system", "trainable parameters", "training scale",
                    "seeds", "mean accuracy",
                    "sd across training seeds (ddof=1)",
                    "95% image-clustered interval", "interval available"],
        "precision": "accuracies and effects to 5 decimal places, matching "
                     "the canonical stored precision; never rounded further "
                     "in a way that changes a reported comparison",
        "uncertainty_notation": "SD in a column headed 'sd (training seeds)', "
                                "interval in a separate column headed "
                                "'95% CI (image-clustered)'. Where no "
                                "interval exists the cell reads "
                                "'not available' and never a dash that could "
                                "be read as zero width.",
        "bolding_rule": "no bolding. Bolding a maximum in a table where some "
                        "rows have no interval would imply a comparison the "
                        "evidence does not support",
        "highlight_rule": "no significance stars, no colour by direction",
        "footnotes": [
            "v2_07 rows carry the ACCEPTED DOCUMENTED LIMITATION: no "
            "image-clustered evaluation interval exists for that family, and "
            "one of forty cells differs from reconstruction by one "
            "numerically tied row.",
            "All rows use the V2-era closed-vocabulary scorer on the 7,714 "
            "in-vocabulary development rows over 768 represented images.",
            "Checkpoints were selected at the best development epoch.",
        ],
        "placement": "MAIN_TEXT",
        "dissertation_section": "R1_BASELINES_AND_FUSION",
    },
    {
        "table_id": "VE0-TAB-02",
        "working_title": "Capacity-matched controls and interaction-feature "
                         "decomposition",
        "scientific_question": "How much of the fusion advantage survives at "
                               "a matched budget, and which term carries it?",
        "evidence_ids": [],
        "selectors": [
            {"evidence_class": "CON", "experiment_family": ["v2_02", "v2_03",
                                                            "v2_04"]},
        ],
        "columns": ["contrast", "comparison class", "effect",
                    "95% image-clustered interval", "excludes zero",
                    "sd across training seeds (ddof=1)", "n questions",
                    "n images"],
        "precision": "5 decimal places",
        "uncertainty_notation": "interval as [lower, upper] with the cluster "
                                "unit stated in the caption; SD in its own "
                                "column and never inside the brackets",
        "bolding_rule": "no bolding by outcome. The comparison-class column "
                        "carries the emphasis instead, because whether a "
                        "contrast is capacity-matched matters more than "
                        "whether it is positive",
        "highlight_rule": "the 'excludes zero' column is a plain true/false; "
                          "it is not rendered as a significance star and "
                          "never as a p-value",
        "footnotes": [
            "Every interval is nominal 95 per cent, image-clustered and "
            "UNCORRECTED for multiplicity. No p-value is computed anywhere in "
            "this project.",
            "An interval containing zero is an absence of a detected effect. "
            "It never establishes equivalence or the absence of an effect.",
            "Trainable parameter counts: concat 576,100; fusion 1,100,388; "
            "concat_wide and fusion_narrow are the matched budgets.",
        ],
        "placement": "MAIN_TEXT",
        "dissertation_section": "R2_CAPACITY_AND_FEATURES",
    },
    {
        "table_id": "VE0-TAB-03",
        "working_title": "Training-scale effects, frozen-SLM families",
        "scientific_question": "How much does labelled training scale move "
                               "each SLM-interface system?",
        "evidence_ids": [
            "EV-CON-E8B.scale_effect_B1",
            "EV-CON-E8B.scale_effect_B2",
            "EV-CON-E8B.scale_effect_B3",
            "EV-CON-E10.scale_effect_B4",
            "EV-CON-E10.scale_effect_B4r",
        ],
        "selectors": [],
        "columns": ["system", "language model", "effect (250k minus 40k)",
                    "95% image-clustered interval", "excludes zero",
                    "sd across training seeds (ddof=1)"],
        "precision": "5 decimal places",
        "uncertainty_notation": "as VE0-TAB-02",
        "bolding_rule": "no bolding",
        "highlight_rule": "none",
        "footnotes": [
            "All rows use the pinned G21 normalised scorer. They are never "
            "placed on a shared scale with the V2-era closed-vocabulary "
            "scorer used by Table VE0-TAB-01.",
            "E8B and E10 use the frozen fixed-22-epoch rule with no early "
            "stopping.",
        ],
        "placement": "MAIN_TEXT",
        "dissertation_section": "R3_SCALING_BEHAVIOUR",
    },
    {
        "table_id": "VE0-TAB-04",
        "working_title": "Frozen-SLM interface comparison",
        "scientific_question": "What does each frozen-SLM configuration "
                               "reach, on each side of the interface?",
        "evidence_ids": [],
        "selectors": [
            {"evidence_class": "SEED", "experiment_family": ["E8A", "E8B",
                                                             "E10"]},
        ],
        "columns": ["arm", "interface side", "language model",
                    "pretrained or random", "training scale", "seeds",
                    "mean accuracy", "sd across training seeds (ddof=1)",
                    "per-seed values"],
        "precision": "5 decimal places",
        "uncertainty_notation": "SD only; this table reports no interval, "
                                "because it reports systems rather than "
                                "contrasts. The contrasts are in "
                                "VE0-TAB-05 and Figure VE0-FIG-08",
        "bolding_rule": "no bolding",
        "highlight_rule": "rows flagged for high seed dispersion are marked "
                          "with a dagger and their per-seed values are "
                          "printed in full",
        "footnotes": [
            "The per-seed column is mandatory: E8B B3 at train_40k has a seed "
            "sd of 0.02795 and its mean alone is misleading.",
            "Every language model is frozen and was never trained or "
            "fine-tuned. The only newly trainable component in these paths is "
            "one linear projection per configuration.",
            "E8A's projected interface is a learned common width, not a "
            "naturally shared pretrained embedding space.",
        ],
        "placement": "MAIN_TEXT",
        "dissertation_section": ["R6_QUESTION_SIDE_SLM",
                                 "R7_ANSWER_SIDE_AND_CAPACITY"],
    },
    {
        "table_id": "VE0-TAB-05",
        "working_title": "Primary pretraining contrasts and the difference "
                         "in differences",
        "scientific_question": "Is there a reliable pretrained-over-random "
                               "advantage, and does it grow with scale?",
        "evidence_ids": [
            "EV-CON-E8A.A1_minus_A1r.train_40k",
            "EV-CON-E8A.A1_minus_A1r.train_250k",
            "EV-CON-E8B.B3_-_B2.train_40k",
            "EV-CON-E8B.B3_-_B2.train_250k",
            "EV-CON-E10.pretraining_effect.train_40k",
            "EV-CON-E10.pretraining_effect.train_250k",
            "EV-CON-E10.difference_in_differences",
        ],
        "selectors": [],
        "columns": ["contrast", "interface side", "language model size",
                    "reference condition", "effect",
                    "95% image-clustered interval", "excludes zero",
                    "per-seed effects", "comparison class"],
        "precision": "5 decimal places",
        "uncertainty_notation": "interval as [lower, upper]; per-seed effects "
                                "printed so a reader can see where seeds "
                                "disagree in sign",
        "bolding_rule": "no bolding. Bolding the question-side row because it "
                        "is positive would visually assert the very "
                        "asymmetry the reader should judge from the intervals",
        "highlight_rule": "none; direction is carried by the sign and the "
                          "interval, not by colour",
        "footnotes": [
            "Every contrast is pretrained minus its architecture-matched "
            "random control at the SAME model size.",
            "Both E10 intervals contain zero and the seeds disagree in sign; "
            "they establish NEITHER equivalence NOR the absence of an effect.",
            "E10 does not reproduce E8B's directional negative 40k result.",
            "The difference in differences is not directional.",
        ],
        "placement": "MAIN_TEXT",
        "dissertation_section": "R7_ANSWER_SIDE_AND_CAPACITY",
    },
    {
        "table_id": "VE0-TAB-06",
        "working_title": "Measured efficiency, by node and timing class",
        "scientific_question": "What does each system cost per query, and "
                               "under exactly which measurement conditions?",
        "evidence_ids": [],
        "selectors": [
            {"evidence_class": "EFF"},
        ],
        "columns": ["system", "experiment family", "timing class", "node",
                    "precision", "batch size",
                    "warm median serial latency (ms)",
                    "comparable with end-to-end", "evidence status"],
        "precision": "latency to 4 decimal places as stored",
        "uncertainty_notation": "no interval; a warm median. The across-pass "
                                "spread is stated in the caption",
        "bolding_rule": "no bolding of the fastest system. The table spans "
                        "two nodes and three timing classes, and a bold "
                        "minimum would invite exactly the cross-node reading "
                        "that is prohibited",
        "highlight_rule": "rows are visually grouped by node, with a rule "
                          "between the groups",
        "footnotes": [
            "COMPUTATIONAL efficiency only. No power or energy measurement "
            "exists anywhere in this project.",
            "E7b (otter155) and E9 (otter159) are separate measurement "
            "groups and are never merged: the fusion bridge control failed "
            "its pre-registered 10 per cent tolerance at -18.69 per cent and "
            "no adjustment factor is applied.",
            "Only END_TO_END_SERIAL rows support an end-to-end latency claim. "
            "E7a's additive sums are superseded and appear here only as a "
            "recorded supersession.",
            "E8B's canonical R1 readout has no serial latency by design; R2 "
            "and R3 are never presented as R1's cost. E10's B4 and B4r have "
            "no timing evidence of any kind.",
        ],
        "placement": "MAIN_TEXT",
        "dissertation_section": "R8_COMPUTATIONAL_EFFICIENCY",
        "superseded_evidence_justification": "this table deliberately "
            "includes the E7a additive-sum row so a reader who remembers the "
            "older full_pipeline_ms and amortised_ms figures can find, in the "
            "same table, the record that they are superseded. The row is "
            "listed with evidence_status SUPERSEDED_FOR_END_TO_END_CLAIMS and "
            "carries no latency value, so it cannot be read as a measurement "
            "or ranked against one.",
    },
    {
        "table_id": "VE0-TAB-07",
        "working_title": "Claim, evidence and limitation ledger",
        "scientific_question": "What exactly does this dissertation claim, on "
                               "what evidence, and with what boundary?",
        "evidence_ids": [],
        "selectors": [],
        "consumes_claim_ledger": True,
        "columns": ["claim id", "claim as stated", "status", "evidence ids",
                    "mandatory caveat", "forbidden stronger version"],
        "precision": "NOT_APPLICABLE",
        "uncertainty_notation": "NOT_APPLICABLE",
        "bolding_rule": "no bolding",
        "highlight_rule": "NOT_SUPPORTED rows are kept in the table, not "
                          "deleted; they record prohibitions",
        "footnotes": [
            "This table is generated from the VE-0 claim ledger. The "
            "Abstract, Discussion and Conclusion may not state a stronger "
            "version of any row here.",
        ],
        "placement": "MAIN_TEXT",
        "dissertation_section": "R_LIMITATIONS",
    },
    # ---------------------------------------------------------------- appendix
    {
        "table_id": "VE0-TAB-A1",
        "working_title": "Full slice statistics",
        "scientific_question": "What is every per-type, per-structure and "
                               "per-step-bucket value, with its counts?",
        "evidence_ids": [],
        "selectors": [{"evidence_class": "SLC"}],
        "columns": ["slice id", "family", "system", "scale", "slice kind",
                    "slice", "quantity", "point estimate",
                    "95% image-clustered interval", "n questions", "n images",
                    "counts provenance"],
        "precision": "5 decimal places",
        "uncertainty_notation": "interval as [lower, upper]; the n questions "
                                "and n images columns are mandatory and are "
                                "never hidden",
        "bolding_rule": "no bolding",
        "highlight_rule": "none",
        "footnotes": [
            "Slices are unequal and dependent; clustering is on each slice's "
            "own represented development images.",
            "Intervals are uncorrected for multiplicity.",
            "Two metrics appear; the family column identifies which, and no "
            "value is compared across metrics.",
        ],
        "placement": "APPENDIX",
        "dissertation_section": "R_APPENDIX_SLICES",
        "mixed_metric_justification": "This is a complete reproducibility "
            "dump, not a comparison. Every row carries its family and hence "
            "its metric, and the table performs no arithmetic across rows.",
    },
    {
        "table_id": "VE0-TAB-A2",
        "working_title": "Full training-seed variability table",
        "scientific_question": "What is the per-seed value behind every "
                               "reported mean?",
        "evidence_ids": [],
        "selectors": [{"evidence_class": "SEED"}],
        "columns": ["row id", "family", "system", "scale", "seed set",
                    "per-seed accuracies", "mean",
                    "sd across training seeds (ddof=1)", "range",
                    "high dispersion flag"],
        "precision": "5 decimal places",
        "uncertainty_notation": "SD only. This table contains no confidence "
                                "interval and no column may be headed CI",
        "bolding_rule": "no bolding",
        "highlight_rule": "high-dispersion rows flagged in their own column",
        "footnotes": [
            "The standard deviation here is across independent TRAINING "
            "seeds (ddof=1). It is not a confidence interval and not "
            "evaluation-sampling uncertainty.",
            "Seed sets are never pooled across families.",
        ],
        "placement": "APPENDIX",
        "dissertation_section": "R_APPENDIX_STATISTICS",
        "mixed_metric_justification": "A reproducibility dump of seed "
            "dispersion. The family column carries the metric and no "
            "arithmetic crosses families.",
    },
    {
        "table_id": "VE0-TAB-A3",
        "working_title": "SigLIP against CLIP: full arm accuracies at both "
                         "scales",
        "scientific_question": "What is the encoder-swap direction at 250k, "
                               "where no clustered contrast exists?",
        "evidence_ids": [],
        "selectors": [
            {"evidence_class": "ARM", "experiment_family": ["e2"]},
            {"evidence_class": "SEED", "experiment_family": ["E2", "v2_07"]},
        ],
        "columns": ["encoder", "system", "scale", "seeds", "mean accuracy",
                    "sd across training seeds (ddof=1)",
                    "95% image-clustered interval", "interval available"],
        "precision": "5 decimal places",
        "uncertainty_notation": "interval where it exists; the words 'not "
                                "available' where it does not",
        "bolding_rule": "no bolding",
        "highlight_rule": "none",
        "footnotes": [
            "At train_250k no SigLIP-minus-CLIP clustered contrast can be "
            "formed, because the CLIP partner is the stopped v2_07 family. "
            "The 250k direction rests on stored per-seed means alone.",
            "Encoder, embedding width (512 against 768) and head input width "
            "move together; representation quality is not isolated and is not "
            "shown to be the sole or main bottleneck.",
        ],
        "placement": "APPENDIX",
        "dissertation_section": "R_APPENDIX_REPRESENTATION",
    },
    {
        "table_id": "VE0-TAB-A4",
        "working_title": "Complete contrast registry",
        "scientific_question": "What is every contrast this project computed, "
                               "including the ones whose intervals contain "
                               "zero?",
        "evidence_ids": [],
        "selectors": [{"evidence_class": "CON"}],
        "columns": ["contrast id", "family", "analysis role",
                    "comparison class", "effect",
                    "95% image-clustered interval", "excludes zero",
                    "n questions", "n images", "training seeds"],
        "precision": "5 decimal places",
        "uncertainty_notation": "interval as [lower, upper]",
        "bolding_rule": "no bolding",
        "highlight_rule": "none. Every contrast is listed whether or not its "
                          "interval excludes zero; twenty of them do not",
        "footnotes": [
            "Multiplicity is disclosed and deliberately uncorrected. No "
            "p-value is computed and no new hypothesis test was introduced.",
            "Analysis role and comparison class were fixed before any "
            "reconstructed number existed and no row was reclassified after "
            "its interval was computed.",
            "Two metrics appear; the family column identifies which.",
        ],
        "placement": "APPENDIX",
        "dissertation_section": "R_APPENDIX_STATISTICS",
        "mixed_metric_justification": "A complete registry, not a "
            "comparison. Effects are never differenced or ranked across "
            "metrics.",
    },
]
