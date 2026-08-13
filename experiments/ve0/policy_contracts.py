"""The frozen contracts: qualitative protocol and record schema, reporting
style, provenance, efficiency visualisation, and the results-section mapping.

Nothing here is populated with examples. VE-0 fixes the rules; VE-2 applies
them. The selection salt is frozen in ve0_common and is recorded in this
protocol, in the manifest and in the VE-0 report, so a reader can verify it
was fixed before any example was inspected.
"""

from __future__ import annotations

from . import ve0_common as vc

# --------------------------------------------------------------------------
# results-section mapping
# --------------------------------------------------------------------------

RESULTS_SECTIONS = [
    {"section_id": "R0_METHOD_OVERVIEW",
     "title": "Systems, data and evaluation protocol",
     "purpose": "Establishes the frozen encoders, the trainable components, "
                "the image-disjoint partition, the closed answer set and the "
                "two scorers, so that every later number has a defined "
                "meaning.",
     "placement": "MAIN_TEXT"},
    {"section_id": "R1_BASELINES_AND_FUSION",
     "title": "Baselines and low-data multimodal fusion",
     "purpose": "What the question-only, image-only, concatenation and fusion "
                "heads reach at train_40k.",
     "placement": "MAIN_TEXT"},
    {"section_id": "R2_CAPACITY_AND_FEATURES",
     "title": "Capacity and feature decomposition",
     "purpose": "How much of the fusion advantage survives at a matched "
                "trainable-parameter budget, and which interaction term "
                "carries it.",
     "placement": "MAIN_TEXT"},
    {"section_id": "R3_SCALING_BEHAVIOUR",
     "title": "Scaling behaviour",
     "purpose": "How accuracy and the fusion advantage move from 40k to 100k "
                "to 250k labelled questions, and how the frozen-SLM families "
                "respond to the same axis.",
     "placement": "MAIN_TEXT"},
    {"section_id": "R4_DEPTH_AND_RELIANCE",
     "title": "Reasoning depth and visual reliance",
     "purpose": "The step-bucket profile, the pooled four-or-more-step "
                "deficit, and how much each system loses under a wrong or "
                "removed image.",
     "placement": "MAIN_TEXT"},
    {"section_id": "R5_LATENT_REASONING_AND_REPRESENTATION",
     "title": "Latent-query reasoning and representation quality",
     "purpose": "The token-level reasoner against the global heads, and the "
                "frozen encoder swap.",
     "placement": "MAIN_TEXT"},
    {"section_id": "R6_QUESTION_SIDE_SLM",
     "title": "Question-side small-language-model pretraining",
     "purpose": "The A1 against A1r within-size contrast and the A0p system "
                "comparison.",
     "placement": "MAIN_TEXT"},
    {"section_id": "R7_ANSWER_SIDE_AND_CAPACITY",
     "title": "Answer-side pretraining and capacity sensitivity",
     "purpose": "The 135M and 360M answer-side readouts, their random "
                "controls, the scale effects and the difference in "
                "differences.",
     "placement": "MAIN_TEXT"},
    {"section_id": "R8_COMPUTATIONAL_EFFICIENCY",
     "title": "Computational efficiency",
     "purpose": "Measured end-to-end serial latency per node, the component "
                "and cached costs, and the compact-VLM context.",
     "placement": "MAIN_TEXT"},
    {"section_id": "R9_QUALITATIVE_AND_ERROR_ANALYSIS",
     "title": "Qualitative and error analysis",
     "purpose": "Deterministically selected development examples "
                "illustrating agreement and disagreement between systems.",
     "placement": "MAIN_TEXT"},
    {"section_id": "R10_HELD_OUT_EVALUATION_PLACEHOLDER",
     "title": "Held-out clean-test evaluation",
     "purpose": "PLACEHOLDER ONLY. Remains empty until F1 (model freeze) and "
                "F2 (blinded clean-test evaluation) are authorised and "
                "executed. No clean-test value, statistic, coverage figure or "
                "answer distribution may appear here or anywhere else before "
                "F2.",
     "placement": "MAIN_TEXT",
     "placeholder": True,
     "unblocked_by": "F2"},
    {"section_id": "R_DISCUSSION",
     "title": "Discussion",
     "purpose": "What the assembled evidence supports about efficient "
                "embedding-space visual question answering, and what it does "
                "not.",
     "placement": "MAIN_TEXT"},
    {"section_id": "R_LIMITATIONS",
     "title": "Limitations and evidence boundaries",
     "purpose": "The claim ledger rendered as a table, including the "
                "prohibitions.",
     "placement": "MAIN_TEXT"},
    {"section_id": "R_APPENDIX_STATISTICS",
     "title": "Appendix: full statistical tables",
     "purpose": "The complete contrast registry and the per-seed variability "
                "table.",
     "placement": "APPENDIX"},
    {"section_id": "R_APPENDIX_SLICES",
     "title": "Appendix: slice statistics",
     "purpose": "Every per-type, per-structure and per-step-bucket value "
                "with its question and image counts.",
     "placement": "APPENDIX"},
    {"section_id": "R_APPENDIX_ANSWER_SET",
     "title": "Appendix: answer-set size",
     "purpose": "The top-1000 vocabulary experiment, kept separate because "
                "its rows are not comparable with the top-100 view.",
     "placement": "APPENDIX"},
    {"section_id": "R_APPENDIX_REPRESENTATION",
     "title": "Appendix: encoder-swap detail",
     "purpose": "Full SigLIP and CLIP arm accuracies at both scales, "
                "including the 250k rows that carry no clustered interval.",
     "placement": "APPENDIX"},
]

SECTION_IDS = [s["section_id"] for s in RESULTS_SECTIONS]

# --------------------------------------------------------------------------
# qualitative protocol
# --------------------------------------------------------------------------

QUALITATIVE_SELECTION_RULE = {
    "method": "deterministic hash rank",
    "formula": "rank_key = SHA256(salt + '|' + category_id + '|' + "
               "question_id).hexdigest(); candidates sorted ascending by "
               "rank_key; the lowest ranks are taken in order",
    "salt": vc.QUALITATIVE_SELECTION_SALT,
    "salt_frozen_before_inspection": True,
    "salt_freeze_statement": "The salt was fixed in "
        "experiments/ve0/ve0_common.py at VE-0 time, before any candidate "
        "example was listed, inspected or rendered. Changing it changes every "
        "rank, so it is recorded here, in the VE-0 manifest and in the VE-0 "
        "report, and a later change would be visible as a manifest "
        "difference.",
    "tie_breaking": "rank_key collisions are broken by ascending question_id; "
                    "a collision has never been observed and would be "
                    "recorded if it occurred",
    "examples_per_category": 2,
    "substitution_policy": "NO SUBSTITUTION FOR APPEARANCE. An example may be "
                           "skipped ONLY for a recorded mechanical reason: "
                           "the image file is unreadable, or the row fails "
                           "the category predicate on re-check. Every skip is "
                           "recorded in the VE-2 record with its rank, its "
                           "question_id and its reason, and the next rank is "
                           "taken. A skip for any aesthetic, rhetorical or "
                           "outcome-related reason is forbidden.",
    "candidate_pool": "the in-vocabulary development rows only "
                      "(data/v2/dev.csv, 7,714 rows over 768 represented "
                      "images), except where a category explicitly declares "
                      "the raw development view. The clean test is never a "
                      "candidate pool.",
    "predicate_evaluation": "each category names the exact systems, scale, "
                            "seed and condition its predicate is evaluated "
                            "at, fixed here and not chosen after seeing the "
                            "candidates. Where a category needs a single "
                            "seed, seed 0 is used, declared in advance.",
    "clean_test": "NEVER. No qualitative example may come from the clean "
                  "test, and no VE-2 script may resolve the embargoed path.",
}

# Availability is a statement about canonical prediction evidence, not a
# preference. A category whose evidence does not exist is declared
# UNAVAILABLE and is not quietly dropped.
PREDICTION_EVIDENCE_SOURCES = {
    "closure_row_evidence": {
        "path_pattern": "results/closure/row_evidence/<family>/"
                        "<family>__<arm>__<scale>__seed<k>__<condition>.npz",
        "families": ["v2_02", "v2_03", "v2_04", "v2_06", "e2", "e3"],
        "fields": ["question_id", "image_id", "gold_answer", "gold_label",
                   "prediction_answer", "prediction_label", "correct"],
        "gives_prediction_text": True,
        "hash_bound_in": "results/closure/"
                         "C_reconstructed_evidence_manifest.json",
        "status": "AVAILABLE",
    },
    "e8a_prediction_tables": {
        "path_pattern": "results/experiments/e8a_question_encoder/"
                        "predictions_g21_v1/predictions_<arm>_<scale>_"
                        "seed<k>.csv.gz",
        "families": ["E8A"],
        "fields": ["questionId", "imageId", "gold_answer", "predicted_answer",
                   "normalized_gold_answer", "normalized_predicted_answer",
                   "raw_correct", "normalized_correct"],
        "gives_prediction_text": True,
        "hash_bound_in": "results/experiments/e8a_question_encoder/"
                         "ARTEFACT_MANIFEST.json",
        "status": "AVAILABLE",
    },
    "e10_per_row": {
        "path_pattern": "results/experiments/e10_capacity_360m_core/"
                        "e10_core_<arm>_<scale>_seed<k>_per_row.npz",
        "families": ["E10"],
        "fields": ["question_id", "image_id", "canonical_predictions",
                   "labels", "raw_correct", "normalized_correct",
                   "canonical_margins"],
        "gives_prediction_text": False,
        "prediction_text_note": "predictions are stored as answer-vocabulary "
                                "indices; the answer string is recovered "
                                "through data/v2/answer_vocab_v2.json, which "
                                "is a lookup, not a re-evaluation",
        "hash_bound_in": "results/experiments/e10_capacity_360m_core/"
                         "ARTEFACT_MANIFEST.json",
        "status": "AVAILABLE",
        "read_only_note": "E10 is CLOSED. VE-2 may READ these arrays and must "
                          "not write into the E10 tree, re-run an E10 cell or "
                          "touch any E10 source file.",
    },
    "e8b_per_row": {
        "path_pattern": "results/experiments/e8b_readout_generation/"
                        "e8b_core_<arm>_<scale>_seed<k>_final_eval_per_row"
                        ".npz",
        "families": ["E8B"],
        "fields": ["in_vocabulary", "labels", "r1_hits_normal",
                   "r1_hits_fixed_image", "r1_hits_fixed_question",
                   "r1_hits_shuffled_image"],
        "gives_prediction_text": False,
        "prediction_text_note": "correctness only. No predicted answer string "
                                "or index is stored, so an E8B row can be "
                                "shown as correct or incorrect but its "
                                "predicted answer cannot be displayed",
        "row_alignment_note": "these arrays carry no questionId and are "
                              "positionally aligned to the 10,004-row raw "
                              "development order in data/v2/dev_raw.csv. VE-2 "
                              "MUST verify that alignment explicitly before "
                              "using an E8B row, and must refuse the category "
                              "if it cannot",
        "hash_bound_in": "results/closure/"
                         "manifest_e8b_readout_generation.json",
        "status": "AVAILABLE_CORRECTNESS_ONLY",
    },
    "e9_tokens": {
        "path_pattern": "results/experiments/e9_compact_vlm/"
                        "e9_smolvlm_<size>_<mode>_<condition>.tokens.npz",
        "families": ["E9"],
        "fields": ["questionIds", "tokens", "lengths"],
        "gives_prediction_text": False,
        "prediction_text_note": "generated token identifiers only. Turning "
                                "them into an answer string requires the "
                                "SmolVLM tokenizer, and per-row correctness "
                                "is not stored as an array",
        "hash_bound_in": "results/experiments/e9_compact_vlm/"
                         "ARTEFACT_MANIFEST.json",
        "status": "REQUIRES_VE2_VERIFICATION",
        "verification_required": "VE-2 must first establish, from the frozen "
                                 "E9 artefacts alone, whether a per-row "
                                 "correctness vector can be recovered by "
                                 "DECODING stored tokens and comparing them "
                                 "with stored gold answers. Decoding stored "
                                 "bytes is a lookup. RE-SCORING, re-running "
                                 "generation, or loading SmolVLM weights is a "
                                 "NEW EVALUATION and is OUT OF SCOPE without "
                                 "fresh explicit user authorisation. If the "
                                 "verification fails, the category is "
                                 "recorded UNAVAILABLE and dropped, not "
                                 "worked around.",
    },
}

QUALITATIVE_CATEGORIES = [
    {
        "category_id": "QC01_all_correct",
        "description": "every compared global-embedding system answers "
                       "correctly",
        "predicate": "correct == 1 for question_only, concat and fusion",
        "systems": ["v2_02/question_only", "v2_02/concat", "v2_02/fusion"],
        "scale": "train_40k", "seed": 0, "condition": "normal",
        "evidence_source": "closure_row_evidence",
        "status": "AVAILABLE",
        "placement": "APPENDIX",
        "interpretation_bound": "an easy row for every system; illustrative "
                                "of the task, not of any system's advantage",
    },
    {
        "category_id": "QC02_all_wrong",
        "description": "every compared global-embedding system answers "
                       "incorrectly",
        "predicate": "correct == 0 for question_only, concat and fusion",
        "systems": ["v2_02/question_only", "v2_02/concat", "v2_02/fusion"],
        "scale": "train_40k", "seed": 0, "condition": "normal",
        "evidence_source": "closure_row_evidence",
        "status": "AVAILABLE",
        "placement": "MAIN_TEXT",
        "interpretation_bound": "a shared failure. It does not identify a "
                                "cause and must not be captioned as a "
                                "reasoning failure without independent "
                                "evidence",
    },
    {
        "category_id": "QC03_fusion_right_concat_wrong",
        "description": "fusion correct where concatenation is wrong",
        "predicate": "fusion correct == 1 and concat correct == 0",
        "systems": ["v2_02/fusion", "v2_02/concat"],
        "scale": "train_40k", "seed": 0, "condition": "normal",
        "evidence_source": "closure_row_evidence",
        "status": "AVAILABLE",
        "placement": "MAIN_TEXT",
        "interpretation_bound": "one row where the two systems disagree. It "
                                "illustrates the aggregate contrast; it is "
                                "not evidence for it, and the aggregate "
                                "contrast is capacity-confounded",
    },
    {
        "category_id": "QC04_representation_improvement",
        "description": "the SigLIP fusion head correct where the CLIP fusion "
                       "head is wrong",
        "predicate": "e2 fusion correct == 1 and v2_02 fusion correct == 0",
        "systems": ["e2/fusion", "v2_02/fusion"],
        "scale": "train_40k", "seed": 0, "condition": "normal",
        "evidence_source": "closure_row_evidence",
        "status": "AVAILABLE",
        "placement": "APPENDIX",
        "interpretation_bound": "representation quality is not isolated; the "
                                "encoder and two widths changed together, so "
                                "the example illustrates a system difference",
    },
    {
        "category_id": "QC05_question_side_pretraining",
        "description": "the pretrained question encoder A1 correct where its "
                       "random control A1r is wrong",
        "predicate": "A1 normalized_correct == 1 and A1r "
                     "normalized_correct == 0",
        "systems": ["E8A/A1", "E8A/A1r"],
        "scale": "train_40k", "seed": 0, "condition": "normal",
        "evidence_source": "e8a_prediction_tables",
        "status": "AVAILABLE",
        "placement": "MAIN_TEXT",
        "interpretation_bound": "illustrates the only clean within-size "
                                "question-side pretraining contrast; a single "
                                "row is not evidence of the effect",
    },
    {
        "category_id": "QC06_answer_side_disagreement",
        "description": "the pretrained 360M readout B4 and its random control "
                       "B4r disagree",
        "predicate": "B4 normalized_correct != B4r normalized_correct",
        "systems": ["E10/B4", "E10/B4r"],
        "scale": "train_40k", "seed": 0, "condition": "normal",
        "evidence_source": "e10_per_row",
        "status": "AVAILABLE",
        "placement": "APPENDIX",
        "interpretation_bound": "the aggregate contrast is not directional "
                                "and its interval contains zero. Both "
                                "directions of disagreement must be shown, "
                                "and the caption must not imply the "
                                "pretrained side is better",
        "balance_requirement": "equal numbers of B4-correct and B4r-correct "
                               "rows, taken by rank within each direction, so "
                               "the gallery cannot suggest a direction the "
                               "statistics do not support",
    },
    {
        "category_id": "QC07_visual_reliance",
        "description": "fusion correct under the normal image and wrong "
                       "under the shuffled image",
        "predicate": "v2_06 fusion normal correct == 1 and v2_06 fusion "
                     "shuffled correct == 0",
        "systems": ["v2_06/fusion"],
        "scale": "train_40k", "seed": 0,
        "condition": "normal and shuffled, paired on the same question",
        "evidence_source": "closure_row_evidence",
        "status": "AVAILABLE",
        "placement": "MAIN_TEXT",
        "interpretation_bound": "demonstrates reliance on the image for this "
                                "row. It is not proof of grounding, and the "
                                "wrong-image partner must be shown so a "
                                "reader can see what replaced it",
    },
    {
        "category_id": "QC08_short_reasoning",
        "description": "a two-step-or-fewer question",
        "predicate": "n_steps <= 2 in data/v2/metadata/dev_types.csv",
        "systems": ["v2_02/fusion"],
        "scale": "train_40k", "seed": 0, "condition": "normal",
        "evidence_source": "closure_row_evidence",
        "status": "AVAILABLE",
        "placement": "APPENDIX",
        "interpretation_bound": "step count is the GQA semantic program "
                                "length, a proxy for reasoning depth",
    },
    {
        "category_id": "QC09_deep_reasoning_failure",
        "description": "a question with four or more program steps that "
                       "fusion answers incorrectly",
        "predicate": "n_steps >= 4 and v2_02 fusion correct == 0",
        "systems": ["v2_02/fusion"],
        "scale": "train_40k", "seed": 0, "condition": "normal",
        "evidence_source": "closure_row_evidence",
        "status": "AVAILABLE",
        "placement": "MAIN_TEXT",
        "interpretation_bound": "illustrates where the pooled "
                                "four-or-more-step deficit lives. It does not "
                                "show the model failed BECAUSE the question "
                                "was deep; no causal attribution may be made "
                                "from an example",
    },
    {
        "category_id": "QC10_compact_vlm_disagreement",
        "description": "the compact VLM and the global-embedding head "
                       "disagree",
        "predicate": "SmolVLM normalised correctness != fusion normalised "
                     "correctness on the same raw development row",
        "systems": ["E9/SmolVLM-500M-Instruct", "v2_02/fusion"],
        "scale": "NOT_APPLICABLE for the VLM; train_40k for the head",
        "seed": 0, "condition": "normal",
        "evidence_source": "e9_tokens",
        "status": "CONDITIONAL_REQUIRES_VE2_VERIFICATION",
        "placement": "APPENDIX",
        "interpretation_bound": "contextual positioning only. The two systems "
                                "differ in training history, answer support "
                                "and output format, and the row denominators "
                                "differ, so the example illustrates a "
                                "difference and never a superiority",
        "drop_if_unverified": True,
    },
]

QUALITATIVE_RECORD_SCHEMA = {
    "description": "the exact record VE-2 populates, one per selected "
                   "example. Fields that do not apply carry an explicit "
                   "NOT_APPLICABLE or NOT_AVAILABLE.",
    "fields": [
        {"name": "qualitative_id",
         "type": "string",
         "note": "VE2-QUAL-<category_id>-<two-digit rank>"},
        {"name": "category_id", "type": "string",
         "note": "the declared selection category"},
        {"name": "selection_rank", "type": "integer",
         "note": "1-based rank in the deterministic ordering, after any "
                 "recorded skip"},
        {"name": "selection_rank_key", "type": "string",
         "note": "the SHA-256 hex digest that produced the rank, recorded so "
                 "the ordering can be recomputed"},
        {"name": "skipped_ranks", "type": "list",
         "note": "every rank skipped before this one, each with its "
                 "question_id and its recorded mechanical reason. Empty list "
                 "if none"},
        {"name": "question_id", "type": "string"},
        {"name": "image_id", "type": "string"},
        {"name": "image_path", "type": "string",
         "note": "path under the GQA image directory; the image bytes are "
                 "not copied into the record"},
        {"name": "image_sha256", "type": "string",
         "note": "hash of the image file actually rendered"},
        {"name": "question_text", "type": "string",
         "note": "read verbatim from data/v2/dev.csv"},
        {"name": "gold_answer", "type": "string"},
        {"name": "semantic_type", "type": "string",
         "note": "from data/v2/metadata/dev_types.csv"},
        {"name": "structural_type", "type": "string",
         "note": "from data/v2/metadata/dev_types.csv"},
        {"name": "n_program_steps", "type": "integer",
         "note": "GQA semantic program length; a proxy for reasoning depth, "
                 "never described as a measured reasoning chain"},
        {"name": "systems", "type": "list",
         "note": "one entry per compared system: system_id, arm, scale, seed, "
                 "condition, predicted_answer or NOT_AVAILABLE, correct, "
                 "evidence_path, evidence_sha256"},
        {"name": "selection_category_interpretation", "type": "string",
         "note": "the interpretation_bound copied verbatim from the category, "
                 "so a caption cannot drift from it"},
        {"name": "limitation", "type": "string",
         "note": "must state that a single example is illustrative and "
                 "supports no quantitative claim"},
        {"name": "source_hashes", "type": "object",
         "note": "sha256 of every artefact read to build this record"},
        {"name": "placement", "type": "string",
         "note": "MAIN_TEXT or APPENDIX, inherited from the category"},
    ],
    "prohibitions": [
        "No scene graph, program trace or reasoning chain may be shown that "
        "is not present in the canonical evidence. GQA program length is a "
        "count, not a chain, and must not be rendered as one.",
        "No attention map may be presented as a causal explanation of a "
        "prediction. The v3_02a attention diagnostics are diagnostics.",
        "No predicted answer may be invented, paraphrased or normalised "
        "beyond the scorer's own normalisation, and where no prediction "
        "string is stored the field reads NOT_AVAILABLE.",
        "No example may come from the clean test.",
        "No example may be replaced because a different one reads better.",
    ],
}

# --------------------------------------------------------------------------
# reporting style contract
# --------------------------------------------------------------------------

REPORTING_STYLE_CONTRACT = {
    "four_part_rule": "every main-text figure and table communicates four "
                      "things: the MESSAGE (what the reader should take "
                      "away), the NUMBER (the point estimate), the "
                      "UNCERTAINTY (which kind, named), and the CAVEAT (what "
                      "it does not show).",
    "uncertainty_rules": {
        "across_training_seed": "use the sample standard deviation across "
            "independent training seeds with ddof=1, and label it SD. Never "
            "label it CI, never draw it as a confidence band, and never pool "
            "seed sets across families.",
        "evaluation_sampling": "use the approved image-clustered 95 per cent "
            "interval where it exists, and name the cluster unit and the "
            "fixed trained seed set it conditions on.",
        "never_interchange": "SD and CI are different quantities and are "
            "never used interchangeably, never combined into one error bar, "
            "and never plotted on the same glyph.",
        "missing_interval": "a missing clustered interval is NEVER drawn as a "
            "zero-width bar, a bare point, or an empty cell that could be "
            "read as zero uncertainty. It is written 'not available' with the "
            "reason, and the affected rows are visually distinguished.",
        "v2_07_rule": "for v2_07-affected evidence show the training-seed "
            "variation only, carry the ACCEPTED DOCUMENTED LIMITATION in the "
            "caption, and never draw an interval.",
        "zero_containing_interval": "an interval that contains zero is "
            "reported as an absence of a detected effect. It is never "
            "described as equivalence, as no difference, or as proof of "
            "absence.",
        "multiplicity": "every interval is nominal 95 per cent and "
            "UNCORRECTED for multiplicity. This is stated wherever more than "
            "a handful of intervals appear together. No p-value is computed "
            "anywhere in this project.",
    },
    "forest_plot_row_requirements": [
        "effect orientation, written as 'A minus B' with the sign convention "
        "spelled out",
        "the reference condition",
        "the interval type (image-clustered evaluation-sampling)",
        "the cluster unit",
        "the training scale",
        "the comparison class: clean paired control, system-level, "
        "capacity-confounded or representation-confounded",
        "n questions and n unique images",
    ],
    "axis_and_label_rules": [
        "axes are labelled with the quantity and its unit; accuracy axes say "
        "accuracy, latency axes say milliseconds",
        "no truncated axis without an explicit stated reason in the caption",
        "a log axis is permitted for latency and is labelled as log",
        "no decorative three-dimensional charts, no drop shadows, no "
        "gradients carrying no information",
        "no unexplained abbreviation: every arm code (A0p, A1, A1r, B1, B2, "
        "B3, B4, B4r) is expanded at first use in the caption or in a legend",
        "sample counts are never hidden: n questions and n images appear in "
        "the caption or in the table",
        "colour is never the only channel carrying meaning",
    ],
    "negative_results_rule": "negative and null results are shown, not "
        "omitted. Twenty of the seventy contrasts have intervals containing "
        "zero and all of them are reported. A figure may not silently drop a "
        "row because its interval crosses zero.",
    "bolding_rule": "no value is bolded merely because it is the largest or "
        "because it is positive. Bolding is reserved for a row a caption "
        "explicitly directs the reader to, and is never used as an implicit "
        "significance marker.",
    "terminology": {
        "efficiency": "say computational efficiency or latency efficiency. "
                      "The words energy, power, watt, joule and carbon are "
                      "PROHIBITED as properties of any system measured here.",
        "reliance_vs_grounding": "say the system relies on the image. Do not "
                                 "say it grounds, understands or reasons "
                                 "about the image.",
        "absence_vs_equivalence": "say no reliable effect was detected. Do "
                                  "not say there is no effect, or that the "
                                  "systems are equivalent.",
        "development_vs_test": "every result is a development-set result "
                               "until F2. Say development set, not test set.",
        "step_count": "say program steps or program length, not reasoning "
                      "steps performed by the model.",
    },
    "presentation_notes": "every main-text figure carries a likely "
        "supervisor or viva question and a concise answer in its "
        "specification, so the deck and the viva preparation draw on the same "
        "prepared wording as the dissertation.",
    "consistency_rule": "the dissertation and the supervisor presentation "
        "draw from this single registry. A number that appears on a slide but "
        "not in the inventory is a defect.",
}

# --------------------------------------------------------------------------
# provenance contract
# --------------------------------------------------------------------------

PROVENANCE_CONTRACT = {
    "rule": "every VE-1 and VE-2 artefact must be reproducible from VE-0 "
            "alone, given the repository at the recorded HEAD.",
    "required_fields": [
        {"field": "artefact_id",
         "note": "stable identifier, for example VE0-FIG-02 or "
                 "VE2-QUAL-QC03-01"},
        {"field": "builder_script", "note": "repository-relative path"},
        {"field": "builder_sha256",
         "note": "hash of the builder source at build time"},
        {"field": "source_evidence_ids",
         "note": "every evidence_id consumed, resolved and explicit"},
        {"field": "source_paths",
         "note": "every file read, repository-relative"},
        {"field": "source_sha256",
         "note": "hash of every file read"},
        {"field": "repository_head", "note": "full commit SHA"},
        {"field": "build_command",
         "note": "the exact command that reproduces the artefact"},
        {"field": "deterministic_seed_or_salt",
         "note": "any seed or salt that affects content; for qualitative "
                 "selection this is the frozen VE-0 salt"},
        {"field": "output_sha256", "note": "hash of the produced file"},
        {"field": "content_sha256",
         "note": "hash of the scientific content with the provenance block "
                 "removed. This is the value a later rebuild must reproduce"},
        {"field": "generated_utc",
         "note": "OPTIONAL metadata. It must not enter content_sha256, and "
                 "VE-0 writes no timestamp into any artefact at all, so a "
                 "VE-0 rebuild is byte-identical, not merely "
                 "content-identical"},
    ],
    "determinism_rules": [
        "JSON is written with sorted keys and a fixed indent",
        "no wall-clock timestamp enters any scientific content",
        "no filesystem iteration order affects output; every collection is "
        "sorted by an explicit key",
        "a second build from unchanged inputs produces identical bytes",
    ],
    "traceability_statement": "a figure in the dissertation, a table in the "
        "dissertation and a slide in the supervisor deck must each resolve, "
        "through this contract, to the same evidence_id and hence to the same "
        "frozen artefact bytes.",
    "read_only_trees": {
        "results/experiments/e10_capacity_360m_core": "E10 is CLOSED. Read "
            "only. No VE-1 or VE-2 process may write here, re-run a cell, or "
            "modify any file inside experiments/e10_capacity_360m.",
        "results/closure": "the approved statistical and efficiency closure. "
            "Read only. VE-0, VE-1 and VE-2 consume it and never rewrite it.",
        "tests/run_all.py": "inside E10's frozen SOURCE_PATHS. Editing it "
            "moves the sealed live source digest and breaks all three E10 "
            "phase verifiers. VE-0 ships its own runner, tests/run_ve0.py, "
            "for the same reason the closure shipped tests/run_closure.py.",
        # The embargoed target file is deliberately NOT named here. The only
        # place its name appears in VE-0 source is the EMBARGOED_NAME
        # constant that the guard uses to refuse it, and write_json refuses
        # any payload carrying the token, so no VE-0 output can contain it.
        "the embargoed clean-test targets under data/v2/": "EMBARGOED. Never "
            "opened, never resolved, never hashed. VE-0 code names the file "
            "only inside the refusal guard, and every VE-0 writer refuses a "
            "payload containing that token, so no VE-0, VE-1 or VE-2 artefact "
            "can carry it even by accident.",
    },
}

# --------------------------------------------------------------------------
# efficiency visualisation contract
# --------------------------------------------------------------------------

EFFICIENCY_VISUALISATION_CONTRACT = {
    "hard_rule": "E7b on otter155 and E9 on otter159 are two measurement "
                 "groups and are NEVER merged into one frontier, one line, "
                 "one ranking or one scatter with a shared trend.",
    "why": "the fusion bridge control, the same system measured on both "
           "nodes, differs by -18.69 per cent against a pre-registered 10 per "
           "cent tolerance. A cross-node latency difference is a fact about "
           "the hardware, not about the systems, and no adjustment factor is "
           "applied.",
    "permitted_layouts": [
        "two side-by-side panels, one per node, each with its own axis range "
        "and its own node label",
        "one axis with two independently labelled groups, visually separated, "
        "with no connecting line, no shared frontier and no shared trend",
        "separate contextual markers on a single panel, provided each marker "
        "carries its node and no frontier is drawn across nodes",
    ],
    "forbidden_layouts": [
        "one Pareto frontier spanning both nodes",
        "any ranking, ordering or 'fastest system' statement that mixes nodes",
        "any interpolation, adjustment factor or normalisation between nodes",
        "plotting a cached or head-only figure on an end-to-end axis",
        "plotting E7a's additive full_pipeline_ms or amortised_ms as current "
        "end-to-end evidence",
    ],
    "mandatory_point_annotations": [
        "timing_kind", "node", "precision", "batch_size",
        "serial or cached", "comparable_with_end_to_end",
        "the accuracy denominator used for the vertical position",
    ],
    "timing_classes_plottable_on_an_end_to_end_axis": ["END_TO_END_SERIAL"],
    "e7a_status": "PARTLY_SUPERSEDED. Component measurements remain valid as "
                  "components and may appear in the appendix decomposition "
                  "figure with their class printed. The additive "
                  "gpu_encoder_plus_head_ms, full_pipeline_ms and "
                  "amortised_ms fields and every Pareto front derived from "
                  "them are superseded by E7b for any end-to-end claim.",
    "missing_evidence": {
        "E10": "B4 and B4r have NO timing evidence of any kind. This is "
               "recorded, never estimated, and no proxy from E8B may stand "
               "in for it.",
        "E8B_R1": "E8B's canonical R1 readout has no serial latency by "
                  "design. R2 and R3 are never presented as R1's cost.",
    },
    "energy": "PROHIBITED. No power or energy measurement exists anywhere in "
              "this project. Axis labels, captions and slide titles say "
              "latency or computational efficiency.",
}
