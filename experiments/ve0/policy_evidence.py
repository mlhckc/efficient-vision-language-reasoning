"""Authored evidence policy: story stage, reporting role, claim boundaries.

Everything in this module is a judgement, not a measurement. The numbers in
the inventory are read from frozen artefacts; the allowed claim, the forbidden
stronger claim and the mandatory limitation attached to each number are
authored here, once, so that no figure, table or chapter can quietly attach a
different boundary to the same evidence.

Every boundary encoded here traces to a decision already accepted in the
independent statistical and efficiency closure review
(.agent-bridge/tasks/closure-evidence-statistical-efficiency/40-claude-review-01.md,
verdict STATISTICS_EFFICIENCY_CLOSURE_PASS) or to the E10 closure. VE-0 does
not weaken any of them and does not invent a new one.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# the canonical scientific sequence
# --------------------------------------------------------------------------
# A stage exists because evidence answers a question the dissertation asks. An
# experiment is not given a stage merely because it was run: E3 and the V1
# prototype are deliberately outside the main sequence.

STORY_STAGES = {
    "S01_BASELINE_REPRESENTATION": "Baseline and representation: what a "
        "lightweight head over frozen global CLIP features achieves from the "
        "question alone, the image alone and both.",
    "S02_MULTIMODAL_FUSION": "Multimodal fusion: whether explicit interaction "
        "features beat plain concatenation in the low-data regime.",
    "S03_CAPACITY_MATCHING": "Capacity matching: how much of the fusion "
        "advantage survives at an equal trainable-parameter budget.",
    "S04_FEATURE_ABLATION": "Feature ablation: which interaction term carries "
        "the effect, and whether the two terms are redundant.",
    "S05_TRAINING_SCALE": "Training-data scale: how the advantages move from "
        "40k to 100k to 250k labelled questions.",
    "S06_TYPE_AND_DEPTH": "Question type and reasoning depth: where accuracy "
        "concentrates and where the multi-step deficit sits.",
    "S07_VISUAL_RELIANCE": "Visual reliance: how much a wrong or removed "
        "image costs each system.",
    "S08_LATENT_REASONER": "Token-level latent reasoning: whether a "
        "question-conditioned latent-query reasoner over token features beats "
        "the global-embedding heads.",
    "S09_REPRESENTATION_QUALITY": "Representation-quality sensitivity: what a "
        "stronger frozen dual encoder changes.",
    "S10_QUESTION_SIDE_SLM": "Question-side SLM pretraining: whether a frozen "
        "pretrained small language model beats its random control as the "
        "question encoder.",
    "S11_ANSWER_SIDE_SLM_135M": "Answer-side SLM pretraining at 135M: whether "
        "a frozen pretrained readout beats its random control.",
    "S12_ANSWER_SIDE_SLM_360M": "Answer-side capacity sensitivity at 360M: "
        "whether the answer-side conclusion changes with a larger frozen "
        "language model.",
    "S13_COMPACT_VLM_CONTEXT": "Compact-VLM context: where a frozen compact "
        "integrated VLM sits relative to these systems.",
    "S14_EFFICIENCY_TRADEOFF": "Efficiency and accuracy: measured end-to-end "
        "serial latency against accuracy, per measurement node.",
    "S00_OFF_SEQUENCE": "Kept as supporting or appendix evidence, "
        "deliberately outside the main argument.",
}

MAIN_SEQUENCE = [
    "S01_BASELINE_REPRESENTATION", "S02_MULTIMODAL_FUSION",
    "S03_CAPACITY_MATCHING", "S04_FEATURE_ABLATION", "S05_TRAINING_SCALE",
    "S06_TYPE_AND_DEPTH", "S07_VISUAL_RELIANCE", "S08_LATENT_REASONER",
    "S09_REPRESENTATION_QUALITY", "S10_QUESTION_SIDE_SLM",
    "S11_ANSWER_SIDE_SLM_135M", "S12_ANSWER_SIDE_SLM_360M",
    "S13_COMPACT_VLM_CONTEXT", "S14_EFFICIENCY_TRADEOFF",
]

SCIENTIFIC_ROLES = {
    "CORE": "carries a main-text claim; a reader who skipped it would miss "
            "part of the argument",
    "SUPPORTING": "strengthens or bounds a core claim without carrying one of "
                  "its own",
    "DIAGNOSTIC": "explains behaviour or checks an assumption; never a "
                  "headline number",
    "APPENDIX": "recorded in full for completeness and reproducibility, "
                "reported outside the main text",
    "SUPERSEDED": "replaced by a later artefact; never re-quoted",
    "NOT_FOR_REPORTING": "must not appear as a result in any form",
}

# --------------------------------------------------------------------------
# family -> stage and default reporting role
# --------------------------------------------------------------------------

FAMILY_STAGE = {
    "v2_02": "S02_MULTIMODAL_FUSION",
    "v2_03": "S03_CAPACITY_MATCHING",
    "v2_04": "S04_FEATURE_ABLATION",
    "v2_05c": "S06_TYPE_AND_DEPTH",
    "v2_06": "S07_VISUAL_RELIANCE",
    "v2_07": "S05_TRAINING_SCALE",
    "v2_00": "S01_BASELINE_REPRESENTATION",
    "v2_01": "S01_BASELINE_REPRESENTATION",
    "v3_01": "S08_LATENT_REASONER",
    "v3_02a": "S08_LATENT_REASONER",
    "v3_03": "S08_LATENT_REASONER",
    "e2": "S09_REPRESENTATION_QUALITY",
    "E2": "S09_REPRESENTATION_QUALITY",
    "e3": "S00_OFF_SEQUENCE",
    "E3": "S00_OFF_SEQUENCE",
    "E7a": "S14_EFFICIENCY_TRADEOFF",
    "E7b": "S14_EFFICIENCY_TRADEOFF",
    "E8A": "S10_QUESTION_SIDE_SLM",
    "E8B": "S11_ANSWER_SIDE_SLM_135M",
    "E9": "S13_COMPACT_VLM_CONTEXT",
    "E10": "S12_ANSWER_SIDE_SLM_360M",
}

FAMILY_ROLE = {
    "v2_02": "CORE",
    "v2_03": "CORE",
    "v2_04": "SUPPORTING",
    "v2_05c": "SUPPORTING",
    "v2_06": "CORE",
    "v2_07": "CORE",
    "v2_00": "SUPPORTING",
    "v2_01": "APPENDIX",
    "v3_01": "SUPPORTING",
    "v3_02a": "CORE",
    "v3_03": "CORE",
    "e2": "CORE",
    "E2": "CORE",
    "e3": "APPENDIX",
    "E3": "APPENDIX",
    "E7a": "DIAGNOSTIC",
    "E7b": "CORE",
    "E8A": "CORE",
    "E8B": "CORE",
    "E9": "SUPPORTING",
    "E10": "CORE",
}

# The metric identity of each family. Two systems may only share a figure axis
# or a table column block if they share a metric_id, or if the consuming
# specification declares and justifies the mix. This is the mechanism that
# stops the V2-era closed-vocabulary scorer being averaged with the pinned G21
# normalised scorer.

FAMILY_METRIC = {
    "v2_02": "v2_closed_vocab_top1_index_match",
    "v2_03": "v2_closed_vocab_top1_index_match",
    "v2_04": "v2_closed_vocab_top1_index_match",
    "v2_05c": "v2_closed_vocab_top1_index_match",
    "v2_06": "v2_closed_vocab_top1_index_match",
    "v2_07": "v2_closed_vocab_top1_index_match",
    "v2_00": "structural_count_or_share",
    "v2_01": "v2_closed_vocab_top1_index_match",
    "v3_01": "v2_closed_vocab_top1_index_match",
    "v3_02a": "v2_closed_vocab_top1_index_match",
    "v3_03": "v2_closed_vocab_top1_index_match",
    "e2": "v2_closed_vocab_top1_index_match",
    "E2": "v2_closed_vocab_top1_index_match",
    "e3": "v2_closed_vocab_top1_index_match",
    "E3": "v2_closed_vocab_top1_index_match",
    "E8A": "g21_pinned_normalised_exact_match",
    "E8B": "g21_pinned_normalised_exact_match",
    "E9": "g21_pinned_normalised_exact_match",
    "E10": "g21_pinned_normalised_exact_match",
    "E7a": "latency_and_memory_components",
    "E7b": "warm_median_serial_latency_ms",
}

METRIC_DEFINITIONS = {
    "v2_closed_vocab_top1_index_match": "argmax over the closed answer set "
        "compared with the gold label index, on the in-vocabulary "
        "development rows. The V2-era scorer.",
    "g21_pinned_normalised_exact_match": "the pinned G21 VQA-style normalised "
        "exact-match scorer used by E8A, E8B, E9 and E10.",
    "structural_count_or_share": "a counted quantity or a share of counts "
        "read from a frozen protocol artefact; not an estimate and carrying "
        "no sampling uncertainty.",
    "warm_median_serial_latency_ms": "warm median wall-clock milliseconds for "
        "one raw image plus raw question query at batch 1, measured serially "
        "on one named node.",
    "latency_and_memory_components": "isolated component latencies and peak "
        "memory; valid as components only.",
}

MIXED_METRIC_RULE = (
    "No figure axis and no table column block may place two metric_ids on a "
    "shared quantitative scale unless the specification sets "
    "mixed_metric_justification and carries the non-comparability footnote. "
    "The V2-era closed-vocabulary scorer and the pinned G21 normalised scorer "
    "are never averaged, differenced or ranked against each other.")

# Checkpoint-selection identity. This is a real cross-family comparability
# fact, not a defect: the V2/V3 protocol selects the best development epoch,
# which is legitimate because every development decision is allowed to use
# data/v2/dev.csv, while E8B and E10 froze the fixed-22 rule with no early
# stopping. A best-on-development number and a fixed-epoch number are not
# selected the same way, so any figure or table that puts them on one axis
# must carry the caveat below.

# The comparison is on the CLASS, never on the prose. An earlier revision
# compared the human descriptions, so two families following the identical
# best-on-development rule but described at different lengths looked like a
# protocol difference. A controlled class fixes that and keeps the detail.

CHECKPOINT_SELECTION_CLASSES = {
    "BEST_ON_DEVELOPMENT": "the checkpoint is the epoch with the best "
        "development accuracy. Legitimate under this protocol, because every "
        "development decision is allowed to use data/v2/dev.csv, but it means "
        "the reported development accuracy is selected on the same split it "
        "is reported on",
    "FIXED_EPOCH_22": "the checkpoint is epoch 22 by the frozen rule, with no "
        "early stopping and no best-of selection. E8B and E10 only",
    "NO_TRAINING": "nothing was trained; evaluation or measurement only",
}

FAMILY_CHECKPOINT_SELECTION = {
    "v2_02": {"class": "BEST_ON_DEVELOPMENT",
              "detail": "best development epoch (early stopping on "
                        "development accuracy, best-on-development "
                        "checkpointing)"},
    "v2_03": {"class": "BEST_ON_DEVELOPMENT",
              "detail": "best development epoch"},
    "v2_04": {"class": "BEST_ON_DEVELOPMENT",
              "detail": "best development epoch"},
    "v2_05c": {"class": "BEST_ON_DEVELOPMENT",
               "detail": "best development epoch"},
    "v2_06": {"class": "BEST_ON_DEVELOPMENT",
              "detail": "best development epoch"},
    "v2_07": {"class": "BEST_ON_DEVELOPMENT",
              "detail": "best development epoch"},
    "v3_01": {"class": "BEST_ON_DEVELOPMENT",
              "detail": "best development epoch, patience 10"},
    "v3_02a": {"class": "BEST_ON_DEVELOPMENT",
               "detail": "best development epoch, patience 10"},
    "v3_03": {"class": "BEST_ON_DEVELOPMENT",
              "detail": "best development epoch, patience 10"},
    "e2": {"class": "BEST_ON_DEVELOPMENT",
           "detail": "best development epoch"},
    "E2": {"class": "BEST_ON_DEVELOPMENT",
           "detail": "best development epoch"},
    "e3": {"class": "BEST_ON_DEVELOPMENT",
           "detail": "best development epoch"},
    "E3": {"class": "BEST_ON_DEVELOPMENT",
           "detail": "best development epoch"},
    "E8A": {"class": "BEST_ON_DEVELOPMENT",
            "detail": "best development epoch, patience 10, at most 100 "
                      "epochs"},
    "E8B": {"class": "FIXED_EPOCH_22",
            "detail": "fixed epoch 22, no early stopping; epoch 22 is the "
                      "sole primary"},
    "E10": {"class": "FIXED_EPOCH_22",
            "detail": "fixed epoch 22, no early stopping; epoch 22 is the "
                      "sole primary"},
    "E9": {"class": "NO_TRAINING", "detail": "no training; evaluation only"},
    "E7a": {"class": "NO_TRAINING",
            "detail": "no training; measurement only"},
    "E7b": {"class": "NO_TRAINING",
            "detail": "no training; measurement only"},
    "v2_00": {"class": "NO_TRAINING",
              "detail": "no training; a protocol artefact"},
    "v2_01": {"class": "NO_TRAINING",
              "detail": "no training; a zero-shot protocol check"},
}

CROSS_FAMILY_SELECTION_CAVEAT = (
    "Checkpoint selection differs across families. V2, V3, E2, E3 and E8A "
    "select the best development epoch; E8B and E10 use the frozen fixed-22 "
    "rule with no early stopping. A best-on-development number and a "
    "fixed-epoch number are not selected the same way, so a side-by-side "
    "reading is a system-level comparison, not a matched one.")

# The answer-set identity. Top-100 and top-1000 rows are never paired: they
# use different development rows, a different class count and different
# training rows.
FAMILY_ANSWER_SET = {
    "e3": 1000,
    "E3": 1000,
}
DEFAULT_ANSWER_SET = 100

# --------------------------------------------------------------------------
# claim boundaries by comparison class
# --------------------------------------------------------------------------
# comparison_class is carried through from the closure's comparison_status,
# which was fixed in the planning audit before any reconstructed number
# existed. VE-0 adds the allowed and forbidden wording that must travel with
# each class.

COMPARISON_CLASSES = {
    "CLEAN_PAIRED_CONTROL": {
        "definition": "the two sides differ in the one factor under study and "
                      "are evaluated on identical development rows",
        "allowed_claim": "a causal statement about the single factor that "
                         "differs, bounded to this task, this split and these "
                         "training seeds",
        "forbidden_stronger_claim": "any generalisation beyond the tested "
                                    "setting, and any equivalence claim drawn "
                                    "from an interval that contains zero",
    },
    "CAPACITY_CONFOUNDED": {
        "definition": "the two sides differ in trainable capacity as well as "
                      "in the factor under study",
        "allowed_claim": "a descriptive statement that the configuration is "
                         "better, with the capacity difference named",
        "forbidden_stronger_claim": "attributing the difference to the "
                                    "architecture or the feature alone",
    },
    "REPRESENTATION_CONFOUNDED": {
        "definition": "the two sides differ in representation space, "
                      "embedding width or interface as well as in the factor "
                      "under study",
        "allowed_claim": "a system-level statement that one configuration is "
                         "better under the tested conditions",
        "forbidden_stronger_claim": "isolating representation quality, "
                                    "semantics or alignment as the cause, or "
                                    "calling it the sole or main bottleneck",
    },
    "SYSTEM_LEVEL_COMPARISON": {
        "definition": "whole systems differ in several respects at once",
        "allowed_claim": "a statement about the systems as built",
        "forbidden_stronger_claim": "attributing the outcome to any single "
                                    "component, such as token-level access, "
                                    "in isolation",
    },
    "DESCRIPTIVE_ONLY": {
        "definition": "a described quantity, not a contrast between systems",
        "allowed_claim": "a description of the measured profile",
        "forbidden_stronger_claim": "any causal reading, and any ranking "
                                    "against a quantity measured differently",
    },
}

ANALYSIS_ROLES = {
    "PRIMARY": "pre-registered or planned before the number existed; carries "
               "a headline claim",
    "SECONDARY": "planned, reported in full, but not a headline claim",
    "EXPLORATORY": "not confirmatory; never selects a recipe and never "
                   "supports a superiority claim",
    "DIAGNOSTIC": "checks an assumption or explains behaviour",
}

# --------------------------------------------------------------------------
# mandatory limitations that attach to particular evidence, by family
# --------------------------------------------------------------------------

UNIVERSAL_LIMITATION = (
    "Development-set result on one GQA subset. The clean test is embargoed "
    "and unread, so no held-out generalisation is claimed.")

FAMILY_LIMITATION = {
    "v2_07": (
        "v2_07 ACCEPTED DOCUMENTED LIMITATION. No image-clustered evaluation "
        "interval is available for this family: row-level reconstruction was "
        "attempted and stopped by a point-estimate reproduction mismatch, and "
        "one of forty cells differs from reconstruction by one numerically "
        "tied row (v2_07::question_only::train_250k::seed2::normal, stored "
        "0.49767 against recomputed 0.49754, a single float32 argmax tie on "
        "row index 6213 of 7,714). Report per-seed means and the "
        "across-training-seed sample sd only. No post-hoc partial-family "
        "reconstruction is authorised."),
    "e2": (
        "Representation quality is not isolated: encoder, embedding width "
        "(512 against 768) and head input width all move together. The "
        "training streams are not paired and the seed label does not pair two "
        "runs. At train_250k no clustered contrast exists, because the CLIP "
        "partner is the stopped v2_07 family."),
    "E2": (
        "Representation quality is not isolated: encoder, embedding width "
        "(512 against 768) and head input width all move together. At "
        "train_250k the direction rests on stored per-seed means with no "
        "clustered interval."),
    "e3": (
        "Top-100 and top-1000 results are never paired: different "
        "development rows (7,714 against 9,823), a different class count and "
        "different training rows. Class competition and the smaller "
        "head-answer share of a fixed budget are confounded."),
    "E3": (
        "Top-100 and top-1000 results are never paired: different "
        "development rows (7,714 against 9,823), a different class count and "
        "different training rows."),
    "v2_06": (
        "A drop under a wrong or removed image demonstrates reliance on the "
        "image. It is NOT proof of robust visual reasoning, of grounding or "
        "of compositional understanding. The question-only drop is exactly "
        "zero by construction and is a control, not a result."),
    "v3_01": (
        "A system-level comparison. Input granularity, architecture and "
        "parameter count change together, so token-level access is not "
        "isolated as the cause. These 40k values carry NO image-clustered "
        "evaluation interval: the closure reconstructed the global-head "
        "families only. Report the across-training-seed mean and sd over "
        "three seeds, and the same-seed gap, with no interval. The original "
        "v3_01 pooled step statistics are superseded by v3_02a."),
    "v3_02a": (
        "A system-level comparison. Input granularity, architecture and "
        "parameter count change together, so token-level access is not "
        "isolated as the cause."),
    "v3_03": (
        "A system-level comparison. Input granularity, architecture and "
        "parameter count change together, so token-level access is not "
        "isolated as the cause."),
    "E8A": (
        "E8A's projected interface is a learned common width, not a naturally "
        "shared pretrained embedding space. A1 minus A1r is the only clean "
        "within-size question-side pretraining contrast; A1 minus A0p also "
        "changes the representation space and the interface."),
    "E8B": (
        "An interval containing zero is an absence of a detected effect and "
        "establishes neither equivalence nor the absence of an effect. B3 at "
        "train_40k has an across-training-seed sd of 0.02795 and must always "
        "be shown with its seed spread."),
    "E10": (
        "Both pretraining-effect intervals contain zero and the seeds "
        "disagree in sign; they establish neither equivalence nor the absence "
        "of an effect. The 135M-to-360M step is whole-system capacity "
        "sensitivity, not an isolated causal language-model-size effect: "
        "trainable capacity moves too, 21,343,808 to 21,540,800 parameters, "
        "because the projection width follows the hidden size."),
    "E9": (
        "Contextual positioning only, never a leaderboard or fair-protocol "
        "superiority claim. Training history, multimodal pretraining, answer "
        "support and output format all differ, and SmolVLM-256M's text "
        "backbone is the Instruct checkpoint where E8A and E8B use base. No "
        "comparison is made against published official GQA scores in either "
        "direction."),
    "E7a": (
        "Component measurements only. The additive gpu_encoder_plus_head_ms, "
        "full_pipeline_ms and amortised_ms fields and their Pareto fronts are "
        "SUPERSEDED for any end-to-end claim by the measured serial values of "
        "E7b."),
    "E7b": (
        "Latency efficiency only, measured on one node. No power or energy "
        "measurement exists anywhere in this project. E7b (otter155) and E9 "
        "(otter159) are separate frontiers and are never merged: the fusion "
        "bridge control failed its pre-registered 10 per cent tolerance at "
        "-18.69 per cent and no adjustment factor is applied."),
}

# Contrast-level overrides, applied by exact contrast id where the family
# limitation is not specific enough.
CONTRAST_LIMITATION = {
    "v2_02/fusion_minus_concat/train_40k": (
        "Capacity-confounded: fusion carries 1,100,388 trainable parameters "
        "against concat's 576,100. Only the v2_03 matched pairs isolate the "
        "feature effect, and fusion minus concat_wide includes zero."),
    "v2_06/question_only/drop_shuffled/train_40k": (
        "Exactly zero by construction. This is a control that proves the "
        "intervention touched only the image path; it is not a result."),
    "v2_06/question_only/drop_zeroed/train_40k": (
        "Exactly zero by construction. This is a control that proves the "
        "intervention touched only the image path; it is not a result."),
    "E8A/A1_minus_A0p/train_40k": (
        "Not a pure causal semantics or alignment effect: the two sides "
        "differ in representation space, in interface, and in whether the "
        "text representation shares a pretrained space with the image "
        "representation."),
    "E8A/A1_minus_A0p/train_250k": (
        "Not a pure causal semantics or alignment effect: the two sides "
        "differ in representation space, in interface, and in whether the "
        "text representation shares a pretrained space with the image "
        "representation."),
    "E10/difference_in_differences": (
        "The difference in differences is not directional; its interval "
        "contains zero. It gives no reliable evidence that any answer-side "
        "pretraining contribution grows with training scale."),
}

# --------------------------------------------------------------------------
# descriptive scalars, each resolved from a canonical artefact by pointer
# --------------------------------------------------------------------------
# A scalar that cannot be resolved from a frozen artefact is not written. No
# value in this table is typed from memory; the builder reads each one and
# fails loudly if the pointer does not resolve.
#
# Each entry carries its OWN scientific identity. An earlier revision let the
# whole class inherit one structural metric identity, which mislabelled three
# real development accuracies as counted quantities and silently defeated the
# mixed-metric guard on the figure that consumes them. Identity is therefore
# per scalar, and the builder refuses an entry that does not declare it.

# Where a count is not stated by the scalar's own artefact but is fixed by the
# split the evaluation ran on, the binding is recorded rather than assumed.
V3_DEV_VIEW_COUNTS_PROVENANCE = (
    "n_questions and n_unique_images are NOT read from the v3_01 artefact, "
    "which does not record them. They are the V3 development view, bound by "
    "the closure's own v3_02a and v3_03 rows on the same split under the "
    "identical frozen v3_01 recipe and the same seed set: 7,714 in-vocabulary "
    "rows over 768 represented development images, with the v3_02a reasoner "
    "step buckets summing to exactly 7,714. The same-seed reasoner-minus-"
    "fusion gap additionally requires the same evaluation rows as v2_02's "
    "fusion arm, which is canonically 7,714 rows over 768 images. WARNING for "
    "any later reader: the v3_01 artefact carries a top-level n_val field of "
    "8000, which is a stale V1-era configuration value and does NOT describe "
    "this evaluation.")

V3_01_NO_INTERVAL_REASON = (
    "NO_INTERVAL_AVAILABLE: this is a development accuracy under the V2-era "
    "closed-vocabulary scorer, not a counted quantity. No image-clustered "
    "evaluation-sampling interval exists for it because the post-E10 closure "
    "reconstructed the global-head families only, so the V3 families have no "
    "reconstructed row-level evidence and no clustered interval was ever "
    "computed for this scalar. The accompanying seed mean and seed sd "
    "summarise variability across three independent TRAINING runs; they are "
    "NOT an evaluation-sampling interval. No interval is invented or "
    "reconstructed here.")

STRUCTURAL_NO_INTERVAL_REASON = (
    "NO_INTERVAL: a counted quantity or a share of counts read from a frozen "
    "protocol artefact, not an estimate, so it carries no sampling "
    "uncertainty.")

_PROTOCOL_SUMMARY = "artifacts/v2_00_protocol/protocol_build_summary.json"

DESCRIPTIVE_SCALARS = [
    {
        "key": "dev.coverage_top100",
        "artefact": _PROTOCOL_SUMMARY,
        "pointer": "/dev/v2_coverage",
        "quantity": "share of raw development questions whose answer is in "
                    "the top-100 closed answer set",
        "quantity_kind": "STRUCTURAL_SHARE",
        "family": "v2_00",
        "story_stage": "S01_BASELINE_REPRESENTATION",
        "scientific_role": "SUPPORTING",
        "metric_id": "structural_count_or_share",
        "comparison_class": "DESCRIPTIVE_ONLY",
        "analysis_role": "DIAGNOSTIC",
        "split": None,
        "training_scale": None,
        "seed_set": None,
        "n_questions": None,
        "n_unique_images": None,
        "counts_provenance": "NOT_APPLICABLE: a protocol-level share",
        "ci_type": STRUCTURAL_NO_INTERVAL_REASON,
    },
    {
        "key": "dev.coverage_top1000",
        "artefact": "results/experiments/e3_vocab1000/results.json",
        "pointer": "/e3_vocab1000/build_summary/dev/coverage_top1000",
        "quantity": "share of raw development questions whose answer is in "
                    "the top-1000 closed answer set",
        "quantity_kind": "STRUCTURAL_SHARE",
        "family": "E3",
        "story_stage": "S00_OFF_SEQUENCE",
        "scientific_role": "APPENDIX",
        "metric_id": "structural_count_or_share",
        "comparison_class": "DESCRIPTIVE_ONLY",
        "analysis_role": "DIAGNOSTIC",
        "split": None,
        "training_scale": None,
        "seed_set": None,
        "n_questions": None,
        "n_unique_images": None,
        "counts_provenance": "NOT_APPLICABLE: a protocol-level share",
        "ci_type": STRUCTURAL_NO_INTERVAL_REASON,
    },
    {
        "key": "dev.n_questions_in_vocabulary",
        "artefact": _PROTOCOL_SUMMARY,
        "pointer": "/manifests/dev/n_questions",
        "quantity": "in-vocabulary development questions in the top-100 view",
        "quantity_kind": "STRUCTURAL_COUNT",
        "family": "v2_00",
        "story_stage": "S01_BASELINE_REPRESENTATION",
        "scientific_role": "SUPPORTING",
        "metric_id": "structural_count_or_share",
        "comparison_class": "DESCRIPTIVE_ONLY",
        "analysis_role": "DIAGNOSTIC",
        "split": "v2_dev",
        "training_scale": None,
        "seed_set": None,
        "n_questions": None,
        "n_unique_images": None,
        "counts_provenance": "NOT_APPLICABLE: this scalar IS a count",
        "ci_type": STRUCTURAL_NO_INTERVAL_REASON,
    },
    {
        "key": "dev.n_raw_questions",
        "artefact": _PROTOCOL_SUMMARY,
        "pointer": "/dev/n_raw_questions",
        "quantity": "raw development questions before the vocabulary gate",
        "quantity_kind": "STRUCTURAL_COUNT",
        "family": "v2_00",
        "story_stage": "S01_BASELINE_REPRESENTATION",
        "scientific_role": "SUPPORTING",
        "metric_id": "structural_count_or_share",
        "comparison_class": "DESCRIPTIVE_ONLY",
        "analysis_role": "DIAGNOSTIC",
        "split": "v2_dev_raw",
        "training_scale": None,
        "seed_set": None,
        "n_questions": None,
        "n_unique_images": None,
        "counts_provenance": "NOT_APPLICABLE: this scalar IS a count",
        "ci_type": STRUCTURAL_NO_INTERVAL_REASON,
    },
    {
        "key": "dev.n_images",
        "artefact": _PROTOCOL_SUMMARY,
        "pointer": "/dev/n_images",
        "quantity": "development images in the image-disjoint partition",
        "quantity_kind": "STRUCTURAL_COUNT",
        "family": "v2_00",
        "story_stage": "S01_BASELINE_REPRESENTATION",
        "scientific_role": "SUPPORTING",
        "metric_id": "structural_count_or_share",
        "comparison_class": "DESCRIPTIVE_ONLY",
        "analysis_role": "DIAGNOSTIC",
        "split": "v2_dev_raw",
        "training_scale": None,
        "seed_set": None,
        "n_questions": None,
        "n_unique_images": None,
        "counts_provenance": "NOT_APPLICABLE: this scalar IS a count",
        "ci_type": STRUCTURAL_NO_INTERVAL_REASON,
    },
    {
        "key": "train_40k.n_questions",
        "artefact": _PROTOCOL_SUMMARY,
        "pointer": "/manifests/train_40k/n_questions",
        "quantity": "labelled training questions at the 40k scale",
        "quantity_kind": "STRUCTURAL_COUNT",
        "family": "v2_00",
        "story_stage": "S05_TRAINING_SCALE",
        "scientific_role": "SUPPORTING",
        "metric_id": "structural_count_or_share",
        "comparison_class": "DESCRIPTIVE_ONLY",
        "analysis_role": "DIAGNOSTIC",
        "split": None,
        "training_scale": "train_40k",
        "seed_set": None,
        "n_questions": None,
        "n_unique_images": None,
        "counts_provenance": "NOT_APPLICABLE: this scalar IS a count",
        "ci_type": STRUCTURAL_NO_INTERVAL_REASON,
    },
    {
        "key": "train_100k.n_questions",
        "artefact": _PROTOCOL_SUMMARY,
        "pointer": "/manifests/train_100k/n_questions",
        "quantity": "labelled training questions at the 100k scale",
        "quantity_kind": "STRUCTURAL_COUNT",
        "family": "v2_00",
        "story_stage": "S05_TRAINING_SCALE",
        "scientific_role": "SUPPORTING",
        "metric_id": "structural_count_or_share",
        "comparison_class": "DESCRIPTIVE_ONLY",
        "analysis_role": "DIAGNOSTIC",
        "split": None,
        "training_scale": "train_100k",
        "seed_set": None,
        "n_questions": None,
        "n_unique_images": None,
        "counts_provenance": "NOT_APPLICABLE: this scalar IS a count",
        "ci_type": STRUCTURAL_NO_INTERVAL_REASON,
    },
    {
        "key": "train_250k.n_questions",
        "artefact": _PROTOCOL_SUMMARY,
        "pointer": "/manifests/train_250k/n_questions",
        "quantity": "labelled training questions at the 250k scale",
        "quantity_kind": "STRUCTURAL_COUNT",
        "family": "v2_00",
        "story_stage": "S05_TRAINING_SCALE",
        "scientific_role": "SUPPORTING",
        "metric_id": "structural_count_or_share",
        "comparison_class": "DESCRIPTIVE_ONLY",
        "analysis_role": "DIAGNOSTIC",
        "split": None,
        "training_scale": "train_250k",
        "seed_set": None,
        "n_questions": None,
        "n_unique_images": None,
        "counts_provenance": "NOT_APPLICABLE: this scalar IS a count",
        "ci_type": STRUCTURAL_NO_INTERVAL_REASON,
    },
    # The 40k latent-query reasoner. The closure registry carries no overall
    # 40k reasoner accuracy, because the closure reconstructed the global-head
    # families only. These three values are read from the hash-pinned v3_01
    # artefact, whose accuracies the supersession map lists as still valid,
    # and they are the only evidence supporting the 40k half of the reasoner
    # claim. They are development ACCURACIES under the V2-era closed-vocabulary
    # scorer, not counted quantities, and they carry no clustered interval.
    {
        "key": "v3_01.reasoner.train_40k.seed_mean",
        "artefact": "results/experiments/v3_01_reasoner/results.json",
        "pointer": "/v3_01_reasoner/final/mean",
        "quantity": "mean development accuracy of the 40k latent-query "
                    "reasoner across three independent training seeds",
        "quantity_kind": "DEVELOPMENT_ACCURACY",
        "family": "v3_01",
        "story_stage": "S08_LATENT_REASONER",
        "scientific_role": "CORE",
        "metric_id": "v2_closed_vocab_top1_index_match",
        "comparison_class": "DESCRIPTIVE_ONLY",
        "comparison_class_reason": "a single system's accuracy summary, not a "
                                   "comparison between systems. The "
                                   "system-level comparison is carried by the "
                                   "reasoner-minus-fusion gap row.",
        "analysis_role": "SECONDARY",
        "split": "v2_dev",
        "training_scale": "train_40k",
        "seed_set": [0, 1, 2],
        "n_questions": 7714,
        "n_unique_images": 768,
        "counts_provenance": V3_DEV_VIEW_COUNTS_PROVENANCE,
        "ci_type": V3_01_NO_INTERVAL_REASON,
    },
    {
        "key": "v3_01.reasoner.train_40k.seed_sd",
        "artefact": "results/experiments/v3_01_reasoner/results.json",
        "pointer": "/v3_01_reasoner/final/std",
        "quantity": "standard deviation of the 40k latent-query reasoner's "
                    "development accuracy across three independent TRAINING "
                    "seeds. This is training-run variability, NOT a confidence "
                    "interval and NOT evaluation-sampling uncertainty",
        "quantity_kind": "TRAINING_SEED_DISPERSION",
        "family": "v3_01",
        "story_stage": "S08_LATENT_REASONER",
        "scientific_role": "CORE",
        "metric_id": "v2_closed_vocab_top1_index_match",
        "comparison_class": "DESCRIPTIVE_ONLY",
        "comparison_class_reason": "a dispersion summary for one system, not a "
                                   "comparison between systems.",
        "analysis_role": "SECONDARY",
        "split": "v2_dev",
        "training_scale": "train_40k",
        "seed_set": [0, 1, 2],
        "n_questions": 7714,
        "n_unique_images": 768,
        "counts_provenance": V3_DEV_VIEW_COUNTS_PROVENANCE,
        "ci_type": V3_01_NO_INTERVAL_REASON,
    },
    {
        "key": "v3_01.reasoner_minus_fusion.train_40k.seed_mean",
        "artefact": "results/experiments/v3_01_reasoner/results.json",
        "pointer": "/v3_01_reasoner/gaps_same_seeds/reasoner_minus_fusion/"
                   "mean",
        "quantity": "same-seed mean development accuracy gap of the 40k "
                    "latent-query reasoner over the global fusion head",
        "quantity_kind": "DEVELOPMENT_ACCURACY_CONTRAST",
        "family": "v3_01",
        "story_stage": "S08_LATENT_REASONER",
        "scientific_role": "CORE",
        "metric_id": "v2_closed_vocab_top1_index_match",
        "comparison_class": "SYSTEM_LEVEL_COMPARISON",
        "comparison_class_reason": "this row IS a contrast between two whole "
                                   "systems that differ in input granularity, "
                                   "architecture and parameter count at once. "
                                   "DESCRIPTIVE_ONLY would understate what the "
                                   "quantity is and would drop the "
                                   "no-single-component boundary that must "
                                   "travel with it.",
        "analysis_role": "SECONDARY",
        "split": "v2_dev",
        "training_scale": "train_40k",
        "seed_set": [0, 1, 2],
        "n_questions": 7714,
        "n_unique_images": 768,
        "counts_provenance": V3_DEV_VIEW_COUNTS_PROVENANCE,
        "ci_type": V3_01_NO_INTERVAL_REASON,
    },
]

# --------------------------------------------------------------------------
# accuracy pairing for the efficiency figure
# --------------------------------------------------------------------------
# An accuracy-against-latency figure needs a vertical coordinate, and the
# efficiency table deliberately carries none: the 64-row timing sample
# supports no accuracy claim. The closure fixes the convention, the common
# 10,004-row raw-distribution denominator, and this table fixes the exact
# pointer per system so VE-1 cannot pair a latency with the wrong accuracy.
#
# Rows that cannot be bound without a guess are marked PAIRING_UNRESOLVED with
# the precise question that has to be answered first. They are NOT paired
# speculatively and NOT dropped: an unresolved row may be plotted latency-only
# or omitted with the omission stated, and either way VE-1 must record which.

EFFICIENCY_ACCURACY_DENOMINATOR = (
    "common 10,004-row raw-distribution development denominator, the "
    "convention the closure fixed for both measurement nodes")

EFFICIENCY_ACCURACY_PAIRING = {
    "E7b::concat": {
        "status": "RESOLVED",
        "artefact": "results/experiments/e7b_serial_efficiency/"
                    "e7b_results.json",
        "pointer": "/e7b_analysis/systems/concat/accuracy/"
                   "raw_distribution_accuracy_common_denominator"},
    "E7b::e8a_a0p": {
        "status": "RESOLVED",
        "artefact": "results/experiments/e7b_serial_efficiency/"
                    "e7b_results.json",
        "pointer": "/e7b_analysis/systems/e8a_a0p/accuracy/"
                   "raw_distribution_accuracy_common_denominator"},
    "E7b::e8a_a1": {
        "status": "RESOLVED",
        "artefact": "results/experiments/e7b_serial_efficiency/"
                    "e7b_results.json",
        "pointer": "/e7b_analysis/systems/e8a_a1/accuracy/"
                   "raw_distribution_accuracy_common_denominator"},
    "E7b::fusion": {
        "status": "RESOLVED",
        "artefact": "results/experiments/e7b_serial_efficiency/"
                    "e7b_results.json",
        "pointer": "/e7b_analysis/systems/fusion/accuracy/"
                   "raw_distribution_accuracy_common_denominator"},
    "E7b::question_only": {
        "status": "RESOLVED",
        "artefact": "results/experiments/e7b_serial_efficiency/"
                    "e7b_results.json",
        "pointer": "/e7b_analysis/systems/question_only/accuracy/"
                   "raw_distribution_accuracy_common_denominator"},
    "E7b::reasoner": {
        "status": "RESOLVED",
        "artefact": "results/experiments/e7b_serial_efficiency/"
                    "e7b_results.json",
        "pointer": "/e7b_analysis/systems/reasoner/accuracy/"
                   "raw_distribution_accuracy_common_denominator"},
    "E7b::siglip_fusion": {
        "status": "RESOLVED",
        "artefact": "results/experiments/e7b_serial_efficiency/"
                    "e7b_results.json",
        "pointer": "/e7b_analysis/systems/siglip_fusion/accuracy/"
                   "raw_distribution_accuracy_common_denominator"},
    "E7b::vocab1000_product": {
        "status": "RESOLVED",
        "artefact": "results/experiments/e7b_serial_efficiency/"
                    "e7b_results.json",
        "pointer": "/e7b_analysis/systems/vocab1000_product/accuracy/"
                   "raw_distribution_accuracy_common_denominator"},
    "E9::e9_256m_open": {
        "status": "RESOLVED",
        "artefact": "results/experiments/e9_compact_vlm/e9_results.json",
        "pointer": "/e9_results/scored_passes/smolvlm_256m~1open~1normal/views/"
                   "raw_10004/normalised_exact"},
    "E9::e9_256m_constrained": {
        "status": "RESOLVED",
        "artefact": "results/experiments/e9_compact_vlm/e9_results.json",
        "pointer": "/e9_results/scored_passes/"
                   "smolvlm_256m~1constrained~1normal/views/raw_10004/"
                   "normalised_exact"},
    "E9::e9_500m_open": {
        "status": "RESOLVED",
        "artefact": "results/experiments/e9_compact_vlm/e9_results.json",
        "pointer": "/e9_results/scored_passes/smolvlm_500m~1open~1normal/views/"
                   "raw_10004/normalised_exact"},
    "E9::e9_500m_constrained": {
        "status": "RESOLVED",
        "artefact": "results/experiments/e9_compact_vlm/e9_results.json",
        "pointer": "/e9_results/scored_passes/"
                   "smolvlm_500m~1constrained~1normal/views/raw_10004/"
                   "normalised_exact"},
    "E9::e8b_B1": {
        "status": "PAIRING_UNRESOLVED",
        "question": "the stored E8B reference accuracies are recorded per "
                    "training scale (B1/train_40k and B1/train_250k) but the "
                    "timing row does not name the scale that was timed. VE-1 "
                    "must establish the timed configuration from the E9 "
                    "frozen protocol record before plotting a vertical "
                    "coordinate for this point."},
    "E9::e8b_B2_R2": {
        "status": "PAIRING_UNRESOLVED",
        "question": "the timed configuration is the R2 constrained-generation "
                    "readout, while the stored E8B reference accuracy block "
                    "carries the R1 readout. The training scale is also not "
                    "named on the timing row. VE-1 must resolve both before "
                    "plotting a vertical coordinate."},
    "E9::e8b_B2_R3": {
        "status": "PAIRING_UNRESOLVED",
        "question": "as E9::e8b_B2_R2, for the R3 free bounded-generation "
                    "readout."},
    "E9::e8b_B3_R2": {
        "status": "PAIRING_UNRESOLVED",
        "question": "as E9::e8b_B2_R2, for B3."},
    "E9::e8b_B3_R3": {
        "status": "PAIRING_UNRESOLVED",
        "question": "as E9::e8b_B2_R3, for B3."},
    "E9::fusion": {
        "status": "PAIRING_UNRESOLVED",
        "question": "E9 re-measured this reference system's latency on its "
                    "own node. The E9 efficiency artefact does not restate "
                    "the checkpoint identity behind that measurement, so "
                    "VE-1 must confirm from the E9 frozen protocol record "
                    "that it is the same checkpoint whose accuracy E7b "
                    "reports before reusing that accuracy as the vertical "
                    "coordinate. Accuracy is node-independent; the "
                    "constraint is checkpoint identity, not hardware."},
    "E9::vocab1000_product": {
        "status": "PAIRING_UNRESOLVED",
        "question": "as E9::fusion, for the top-1000 product reference."},
}


# --------------------------------------------------------------------------
# supersession decisions
# --------------------------------------------------------------------------
# Every important historical artefact gets one status. PARTLY_SUPERSEDED
# entries name the fields that remain valid and the fields that do not, so a
# later reader never has to guess which half of an artefact survives.

SUPERSESSION_STATUSES = {
    "CURRENT_CANONICAL": "the artefact a dissertation number must come from",
    "PARTLY_SUPERSEDED": "named fields remain valid; other named fields do not",
    "SUPERSEDED": "replaced in full by a later artefact; never re-quoted",
    "HISTORICAL_ONLY": "kept for provenance and narrative; never a reported "
                       "result",
    "NOT_FOR_REPORTING": "must not appear as a result in any form",
}

SUPERSESSION_ENTRIES = [
    {
        "artefact_group": "V1 stage results (results/v1 stage artefacts, the "
                          "five numbered stage scripts and their outputs)",
        "paths": ["results/stage3_baselines.json",
                  "results/stage4_fusion.json",
                  "results/stage5_evaluation.json"],
        "family": "V1",
        "status": "NOT_FOR_REPORTING",
        "valid_fields": [],
        "invalid_fields": ["every accuracy", "every efficiency figure"],
        "superseded_by": ["v2_02", "E7b"],
        "reason": "V1 is a completed legacy prototype. Its validation set was "
                  "reused for checkpoint selection and reporting, its "
                  "vocabulary indices differ from the V2 vocabulary for 11 "
                  "answers, and its stage-5 latencies were superseded first "
                  "by E7a components and then by E7b serial measurement. V1 "
                  "may be described as prototype history and must not supply "
                  "a reported number.",
    },
    {
        "artefact_group": "v2_01 zero-shot embedding floor",
        "paths": ["results/experiments/v2_01_embeddings/zero_shot.json"],
        "family": "v2_01",
        "status": "HISTORICAL_ONLY",
        "valid_fields": ["the zero-shot floor as a protocol sanity check"],
        "invalid_fields": [],
        "superseded_by": [],
        "reason": "The zero-shot floor established that the cached embeddings "
                  "carry signal. It is a protocol check, not a system result, "
                  "and no trained system is compared against it as a "
                  "baseline.",
    },
    {
        "artefact_group": "v2_02 five-seed baselines and fusion",
        "paths": ["results/experiments/v2_02_multiseed/results.json"],
        "family": "v2_02",
        "status": "CURRENT_CANONICAL",
        "valid_fields": ["per-seed accuracies", "across-seed sd",
                         "the closure-reconstructed row evidence and "
                         "clustered intervals"],
        "invalid_fields": [],
        "superseded_by": [],
        "reason": "Hash-verified, fully reconstructed by the closure, and the "
                  "source of the low-data fusion result.",
    },
    {
        "artefact_group": "v2_03 parameter-matched capacity controls",
        "paths": ["results/experiments/v2_03_param_match/results.json"],
        "family": "v2_03",
        "status": "CURRENT_CANONICAL",
        "valid_fields": ["per-seed accuracies", "across-seed sd",
                         "the closure-reconstructed clustered contrasts"],
        "invalid_fields": [],
        "superseded_by": [],
        "reason": "The only evidence that bounds how much of the fusion "
                  "advantage is capacity rather than features.",
    },
    {
        "artefact_group": "v2_04 interaction-feature ablation",
        "paths": ["results/experiments/v2_04_ablation/results.json"],
        "family": "v2_04",
        "status": "CURRENT_CANONICAL",
        "valid_fields": ["per-seed accuracies", "across-seed sd",
                         "all nine closure-reconstructed clustered gaps"],
        "invalid_fields": [],
        "superseded_by": [],
        "reason": "Complete systematic grid, no post-hoc selection.",
    },
    {
        "artefact_group": "v2_05b per-type gaps with row-level intervals",
        "paths": ["results/experiments/v2_05_types/addendum.json"],
        "family": "v2_05b",
        "status": "SUPERSEDED",
        "valid_fields": [],
        "invalid_fields": ["every row-level normal-approximation interval"],
        "superseded_by": ["v2_05c"],
        "reason": "The row-level normal approximation ignores within-image "
                  "dependence. v2_05c recomputed the same quantities with the "
                  "image-clustered bootstrap. The superseded intervals are "
                  "never re-quoted; the v2_05b per-bucket priors remain in "
                  "use as the fixed reference for the step deficit.",
    },
    {
        "artefact_group": "v2_05c image-clustered per-type gap intervals",
        "paths": ["results/experiments/v2_05_types/addendum_clustered.json"],
        "family": "v2_05c",
        "status": "CURRENT_CANONICAL",
        "valid_fields": ["all 42 clustered interval and count values",
                         "the 14 seed-42 intervals"],
        "invalid_fields": [],
        "superseded_by": [],
        "reason": "Independently reproduced value for value during the "
                  "closure review, and the source of the project's single "
                  "bootstrap method string.",
    },
    {
        "artefact_group": "v2_06 visual-reliance interventions",
        "paths": ["results/experiments/v2_06_reliance/results.json"],
        "family": "v2_06",
        "status": "CURRENT_CANONICAL",
        "valid_fields": ["per-seed accuracies under all three conditions",
                         "the 75 closure-reconstructed intervention cells"],
        "invalid_fields": [],
        "superseded_by": [],
        "reason": "Self-validating: a different image permutation could not "
                  "have reproduced 75 of 75 cells at five-decimal precision.",
    },
    {
        "artefact_group": "v2_07 scaling of the global heads",
        "paths": ["results/experiments/v2_07_scaling/results.json",
                  "results/experiments/v2_07_scaling/addendum_pooled.json"],
        "family": "v2_07",
        "status": "PARTLY_SUPERSEDED",
        "valid_fields": ["stored per-seed accuracies at 40k, 100k and 250k",
                         "across-training-seed sample sd",
                         "the stored pooled four-or-more-step deficit as a "
                         "seed-42, 250k, no-interval figure"],
        "invalid_fields": ["any image-clustered evaluation interval for this "
                           "family, which does not exist",
                           "any newly reconstructed row-level evidence, which "
                           "was attempted and stopped"],
        "superseded_by": [],
        "reason": "ACCEPTED DOCUMENTED LIMITATION, not a blocker. The "
                  "historical aggregate and seed evidence is untouched, "
                  "hash-verified and reproduced in 39 of 40 cells. The single "
                  "failing cell is one float32 argmax tie. The family is "
                  "stopped in full rather than partially rescued, because "
                  "admitting the 39 passing cells after seeing which one "
                  "failed would be outcome-conditioned selection.",
    },
    {
        "artefact_group": "v3_01 latent-query reasoner at 40k",
        "paths": ["results/experiments/v3_01_reasoner/results.json"],
        "family": "v3_01",
        "status": "PARTLY_SUPERSEDED",
        "valid_fields": ["the frozen training recipe inherited by v3_02a, "
                         "v3_03 and E8A/E8B",
                         "the 40k reasoner accuracies as originally reported"],
        "invalid_fields": ["the original pooled step statistics, repaired in "
                           "v3_02a"],
        "superseded_by": ["v3_02a"],
        "reason": "v3_02a repaired the pooled step statistics and added the "
                  "image-clustered uncertainty, the direct-linear and "
                  "mean-patch references and the attention diagnostics. Step "
                  "statistics must be quoted from v3_02a.",
    },
    {
        "artefact_group": "v3_02a repaired step statistics and references",
        "paths": ["results/experiments/v3_02a_refs/step_statistics.json"],
        "family": "v3_02a",
        "status": "CURRENT_CANONICAL",
        "valid_fields": ["per-bucket accuracies and counts",
                         "the clustered step deficits and deficit "
                         "differences at 40k"],
        "invalid_fields": [],
        "superseded_by": [],
        "reason": "The repaired and clustered version of the step analysis.",
    },
    {
        "artefact_group": "v3_03 reasoner scaling to 100k and 250k",
        "paths": ["results/experiments/v3_03_scaling/results.json"],
        "family": "v3_03",
        "status": "CURRENT_CANONICAL",
        "valid_fields": ["per-seed reasoner accuracies at 100k and 250k",
                         "the clustered paired deficit differences against "
                         "fusion"],
        "invalid_fields": [],
        "superseded_by": [],
        "reason": "The reasoner's scaling evidence and its deficit "
                  "comparison against the global fusion head.",
    },
    {
        "artefact_group": "E2 frozen SigLIP-B/16 encoder swap",
        "paths": ["results/experiments/e2_siglip/results.json"],
        "family": "E2",
        "status": "CURRENT_CANONICAL",
        "valid_fields": ["per-seed accuracies at 40k and 250k",
                         "the closure-reconstructed clustered SigLIP minus "
                         "CLIP contrasts at 40k"],
        "invalid_fields": ["any clustered SigLIP minus CLIP contrast at 250k, "
                           "which could not be formed because the CLIP "
                           "partner is the stopped v2_07 family"],
        "superseded_by": [],
        "reason": "The 40k comparison carries valid reconstructed clustered "
                  "evidence on identical development rows. At 250k only the "
                  "stored per-seed direction survives.",
    },
    {
        "artefact_group": "E3 top-1000 answer vocabulary",
        "paths": ["results/experiments/e3_vocab1000/results.json"],
        "family": "E3",
        "status": "CURRENT_CANONICAL",
        "valid_fields": ["coverage", "per-seed accuracies on the E3 view",
                         "the closure-reconstructed clustered contrasts "
                         "within the E3 view"],
        "invalid_fields": ["any pairing of an E3 row with a top-100 row"],
        "superseded_by": [],
        "reason": "Valid within its own answer set and its own development "
                  "view. Deliberately kept out of the main sequence: it "
                  "answers an answer-set-design question rather than one of "
                  "the project's reasoning or efficiency questions.",
    },
    {
        "artefact_group": "E7a component efficiency measurements",
        "paths": ["results/experiments/e7a_efficiency/results.json"],
        "family": "E7a",
        "status": "PARTLY_SUPERSEDED",
        "valid_fields": ["isolated encoder component latencies",
                         "isolated head component latencies",
                         "peak memory components",
                         "parameter counts"],
        "invalid_fields": ["gpu_encoder_plus_head_ms",
                           "full_pipeline_ms",
                           "amortised_ms",
                           "every Pareto front derived from those additive "
                           "sums"],
        "superseded_by": ["E7b"],
        "reason": "The component measurements remain valid as components. The "
                  "additive end-to-end sums and the fronts built on them are "
                  "superseded for any end-to-end claim by E7b's measured "
                  "serial latencies. The README statements 'cuts a query from "
                  "6.35 to 2.25 ms' and 'on both end-to-end latency Pareto "
                  "fronts' were removed for this reason and must not return.",
    },
    {
        "artefact_group": "E7b serial end-to-end efficiency",
        "paths": ["results/experiments/e7b_serial_efficiency/"
                  "e7b_results.json"],
        "family": "E7b",
        "status": "CURRENT_CANONICAL",
        "valid_fields": ["warm median serial latency at batch 1 on otter155",
                         "the recorded primary frontier on that node",
                         "the cached and head-only classes, each labelled "
                         "not comparable with end-to-end"],
        "invalid_fields": ["any merge with the E9 otter159 measurements"],
        "superseded_by": [],
        "reason": "The authoritative end-to-end latency evidence. One node, "
                  "one protocol, never merged across nodes.",
    },
    {
        "artefact_group": "E8A frozen-SLM question-encoder core analysis",
        "paths": ["results/experiments/e8a_question_encoder/"
                  "core_analysis_g21.json"],
        "family": "E8A",
        "status": "CURRENT_CANONICAL",
        "valid_fields": ["the A1 minus A1r within-size contrasts",
                         "the A1 minus A0p and A0p minus A1r system "
                         "comparisons, each with its confound stated",
                         "the per-slice accuracies"],
        "invalid_fields": [],
        "superseded_by": [],
        "reason": "The question-side pretraining evidence, with 102 "
                  "image-clustered intervals and multiplicity disclosed.",
    },
    {
        "artefact_group": "E8B recipe search, grid points 1 to 8",
        "paths": ["results/experiments/e8b_readout_generation/"
                  "superseded_evidence_20260807.json"],
        "family": "E8B",
        "status": "NOT_FOR_REPORTING",
        "valid_fields": [],
        "invalid_fields": ["every ranking statistic derived from grid points "
                           "1 to 3", "grid points 4 to 8, which never ran"],
        "superseded_by": [],
        "reason": "The eight-point recipe search is permanently abandoned. "
                  "Grid points 1 to 3 are exploratory protocol-diagnostic "
                  "evidence only; every statistic derived from them for "
                  "ranking purposes is INVALIDATED. The honest narrative is "
                  "that the exploratory search could not support a winner "
                  "claim, so the pre-result default recipe was frozen "
                  "outcome-independently. The historical observation that "
                  "grid point 1 also showed the highest bf16 maximum may be "
                  "mentioned only as a superseded, non-canonical fact.",
    },
    {
        "artefact_group": "E8B answer-readout core matrix",
        "paths": ["results/experiments/e8b_readout_generation/"
                  "e8b_final_report_20260810.json"],
        "family": "E8B",
        "status": "CURRENT_CANONICAL",
        "valid_fields": ["the epoch-22 fp32 primary accuracies",
                         "the B3 minus B2 within-size contrasts",
                         "the B2 minus B1 and B3 minus B1 system comparisons",
                         "the image-reliance drops"],
        "invalid_fields": ["every bf16 figure as a scientific accuracy",
                           "every best-of-22 figure as a primary result"],
        "superseded_by": [],
        "reason": "Canonical scientific evaluation is fp32 and the epoch-22 "
                  "checkpoint is the sole primary. bf16 figures are secondary "
                  "deployment diagnostics and best-of-22 figures are clearly "
                  "labelled secondary diagnostics; neither determines a "
                  "primary comparison.",
    },
    {
        "artefact_group": "E9 evaluation-only compact-VLM baselines",
        "paths": ["results/experiments/e9_compact_vlm/e9_results.json"],
        "family": "E9",
        "status": "CURRENT_CANONICAL",
        "valid_fields": ["the normalised accuracies on the raw development "
                         "partition",
                         "the constrained against open generation contrasts",
                         "the derangement reliance contrast",
                         "the same-node otter159 latencies"],
        "invalid_fields": ["any merge of the otter159 latencies with E7b's "
                           "otter155 frontier",
                           "any comparison against published official GQA "
                           "scores"],
        "superseded_by": [],
        "reason": "Contextual positioning evidence. The fusion bridge control "
                  "failed its pre-registered 10 per cent tolerance at -18.69 "
                  "per cent, so the two nodes stay separate and no adjustment "
                  "factor is applied.",
    },
    {
        "artefact_group": "E10 SmolLM2-360M answer-readout capacity matrix",
        "paths": ["results/experiments/e10_capacity_360m_core/"
                  "e10_core_analysis.json"],
        "family": "E10",
        "status": "CURRENT_CANONICAL",
        "valid_fields": ["the twelve cell accuracies",
                         "the two scale effects",
                         "the two pretraining effects",
                         "the difference in differences"],
        "invalid_fields": ["any timing figure, because none was measured"],
        "superseded_by": [],
        "reason": "E10 is CLOSED at E10_PASS and must not be reopened. Its "
                  "grant is operationally SPENT, its completed cells are "
                  "immutable and its live source digest is unchanged.",
    },
    {
        "artefact_group": "post-E10 statistical and efficiency closure",
        "paths": ["results/closure/A_evidence_registry.json",
                  "results/closure/B_primary_contrasts.json",
                  "results/closure/C_reconstructed_evidence_manifest.json",
                  "results/closure/D_slice_statistics.json",
                  "results/closure/E_seed_variability.json",
                  "results/closure/F_multiplicity_status.json",
                  "results/closure/G_claim_evidence.json",
                  "results/closure/H_evidence_readiness.json",
                  "results/closure/efficiency_closure_table.json",
                  "results/closure/reconstructed_statistics.json"],
        "family": "closure",
        "status": "CURRENT_CANONICAL",
        "valid_fields": ["every reconstructed clustered interval",
                         "the seed variability table",
                         "the slice table", "the efficiency table",
                         "the claim and readiness tables"],
        "invalid_fields": ["any v2_07 row-level or clustered value, which the "
                           "closure deliberately did not produce"],
        "superseded_by": [],
        "reason": "Independently reviewed at verdict "
                  "STATISTICS_EFFICIENCY_CLOSURE_PASS with zero outstanding "
                  "blocking findings. It is the immediate parent of this "
                  "evidence contract.",
    },
]
