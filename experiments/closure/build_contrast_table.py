"""Outputs B and F: the canonical primary-contrast table and the
multiplicity / status table.

    python -m experiments.closure.build_contrast_table

B collects, in one place and one schema, every claim-bearing contrast in the
project: those the source experiments already computed with a correct
image-clustered interval, and those this closure reconstructed. Contrasts
that already carry a correct interval are READ, never recomputed; a
competing re-analysis of E8A, E8B, E10, v2_05c, v3_02a or v3_03 would be a
second answer to a settled question.

Every row carries two classifications, both fixed before the numbers were
looked at:

  analysis_role      PRIMARY | SECONDARY | EXPLORATORY | DIAGNOSTIC
  comparison_status  CLEAN_PAIRED_CONTROL | SYSTEM_LEVEL_COMPARISON |
                     CAPACITY_CONFOUNDED | REPRESENTATION_CONFOUNDED |
                     DESCRIPTIVE_ONLY

F records how many intervals exist across the project and states plainly
that none is corrected for multiplicity. No new hypothesis test is
introduced anywhere, and no row is included or excluded on the basis of
whether its interval excludes zero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402
from experiments.closure import closure_common as cc  # noqa: E402

RESULTS_ROOT = config.RESULTS_DIR / "experiments"

# Classification of the reconstructed contrasts is carried on the rows
# themselves (build_intervals assigned it). Classification of the stored
# contrasts is assigned here, from the planning audit, before any number is
# read.
STORED_CLASS = {
    "E10/pretraining_effect/train_40k": ("PRIMARY", "CLEAN_PAIRED_CONTROL"),
    "E10/pretraining_effect/train_250k": ("PRIMARY", "CLEAN_PAIRED_CONTROL"),
    "E10/scale_effect_B4": ("PRIMARY", "CLEAN_PAIRED_CONTROL"),
    "E10/scale_effect_B4r": ("PRIMARY", "CLEAN_PAIRED_CONTROL"),
    "E10/difference_in_differences": ("PRIMARY", "CLEAN_PAIRED_CONTROL"),
    "E8B/B3 - B2/train_40k": ("PRIMARY", "CLEAN_PAIRED_CONTROL"),
    "E8B/B3 - B2/train_250k": ("PRIMARY", "CLEAN_PAIRED_CONTROL"),
    "E8B/B2 - B1/train_40k": ("SECONDARY", "SYSTEM_LEVEL_COMPARISON"),
    "E8B/B2 - B1/train_250k": ("SECONDARY", "SYSTEM_LEVEL_COMPARISON"),
    "E8B/B3 - B1/train_40k": ("SECONDARY", "SYSTEM_LEVEL_COMPARISON"),
    "E8B/B3 - B1/train_250k": ("SECONDARY", "SYSTEM_LEVEL_COMPARISON"),
    "E8A/A1_minus_A1r/train_40k": ("PRIMARY", "CLEAN_PAIRED_CONTROL"),
    "E8A/A1_minus_A1r/train_250k": ("PRIMARY", "CLEAN_PAIRED_CONTROL"),
    "E8A/A1_minus_A0p/train_40k": ("SECONDARY", "REPRESENTATION_CONFOUNDED"),
    "E8A/A1_minus_A0p/train_250k": ("SECONDARY", "REPRESENTATION_CONFOUNDED"),
    "E8A/A0p_minus_A1r/train_40k": ("SECONDARY", "SYSTEM_LEVEL_COMPARISON"),
    "E8A/A0p_minus_A1r/train_250k": ("SECONDARY", "SYSTEM_LEVEL_COMPARISON"),
    "v3_02a/reasoner_minus_fusion_deficit/train_40k":
        ("PRIMARY", "SYSTEM_LEVEL_COMPARISON"),
    "v3_03/reasoner_minus_fusion_deficit/train_100k":
        ("PRIMARY", "SYSTEM_LEVEL_COMPARISON"),
    "v3_03/reasoner_minus_fusion_deficit/train_250k":
        ("PRIMARY", "SYSTEM_LEVEL_COMPARISON"),
}

CAVEAT_FIXED_SEED = (
    "the 95 per cent interval is image-clustered over the development images "
    "and conditions on the fixed trained seed set: it carries sampling "
    "variation over questions and images, not training-seed variation. The "
    "across-seed standard deviation is reported beside it as a separate "
    "quantity")

B1_SELECTION_CAVEAT = (
    "B1 keeps its own frozen section 7.3 classifier recipe and is not forced "
    "onto the fixed-22-epoch rule that governs B2 and B3, so any contrast "
    "against B1 compares two different checkpoint-selection rules as well as "
    "two architectures. It is a system-level comparison, not a clean causal "
    "architecture contrast")


def row(contrast_id, family, effect, ci, *, role, status, per_seed, sd,
        n_questions, n_unique_images, source, note="", directional=None,
        draws=2000, rng_seed=0, seeds=None, analysis_status="READ_FROM_SOURCE",
        metric=""):
    return {
        "contrast_id": contrast_id,
        "family": family,
        "analysis_role": role,
        "comparison_status": status,
        "point_estimate": effect,
        "effect": effect,
        "ci95_image_clustered": ci,
        "excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
        "directional_rule_outcome": directional,
        "cluster_unit": cc.CLUSTER_UNIT,
        "resampling_method": ("image-clustered percentile bootstrap, "
                              "clusters resampled with replacement, "
                              "percentiles 2.5/97.5"),
        "n_resamples": draws,
        "rng_seed": rng_seed,
        "n_questions": n_questions,
        "n_unique_images": n_unique_images,
        "training_seeds": seeds,
        "per_seed_effect": per_seed,
        "across_training_seed_sd_ddof1": sd,
        "interval_kind": "evaluation-sampling, image-clustered",
        "interval_caveat": CAVEAT_FIXED_SEED,
        "metric": metric,
        "source_artefact": source,
        "analysis_status": analysis_status,
        "interpretation_bound": note,
    }


def main() -> int:
    utils.set_seed()
    rows = []

    # ---------------- reconstructed contrasts ------------------------------
    stats_path = cc.CLOSURE_DIR / "reconstructed_statistics.json"
    stats = json.loads(stats_path.read_text())
    for entry in stats["contrasts"]:
        rows.append({
            "contrast_id": entry["contrast_id"],
            "family": entry["contrast_id"].split("/")[0],
            "analysis_role": entry["analysis_role"],
            "comparison_status": entry["comparison_status"],
            "point_estimate": entry["point_estimate"],
            "effect": entry["effect"],
            "ci95_image_clustered": entry["ci95_image_clustered"],
            "excludes_zero": entry["excludes_zero"],
            "directional_rule_outcome": None,
            "cluster_unit": entry["cluster_unit"],
            "resampling_method": entry["resampling_method"],
            "n_resamples": entry["n_resamples"],
            "rng_seed": entry["rng_seed"],
            "n_questions": entry["n_questions"],
            "n_unique_images": entry["n_unique_images"],
            "training_seeds": entry["training_seeds"],
            "per_seed_effect": entry["per_seed_effect"],
            "across_training_seed_sd_ddof1":
                entry["across_training_seed_sd_ddof1"],
            "interval_kind": entry["interval_kind"],
            "interval_caveat": CAVEAT_FIXED_SEED,
            "metric": "v2_closed_vocab_top1_index_match",
            "source_artefact": "results/closure/reconstructed_statistics.json",
            "analysis_status": entry["analysis_status"],
            "interpretation_bound": entry["effect_direction_note"],
        })

    # ---------------- E10 (read, never recomputed) -------------------------
    e10_path = RESULTS_ROOT / "e10_capacity_360m_core" / "e10_core_analysis.json"
    e10 = json.loads(e10_path.read_text())["e10_core_analysis"]
    e10_names = {
        "pretraining_effect_train_40k": "E10/pretraining_effect/train_40k",
        "pretraining_effect_train_250k": "E10/pretraining_effect/train_250k",
        "scale_effect_B4": "E10/scale_effect_B4",
        "scale_effect_B4r": "E10/scale_effect_B4r",
        "pretraining_by_scale_difference_in_differences":
            "E10/difference_in_differences",
    }
    e10_notes = {
        "E10/pretraining_effect/train_40k": (
            "no reliable positive pretrained-over-random advantage was "
            "detected. The interval includes zero and the seeds disagree in "
            "sign; this establishes neither equivalence nor the absence of an "
            "effect. It does NOT reproduce E8B's directional negative 40k "
            "result"),
        "E10/pretraining_effect/train_250k": (
            "no reliable positive pretrained-over-random advantage was "
            "detected. The interval includes zero; this establishes neither "
            "equivalence nor the absence of an effect"),
        "E10/scale_effect_B4": "training-set size is what moves accuracy",
        "E10/scale_effect_B4r": "training-set size is what moves accuracy",
        "E10/difference_in_differences": (
            "not directional; the interval includes zero. The full-magnitude "
            "form, after the Phase-2 halving repair"),
    }
    for key, contrast_id in e10_names.items():
        metric = e10["contrasts"][key]["metrics"]["g21_normalized_correct"]
        role, status = STORED_CLASS[contrast_id]
        rows.append(row(
            contrast_id, "E10", metric["mean_difference"],
            metric["interval_95"], role=role, status=status,
            per_seed=metric["per_seed_differences"],
            sd=metric["sd_difference"], n_questions=7714, n_unique_images=768,
            source="results/experiments/e10_capacity_360m_core/"
                   "e10_core_analysis.json",
            note=e10_notes[contrast_id],
            directional=metric["directional_rule"], seeds=[0, 1, 2],
            metric="G21 pinned normalised exact match"))

    # ---------------- E8B (read) -------------------------------------------
    e8b_path = (RESULTS_ROOT / "e8b_readout_generation"
                / "e8b_final_report_20260810.json")
    e8b = json.loads(e8b_path.read_text())["e8b_final_report"]["PRIMARY"]
    for entry in e8b["paired_contrasts_in_vocabulary"]:
        contrast_id = f"E8B/{entry['contrast']}/{entry['scale']}"
        role, status = STORED_CLASS[contrast_id]
        note = ("the clean within-size answer-side pretraining contrast. It "
                "does not support a reliable positive pretrained advantage"
                if entry["contrast"] == "B3 - B2" else B1_SELECTION_CAVEAT)
        rows.append(row(
            contrast_id, "E8B", entry["mean_paired_difference"],
            [entry["ci95_low"], entry["ci95_high"]], role=role, status=status,
            per_seed=entry["per_seed_differences"],
            sd=None, n_questions=entry["n_rows"],
            n_unique_images=entry["n_image_clusters"],
            source="results/experiments/e8b_readout_generation/"
                   "e8b_final_report_20260810.json",
            note=note, directional=entry["directional_rule_outcome"],
            seeds=[0, 1, 2], metric="G21 pinned normalised exact match"))
    for entry in e8b["scale_effects"]:
        rows.append(row(
            f"E8B/scale_effect_{entry['arm']}", "E8B",
            entry["mean_paired_difference"],
            [entry["ci95_low"], entry["ci95_high"]], role="SECONDARY",
            status="CLEAN_PAIRED_CONTROL",
            per_seed=entry["per_seed_differences"], sd=None,
            n_questions=entry["n_rows"],
            n_unique_images=entry["n_image_clusters"],
            source="results/experiments/e8b_readout_generation/"
                   "e8b_final_report_20260810.json",
            note="the same development rows; only the training set size "
                 "differs", directional=entry["directional_rule_outcome"],
            seeds=[0, 1, 2], metric="G21 pinned normalised exact match"))
    for entry in e8b["image_reliance"]:
        rows.append(row(
            f"E8B/image_reliance/{entry['arm']}/{entry['scale']}", "E8B",
            entry["mean_drop"], [entry["ci95_low"], entry["ci95_high"]],
            role="SECONDARY", status="CLEAN_PAIRED_CONTROL",
            per_seed=entry["per_seed_drops"], sd=None, n_questions=7714,
            n_unique_images=768,
            source="results/experiments/e8b_readout_generation/"
                   "e8b_final_report_20260810.json",
            note="a drop under a deranged image shows the answer depends on "
                 "the correct image. It does not show visual reasoning",
            directional=entry["directional_rule_outcome"], seeds=[0, 1, 2],
            metric="G21 pinned normalised exact match"))

    # ---------------- E8A (read) -------------------------------------------
    e8a_path = (RESULTS_ROOT / "e8a_question_encoder"
                / "core_analysis_g21.json")
    e8a = json.loads(e8a_path.read_text())["e8a_core_analysis_g21"]
    e8a_notes = {
        "A1_minus_A1r": ("the clean within-size question-side pretraining "
                         "contrast: identical architecture and tokenizer, "
                         "pretrained weights against a pinned random "
                         "initialisation"),
        "A1_minus_A0p": ("a system comparison, not a pure causal semantics or "
                         "alignment effect: the two sides differ in "
                         "representation space and in interface as well as in "
                         "the question encoder"),
        "A0p_minus_A1r": "a system-level comparison across two interfaces",
    }
    for scale in ("train_40k", "train_250k"):
        for name, entry in e8a["analyses"][scale][
                "contrasts_normalized"].items():
            contrast_id = f"E8A/{name}/{scale}"
            role, status = STORED_CLASS[contrast_id]
            per_seed = [entry["per_seed_differences"][f"seed{s}"]
                        for s in (0, 1, 2)]
            rows.append(row(
                contrast_id, "E8A", entry["mean"],
                entry["ci95_image_clustered"], role=role, status=status,
                per_seed=per_seed, sd=entry["std"], n_questions=7714,
                n_unique_images=entry["n_clusters"],
                source="results/experiments/e8a_question_encoder/"
                       "core_analysis_g21.json",
                note=e8a_notes[name],
                directional=entry["directional_rule_outcome"],
                draws=entry["n_draws"], rng_seed=entry["rng_seed"],
                seeds=[0, 1, 2], metric="G21 pinned normalised exact match"))

    # ---------------- V3 paired step-deficit differences (read) ------------
    v3_02a_path = RESULTS_ROOT / "v3_02a_refs" / "step_statistics.json"
    v3_02a = json.loads(v3_02a_path.read_text())["v3_02a_step_statistics"]
    paired = v3_02a["reasoner_minus_fusion_deficit_paired"]
    v3_note = ("a system-level comparison of a latent-query reasoner over "
               "token features against a global-embedding fusion head. It "
               "does not isolate token access causally. The interval includes "
               "zero: no model in the set has been shown to reduce the "
               "deficit, which is not the same as showing the models "
               "equivalent")
    rows.append(row(
        "v3_02a/reasoner_minus_fusion_deficit/train_40k", "v3_02a",
        paired["mean"], paired["ci95"], role="PRIMARY",
        status="SYSTEM_LEVEL_COMPARISON", per_seed=None, sd=None,
        n_questions=7714, n_unique_images=v3_02a["bootstrap"]["n_clusters"],
        source="results/experiments/v3_02a_refs/step_statistics.json",
        note=v3_note, draws=v3_02a["bootstrap"]["n_draws"],
        rng_seed=v3_02a["bootstrap"]["rng_seed"], seeds=paired["seeds"],
        metric="v2_closed_vocab_top1_index_match"))

    v3_03_path = RESULTS_ROOT / "v3_03_scaling" / "results.json"
    v3_03 = json.loads(v3_03_path.read_text())["v3_03_scaling"]
    for scale_key, scale in (("100k", "train_100k"), ("250k", "train_250k")):
        entry = v3_03["bootstrap"][scale_key][
            "reasoner_minus_fusion_deficit_paired"]
        rows.append(row(
            f"v3_03/reasoner_minus_fusion_deficit/{scale}", "v3_03",
            entry["mean"], entry["ci95"], role="PRIMARY",
            status="SYSTEM_LEVEL_COMPARISON", per_seed=None, sd=None,
            n_questions=7714, n_unique_images=entry["n_clusters"],
            source="results/experiments/v3_03_scaling/results.json",
            note=v3_note, draws=entry["n_draws"], rng_seed=entry["rng_seed"],
            seeds=entry["seeds"], metric="v2_closed_vocab_top1_index_match"))

    rows.sort(key=lambda r: r["contrast_id"])

    record_b = {
        "closure_output": "B",
        "title": "canonical primary-contrast table",
        "provenance": cc.provenance(
            "experiments/closure/build_contrast_table.py",
            [cc.record_source(stats_path, "reconstructed statistics"),
             cc.record_source(e10_path, "E10 canonical analysis"),
             cc.record_source(e8b_path, "E8B final report"),
             cc.record_source(e8a_path, "E8A core analysis"),
             cc.record_source(v3_02a_path, "v3_02a step statistics"),
             cc.record_source(v3_03_path, "v3_03 scaling")]),
        "policy": (
            "contrasts that already carry a correct image-clustered interval "
            "are READ from their canonical artefact, never recomputed. This "
            "closure does not produce a competing re-analysis of E8A, E8B, "
            "E10, v2_05c, v3_02a or v3_03"),
        "uncertainty_note": stats["uncertainty_note"],
        "classification_note": (
            "analysis_role and comparison_status were fixed in the planning "
            "audit before any reconstructed number existed, and no row was "
            "reclassified after its interval was computed"),
        "contrast_count": len(rows),
        "by_role": {role: sum(1 for r in rows if r["analysis_role"] == role)
                    for role in ("PRIMARY", "SECONDARY", "EXPLORATORY",
                                 "DIAGNOSTIC")},
        "by_status": {status: sum(1 for r in rows
                                  if r["comparison_status"] == status)
                      for status in ("CLEAN_PAIRED_CONTROL",
                                     "SYSTEM_LEVEL_COMPARISON",
                                     "CAPACITY_CONFOUNDED",
                                     "REPRESENTATION_CONFOUNDED",
                                     "DESCRIPTIVE_ONLY")},
        "not_computed": stats["blocked_contrasts"],
        "contrasts": rows,
        "clean_test_accessed": False,
    }
    digest_b = cc.write_json(record_b,
                             cc.CLOSURE_DIR / "B_primary_contrasts.json")
    cc.write_csv(rows, ["contrast_id", "family", "analysis_role",
                        "comparison_status", "effect", "ci95_image_clustered",
                        "excludes_zero", "directional_rule_outcome",
                        "n_questions", "n_unique_images", "cluster_unit",
                        "n_resamples", "rng_seed", "training_seeds",
                        "across_training_seed_sd_ddof1", "metric",
                        "analysis_status", "source_artefact"],
                 cc.CLOSURE_DIR / "B_primary_contrasts.csv")

    # ---------------- Output F: multiplicity and status --------------------
    e8a_constructed = e8a["intervals_constructed"]
    reconstructed_intervals = (len(stats["contrasts"]) + len(stats["arms"] if
                               "arms" in stats else stats["arm_accuracies"])
                               + len(stats["pooled_deficits"])
                               + len(stats["slices"]))
    families = {}
    for entry in rows:
        families.setdefault(entry["family"], {"contrasts": 0, "primary": 0})
        families[entry["family"]]["contrasts"] += 1
        if entry["analysis_role"] == "PRIMARY":
            families[entry["family"]]["primary"] += 1

    record_f = {
        "closure_output": "F",
        "title": "multiplicity and comparison-status table",
        "provenance": cc.provenance(
            "experiments/closure/build_contrast_table.py"),
        "multiplicity_statement": (
            "every interval in this project is nominal 95 per cent, "
            "image-clustered and UNCORRECTED for multiplicity. No family-wise "
            "correction is applied by any experiment or by this closure, no "
            "p-value is computed anywhere, and directional rules rather than "
            "significance tests govern every conclusion. This is a disclosure, "
            "not a defect to be repaired by an undeclared correction"),
        "no_fishing_statement": (
            "no new hypothesis test was introduced by this closure, no "
            "analysis was chosen after seeing its outcome, and no result was "
            "reported or withheld on the basis of whether its interval "
            "excludes zero. The reconstructed contrast list is the list the "
            "source experiments already published as their own gaps, plus the "
            "arm accuracies those gaps are built from"),
        "interval_counts": {
            "E8A_constructed_in_source": e8a_constructed["total"],
            "E8A_headline": e8a_constructed["headline"],
            "E8A_per_slice": e8a_constructed["per_slice"],
            "E8B_in_final_report": (len(e8b["paired_contrasts_in_vocabulary"])
                                    + len(e8b["scale_effects"])
                                    + len(e8b["image_reliance"])),
            "E10_in_core_analysis": len(e10["contrasts"]),
            "constructed_by_this_closure": reconstructed_intervals,
            "in_output_B": len(rows),
        },
        "role_definitions": {
            "PRIMARY": "a contrast a central dissertation claim rests on",
            "SECONDARY": "a supporting contrast, reported with its claim",
            "EXPLORATORY": "protocol-diagnostic evidence; never a core result "
                           "and never a basis for selection",
            "DIAGNOSTIC": "descriptive detail; supports no comparative claim",
        },
        "status_definitions": {
            "CLEAN_PAIRED_CONTROL": "the two sides differ in the one factor "
                                    "under study, with the control matched "
                                    "on architecture and size",
            "SYSTEM_LEVEL_COMPARISON": "the two sides differ in more than the "
                                       "named factor; the comparison is "
                                       "between whole systems",
            "CAPACITY_CONFOUNDED": "trainable capacity moves with the factor "
                                   "under study",
            "REPRESENTATION_CONFOUNDED": "the representation space or width "
                                         "moves with the factor under study",
            "DESCRIPTIVE_ONLY": "different denominators, rows or supports; no "
                                "paired contrast is defined",
        },
        "by_family": families,
        "by_role": record_b["by_role"],
        "by_status": record_b["by_status"],
        "exploratory_evidence_permanently_invalidated": {
            "E8B grid points 1-3": (
                "exploratory protocol-diagnostic evidence only. Every "
                "statistic derived from them for ranking purposes is "
                "invalidated and supports no superiority claim in either "
                "direction. Grid points 4-8 were permanently abandoned and "
                "never ran"),
        },
        "clean_test_accessed": False,
    }
    digest_f = cc.write_json(record_f,
                             cc.CLOSURE_DIR / "F_multiplicity_status.json")

    print(f"B: {len(rows)} contrasts  {record_b['by_role']}")
    print(f"B_primary_contrasts.json sha256 {digest_b}")
    print(f"F_multiplicity_status.json sha256 {digest_f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
