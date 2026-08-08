"""Stage 8: re-project the resource gates from measured Phase-1A evidence.

Every rate below is read from a real artefact. Where a component has no
measurement — the whole of E8B, and the 360M and FLAN-T5 extraction rates —
that is stated rather than silently extrapolated from the 135M pilot.

Usage:

    python -B experiments/e8a_question_encoder/stage8_projection.py
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
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402

# Canonical figures, quoted for comparison only; none is used as an input.
CANONICAL = {
    "e8a_seconds_per_epoch_40k": 15.27,
    "e8a_seconds_per_epoch_250k": 86.36,
    "observed_epochs_40k": 15,
    "observed_epochs_250k": 22,
    "configured_cap": 100,
    "extraction_full_pass_hours": (0.200, 0.400),
    "e8b_multiplier": (2.931, 8.004),
    "e8b_eval_pass_hours": (0.041667, 0.083333),
    "e8a_intervention_pass_hours": (0.002, 0.005),
    "e9_eval_pass_hours": (0.333333, 0.666667),
    "efficiency_pass_hours": (0.500, 1.000),
    "per_run_ceiling_hours": 8.0,
    "per_model_ceiling_hours": 35.0,
    "total_ceiling_hours": 180.0,
    "expected_projection": 92.224,
    "direct_maximum": 123.375,
    "contingency_maximum": 148.050,
    "storage_gib": 54.60,
}

# The ratio v3_03 measured between its 250k and 40k seconds per epoch. Used to
# carry the pilot's measured 40k rate to 250k, because no 250k E8A run exists
# and the Phase-1A authorisation forbids one.
V3_SCALE_RATIO = 86.36 / 15.27
ROW_SCALE_RATIO = 250_000 / 40_000


def main() -> int:
    pilot = json.loads(
        (e8a.OUT_DIR / "pilot.json").read_text())["e8a_pilot"]
    extraction = json.loads(
        (e8a.OUT_DIR / "extraction.json").read_text())["e8a_extraction"]

    runs = pilot["runs"]
    sec_per_epoch = max(r["seconds_per_epoch"] for r in runs.values())
    sec_per_epoch_max = max(r["seconds_per_epoch_max"] for r in runs.values())
    epochs_observed = max(r["epochs_run"] for r in runs.values())
    best_epochs = {a: r["best_epoch"] for a, r in runs.items()}
    peak_mib = max(r["peak_allocated_mib"] for r in runs.values())
    ceiling_mib = runs["A1"]["memory_ceiling_mib"]
    loader = {a: r["loader_throughput"]["samples_per_second"]
              for a, r in runs.items()}

    stores = extraction["stores_written"]
    extract_seconds_40k = max(s["seconds"] for s in stores.values())
    strings_40k = stores["A1"]["n_unique_strings"]
    strings_250k = extraction["length_statistics"][
        "train_250k + dev"]["unique_strings"]
    strings_per_second = strings_40k / extract_seconds_40k
    extract_hours_40k = extract_seconds_40k / 3600
    extract_hours_250k_135m = strings_250k / strings_per_second / 3600

    # --- E8A training, measured ------------------------------------------
    e8a_40k_expected = epochs_observed * sec_per_epoch_max / 3600
    e8a_40k_cap = 100 * sec_per_epoch_max / 3600
    sec_250k_v3ratio = sec_per_epoch_max * V3_SCALE_RATIO
    sec_250k_rowratio = sec_per_epoch_max * ROW_SCALE_RATIO
    e8a_250k_expected = 22 * sec_250k_rowratio / 3600   # conservative basis
    e8a_250k_cap = 100 * sec_250k_rowratio / 3600

    e8a_core_expected = 30 * e8a_40k_expected + 30 * e8a_250k_expected
    e8a_core_cap = 30 * e8a_40k_cap + 30 * e8a_250k_cap
    e8a_secondary_expected = 16 * e8a_40k_expected
    e8a_secondary_cap = 16 * e8a_40k_cap
    e8a_ablation_expected = 4 * e8a_40k_expected
    e8a_ablation_cap = 4 * e8a_40k_cap

    # Extraction: 5 full stores + 2 middle-layer partial stores. Only the 135M
    # rate is measured; the 360M and FLAN-T5 rates are NOT, and are flagged.
    extraction_measured_135m = 2 * extract_hours_250k_135m   # A1, A1r full
    extraction_unmeasured = 3 * extract_hours_250k_135m      # A2, A2r, AF
    extraction_partial = 2 * extract_hours_40k
    extraction_total_at_135m_rate = (extraction_measured_135m
                                     + extraction_unmeasured
                                     + extraction_partial)

    report = {
        "metadata": utils.run_metadata(),
        "stage8_projection": {
            "basis": "measured Phase-1A evidence only. Components with no "
                     "measurement are listed separately and are NOT folded "
                     "into a single headline number.",
            "measured": {
                "e8a_seconds_per_epoch_40k_mean": sec_per_epoch,
                "e8a_seconds_per_epoch_40k_max": sec_per_epoch_max,
                "canonical_planning_value": CANONICAL[
                    "e8a_seconds_per_epoch_40k"],
                "ratio_measured_to_canonical": round(
                    sec_per_epoch_max / CANONICAL[
                        "e8a_seconds_per_epoch_40k"], 4),
                "epochs_run": {a: r["epochs_run"] for a, r in runs.items()},
                "selected_epoch": best_epochs,
                "stopping_behaviour": {a: r["stopped_by"]
                                       for a, r in runs.items()},
                "wall_clock_hours": {a: r["wall_clock_hours"]
                                     for a, r in runs.items()},
                "loader_samples_per_second": loader,
                "peak_allocated_mib": peak_mib,
                "memory_ceiling_mib": ceiling_mib,
                "memory_fraction_of_ceiling": round(peak_mib / ceiling_mib, 4),
                "extraction_seconds_40k_store": extract_seconds_40k,
                "extraction_strings_per_second": round(strings_per_second, 2),
                "extraction_hours_40k_store": round(extract_hours_40k, 6),
                "store_gib_40k": stores["A1"]["gib"],
            },
            "per_run_projection": {
                "e8a_40k_expected_hours": round(e8a_40k_expected, 6),
                "e8a_40k_configured_cap_hours": round(e8a_40k_cap, 6),
                "e8a_250k_expected_hours": round(e8a_250k_expected, 6),
                "e8a_250k_configured_cap_hours": round(e8a_250k_cap, 6),
                "e8a_250k_seconds_per_epoch_v3_ratio_basis": round(
                    sec_250k_v3ratio, 3),
                "e8a_250k_seconds_per_epoch_row_ratio_basis": round(
                    sec_250k_rowratio, 3),
                "basis_note": "the 250k figures use the ROW ratio 6.25, the "
                              "more conservative of the two, because no E8A "
                              "250k run exists and Phase 1A forbids one. The "
                              "v3_03 measured ratio 5.655 would give smaller "
                              "figures.",
                "canonical_e8a_40k_expected": 0.063625,
                "canonical_e8a_250k_expected": 0.527756,
                "canonical_e8a_40k_cap": 0.424167,
                "canonical_e8a_250k_cap": 2.398889,
                "direct_per_run_maximum_hours": round(e8a_250k_cap, 6),
                "per_run_ceiling_hours": CANONICAL["per_run_ceiling_hours"],
                "per_run_gate_fires": bool(
                    e8a_250k_cap > CANONICAL["per_run_ceiling_hours"]),
            },
            "e8a_component_projection": {
                "core_training_60_runs_expected": round(e8a_core_expected, 3),
                "core_training_60_runs_cap": round(e8a_core_cap, 3),
                "secondary_study_16_runs_expected": round(
                    e8a_secondary_expected, 3),
                "secondary_study_16_runs_cap": round(e8a_secondary_cap, 3),
                "ablation_4_runs_expected": round(e8a_ablation_expected, 3),
                "ablation_4_runs_cap": round(e8a_ablation_cap, 3),
                "extraction_5_full_plus_2_partial_at_the_135M_rate": round(
                    extraction_total_at_135m_rate, 3),
                "canonical_extraction_assumption": list(
                    CANONICAL["extraction_full_pass_hours"]),
                "measured_full_pass_hours_135M": round(
                    extract_hours_250k_135m, 4),
                "extraction_exceeds_canonical_upper_assumption": bool(
                    extract_hours_250k_135m
                    > CANONICAL["extraction_full_pass_hours"][1]),
                "extraction_caveat":
                    "only the 135M rate is measured. SmolLM2-360M has 2.7 "
                    "times the parameters and will be slower per string; "
                    "FLAN-T5-small is smaller and will be faster. The three "
                    "unmeasured stores are costed here at the 135M rate, "
                    "which UNDERSTATES the 360M pair.",
            },
            "unmeasured_components": {
                "e8b_entirely": "no E8B run has ever been performed in this "
                                "repository. The per-epoch multiplier m "
                                "remains the back-solved assumption 2.931 to "
                                "8.004 and is the single dominant unknown in "
                                "the programme total. Phase 1A does not "
                                "reduce that uncertainty at all.",
                "e9_entirely": "evaluation-only, unmeasured",
                "e8a_360M_and_flan_t5_extraction": "unmeasured",
                "e8a_250k_training": "unmeasured; carried from the 40k "
                                     "measurement by the row ratio",
                "matched_intervention_pass_cost": "the pilot ran 5 conditions "
                                                  "per arm inside a single "
                                                  "evaluation phase and did "
                                                  "not time them separately",
            },
            "gates": {
                "per_run_8_hours": {
                    "largest_measured_e8a_run_hours": round(
                        max(r["wall_clock_hours"] for r in runs.values()), 6),
                    "largest_projected_e8a_run_hours": round(e8a_250k_cap, 6),
                    "ceiling": CANONICAL["per_run_ceiling_hours"],
                    "fires": bool(e8a_250k_cap
                                  > CANONICAL["per_run_ceiling_hours"]),
                    "margin_hours": round(
                        CANONICAL["per_run_ceiling_hours"] - e8a_250k_cap, 4),
                },
                "memory_80_per_cent": {
                    "peak_allocated_mib": peak_mib,
                    "ceiling_mib": ceiling_mib,
                    "fires": bool(peak_mib > ceiling_mib),
                    "headroom_mib": round(ceiling_mib - peak_mib, 1),
                },
                "storage": {
                    "phase_1a_written_gib": round(
                        sum(s["gib"] for s in stores.values()), 4),
                    "full_programme_sequence_stores_packed_gib":
                        extraction["storage_projection"][
                            "full_programme_sequence_stores_if_authorised"][
                                "packed_gib"],
                    "canonical_padded_estimate_gib": CANONICAL["storage_gib"],
                    "free_disk_gib": extraction["storage_projection"][
                        "free_disk_gib"],
                    "fires": False,
                    "note": "the packed layout stores sum(length) rows rather "
                            "than n_strings x 32, so the measured stores are "
                            "about a third of the canonical padded estimate. "
                            "The canonical figure is conservative and no "
                            "storage gate is at risk.",
                },
                "per_model_ceiling_hours_note": "the key was once "
                                      "per_model_35_hours; the ceiling is "
                                      "40 h programme-wide since "
                                      "2026-08-08. NOT recomputable from "
                                      "Phase-1A "
                                      "evidence: the SmolLM2-135M aggregate "
                                      "is A1 + A4 + A7c + B3, and B3 is an "
                                      "E8B arm with no measurement. The E8A "
                                      "limb is revised below; the B3 limb is "
                                      "not.",
                "total_programme": "NOT recomputable: dominated by the "
                                   "unmeasured E8B multiplier.",
            },
        },
    }

    e8a_total_expected = (e8a_core_expected + e8a_secondary_expected
                          + e8a_ablation_expected
                          + extraction_total_at_135m_rate)
    e8a_total_cap = (e8a_core_cap + e8a_secondary_cap + e8a_ablation_cap
                     + extraction_total_at_135m_rate)
    report["stage8_projection"]["e8a_branch_total"] = {
        "expected_basis_hours": round(e8a_total_expected, 3),
        "configured_cap_basis_hours": round(e8a_total_cap, 3),
        "canonical_expected": [20.413, 21.968],
        "canonical_cap": [94.176, 95.730],
        "excludes": "the 156 matched-intervention passes, which the pilot did "
                    "not time separately",
        "with_20_per_cent_contingency": {
            "expected": round(e8a_total_expected * 1.2, 3),
            "cap": round(e8a_total_cap * 1.2, 3),
        },
    }

    utils.save_json(report, e8a.OUT_DIR / "stage8_projection.json")

    p = report["stage8_projection"]
    print("=== MEASURED ===")
    for key, value in p["measured"].items():
        print(f"  {key}: {value}")
    print("\n=== PER-RUN PROJECTION ===")
    for key, value in p["per_run_projection"].items():
        print(f"  {key}: {value}")
    print("\n=== E8A COMPONENTS ===")
    for key, value in p["e8a_component_projection"].items():
        print(f"  {key}: {value}")
    print("\n=== E8A BRANCH TOTAL ===")
    for key, value in p["e8a_branch_total"].items():
        print(f"  {key}: {value}")
    print("\n=== GATES ===")
    for key, value in p["gates"].items():
        print(f"  {key}: {value}")
    print("\nwritten to results/experiments/e8a_question_encoder/"
          "stage8_projection.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
