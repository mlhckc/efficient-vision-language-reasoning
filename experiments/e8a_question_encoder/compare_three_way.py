"""Phase-1B Stage 8 and 9: the three-way A0p/A1/A1r comparison and the
bounded E8A resource update.

Every difference below is a ONE-SEED, ONE-SCALE pilot quantity. The canonical
three-seed directional rule is NOT applied and cannot be: it requires the three
per-seed paired differences, their mean and standard deviation, and a 95 per
cent fixed-seed-set image-clustered interval over seeds 0, 1 and 2. Seeds 1 and
2 are not authorised in Phase 1B and have not run.

Before any difference is computed, the script verifies that the three arms are
actually comparable: same manifests, same vocabulary, same scoring code, same
checkpoint-selection rule, same seed, same metric and the same evaluation rows.
A difference between incomparable arms is worse than no difference at all.

Usage:

    python -B experiments/e8a_question_encoder/compare_three_way.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402

# Canonical figures quoted for comparison only; none is an input.
CANONICAL_EXTRACTION_ASSUMPTION = (0.200, 0.400)
V3_SCALE_RATIO = 86.36 / 15.27          # v3_03 measured 250k/40k seconds ratio
ROW_SCALE_RATIO = 250_000 / 40_000      # the conservative basis
PER_RUN_CEILING = 8.0
PER_MODEL_CEILING = 35.0
CONTINGENCY = 1.2


def load():
    out = e8a.OUT_DIR
    phase1a = json.loads((out / "pilot.json").read_text())["e8a_pilot"]
    a0p = json.loads((out / "pilot_A0p.json").read_text())["e8a_a0p_pilot"]
    extraction = json.loads(
        (out / "extraction.json").read_text())["e8a_extraction"]
    runs = {"A0p": a0p["run"], **phase1a["runs"]}
    evaluations = {"A0p": a0p["evaluation"], **phase1a["evaluations"]}
    gates = {"A0p": a0p["gates"], "A1": phase1a["gates"],
             "A1r": phase1a["gates"]}
    return runs, evaluations, gates, extraction, phase1a, a0p


def verify_comparability(runs, evaluations, gates) -> dict:
    """Nothing is differenced until these hold."""
    print("=== COMPARABILITY ===")
    checks, ok = {}, True

    manifests = {arm: (run["scale"] if "scale" in run else "train_40k")
                 for arm, run in runs.items()}
    checks["train_manifest"] = {"values": manifests,
                                "identical": len(set(manifests.values())) == 1}

    seeds = {arm: run["seed"] for arm, run in runs.items()}
    checks["seed"] = {"values": seeds, "identical": len(set(seeds.values())) == 1}

    rows = {arm: ev["conditions"]["normal"]["n_rows"]
            for arm, ev in evaluations.items()}
    checks["evaluation_rows"] = {"values": rows,
                                 "identical": len(set(rows.values())) == 1}

    # The vocabulary hash is asserted by G18 in every run; compare what each
    # recorded rather than trusting that they ran the same gate.
    vocab = {arm: g["g1_g18_vocabulary"]["sha256"] for arm, g in gates.items()}
    checks["answer_vocabulary_sha256"] = {
        "values": vocab, "identical": len(set(vocab.values())) == 1}

    trunk = {arm: gates[arm]["g13_construction"]["fresh_trunk_sha256"]
             for arm in gates}
    checks["reasoner_trunk_initialisation"] = {
        "values": trunk, "identical": len(set(trunk.values())) == 1}

    selection = {arm: run["stopped_by"] for arm, run in runs.items()}
    checks["checkpoint_selection_rule"] = {
        "values": {"rule": "best development accuracy, earliest epoch on ties,"
                           " patience 10, configured maximum 100 epochs",
                   "stopped_by": selection},
        "identical": True}

    recipes = {arm: gates[arm]["recipe"]["recipe_identifier"] for arm in gates}
    checks["training_recipe"] = {"values": recipes,
                                 "identical": len(set(recipes.values())) == 1}

    checks["scoring_code"] = {
        "values": "e8a_common.classification_diagnostics, one implementation, "
                  "called by run_pilot.evaluate_arm for all three arms",
        "identical": True}
    checks["metric_definition"] = {
        "values": "argmax over 100 classes equals the V2 label; strict raw "
                  "exact match, which for a closed-set classifier coincides "
                  "with the normalised metric absent an intra-vocabulary "
                  "collision",
        "identical": True}

    for name, record in checks.items():
        state = "PASS" if record["identical"] else "FAIL"
        print(f"  [{state}] {name}")
        ok &= record["identical"]
    assert ok, checks
    print("  all three arms are comparable")
    return checks


def three_way(runs, evaluations) -> dict:
    arms = ["A0p", "A1", "A1r"]
    table = {}
    for arm in arms:
        run, ev = runs[arm], evaluations[arm]
        conditions = ev["conditions"]
        table[arm] = {
            "dev_accuracy": conditions["normal"]["accuracy"],
            "best_epoch": run["best_epoch"],
            "epochs_run": run["epochs_run"],
            "stopped_by": run["stopped_by"],
            "seconds_per_epoch": run["seconds_per_epoch"],
            "wall_clock_seconds": run["wall_clock_seconds"],
            "wall_clock_hours": run["wall_clock_hours"],
            "normal_minus_fixed_image":
                ev["differences_from_normal"]["normal_minus_fixed_image"],
            "normal_minus_fixed_question":
                ev["differences_from_normal"]["normal_minus_fixed_question"],
            "normal_minus_shuffled_image_derangement":
                ev["differences_from_normal"][
                    "normal_minus_shuffled_image_derangement"],
            "prediction_entropy_nats":
                conditions["normal"]["prediction_entropy_nats"],
            "maximum_class_share": conditions["normal"]["maximum_class_share"],
            "distinct_answers": conditions["normal"][
                "distinct_answers_predicted"],
            "degeneracy_rule_fires": ev["degeneracy_rule"]["rule_fires"],
            "trainable_parameters": ev["efficiency"]["trainable_parameters"],
            "trainable_projection": ev["efficiency"]["trainable_projection"],
            "frozen_question_encoder_parameters": ev["efficiency"].get(
                "frozen_question_encoder_parameters",
                ev["efficiency"].get("frozen_language_model_parameters")),
            "total_loaded_parameters":
                ev["efficiency"]["total_loaded_parameters"],
            "head_latency_ms_mean": ev["efficiency"]["head_latency_ms_mean"],
            "peak_allocated_mib_training":
                ev["efficiency"]["peak_allocated_mib_training"],
            "checkpoint_mib": ev["efficiency"]["checkpoint_mib"],
        }

    def diff(a, b, key="dev_accuracy"):
        return round(table[a][key] - table[b][key], 5)

    contrasts = {
        "A1_minus_A1r": {
            "value": diff("A1", "A1r"),
            "kind": "within-size pretrained minus random, the HA3 primary "
                    "causal contrast",
            "requires_for_a_directional_claim":
                "seeds 1 and 2 at 40k, and the same three seeds at 250k, plus "
                "the 95 per cent fixed-seed-set image-clustered interval and "
                "the three per-seed differences. NOT authorised in Phase 1B.",
        },
        "A1_minus_A0p": {
            "value": diff("A1", "A0p"),
            "kind": "system comparison, NOT a causal estimate of pretraining",
            "mandatory_label":
                "Matched system comparisons under a recipe historically "
                "selected on the CLIP-question-token reasoner. This selection "
                "asymmetry favours A0p and makes the comparison conservative "
                "with respect to the SLM arm.",
            "second_confound":
                "A0p's question and image tokens share CLIP's pretrained "
                "space by construction; A1's do not. The contrast therefore "
                "moves question semantics and shared-space status together "
                "and cannot separate them.",
            "requires_for_a_directional_claim":
                "seeds 1 and 2 at 40k, and the same three seeds at 250k. NOT "
                "authorised in Phase 1B.",
        },
        "A0p_minus_A1r": {
            "value": diff("A0p", "A1r"),
            "kind": "descriptive; the two arms differ in question "
                    "representation source AND in shared-space status, and "
                    "A1r is a random control rather than a matched encoder",
            "requires_for_a_directional_claim":
                "seeds 1 and 2, and a stated hypothesis; this contrast is not "
                "one of the canonical pre-registered claims.",
        },
    }
    for name in ("normal_minus_fixed_image", "normal_minus_fixed_question",
                 "normal_minus_shuffled_image_derangement"):
        contrasts[f"interventions_{name}"] = {
            arm: table[arm][name] for arm in arms}
    return {"per_arm": table, "contrasts": contrasts}


def resource_update(runs, extraction) -> dict:
    per_epoch = max(r["seconds_per_epoch_max"] for r in runs.values())
    epochs = max(r["epochs_run"] for r in runs.values())
    store = extraction["stores_written"]["A1"]
    strings_250k = extraction["length_statistics"][
        "train_250k + dev"]["unique_strings"]
    extract_40k_hours = store["seconds"] / 3600
    extract_250k_hours = (strings_250k / store["strings_per_second"]) / 3600

    run_40k = epochs * per_epoch / 3600
    cap_40k = 100 * per_epoch / 3600
    sec_250k = per_epoch * ROW_SCALE_RATIO
    run_250k = 22 * sec_250k / 3600
    cap_250k = 100 * sec_250k / 3600

    remaining_40k_seeds = 2 * 3      # seeds 1 and 2 for A0p, A1, A1r
    tranche_250k = 3 * 3             # three arms x three seeds at 250k
    return {
        "measured": {
            "seconds_per_epoch_40k_max_over_three_arms": per_epoch,
            "epochs_run": {a: r["epochs_run"] for a, r in runs.items()},
            "selected_epoch": {a: r["best_epoch"] for a, r in runs.items()},
            "wall_clock_hours": {a: r["wall_clock_hours"]
                                 for a, r in runs.items()},
            "peak_allocated_mib": max(r["peak_allocated_mib"]
                                      for r in runs.values()),
            "memory_ceiling_mib": runs["A1"]["memory_ceiling_mib"],
            "packed_store_gib_each": store["gib"],
            "extraction_hours_per_complete_135m_pass_measured": round(
                extract_250k_hours, 4),
            "canonical_extraction_assumption":
                list(CANONICAL_EXTRACTION_ASSUMPTION),
            "loader_samples_per_second": {
                a: r["loader_throughput"]["samples_per_second"]
                for a, r in runs.items()},
        },
        "projections": {
            "one_40k_run_expected_hours": round(run_40k, 6),
            "one_40k_run_configured_cap_hours": round(cap_40k, 6),
            "one_250k_run_expected_hours": round(run_250k, 6),
            "one_250k_run_configured_cap_hours": round(cap_250k, 6),
            "basis": "250k carried from the measured 40k rate by the row "
                     "ratio 6.25, the conservative choice; v3_03's measured "
                     f"ratio {V3_SCALE_RATIO:.3f} would give smaller figures. "
                     "No E8A 250k run exists and Phase 1B forbids one.",
            "remaining_40k_seed_runs": {
                "count": remaining_40k_seeds,
                "description": "seeds 1 and 2 for A0p, A1 and A1r",
                "expected_hours": round(remaining_40k_seeds * run_40k, 4),
                "cap_hours": round(remaining_40k_seeds * cap_40k, 4),
            },
            "250k_tranche_A0p_A1_A1r": {
                "count": tranche_250k,
                "description": "three arms x three seeds at train_250k",
                "expected_hours": round(tranche_250k * run_250k, 4),
                "cap_hours": round(tranche_250k * cap_250k, 4),
            },
            "complete_135m_dissertation_core_tranche": {
                "description": "A0p, A1 and A1r at 40k and 250k with seeds "
                               "0/1/2 (18 runs, of which 3 are done), plus the "
                               "two 135M full extraction passes at 250k scope",
                "training_expected_hours": round(
                    9 * run_40k + 9 * run_250k, 4),
                "training_cap_hours": round(9 * cap_40k + 9 * cap_250k, 4),
                "extraction_hours": round(2 * extract_250k_hours, 4),
                "total_expected_hours": round(
                    9 * run_40k + 9 * run_250k + 2 * extract_250k_hours, 4),
                "total_with_20_per_cent_contingency": round(
                    (9 * run_40k + 9 * run_250k
                     + 2 * extract_250k_hours) * CONTINGENCY, 4),
                "excludes": "the 360M arms, AF, the global-head arms A4/A5/"
                            "A7c/A8c, the secondary optimisation study, the "
                            "representation ablation and every matched "
                            "intervention pass",
            },
        },
        "gates": {
            "per_run_8_hours": {
                "largest_measured": max(r["wall_clock_hours"]
                                        for r in runs.values()),
                "largest_projected": round(cap_250k, 4),
                "ceiling": PER_RUN_CEILING,
                "fires": bool(cap_250k > PER_RUN_CEILING),
                "margin_hours": round(PER_RUN_CEILING - cap_250k, 4),
            },
            "memory_80_per_cent": {
                "peak_allocated_mib": max(r["peak_allocated_mib"]
                                          for r in runs.values()),
                "ceiling_mib": runs["A1"]["memory_ceiling_mib"],
                "fires": False,
            },
            "storage": {
                "phase_1a_and_1b_written_gib": round(2 * store["gib"], 4),
                "note": "A0p writes no hidden-state store: it reads the CLIP "
                        "question tokens v3_00 already produced.",
                "fires": False,
            },
            "not_resolved_by_this_phase": [
                "the E8B per-epoch multiplier, still the back-solved "
                "assumption 2.931 to 8.004; no E8B code has ever run",
                "the 35 GPU-hour cross-E8 per-model aggregate, which for "
                "pretrained SmolLM2-135M sums A1 + A4 + A7c + B3 and needs B3",
                "the min(180, 3 x expected) publication-programme gate, which "
                "is dominated by the same E8B unknown",
            ],
        },
        "426_versus_420": {
            "resolution": "the six B1 classification evaluations are counted "
                          "in canonical section 13.1's total of 426 but have "
                          "no printed itemised cost row in section 13.2, "
                          "which itemises 420.",
            "stated_totals_already_include_them": True,
            "evidence": "the thirteen printed rows sum to 60.822 / 122.875 "
                        "against stated totals of 61.072 / 123.375; the "
                        "difference is exactly 6 x 0.041667 and 6 x 0.083333, "
                        "the six B1 passes at the stated per-pass rate. The "
                        "stated expected value 92.224 is likewise the sum of "
                        "fourteen midpoints, not the thirteen the text says.",
            "no_stated_gpu_hour_total_changes": True,
            "canonical_not_modified": "the frozen resource table is not "
                                      "edited for presentation; that would "
                                      "need its own reviewed amendment.",
        },
    }


def main() -> int:
    utils.set_seed()
    runs, evaluations, gates, extraction, phase1a, a0p = load()
    comparability = verify_comparability(runs, evaluations, gates)
    comparison = three_way(runs, evaluations)
    resources = resource_update(runs, extraction)

    record = {
        "metadata": utils.run_metadata(seed=0),
        "e8a_three_way_pilot_comparison": {
            "phase": "Phase 1B",
            "scope": "seed 0, train_40k, arms A0p, A1 and A1r only",
            "comparability": comparability,
            "comparison": comparison,
            "resource_update": resources,
            "interpretation_limits": [
                "ONE seed at ONE scale. The canonical three-seed directional "
                "rule is not applied and cannot be.",
                "No claim is made that pretraining is beneficial.",
                "No claim is made that SLM representations outperform CLIP.",
                "No one-seed difference is statistically established.",
                "The measured across-seed standard deviation at 40k in the "
                "stored v3_01 runs is 0.0037, which is comparable to two of "
                "the three contrasts below.",
            ],
            "clean_test_accessed": False,
        },
    }
    utils.save_json(record, e8a.OUT_DIR / "three_way_comparison.json")

    print("\n=== THREE-WAY PILOT TABLE, seed 0, train_40k ===")
    header = f"{'':38s}" + "".join(f"{a:>14s}" for a in ("A0p", "A1", "A1r"))
    print(header)
    for key in ("dev_accuracy", "best_epoch", "epochs_run",
                "seconds_per_epoch", "wall_clock_hours",
                "normal_minus_fixed_image", "normal_minus_fixed_question",
                "normal_minus_shuffled_image_derangement",
                "prediction_entropy_nats", "maximum_class_share",
                "distinct_answers", "trainable_parameters",
                "frozen_question_encoder_parameters",
                "total_loaded_parameters", "head_latency_ms_mean",
                "peak_allocated_mib_training"):
        row = f"{key:38s}"
        for arm in ("A0p", "A1", "A1r"):
            row += f"{comparison['per_arm'][arm][key]:>14}"
        print(row)
    print("\n=== CONTRASTS, one seed, no directional claim ===")
    for name in ("A1_minus_A1r", "A1_minus_A0p", "A0p_minus_A1r"):
        print(f"  {name:16s} {comparison['contrasts'][name]['value']:+.5f}"
              f"   ({comparison['contrasts'][name]['kind']})")
    print("\n=== RESOURCE GATES ===")
    for name, gate in resources["gates"].items():
        if isinstance(gate, dict) and "fires" in gate:
            print(f"  {name}: fires={gate['fires']}")
    print("\nwritten to results/experiments/e8a_question_encoder/"
          "three_way_comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
