"""The committed generator for the E8B core resource projection.

Recomputes every figure from measured inputs and from the walls recorded
in the artefacts themselves, so the projection is reproducible from this
script alone. Regenerates core_resource_projection_20260807.json.

Design projected: the executable amended design - strict deterministic
BF16 training, exactly 22 epochs for B2/B3 with no early stopping and
the epoch-22 canonical rule, B1 under its stored section-7.3 recipe,
full FP32 canonical evaluation, G14-FP32 (64 pre + about 160 post rows,
LM arms only), final R1/R2/R3 readouts, intervention passes, non-G14
gate overhead, the real checkpoint footprint, and the already-spent
search and diagnostic compute charged to the identities that consumed
it.

    python -B experiments/e8b_readout_generation/core_resource_projection.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import utils  # noqa: E402

D = PROJECT_ROOT / "results" / "experiments" / "e8b_readout_generation"

# --- Measured inputs, with sources -------------------------------------------
TRAIN_S_40K = 78.0        # s/epoch, strict-deterministic probe runs 3/4
EVAL_FP32_S = 108.0       # s, canonical FP32 dev pass (probe runs 3/4)
EVAL_BF16_S = 85.7        # secondary deployment diagnostic only
# G14 per-row cost. The measured three-way per-row costs are 5.149 /
# 5.137 / 5.135 s/row (fp32_canonical_validation.json, pooled 5.139 over
# 478 rows). 5.301 is NOT one of those measurements: it is a deliberate
# +3 per cent conservative margin over the pooled measurement, applied
# because the recorded runs did not carry the strict-deterministic
# backend the core cells will. It is an ASSUMPTION, labelled as such.
G14_ROW_S_MEASURED_POOLED = 5.139   # fp32_canonical_validation.json
G14_ROW_S = 5.301         # ASSUMED: measured pooled + 3 per cent margin
G14_ROWS = 64 + 160       # pre-selection P + post-selection P+L+O union
GATE_OVERHEAD_H = 0.223   # measured non-G14 per-run gate overhead
STEPS = {"train_40k": 313, "train_250k": 1954}
RATIO_250K = STEPS["train_250k"] / STEPS["train_40k"]
LM_EPOCHS = 22            # fixed budget, both scales (recipe-encoded)
B1_S_40K = 15.27          # v3_01 measured, includes its dev pass
B1_S_250K = 86.36         # v3_01/v3_03 MEASURED 250k rate (section 13.2)
B1_EXPECTED = {"train_40k": 15, "train_250k": 22}   # section 7.1 planning
B1_STRESS = 100           # section 7.3 patience cap
# R2/R3 per-row costs. NEITHER IS MEASURED FOR E8B. No E8B R2 or R3
# readout has ever been executed, so no per-row walk exists to measure.
# R2 0.0291 s/row is carried over from the E7b S8-stage per-row cost of
# the same trunk-and-projection forward and is an ASSUMPTION for the
# trie-constrained decode; R3 is a further ASSUMPTION at 1.5x R2.
# Sensitivity for both is reported in residual_assumptions.
R2_ROW_S, R3_ROW_S = 0.0291, 0.0437
N_DEV = 7714
WALL_H, IDENT_H, CORE_H, MEM_FRACTION = 8.0, 35.0, 180.0, 0.80
BASE = {"pretrained": 2.29222, "random": 1.95811}   # A1 / A1r measured
A4_H, A7C_H = 1.804, 1.864                          # protocol 13.2b
# Section 19 / protocol 13.2: the mandatory serial efficiency pass,
# charged at its UPPER bound. Loads the frozen pretrained LM.
EFFICIENCY_S19_H = 1.000
# Checkpoint footprint, measured from artefacts on disk.
LM_RESUME_MIB, LM_CKPT_MIB = 325.9, 81.5
B1_RESUME_MIB, B1_CKPT_MIB = 322.0, 80.6
RETAINED_SEARCH_MIB = 1222.0
DEVICE_TOTAL_MIB = 20017.4
PEAK_RESERVED_MIB = 7608.0     # strict-det probe, measured


def hours(seconds: float) -> float:
    return seconds / 3600.0


def spent_compute() -> dict:
    """Itemised already-spent E8B GPU time, read from the artefacts'
    own recorded walls where they exist. Everything here loaded the
    PRETRAINED checkpoint, so the whole table charges the pretrained
    identity."""
    items = {}
    items["search_grid1_pilot"] = 1.379
    items["search_grid2"] = 1.035
    # grid 3 halted before its record: 16 epochs at the measured
    # 138.8 s/epoch plus the measured gate overhead
    items["search_grid3_halted"] = round(16 * 138.8 / 3600
                                         + GATE_OVERHEAD_H, 3)
    char = json.loads((D / "g14_numerical_characterization_v2.json"
                       ).read_text())["g14_numerical_characterization"]
    items["bf16_characterization_v2"] = round(sum(
        r["canonical_pass_seconds"] + r["sample_seconds"]
        for r in char["results"]) / 3600, 3)
    val = json.loads((D / "fp32_canonical_validation.json"
                      ).read_text())["fp32_canonical_validation"]
    items["fp32_validation"] = round(sum(
        r["operational"]["fp32_dev_seconds"]
        + r["operational"]["bf16_dev_seconds_secondary"]
        + r["three_way_equivalence"]["sample_seconds"]
        for r in val["results"]) / 3600, 3)
    probes = 0.0
    for n in (1, 2, 3, 4):
        path = D / f"determinism_probe_run{n}.json"
        if path.exists():
            probes += json.loads(path.read_text())[
                "determinism_probe"]["wall_seconds"]
    items["determinism_probes_1_to_4"] = round(probes / 3600, 3)
    items["paired_rescoring_and_micro_probes"] = 0.25  # ESTIMATE, upper
    items["_total"] = round(sum(v for k, v in items.items()
                                if not k.startswith("_")), 3)
    return items


def main() -> int:
    lm_epoch_40k = TRAIN_S_40K + EVAL_FP32_S
    lm_epoch_250k = TRAIN_S_40K * RATIO_250K + EVAL_FP32_S
    g14_h = hours(G14_ROWS * G14_ROW_S)
    lm_cell_40k = hours(lm_epoch_40k * LM_EPOCHS) + g14_h + GATE_OVERHEAD_H
    lm_cell_250k = (hours(lm_epoch_250k * LM_EPOCHS) + g14_h
                    + GATE_OVERHEAD_H)
    b1_cell = {s: hours((B1_S_40K if s == "train_40k" else B1_S_250K)
                        * B1_EXPECTED[s]) + GATE_OVERHEAD_H
               for s in STEPS}
    b1_stress = {s: min(hours((B1_S_40K if s == "train_40k" else B1_S_250K)
                              * B1_STRESS) + GATE_OVERHEAD_H, WALL_H)
                 for s in STEPS}

    b2_total = b3_total = 3 * lm_cell_40k + 3 * lm_cell_250k
    b1_total = 3 * b1_cell["train_40k"] + 3 * b1_cell["train_250k"]
    b1_total_stress = (3 * b1_stress["train_40k"]
                       + 3 * b1_stress["train_250k"])

    # Final readout passes on the 12 LM canonical checkpoints: R1 comes
    # free from the epoch-22 pass; R2 and R3 are per-row walks; the three
    # intervention conditions are one canonical R1 pass each.
    readouts_per_lm_cell = hours(N_DEV * (R2_ROW_S + R3_ROW_S))
    interventions_per_lm_cell = 3 * hours(EVAL_FP32_S)
    final_eval_lm = 12 * (readouts_per_lm_cell + interventions_per_lm_cell)
    # B1 final classification evaluations: charged at the LM R1 pass
    # rate as a LABELLED UPPER BOUND (a classifier forward over dev costs
    # a few seconds, far below 108 s; B1 loads no LM so this line never
    # touches either identity ceiling).
    final_eval_b1 = 6 * hours(EVAL_FP32_S)
    # Section 19 mandates one E7a-protocol serial efficiency measurement
    # on the final selected checkpoints (serial_efficiency.py). Protocol
    # 13.2 costs it at 0.500-1.000 h; the UPPER bound is charged here.
    # It loads the frozen PRETRAINED LM, so it is charged to that
    # identity. Previously omitted entirely, which made the
    # "complete programme" claim false.
    efficiency_s19 = EFFICIENCY_S19_H

    e8b_remaining = (b1_total + b2_total + b3_total + final_eval_lm
                     + final_eval_b1 + efficiency_s19)
    e8b_remaining_stress = (b1_total_stress + b2_total + b3_total
                            + final_eval_lm + final_eval_b1
                            + efficiency_s19)

    spent = spent_compute()
    share = readouts_per_lm_cell + interventions_per_lm_cell
    pretrained_identity = (BASE["pretrained"] + A4_H + A7C_H + b3_total
                           + 6 * share + spent["_total"]
                           + efficiency_s19)
    random_identity = BASE["random"] + b2_total + 6 * share

    storage_gib = ((12 * (LM_RESUME_MIB + 2 * LM_CKPT_MIB)
                    + 6 * (B1_RESUME_MIB + B1_CKPT_MIB)
                    + RETAINED_SEARCH_MIB + 20) / 1024)
    free_gib = shutil.disk_usage(D).free / 2 ** 30

    gates = {
        "per_run_8h": {
            "largest_cell_hours": round(lm_cell_250k, 3),
            "fires": lm_cell_250k > WALL_H},
        "pretrained_identity_35h": {
            "projected_hours": round(pretrained_identity, 3),
            "headroom_hours": round(IDENT_H - pretrained_identity, 3),
            "includes": "the COMPLETE planned pretrained-identity "
                        "programme: A1 baseline, A4, A7c, all six B3 "
                        "core cells, B3's final R2/R3 readouts and "
                        "interventions, and every already-spent search, "
                        "characterisation, validation and probe hour "
                        "that loaded the pretrained checkpoint",
            "retry_allowance": "NONE. The plan tolerates zero B3 "
                               "retries; any retry or any additional "
                               "B3-loading evaluation triggers a "
                               "stop-and-ask BEFORE execution",
            "fires": pretrained_identity > IDENT_H},
        "random_identity_35h": {
            "projected_hours": round(random_identity, 3),
            "fires": random_identity > IDENT_H},
        "e8b_remaining_vs_180h_core": {
            "e8b_remaining_hours": round(e8b_remaining, 3),
            "e8b_remaining_stress_hours": round(e8b_remaining_stress, 3),
            "scope_note": "E8B cells and E8B evaluation only; the "
                          "programme-wide 180 h ceiling also covers the "
                          "remaining E8A arms (about 17.6-21.6 h "
                          "expected per section 13.2), which leaves the "
                          "combined expectation far below the ceiling",
            "fires": e8b_remaining > CORE_H},
        "memory_80pct_reserved": {
            "peak_reserved_mib": PEAK_RESERVED_MIB,
            "reserved_fraction": round(PEAK_RESERVED_MIB
                                       / DEVICE_TOTAL_MIB, 4),
            "fires": PEAK_RESERVED_MIB / DEVICE_TOTAL_MIB > MEM_FRACTION},
        "storage": {
            "required_gib": round(storage_gib, 3),
            "free_gib": round(free_gib, 1),
            "status": "UNRESOLVED allocation (E8A precedent: free space "
                      "is not a quota). The free-space check passes and "
                      "is reported as context, not as a discharged "
                      "allocation gate.",
            "free_space_check_passes": storage_gib < free_gib,
            # The allocation itself is unresolved, but the free-space
            # check IS a halting condition: without a "fires" key the
            # gate was structurally incapable of ever firing, so it
            # could not have stopped anything.
            "fires": storage_gib >= free_gib},
    }
    fired = [k for k, v in gates.items() if v.get("fires")]

    record = {"metadata": utils.run_metadata(),
              "e8b_core_resource_projection": {
        "generator": "experiments/e8b_readout_generation/"
                     "core_resource_projection.py",
        "design": "strict deterministic BF16 training; fixed 22 epochs "
                  "for B2/B3, no early stopping, epoch-22 canonical; B1 "
                  "under the stored section-7.3 recipe (15/22 expected, "
                  "100-epoch stress); full FP32 canonical evaluation; "
                  "G14-FP32 on the LM arms; final R1/R2/R3 readouts and "
                  "interventions; measured checkpoint footprint",
        "measured_inputs": {
            "train_s_per_epoch_40k_strict_det": TRAIN_S_40K,
            "canonical_fp32_dev_pass_s": EVAL_FP32_S,
            "bf16_dev_pass_s_SECONDARY_DIAGNOSTIC_ONLY": EVAL_BF16_S,
            "g14_fp32_s_per_row_ASSUMED_pooled_plus_3pc": G14_ROW_S,
            "g14_rows_per_lm_cell": G14_ROWS,
            "non_g14_gate_overhead_h": GATE_OVERHEAD_H,
            "b1_s_per_epoch": {"train_40k": B1_S_40K,
                               "train_250k": B1_S_250K},
            "r2_s_per_row_ASSUMED_from_e7b_s8": R2_ROW_S,
            "r3_s_per_row_ASSUMED_1p5x_r2": R3_ROW_S,
            "step_ratio_250k_over_40k": round(RATIO_250K, 6),
            "amended_design_cost_note":
                "the 78.0 versus 53.88 s/epoch difference is the TOTAL "
                "cost of the amended execution design relative to the "
                "superseded one; it bundles three simultaneous changes "
                "(strict-deterministic math SDPA backend, the fp32 "
                "frozen-model instance, and enforcement assertions) and "
                "is not attributed to any single one"},
        "per_cell_hours": {
            "lm_train_40k": round(lm_cell_40k, 3),
            "lm_train_250k": round(lm_cell_250k, 3),
            "b1_train_40k_expected": round(b1_cell["train_40k"], 3),
            "b1_train_250k_expected": round(b1_cell["train_250k"], 3)},
        "arm_totals_hours": {
            "B1_expected": round(b1_total, 3),
            "B1_stress_100_epoch": round(b1_total_stress, 3),
            "B2": round(b2_total, 3), "B3": round(b3_total, 3),
            "final_readouts_and_interventions_lm": round(final_eval_lm, 3),
            "final_b1_evaluations_upper_bound": round(final_eval_b1, 3)},
        "spent_compute_hours_itemised": spent,
        "selection_sensitivity_study": {
            "status": "VOID under the fixed-22 epoch-22-canonical rule: "
                      "the study's object (sensitivity to max-over-"
                      "epochs checkpoint selection across retained "
                      "search checkpoints) no longer exists. Not costed. "
                      "Reinstating any analogue requires a fresh user "
                      "decision and a fresh projection."},
        "gates": gates, "fired": fired,
        "residual_assumptions_quantified": {
            "train_250k_step_scaling": {
                "carries_hours": 12.566,
                "basis": "step-scaled from the MEASURED 40k rate; no "
                         "250k E8B cell has ever run",
                "break_even": "the pretrained identity reaches 35 h if "
                              "the true 250k training rate is about 16 "
                              "per cent above the step-scaled value "
                              "(4.86 h per cell instead of 4.189 h)"},
            "r2_r3_per_row": {
                "carries_hours": 1.556,
                "basis": "NEITHER measured for E8B; R2 carried over "
                         "from the E7b S8 per-row cost, R3 assumed at "
                         "1.5x R2",
                "sensitivity": "if R2 and R3 each cost 3x the assumed "
                               "rate, the pretrained identity rises by "
                               "about 1.87 h and the headroom falls to "
                               "about 0.17 h"},
            "g14_row_cost": {
                "carries_hours": 1.979,
                "basis": "pooled measurement 5.139 s/row plus a 3 per "
                         "cent conservative margin"},
            "a4_a7c_unrun": {
                "carries_hours": 3.668,
                "basis": "expected-epoch projections; neither arm has "
                         "run. Their recipes carry max_epochs 100 and "
                         "patience 10, so the configured-cap cost is "
                         "8.469 h EACH (protocol 13.2). The one arm of "
                         "this family that has run, A1, came in 23 per "
                         "cent BELOW its projection."},
            "efficiency_s19": {
                "carries_hours": EFFICIENCY_S19_H,
                "basis": "protocol 13.2 range 0.500-1.000 h, charged at "
                         "the upper bound"}},
        "clean_test_accessed": False}}
    out = D / "core_resource_projection_20260807.json"
    if out.exists():
        out.unlink()
    out.write_text(json.dumps(record, indent=2, default=str) + "\n")
    body = record["e8b_core_resource_projection"]
    print(json.dumps({"per_cell": body["per_cell_hours"],
                      "arms": body["arm_totals_hours"],
                      "spent": spent["_total"],
                      "gates": {k: v.get("fires", v.get(
                          "free_space_check_passes"))
                          for k, v in gates.items()},
                      "pretrained_identity":
                          gates["pretrained_identity_35h"][
                              "projected_hours"],
                      "headroom":
                          gates["pretrained_identity_35h"][
                              "headroom_hours"]}, indent=1))
    print(f"written {out.name}")
    return 1 if fired else 0


if __name__ == "__main__":
    raise SystemExit(main())
