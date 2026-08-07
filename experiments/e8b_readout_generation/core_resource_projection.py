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
# MAX over the strict-deterministic probe runs, following the project's
# own convention elsewhere ("max over three seeds"). Run 3 gives
# 77.5/78.0/78.0 and run 4 gives 78.3; the earlier 78.0 was the low end.
TRAIN_S_40K = 78.3        # s/epoch, MAX over probe runs 3 and 4
# MAX over every recorded canonical FP32 dev pass. The probes give
# 107.9 (run 3) and 109.2-109.4 (run 4); the only STANDALONE
# measurements, in fp32_canonical_validation.json, are 112.0/112.2/112.3.
# The earlier 108.0 was a rounded mid of the lowest set.
EVAL_FP32_S = 112.3       # s, MAX over all recorded canonical FP32 passes
EVAL_BF16_S = 85.7        # secondary deployment diagnostic only
# G14 per-row cost. The measured per-row costs are 5.149 / 5.137 /
# 5.135 s/row (fp32_canonical_validation.json, pooled 5.1387 over 478
# rows). 5.301 is NOT one of those measurements: it is the pooled figure
# plus a 3.15 per cent margin. It is an ASSUMPTION, labelled as such.
#
# DISCLOSED SHORTFALL: the measurement covers R1 brute force plus R1
# cached only, while the LIVE gate additionally runs r2_brute_force and
# r2_cached per row (clause C3). A 3.15 per cent margin nominally
# covering the deterministic backend does NOT cover two extra
# constrained walks over 224 rows x 12 cells, so this input is
# optimistic by an unquantified amount, not conservative. It is an open
# risk alongside the unsourced R2/R3 rates.
G14_ROW_S_MEASURED_POOLED = 5.139   # fp32_canonical_validation.json
G14_ROW_S = 5.301         # ASSUMED: measured pooled + 3 per cent margin
G14_ROWS = 64 + 160       # pre-selection P + post-selection P+L+O union
GATE_OVERHEAD_H = 0.223   # measured non-G14 per-run gate overhead
STEPS = {"train_40k": 313, "train_250k": 1954}
RATIO_250K = STEPS["train_250k"] / STEPS["train_40k"]
# MEASURED on 2026-08-07 (throughput_calibration_20260807.json): 60 timed
# steps on the real train_250k loader and the real strict-deterministic
# BF16 path, after 20 warmup steps. B2 was used and its validity as a
# proxy for B3 was proven both statically (identical compute graph) and
# empirically (0.2343 against 0.2348 s/step at 40k, within noise).
# The P90 step time is used rather than the mean, and then CORRECTED by
# the same-scale bias: extrapolating a steady-state step window to an
# epoch under-predicts by 3.8 per cent measured against twelve real
# full-epoch times at 40k, because the real loop also pays the scheduler
# step, the wall check and a per-step host synchronisation. After
# correction the measured 250k rate is ABOVE the step-scaled assumption
# it replaces, not below it.
TRAIN_S_250K_MEASURED = 492.5   # s/epoch, bias-corrected
TRAIN_S_250K_MEAN = 461.99   # uncorrected, reported
TRAIN_S_250K_ASSUMED_STEP_SCALED = 78.3 * RATIO_250K
LM_EPOCHS = 22            # fixed budget, both scales (recipe-encoded)
B1_S_40K = 15.27          # v3_01 measured, includes its dev pass
B1_S_250K = 86.36         # v3_01/v3_03 MEASURED 250k rate (section 13.2)
B1_EXPECTED = {"train_40k": 15, "train_250k": 22}   # section 7.1 planning
B1_STRESS = 100           # section 7.3 patience cap
# MEASURED on 2026-08-07 (readout_cost_measurement_20260807.json): the
# complete evaluation pass -- prefix, R1, R2 and R3 for every row -- run
# end to end at batch 128 over rows sampled across the raw denominator.
# Measuring the pass as it actually runs, rather than composing per-part
# timings, is what caught the real error here: the pipeline had been
# costed with the per-row R1 cross-check scorer (1.42 s/row) instead of
# the canonical batched one (0.093 s/row at batch 16, less at 128).
EVAL_PASS_S_PER_ROW = 0.075127
# The per-part warm figures, retained for reporting. Each includes the
# prefix-cache forward that readout must build for itself: r2_cached
# CONSUMES the state it is given, so R2 and R3 cannot share one.
R2_ROW_S_MEASURED, R3_ROW_S_MEASURED = 0.031099, 0.030677

# SUPERSEDED. The per-row costs below were UNSOURCED. No E8B R2 or R3 readout has
# ever been executed, so no per-row walk exists to measure.
#
# CORRECTION of 2026-08-07: an earlier version of this comment claimed
# R2 0.0291 s/row was "carried over from the E7b S8-stage per-row cost
# of the same trunk-and-projection forward". THAT WAS FALSE. The E7b
# figure is S8_projection = 0.0291 MILLISECONDS (pilot_e8a_a1.json,
# runs[2].stage_medians_ms), a LayerNorm-plus-Linear projection stage --
# a different operation, in units 1000x apart. The matching digits are a
# coincidence and are not a basis. No replacement provenance has been
# invented: the number is retained only so the projection remains
# comparable with its own history, and is declared UNSOURCED.
#
# A trie-constrained decode walks up to four sequential LM forwards per
# row unbatched, so 29.1 ms/row is plausible in order of magnitude but
# rests on nothing measured. The exposure is quantified in
# residual_assumptions_quantified and is an OPEN RISK for the user to
# accept or to close with a measurement before core execution.
R2_ROW_S, R3_ROW_S = 0.0291, 0.0437   # superseded by measurement
N_DEV = 7714       # in-vocabulary development rows
# Canonical plan section 6: "Every readout evaluation runs once over the
# 10,004-row raw denominator", and section 6.1 requires every final
# checkpoint to receive raw-distribution evaluation. Budgeting the
# readouts at N_DEV understated them; HB3b was neither implemented nor
# costed.
N_RAW = 10004      # raw development denominator (plan section 6)
WALL_H, IDENT_H, CORE_H, MEM_FRACTION = 8.0, 35.0, 180.0, 0.80
BASE = {"pretrained": 2.29222, "random": 1.95811}   # A1 / A1r CORE measured
# Protocol 13.2b's A1 row is 2.987 h = core 1.774 + secondary 0.509 +
# ablation 0.127 + extraction 0.400 + mid 0.087 + interventions 0.090.
# BASE above is the MEASURED CORE run only. Substituting it for the full
# row silently dropped FOUR retained, unrun components that load the
# pinned pretrained SmolLM2-135M and are therefore charged to that
# identity: the 7.1b secondary optimisation study (retained but demoted,
# selects nothing), the bounded representation ablation, and the
# middle-layer extraction, and A1's own matched interventions. None is
# descoped, so all four are charged.
# The retained but unrun parts of protocol 13.2b's A1 row. FOUR
# components, not three: an earlier version of this comment said three,
# directly above the paragraph that adds the fourth.
# BASE covers protocol 13.2b's A1 core + extraction only: the measured
# 2.29222 h decomposes exactly as training 1.56262 + extraction 0.729599
# (resource_correction_g21.json), and that record states interventions
# are "reported separately from, and never added to, the historical
# training figure". So A1's interventions row is charged here too.
RETAINED_A1_ROW = {
    "secondary_optimisation_7_1b": 0.509,
    "representation_ablation": 0.127,
    "middle_layer_extraction": 0.087,
    "a1_matched_interventions": 0.090,
}
RETAINED_A1R_ROW = {          # the A1r row retains the same components
    "representation_ablation": 0.127,
    "middle_layer_extraction": 0.087,
    "a1r_matched_interventions": 0.090,
}
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
    # The 2026-08-07 readiness phase, read from its own records. Only
    # the parts that loaded the PRETRAINED language model are charged
    # here; the calibration's B2 share is charged to the random identity
    # and the CLIP-only token extension touches neither.
    readout = D / "readout_cost_measurement_20260807.json"
    if readout.exists():
        items["readout_cost_measurement_20260807"] = json.loads(
            readout.read_text())["e8b_readout_cost_measurement"][
                "measurement_cost"]["hours"]
    calibration = D / "throughput_calibration_20260807.json"
    if calibration.exists():
        items["throughput_calibration_pretrained_share"] = json.loads(
            calibration.read_text())["e8b_throughput_calibration"][
                "calibration_cost"][
                    "charged_to_pretrained_identity_hours"]
    validation = D / "evaluation_pipeline_validation_20260807.json"
    if validation.exists():
        # The pipeline validation loaded the pretrained model. Its wall
        # was not instrumented, so it is charged at a deliberate upper
        # bound rather than omitted.
        items["evaluation_pipeline_validation_UPPER_BOUND"] = 0.05
    items["_total"] = round(sum(v for k, v in items.items()
                                if not k.startswith("_")), 3)
    return items


def main() -> int:
    lm_epoch_40k = TRAIN_S_40K + EVAL_FP32_S
    # MEASURED, not step-scaled. This was the single largest unmeasured
    # quantity in the plan and the one whose break-even was tightest.
    lm_epoch_250k = TRAIN_S_250K_MEASURED + EVAL_FP32_S
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
    # MEASURED. One evaluation pass computes the prefix once and runs
    # R1, R2 and R3 from it over the RAW denominator, and the plan
    # requires FOUR matched conditions -- normal, fixed image, fixed
    # question and deranged image -- on every trained checkpoint. So a
    # cell's final evaluation is four complete passes, not a readout
    # cost plus a separate intervention cost.
    eval_pass_hours = hours(N_RAW * EVAL_PASS_S_PER_ROW)
    readouts_per_lm_cell = eval_pass_hours           # the normal pass
    interventions_per_lm_cell = 3 * eval_pass_hours  # the three others
    raw_r1_per_cell = eval_pass_hours
    final_eval_lm = 12 * (readouts_per_lm_cell + interventions_per_lm_cell)
    # B1 final classification evaluations: charged at the LM R1 pass
    # rate as a LABELLED UPPER BOUND (a classifier forward over dev costs
    # a few seconds, far below 108 s; B1 loads no LM so this line never
    # touches either identity ceiling).
    # Plan section 7: EVERY trained checkpoint, B1 included, is
    # evaluated under normal, matched fixed-image, matched
    # fixed-question and shuffled-image conditions. The plan's "90
    # passes" figure is the pre-U1 basis of 30 core runs; the matrix is
    # now 18 cells, so 54 passes. The arithmetic below is per cell and
    # is unaffected. Only six plain B1 passes were costed before,
    # omitting B1's 18 intervention passes and its raw pass.
    # B1 is a classifier: no LM, no R2, no R3. Its four conditions cost
    # a canonical dev pass each, charged over the raw denominator as a
    # labelled UPPER BOUND.
    b1_pass = hours(EVAL_FP32_S * N_RAW / N_DEV)
    final_eval_b1 = 6 * 4 * b1_pass
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
    pretrained_identity = (BASE["pretrained"] + sum(RETAINED_A1_ROW.values())
                           + A4_H + A7C_H + b3_total
                           + 6 * share + spent["_total"]
                           + efficiency_s19)
    random_identity = (BASE["random"] + sum(RETAINED_A1R_ROW.values())
                       + b2_total + 6 * share)

    storage_gib = ((12 * (LM_RESUME_MIB + 2 * LM_CKPT_MIB)
                    + 6 * (B1_RESUME_MIB + B1_CKPT_MIB)
                    + RETAINED_SEARCH_MIB + 20) / 1024)
    free_gib = shutil.disk_usage(D).free / 2 ** 30

    headroom = IDENT_H - pretrained_identity
    gates = {
        "per_run_8h": {
            "largest_cell_hours": round(lm_cell_250k, 3),
            "fires": lm_cell_250k > WALL_H},
        "pretrained_identity_35h": {
            "projected_hours": round(pretrained_identity, 3),
            "ceiling_hours": IDENT_H,
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
            "ceiling_hours": IDENT_H,
            "fires": random_identity > IDENT_H},
        "e8b_remaining_vs_180h_core": {
            "e8b_remaining_hours": round(e8b_remaining, 3),
            "e8b_remaining_stress_hours": round(e8b_remaining_stress, 3),
            "scope_note": "E8B cells and E8B evaluation only. The "
                          "programme-wide 180 h ceiling also covers the "
                          "remaining E8A arms. CORRECTED 2026-08-07: an "
                          "earlier note cited 'about 17.6-21.6 h "
                          "expected per section 13.2'. That range "
                          "appears nowhere in the protocol. Section "
                          "13.2's E8A rows total 20.413-21.968 h for "
                          "ALL of E8A, of which 5.789 h is already "
                          "spent, leaving about 14.6-16.2 h remaining. "
                          "The combined expectation is still far below "
                          "the ceiling.",
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
        "inputs_note": "NOT every entry below is measured. Keys ending "
                       "ASSUMED or UNSOURCED are not measurements; the "
                       "container name is retained for continuity with "
                       "earlier records.",
        "measured_inputs": {
            "efficiency_s19_h": EFFICIENCY_S19_H,
            "train_s_per_epoch_40k_strict_det": TRAIN_S_40K,
            "canonical_fp32_dev_pass_s": EVAL_FP32_S,
            "bf16_dev_pass_s_SECONDARY_DIAGNOSTIC_ONLY": EVAL_BF16_S,
            "g14_fp32_s_per_row_ASSUMED_pooled_plus_3pc": G14_ROW_S,
            "g14_rows_per_lm_cell": G14_ROWS,
            "non_g14_gate_overhead_h": GATE_OVERHEAD_H,
            "b1_s_per_epoch": {"train_40k": B1_S_40K,
                               "train_250k": B1_S_250K},
            "r2_s_per_row_UNSOURCED_no_measurement_exists": R2_ROW_S,
            "r3_s_per_row_ASSUMED_1p5x_r2": R3_ROW_S,
            "step_ratio_250k_over_40k": round(RATIO_250K, 6),
            "amended_design_cost_note":
                f"the {TRAIN_S_40K} versus 53.88 s/epoch difference is "
                f"the TOTAL "
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
            "DERIVATION": "every figure in this block is computed from "
                          "the same constants as the projection above. "
                          "An earlier version hardcoded them, so they "
                          "silently went stale when the constants "
                          "changed and understated the exposure.",
            "train_250k_MEASURED": {
                "carries_hours": round(3 * lm_cell_250k, 3),
                "basis": "MEASURED on 2026-08-07 on the real "
                         "train_250k loader and the real "
                         "strict-deterministic BF16 path, then "
                         "CORRECTED by the same-scale bias: a "
                         "steady-state step window under-predicts a "
                         "real epoch by 3.8 per cent, measured against "
                         "twelve real full-epoch times at 40k. After "
                         "correction the rate is ABOVE the step-scaled "
                         "assumption it replaced. No full 250k E8B "
                         "epoch has been run end to end; the residual "
                         "is that the bias correction is itself "
                         "measured at the other scale.",
                "break_even_per_cell_percent": round(
                    100 * headroom / (3 * lm_cell_250k), 2),
                "break_even_on_training_rate_percent": round(
                    100 * headroom
                    / (3 * hours(lm_epoch_250k * LM_EPOCHS)), 2),
                "note": "the per-cell break-even is smaller than the "
                        "training-rate one because each cell also "
                        "carries a fixed G14 and gate overhead that "
                        "does not scale with the training rate"},
            "evaluation_pass_MEASURED": {
                "carries_hours": round(6 * 4 * hours(
                    N_RAW * EVAL_PASS_S_PER_ROW), 3),
                "s_per_row": EVAL_PASS_S_PER_ROW,
                "basis": "MEASURED end to end at batch 128 on "
                         "2026-08-07, after the correctness fix that "
                         "gave R2 and R3 their own prefix states. This "
                         "REPLACES the unsourced R2/R3 assumption "
                         "(0.0291 and 0.0437 s a row, whose claimed E7b "
                         "provenance was FALSE -- that figure is "
                         "MILLISECONDS for a different operation). The "
                         "assumption no longer enters the budget.",
                "residual_risk": "the measurement is at batch 128, the "
                                 "project default everywhere. At batch "
                                 "16 the same pass costs 0.1405 s a "
                                 "row, which would roughly double this "
                                 "line, so the batch size is pinned and "
                                 "asserted rather than left to a "
                                 "caller.",
                "status": "MEASURED; the residual is the batch size, "
                          "which is pinned"},
            "g14_row_cost": {
                "carries_hours": round(12 * g14_h, 3),
                "basis": f"pooled measurement "
                         f"{G14_ROW_S_MEASURED_POOLED} s/row plus a "
                         f"{round(100 * (G14_ROW_S / G14_ROW_S_MEASURED_POOLED - 1), 2)} "
                         f"per cent margin",
                "direction": "OPTIMISTIC, NOT CONSERVATIVE. The pooled "
                             "measurement covers R1 brute force plus R1 "
                             "cached only, while the live gate also "
                             "runs r2_brute_force and r2_cached per row "
                             "under clause C3. The margin does not "
                             "cover two extra constrained walks over "
                             f"{G14_ROWS} rows and 12 cells.",
                "status": "OPEN and unquantified"},
            "a4_a7c_unrun": {
                "carries_hours": round(A4_H + A7C_H, 3),
                "basis": "expected-epoch projections; neither arm has "
                         "run. Their recipes carry max_epochs 100 and "
                         "patience 10, so the configured-cap cost is "
                         "8.469 h EACH (protocol 13.2).",
                "precedent": "A1 is the one arm of this family that has "
                             "run. LIKE FOR LIKE against the components "
                             "the measurement ACTUALLY COVERS -- core "
                             "1.774 + extraction 0.400 = 2.174 h, since "
                             "resource_correction_g21.json decomposes "
                             "the measured 2.29222 h as training "
                             "1.56262 + extraction 0.729599 and states "
                             "interventions are reported separately and "
                             "never added -- A1 came in 5.44 per cent "
                             "ABOVE projection, and its extraction "
                             "component alone overran by 82.4 per cent. "
                             "Two earlier versions of this record were "
                             "wrong in the same direction: '23 per cent "
                             "BELOW' divided a partial measurement by "
                             "the FULL protocol row, and '1.25 per cent "
                             "ABOVE' credited an interventions "
                             "component the measurement excludes. Both "
                             "picked a more favourable framing than the "
                             "evidence supports. The precedent is "
                             "UNFAVOURABLE, not reassuring."},
            "retained_a1_row": {
                "carries_hours": round(sum(RETAINED_A1_ROW.values()), 3),
                "components": RETAINED_A1_ROW,
                "basis": "protocol 13.2b's own estimates for work that "
                         "has not run. The middle-layer figure in "
                         "particular is an estimate whose measured "
                         "sibling, the full extraction, overran by 82.4 "
                         "per cent."},
            "efficiency_s19": {
                "carries_hours": EFFICIENCY_S19_H,
                "basis": "protocol 13.2 range 0.500-1.000 h, charged at "
                         "the upper bound"},
            "selection_sensitivity_REMOVED_NOT_UNCOSTED": {
                "hours_if_reinstated": round(44 * (
                    hours(N_RAW * (R2_ROW_S + R3_ROW_S)) / 2), 3),
                "status": "REMOVED by the fixed-endpoint amendment, "
                          "which eliminated its object (sensitivity to "
                          "max-over-epochs checkpoint selection). This "
                          "is the ONE component that was removed rather "
                          "than added. Reinstating it would need a "
                          "fresh user decision AND would exceed the "
                          "remaining headroom several times over.",
                "disclosure": "the reconciliation's phrase 'nothing was "
                              "descoped' refers to CORE ARMS AND CELLS, "
                              "none of which was removed. This study "
                              "was removed, deliberately and on stated "
                              "grounds, and that is recorded here so "
                              "the two statements cannot be read as "
                              "contradicting each other."}},
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
