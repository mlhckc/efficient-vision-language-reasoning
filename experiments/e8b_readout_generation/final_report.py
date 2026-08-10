"""Final E8B analysis over the frozen core matrix and its evaluations.

Read-only. Consumes the eighteen frozen core records, the eighteen final
evaluation artefacts and their per-row dumps, and emits the complete set
of dissertation outputs with PRIMARY, SECONDARY DIAGNOSTIC and
EXPLORATORY kept apart.

Statistics are reused, never invented: the universal primary directional
rule is applied through the E8A implementation, exactly as in
core_analysis.py. Intervention drops are reported as absolute
performance and as a drop from that arm's own normal condition, never
against another arm's.

THE EMBARGOED CLEAN TEST IS NEVER TOUCHED.

    python -B experiments/e8b_readout_generation/final_report.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402
from experiments.e8b_readout_generation import run as e8b_run  # noqa: E402
from experiments.e8b_readout_generation import training as e8b_training  # noqa: E402
from experiments.e8a_question_encoder import analyse_core as e8a_stats  # noqa: E402

OUT_DIR = e8b_run.OUT_DIR
SEEDS = (0, 1, 2)
SCALES = ("train_40k", "train_250k")
ARMS = ("B1", "B2", "B3")
LM_ARMS = ("B2", "B3")
CONDITIONS = ("normal", "fixed_image", "fixed_question", "shuffled_image")


def load_evaluations() -> dict:
    """Every evaluation artefact, bound back to its frozen checkpoint."""
    out = {}
    for arm in ARMS:
        for scale in SCALES:
            for seed in SEEDS:
                run_name = e8b_training.core_run_name(arm, scale, seed)
                path = OUT_DIR / f"{run_name}_final_evaluation.json"
                if not path.exists():
                    raise AssertionError(f"{path.name} is missing")
                body = json.loads(path.read_text())["e8b_final_evaluation"]
                if body["clean_test_accessed"] is not False:
                    raise AssertionError(f"{run_name}: clean-test flag")
                if body["manifest_basis_commit"] != "1b4ea42":
                    raise AssertionError(f"{run_name}: wrong basis commit")
                npz = Path(body["per_row_artefact"]["path"])
                live = hashlib.sha256(npz.read_bytes()).hexdigest()
                if live != body["per_row_artefact"]["sha256"]:
                    raise AssertionError(f"{run_name}: per-row hash drift")
                body["_rows"] = np.load(npz)
                out[(arm, scale, seed)] = body
    return out


def metric(body, arm, condition, readout, denominator, scorer):
    """One scored number, arm-aware. B1 has a classifier readout only."""
    block = body["conditions"][condition]
    if arm == "B1":
        if readout != "classifier":
            return None
        key = ("classifier_in_vocabulary" if denominator == "in_vocabulary"
               else "classifier_raw_denominator")
    else:
        if readout == "classifier":
            return None
        key = f"{readout}_{denominator}"
        if denominator == "raw":
            key = f"{readout}_raw_denominator"
    return block[key][f"{scorer}_exact"]


def describe(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {"seeds": [round(float(v), 5) for v in a],
            "mean": round(float(a.mean()), 5),
            "sample_sd": round(float(a.std(ddof=1)), 5)}


def readout_table(evals) -> list:
    """PRIMARY: R1/R2/R3 for the language-model arms, both denominators,
    both scorers, on the normal condition."""
    rows = []
    for arm in LM_ARMS:
        for scale in SCALES:
            for readout in ("R1", "R2", "R3"):
                for denominator in ("in_vocabulary", "raw"):
                    for scorer in ("raw", "normalised"):
                        values = [metric(evals[(arm, scale, s)], arm,
                                         "normal", readout, denominator,
                                         scorer) for s in SEEDS]
                        rows.append({
                            "arm": arm, "scale": scale,
                            "readout": readout,
                            "denominator": denominator,
                            "scorer": scorer, **describe(values)})
    for scale in SCALES:
        for denominator in ("in_vocabulary", "raw"):
            for scorer in ("raw", "normalised"):
                values = [metric(evals[("B1", scale, s)], "B1", "normal",
                                 "classifier", denominator, scorer)
                          for s in SEEDS]
                rows.append({"arm": "B1", "scale": scale,
                             "readout": "classifier",
                             "denominator": denominator,
                             "scorer": scorer, **describe(values)})
    return rows


def r3_validity(evals) -> list:
    """PRIMARY: R3 emission validity, every row kept in the denominator."""
    rows = []
    for arm in LM_ARMS:
        for scale in SCALES:
            counts = [evals[(arm, scale, s)]["conditions"]["normal"][
                "R3_outcomes"]["rates"] for s in SEEDS]
            rows.append({
                "arm": arm, "scale": scale,
                "in_vocabulary_rate": round(float(np.mean(
                    [c["in_vocabulary"] for c in counts])), 5),
                "out_of_vocabulary_rate": round(float(np.mean(
                    [c["out_of_vocabulary"] for c in counts])), 5),
                "empty_rate": round(float(np.mean(
                    [c["empty"] for c in counts])), 6),
                "overlong_rate": round(float(np.mean(
                    [c["overlong"] for c in counts])), 6)})
    return rows


def intervention_table(evals) -> list:
    """PRIMARY: absolute performance and the drop from each arm's OWN
    normal condition. Never compared against another arm's normal."""
    rows = []
    for arm in ARMS:
        readout = "classifier" if arm == "B1" else "R1"
        for scale in SCALES:
            normal = [metric(evals[(arm, scale, s)], arm, "normal",
                             readout, "in_vocabulary", "raw")
                      for s in SEEDS]
            for condition in CONDITIONS:
                values = [metric(evals[(arm, scale, s)], arm, condition,
                                 readout, "in_vocabulary", "raw")
                          for s in SEEDS]
                drops = [round(n - v, 5)
                         for n, v in zip(normal, values)]
                rows.append({
                    "arm": arm, "scale": scale, "condition": condition,
                    "readout": readout, **describe(values),
                    "drop_from_own_normal_mean":
                        round(float(np.mean(drops)), 5),
                    "drop_per_seed": drops})
    return rows


def contrast(evals, arm_a, arm_b, scale, view, label, key) -> dict:
    a = [evals[(arm_a, scale, s)]["_rows"][key].astype(float)
         for s in SEEDS]
    b = [evals[(arm_b, scale, s)]["_rows"][key].astype(float)
         for s in SEEDS]
    per_seed = [round(float(a[i].mean() - b[i].mean()), 5)
                for i in range(len(SEEDS))]
    low, high = e8a_stats.clustered_bootstrap(
        a, b, view["image_index"], view["n_images"])
    return {"contrast": label, "scale": scale, "basis": key,
            "mean_paired_difference": round(float(np.mean(per_seed)), 5),
            "per_seed_differences": per_seed,
            "seeds_above_zero": int(sum(1 for d in per_seed if d > 0)),
            "ci95_low": low, "ci95_high": high,
            "excludes_zero": bool(low > 0 or high < 0),
            "directional_rule_outcome": e8a_stats.directional_rule(
                per_seed, low, high),
            "n_rows": int(len(a[0])),
            "n_image_clusters": view["n_images"]}


def raw_view() -> dict:
    frame = pd.read_csv(config.DATA_DIR / "v2" / "dev_raw.csv",
                        dtype={"questionId": str, "imageId": str},
                        keep_default_na=False)
    codes, uniques = pd.factorize(frame["imageId"], sort=True)
    return {"image_index": np.asarray(codes),
            "n_images": int(len(uniques)),
            "n_rows": int(len(frame))}


def main() -> int:
    utils.set_seed()
    evals = load_evaluations()
    view = raw_view()
    core = json.loads(
        (OUT_DIR / "e8b_core_analysis_20260809.json").read_text()
    )["e8b_core_analysis"]

    # The raw-denominator R1 contrast (HB3b), on the full 10,004 rows.
    raw_contrasts = []
    for scale in SCALES:
        raw_contrasts.append(contrast(evals, "B3", "B2", scale, view,
                                      "B3 - B2", "r1_hits_normal"))
    reliance = []
    for arm in ARMS:
        for scale in SCALES:
            base = [evals[(arm, scale, s)]["_rows"]["r1_hits_normal"]
                    .astype(float) for s in SEEDS]
            shuf = [evals[(arm, scale, s)]["_rows"]["r1_hits_shuffled_image"]
                    .astype(float) for s in SEEDS]
            per_seed = [round(float(base[i].mean() - shuf[i].mean()), 5)
                        for i in range(len(SEEDS))]
            low, high = e8a_stats.clustered_bootstrap(
                base, shuf, view["image_index"], view["n_images"])
            reliance.append({
                "arm": arm, "scale": scale,
                "contrast": "normal - deranged image",
                "mean_drop": round(float(np.mean(per_seed)), 5),
                "per_seed_drops": per_seed,
                "ci95_low": low, "ci95_high": high,
                "directional_rule_outcome": e8a_stats.directional_rule(
                    per_seed, low, high)})

    seconds = {f"{a}/{s}": round(float(np.mean(
        [evals[(a, s, sd)]["wall_seconds"] for sd in SEEDS])), 1)
        for a in ARMS for s in SCALES}
    total_eval_hours = round(float(sum(
        evals[k]["wall_seconds"] for k in evals) / 3600), 5)

    record = {
        "metadata": utils.run_metadata(),
        "e8b_final_report": {
            "dated_utc": "2026-08-10",
            "READ_ONLY": True,
            "basis_commit": "1b4ea42",
            "PRIMARY": {
                "accuracy_by_arm_and_scale":
                    core["descriptive_summary"],
                "paired_contrasts_in_vocabulary":
                    core["paired_contrasts"],
                "scale_effects": core["scale_effects"],
                "readouts": readout_table(evals),
                "readout_contrasts_raw_denominator": raw_contrasts,
                "interventions": intervention_table(evals),
                "image_reliance": reliance,
                "r3_validity": r3_validity(evals)},
            "SECONDARY_DIAGNOSTIC": {
                "stability": core["stability"],
                "note": "best-of-22 never determines a primary "
                        "comparison, a model freeze or a clean-test "
                        "checkpoint"},
            "EXPLORATORY": {
                "abandoned_search": "grid points 1-3 of the eight-point "
                                    "recipe search are exploratory "
                                    "protocol-diagnostic evidence only, "
                                    "support no superiority claim in "
                                    "either direction, and are NOT "
                                    "promoted anywhere in this report"},
            "resources": {
                **core["resources"],
                "final_evaluation_hours_measured": total_eval_hours,
                "final_evaluation_seconds_per_cell_mean": seconds,
                "projection_line_item_lm_hours": 2.8,
                "projection_note":
                    "the measured language-model evaluation exceeds the "
                    "2.8 h projection line item. It remains inside every "
                    "per-identity committed reservation and fired no "
                    "gate; the overrun is recorded rather than absorbed."},
            "b1_selection_rule_caveat": core["b1_selection_rule_caveat"],
            "statistical_procedure": core["statistical_procedure"],
            "clean_test_accessed": False}}
    out = OUT_DIR / "e8b_final_report_20260810.json"
    e8b_run.atomic_write_json(out, record)
    print(f"written {out.name}")
    print(f"  evaluations consumed : {len(evals)}")
    print(f"  measured evaluation  : {total_eval_hours} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
