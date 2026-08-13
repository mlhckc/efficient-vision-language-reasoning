"""Output E: the training-seed variability table.

    python -m experiments.closure.build_seed_table

This table exists to keep two quantities apart that are easy to conflate and
wrong to interchange:

  the standard deviation here      spread across independent TRAINING seeds
  the intervals in Outputs B and D evaluation-sampling uncertainty over
                                   questions and images, conditioning on the
                                   fixed trained seed set

Neither bounds the other. A contrast can have a tight clustered interval and
a large seed spread, which is exactly the case for E8B B3 at train_40k, where
the three seeds are 0.53124, 0.47783 and 0.51880. Reporting that cell's mean
alone would be misleading, so it is flagged.

Seed sets differ by family and are never pooled: V2, E2 and E3 use
{0,1,2,3,42}; the V3 reasoner, E8A, E8B and E10 use {0,1,2}.
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
from experiments.closure import closure_common as cc  # noqa: E402

RESULTS_ROOT = config.RESULTS_DIR / "experiments"
HIGH_DISPERSION_THRESHOLD = 0.01


def make_row(family, arm, scale, seeds, values, metric, source,
             split, note=""):
    array = np.asarray(values, dtype="float64")
    sd = float(array.std(ddof=1)) if len(array) > 1 else None
    return {
        "row_id": f"{family}/{arm}/{scale}",
        "family": family, "arm": arm, "scale": scale,
        "seed_set": list(seeds), "n_training_seeds": len(seeds),
        "per_seed_accuracy": {str(s): round(float(v), 5)
                              for s, v in zip(seeds, values)},
        "mean": round(float(array.mean()), 5),
        "sd_across_training_seeds_ddof1": (round(sd, 5) if sd is not None
                                           else None),
        "min": round(float(array.min()), 5),
        "max": round(float(array.max()), 5),
        "range": round(float(array.max() - array.min()), 5),
        "high_seed_dispersion": bool(sd is not None
                                     and sd >= HIGH_DISPERSION_THRESHOLD),
        "metric": metric, "split": split, "source_artefact": source,
        "quantity": ("sample standard deviation across independent training "
                     "seeds (ddof=1). This is NOT a confidence interval and "
                     "NOT evaluation-sampling uncertainty"),
        "note": note,
    }


def main() -> int:
    utils.set_seed()
    rows = []
    v2_metric = "v2_closed_vocab_top1_index_match"
    g21 = "G21 pinned normalised exact match"
    v2_split = "v2 dev (7,714 rows, 768 images)"
    seeds_v2 = [0, 1, 2, 3, 42]
    seeds_slm = [0, 1, 2]

    def read(family, key):
        path = RESULTS_ROOT / family / "results.json"
        return json.loads(path.read_text())[key], path

    # ---------------- V2 families ------------------------------------------
    v2_02, p = read("v2_02_multiseed", "v2_02_multiseed")
    for arm, entry in v2_02["aggregate"].items():
        rows.append(make_row("v2_02", arm, "train_40k", seeds_v2,
                             [entry["per_seed"][str(s)] for s in seeds_v2],
                             v2_metric, str(p.relative_to(PROJECT_ROOT)),
                             v2_split))
    v2_03, p = read("v2_03_param_match", "v2_03_param_match")
    for arm in ("concat_wide", "fusion_narrow"):
        entry = v2_03["aggregate"][arm]
        rows.append(make_row("v2_03", arm, "train_40k", seeds_v2,
                             [entry["per_seed"][str(s)] for s in seeds_v2],
                             v2_metric, str(p.relative_to(PROJECT_ROOT)),
                             v2_split))
    v2_04, p = read("v2_04_ablation", "v2_04_ablation")
    for arm in ("product_576k", "difference_576k", "product_natural",
                "difference_natural"):
        entry = v2_04["aggregate"][arm]
        rows.append(make_row("v2_04", arm, "train_40k", seeds_v2,
                             [entry["per_seed"][str(s)] for s in seeds_v2],
                             v2_metric, str(p.relative_to(PROJECT_ROOT)),
                             v2_split))
    v2_06, p = read("v2_06_reliance", "v2_06_reliance")
    for arm, entry in v2_06["aggregate"].items():
        for condition in ("shuffled", "zeroed"):
            rows.append(make_row(
                "v2_06", f"{arm}/{condition}", "train_40k", seeds_v2,
                [entry[condition]["per_seed"][str(s)] for s in seeds_v2],
                v2_metric, str(p.relative_to(PROJECT_ROOT)), v2_split,
                note=f"accuracy under the {condition} image intervention"))
    v2_07, p = read("v2_07_scaling", "v2_07_scaling")
    for arm, entry in v2_07["aggregate"].items():
        for scale_key in ("40k", "100k", "250k"):
            rows.append(make_row(
                "v2_07", arm, f"train_{scale_key}", seeds_v2,
                [entry[scale_key]["per_seed"][str(s)] for s in seeds_v2],
                v2_metric, str(p.relative_to(PROJECT_ROOT)), v2_split,
                note=("the stored per-seed values are read here as recorded. "
                      "v2_07 row-level reconstruction was STOPPED by a "
                      "point-estimate reproduction mismatch, so no clustered "
                      "interval is offered for this family in Output B")))

    # ---------------- E2 and E3 --------------------------------------------
    e2, p = read("e2_siglip", "e2_siglip")
    for scale_key in ("40k", "250k"):
        for arm, entry in e2["aggregate"][scale_key].items():
            rows.append(make_row(
                "E2", arm, f"train_{scale_key}", seeds_v2,
                [entry["per_seed"][str(s)] for s in seeds_v2], v2_metric,
                str(p.relative_to(PROJECT_ROOT)),
                "v2 dev (7,714 rows, 768 images), SigLIP store"))
    e3, p = read("e3_vocab1000", "e3_vocab1000")
    for scale_key in ("40k", "250k"):
        for arm, entry in e3["aggregate"][scale_key].items():
            rows.append(make_row(
                "E3", arm, f"train_{scale_key}", seeds_v2,
                [entry["per_seed"][str(s)] for s in seeds_v2], v2_metric,
                str(p.relative_to(PROJECT_ROOT)),
                "E3 dev view (9,823 rows, 776 images)",
                note="a top-1000 accuracy is never comparable row-for-row "
                     "with a top-100 accuracy"))

    # ---------------- V3 reasoner ------------------------------------------
    v3_03, p = read("v3_03_scaling", "v3_03_scaling")
    for scale_key in ("100k", "250k"):
        entry = v3_03["summary"][scale_key]
        rows.append(make_row(
            "v3_03", "reasoner", f"train_{scale_key}", seeds_slm,
            [entry["per_seed"][str(s)] for s in seeds_slm], v2_metric,
            str(p.relative_to(PROJECT_ROOT)), v2_split))

    # ---------------- E8A ---------------------------------------------------
    e8a_path = RESULTS_ROOT / "e8a_question_encoder" / "core_analysis_g21.json"
    e8a = json.loads(e8a_path.read_text())["e8a_core_analysis_g21"]
    for scale in ("train_40k", "train_250k"):
        for arm, entry in e8a["analyses"][scale][
                "per_arm_summary_normalized"].items():
            rows.append(make_row(
                "E8A", arm, scale, seeds_slm, entry["per_seed"], g21,
                str(e8a_path.relative_to(PROJECT_ROOT)), v2_split))

    # ---------------- E8B ---------------------------------------------------
    e8b_path = (RESULTS_ROOT / "e8b_readout_generation"
                / "e8b_final_report_20260810.json")
    e8b = json.loads(e8b_path.read_text())["e8b_final_report"]["PRIMARY"]
    for key, entry in e8b["accuracy_by_arm_and_scale"].items():
        if "scale_change" in key or "seeds" not in entry:
            continue
        arm, scale = key.split("/")
        rows.append(make_row(
            "E8B", arm, scale, seeds_slm, entry["seeds"], g21,
            str(e8b_path.relative_to(PROJECT_ROOT)),
            "v2 dev in-vocabulary (7,714 rows, 768 images)",
            note=("B1 keeps its own frozen classifier recipe and is not "
                  "forced onto the fixed-22-epoch rule"
                  if arm == "B1" else "")))

    # ---------------- E10 ---------------------------------------------------
    e10_path = (RESULTS_ROOT / "e10_capacity_360m_core"
                / "e10_core_analysis.json")
    e10 = json.loads(e10_path.read_text())["e10_core_analysis"]
    for key, entry in e10["seed_summaries"].items():
        arm, scale = key.split("_", 1)
        rows.append(make_row(
            "E10", arm, scale, seeds_slm,
            entry["g21_normalized_correct"]["per_seed"], g21,
            str(e10_path.relative_to(PROJECT_ROOT)), v2_split))

    rows.sort(key=lambda r: r["row_id"])
    flagged = [r["row_id"] for r in rows if r["high_seed_dispersion"]]

    record = {
        "closure_output": "E",
        "title": "training-seed variability table",
        "provenance": cc.provenance("experiments/closure/build_seed_table.py"),
        "separation_statement": (
            "the standard deviation in this table is the sample standard "
            "deviation across independent TRAINING seeds (ddof=1). The "
            "intervals in Outputs B and D are image-clustered "
            "evaluation-sampling intervals that condition on the fixed "
            "trained seed set. The two are different quantities, neither "
            "bounds the other, and neither is described as the other anywhere "
            "in this closure. A clustered interval is never called "
            "training-seed uncertainty"),
        "seed_sets": {
            "V2, E2, E3": [0, 1, 2, 3, 42],
            "V3 reasoner, E8A, E8B, E10": [0, 1, 2],
            "policy": "seed sets are recorded per row and never pooled across "
                      "families",
        },
        "high_dispersion_threshold": HIGH_DISPERSION_THRESHOLD,
        "high_dispersion_rows": flagged,
        "high_dispersion_note": (
            "a flagged row must be shown with its seed spread wherever it is "
            "reported; its mean alone is misleading. E8B/B3/train_40k is the "
            "clearest case, with seeds 0.53124, 0.47783 and 0.51880"),
        "row_count": len(rows),
        "rows": rows,
        "clean_test_accessed": False,
    }
    digest = cc.write_json(record, cc.CLOSURE_DIR / "E_seed_variability.json")
    cc.write_csv(rows, ["row_id", "family", "arm", "scale", "seed_set",
                        "n_training_seeds", "per_seed_accuracy", "mean",
                        "sd_across_training_seeds_ddof1", "min", "max",
                        "range", "high_seed_dispersion", "metric", "split",
                        "source_artefact"],
                 cc.CLOSURE_DIR / "E_seed_variability.csv")
    print(f"E: {len(rows)} seed rows, {len(flagged)} flagged for high "
          f"dispersion")
    print(f"E_seed_variability.json sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
