"""Output D: the slice / type / reasoning-depth statistical table.

    python -m experiments.closure.build_slice_table

Every row is one slice of the development set for one arm: the structural
type, the semantic type or the reasoning-depth bucket. Each carries the two
counts that make a slice interpretable — how many questions it holds and how
many distinct images those questions come from — together with the point
estimate, the image-clustered interval and the clustering unit.

The counts matter because the slices are unequal and dependent: the >=5-step
bucket holds 1,114 questions but far fewer independent images, so a row-level
interval on it would be badly overconfident. Where a source artefact records
its own counts they are read from it. Where a source records an interval but
not the counts, the counts are derived here from data/v2/dev.csv and
data/v2/metadata/dev_types.csv and labelled INFERENCE, so a reader can tell a
read number from a derived one.

The v2_05b row-level normal-approximation intervals are superseded by v2_05c
and are recorded as superseded rather than re-quoted.
"""

from __future__ import annotations

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
from experiments.closure import closure_common as cc  # noqa: E402

RESULTS_ROOT = config.RESULTS_DIR / "experiments"
STRUCTURAL_ORDER = ["verify", "query", "choose", "logical", "compare"]
SEMANTIC_ORDER = ["obj", "attr", "cat", "rel", "global"]
STEP_ORDER = ["<=2", "3", "4", ">=5"]


def slice_counts() -> dict:
    """n_questions and n_unique_images for every named development slice."""
    dev = pd.read_csv(cc.assert_not_embargoed(config.DATA_DIR / "v2"
                                              / "dev.csv"),
                      dtype={"questionId": str, "imageId": str},
                      keep_default_na=False)
    types = pd.read_csv(config.DATA_DIR / "v2" / "metadata" / "dev_types.csv",
                        dtype={"questionId": str}, keep_default_na=False)
    assert list(types["questionId"]) == list(dev["questionId"])
    steps = types["n_steps"].to_numpy()
    bucket = np.where(steps <= 2, "<=2", np.where(
        steps == 3, "3", np.where(steps == 4, "4", ">=5")))
    image = dev["imageId"].to_numpy()

    counts = {}
    for kind, order, column in (("structural", STRUCTURAL_ORDER,
                                 types["structural"].to_numpy()),
                                ("semantic", SEMANTIC_ORDER,
                                 types["semantic"].to_numpy()),
                                ("steps", STEP_ORDER, bucket)):
        for value in order:
            mask = column == value
            counts[f"{kind}:{value}"] = {
                "n_questions": int(mask.sum()),
                "n_unique_images": int(len(set(image[mask])))}
    counts["all"] = {"n_questions": int(len(dev)),
                     "n_unique_images": int(dev["imageId"].nunique())}
    return counts


def main() -> int:
    utils.set_seed()
    counts = slice_counts()
    rows = []

    # ---------------- reconstructed slices ---------------------------------
    stats_path = cc.CLOSURE_DIR / "reconstructed_statistics.json"
    stats = json.loads(stats_path.read_text())
    for entry in stats["slices"]:
        rows.append({
            "slice_id": f"{entry['family']}/{entry['arm']}/{entry['scale']}"
                        f"/{entry['slice_kind']}:{entry['slice']}"
                        f"/{entry['quantity']}",
            "source": "results/closure/reconstructed_statistics.json",
            "family": entry["family"], "arm": entry["arm"],
            "scale": entry["scale"], "slice_kind": entry["slice_kind"],
            "slice": entry["slice"], "quantity": entry["quantity"],
            "point_estimate": entry["point_estimate"],
            "ci95_image_clustered": entry["ci95_image_clustered"],
            "excludes_zero": entry["excludes_zero"],
            "cluster_unit": entry["cluster_unit"],
            "n_questions": entry["n_questions"],
            "n_unique_images": entry["n_unique_images"],
            "counts_provenance": "READ_FROM_ROW_EVIDENCE",
            "n_resamples": entry["n_resamples"], "rng_seed": entry["rng_seed"],
            "training_seeds": entry["training_seeds"],
            "across_training_seed_sd_ddof1":
                entry["across_training_seed_sd_ddof1"],
            "metric": "v2_closed_vocab_top1_index_match",
            "analysis_role": entry["analysis_role"],
            "comparison_status": entry["comparison_status"],
            "analysis_status": entry["analysis_status"],
        })

    # ---------------- v2_05c (read) ----------------------------------------
    v2_05c_path = RESULTS_ROOT / "v2_05_types" / "addendum_clustered.json"
    v2_05c = json.loads(v2_05c_path.read_text())["v2_05c_clustered_correction"]
    for group, seeds, tag in (("seed42_gap_ci", [42], "seed42"),
                              ("multiseed_gap_ci", [0, 1, 2, 3, 42],
                               "five_seed")):
        for entry in v2_05c[group]:
            key = f"{entry['kind']}:{entry['slice']}"
            rows.append({
                "slice_id": f"v2_05c/fusion_minus_concat/train_40k/{key}"
                            f"/{tag}",
                "source": "results/experiments/v2_05_types/"
                          "addendum_clustered.json",
                "family": "v2_05c", "arm": "fusion minus concat",
                "scale": "train_40k", "slice_kind": entry["kind"],
                "slice": entry["slice"], "quantity": "paired accuracy gap",
                "point_estimate": entry.get("gap_seed42", entry.get("gap_mean")),
                "ci95_image_clustered": entry["ci95_image_clustered"],
                "excludes_zero": entry["excludes_zero"],
                "cluster_unit": cc.CLUSTER_UNIT,
                "n_questions": entry["n_rows"],
                "n_unique_images": entry["n_unique_images"],
                "counts_provenance": "READ_FROM_SOURCE",
                "n_resamples": 2000, "rng_seed": 0, "training_seeds": seeds,
                "across_training_seed_sd_ddof1": entry.get("gap_std"),
                "metric": "v2_closed_vocab_top1_index_match",
                "analysis_role": "SECONDARY",
                "comparison_status": "CAPACITY_CONFOUNDED",
                "analysis_status": "READ_FROM_SOURCE",
                "supersedes": ("the v2_05b row-level normal-approximation "
                               "interval for this slice, which is superseded "
                               "and never re-quoted"),
                "superseded_row_level_ci": entry.get(
                    "superseded_row_level_ci"),
            })

    # ---------------- v3_02a step statistics (read) ------------------------
    v3_path = RESULTS_ROOT / "v3_02a_refs" / "step_statistics.json"
    v3 = json.loads(v3_path.read_text())["v3_02a_step_statistics"]
    bucket_counts = {b["bucket"]: b for b in v3["bucket_table"]}
    for arm, entry in v3["confidence_intervals"].items():
        for bucket, interval in entry["bucket_raw_ci"].items():
            table = bucket_counts[bucket]
            rows.append({
                "slice_id": f"v3_02a/{arm}/train_40k/steps:{bucket}/accuracy",
                "source": "results/experiments/v3_02a_refs/"
                          "step_statistics.json",
                "family": "v3_02a", "arm": arm, "scale": "train_40k",
                "slice_kind": "steps", "slice": bucket, "quantity": "accuracy",
                "point_estimate": table.get(f"{arm}_raw"),
                "ci95_image_clustered": interval, "excludes_zero": None,
                "cluster_unit": v3["bootstrap"]["cluster_unit"],
                "n_questions": table["n_questions"],
                "n_unique_images": table["n_unique_images"],
                "counts_provenance": "READ_FROM_SOURCE",
                "n_resamples": v3["bootstrap"]["n_draws"],
                "rng_seed": v3["bootstrap"]["rng_seed"],
                "training_seeds": [0, 1, 2],
                "across_training_seed_sd_ddof1": None,
                "metric": "v2_closed_vocab_top1_index_match",
                "analysis_role": "DIAGNOSTIC",
                "comparison_status": "DESCRIPTIVE_ONLY",
                "analysis_status": "READ_FROM_SOURCE",
            })
        rows.append({
            "slice_id": f"v3_02a/{arm}/train_40k/steps:>=4_combined/deficit",
            "source": "results/experiments/v3_02a_refs/step_statistics.json",
            "family": "v3_02a", "arm": arm, "scale": "train_40k",
            "slice_kind": "steps", "slice": ">=4_combined",
            "quantity": "pooled >=4-step deficit",
            "point_estimate": v3["seed_mean_deficits"].get(arm),
            "ci95_image_clustered": entry["deficit_ci"],
            "excludes_zero": bool(entry["deficit_ci"][0] > 0
                                  or entry["deficit_ci"][1] < 0),
            "cluster_unit": v3["bootstrap"]["cluster_unit"],
            "n_questions": (bucket_counts["4"]["n_questions"]
                            + bucket_counts[">=5"]["n_questions"]),
            "n_unique_images": None,
            "counts_provenance": "DERIVED_FROM_SOURCE_BUCKET_COUNTS",
            "n_resamples": v3["bootstrap"]["n_draws"],
            "rng_seed": v3["bootstrap"]["rng_seed"],
            "training_seeds": [0, 1, 2],
            "across_training_seed_sd_ddof1": None,
            "metric": "v2_closed_vocab_top1_index_match",
            "analysis_role": "PRIMARY",
            "comparison_status": "DESCRIPTIVE_ONLY",
            "analysis_status": "READ_FROM_SOURCE",
        })

    # ---------------- E8A per-slice intervals (read; counts derived) -------
    e8a_path = RESULTS_ROOT / "e8a_question_encoder" / "core_analysis_g21.json"
    e8a = json.loads(e8a_path.read_text())["e8a_core_analysis_g21"]
    kind_of = {"structural_type": "structural", "semantic_type": "semantic",
               "steps": "steps"}
    for key, entry in e8a["type_and_step_slices"]["slices"].items():
        prefix, value = key.split(":", 1)
        kind = kind_of[prefix]
        count = counts[f"{kind}:{value}"]
        for arm_scale, arm_entry in entry["arms"].items():
            arm, scale = arm_scale.split("/")
            rows.append({
                "slice_id": f"E8A/{arm}/{scale}/{kind}:{value}/accuracy",
                "source": "results/experiments/e8a_question_encoder/"
                          "core_analysis_g21.json",
                "family": "E8A", "arm": arm, "scale": scale,
                "slice_kind": kind, "slice": value, "quantity": "accuracy",
                "point_estimate": arm_entry["normalized_exact_mean"],
                "ci95_image_clustered": arm_entry["ci95_image_clustered"],
                "excludes_zero": None, "cluster_unit": cc.CLUSTER_UNIT,
                "n_questions": count["n_questions"],
                "n_unique_images": count["n_unique_images"],
                "counts_provenance": "INFERENCE",
                "n_resamples": 2000, "rng_seed": 0,
                "training_seeds": [0, 1, 2],
                "across_training_seed_sd_ddof1": arm_entry["seed_std"],
                "metric": "G21 pinned normalised exact match",
                "analysis_role": "DIAGNOSTIC",
                "comparison_status": "DESCRIPTIVE_ONLY",
                "analysis_status": "READ_FROM_SOURCE",
            })

    rows.sort(key=lambda r: r["slice_id"])
    record = {
        "closure_output": "D",
        "title": "slice, type and reasoning-depth statistical table",
        "provenance": cc.provenance(
            "experiments/closure/build_slice_table.py",
            [cc.record_source(stats_path, "reconstructed statistics"),
             cc.record_source(v2_05c_path, "v2_05c clustered corrections"),
             cc.record_source(v3_path, "v3_02a step statistics"),
             cc.record_source(e8a_path, "E8A core analysis"),
             cc.record_source(config.DATA_DIR / "v2" / "dev.csv",
                              "split manifest"),
             cc.record_source(config.DATA_DIR / "v2" / "metadata"
                              / "dev_types.csv", "slice metadata")]),
        "counts_provenance_labels": {
            "READ_FROM_SOURCE": "the source artefact recorded the counts",
            "READ_FROM_ROW_EVIDENCE": "counted directly from the "
                                      "reconstructed row-level arrays",
            "DERIVED_FROM_SOURCE_BUCKET_COUNTS": "summed from counts the "
                                                 "source did record",
            "INFERENCE": "the source recorded an interval but not the counts; "
                         "derived here from the development manifest and the "
                         "type metadata",
        },
        "dependence_note": (
            "slices are unequal and dependent. Every interval here is "
            "clustered on the slice's own represented development images, "
            "never on rows, because questions about one image are not "
            "independent observations"),
        "superseded_note": (
            "the v2_05b row-level normal-approximation intervals are "
            "superseded by v2_05c. Where a v2_05c row records the interval it "
            "replaced, that superseded interval is carried as context and is "
            "never quoted as a current result"),
        "slice_counts_reference": counts,
        "row_count": len(rows),
        "slices": rows,
        "clean_test_accessed": False,
    }
    digest = cc.write_json(record, cc.CLOSURE_DIR / "D_slice_statistics.json")
    cc.write_csv(rows, ["slice_id", "family", "arm", "scale", "slice_kind",
                        "slice", "quantity", "point_estimate",
                        "ci95_image_clustered", "n_questions",
                        "n_unique_images", "cluster_unit",
                        "counts_provenance", "n_resamples", "rng_seed",
                        "training_seeds", "across_training_seed_sd_ddof1",
                        "metric", "analysis_role", "comparison_status",
                        "source"],
                 cc.CLOSURE_DIR / "D_slice_statistics.csv")
    print(f"D: {len(rows)} slice rows")
    print(f"D_slice_statistics.json sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
