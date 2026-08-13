"""Dependence-aware statistics over the reconstructed row-level evidence.

    python -m experiments.closure.build_intervals

Runs only after reconstruct_rows.py has passed its point-estimate
reproduction gate. Development rows share images — 7,714 questions over 768
represented images, 9,823 over 776 in the E3 view — so a row-level interval
would assume an independence the data does not have. Every interval here is
image-clustered, uses the project's single resampling procedure, and is
labelled as evaluation-sampling uncertainty conditioning on the fixed trained
seed set. Training-seed spread is reported beside it as a separate quantity
and is never described as a confidence interval.

No new hypothesis test is introduced, no interval is selected for reporting
on whether it excludes zero, and every contrast computed here was named
before its result was seen: the contrast list is the one the source
experiments already report as their own gaps, plus the arm accuracies those
gaps are built from.

Writes results/closure/reconstructed_statistics.json.
"""

from __future__ import annotations

import json
import sys
import zipfile
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
SEEDS_V2 = [0, 1, 2, 3, 42]
STRUCTURAL_ORDER = ["verify", "query", "choose", "logical", "compare"]
SEMANTIC_ORDER = ["obj", "attr", "cat", "rel", "global"]
STEP_ORDER = ["<=2", "3", "4", ">=5"]


def load_rows(manifest: dict) -> dict:
    """Row-level correctness by condition id, from the reconstructed npz."""
    rows = {}
    for entry in manifest["row_evidence"]:
        path = PROJECT_ROOT / entry["row_evidence_path"]
        with zipfile.ZipFile(path) as archive:
            with archive.open("correct.npy") as handle:
                correct = np.load(handle)
        rows[entry["condition_id"]] = correct.astype("float64")
    return rows


def cluster_index_for(split: str) -> tuple:
    manifest = (config.DATA_DIR / "v2" / "dev.csv" if split != "v2_1000_dev"
                else config.DATA_DIR / "v2_1000" / "dev.csv")
    frame = pd.read_csv(cc.assert_not_embargoed(manifest),
                        dtype={"questionId": str, "imageId": str},
                        keep_default_na=False)
    unique = sorted(set(frame["imageId"]))
    position = {value: i for i, value in enumerate(unique)}
    return np.array([position[v] for v in frame["imageId"]]), len(unique)


def contrast(rows: dict, left: str, right: str, cluster_index, label: str,
             role: str, status: str, note: str, seeds=None) -> dict | None:
    """One paired contrast: per-row difference, clustered interval, seed sd."""
    seeds = seeds or SEEDS_V2
    left_ids = [left.format(seed=s) for s in seeds]
    right_ids = [right.format(seed=s) for s in seeds]
    if any(i not in rows for i in left_ids + right_ids):
        return None
    differences = [rows[a] - rows[b] for a, b in zip(left_ids, right_ids)]
    interval = cc.clustered_interval(differences, cluster_index)
    per_seed = [round(float(d.mean()), 5) for d in differences]
    # The source experiments formed a gap by subtracting two accuracies that
    # were already rounded to 5 decimals, then rounding again. That double
    # rounding can move a gap by 1e-5 against a gap computed once from the
    # per-row differences. Both forms are kept: the single-rounded one is the
    # statistic the interval is built on, and the double-rounded one is what
    # the stored tables can be compared against exactly.
    per_seed_stored_convention = [
        round(round(float(rows[a].mean()), 5) - round(float(rows[b].mean()), 5),
              5)
        for a, b in zip(left_ids, right_ids)]
    return {
        "contrast_id": label,
        "left": left.format(seed="{seed}"),
        "right": right.format(seed="{seed}"),
        "analysis_role": role,
        "comparison_status": status,
        "effect_direction_note": note,
        "point_estimate": interval["point_estimate"],
        "effect": interval["point_estimate"],
        **{k: v for k, v in interval.items() if k != "point_estimate"},
        "training_seeds": list(seeds),
        "per_seed_effect": dict(zip(map(str, seeds), per_seed)),
        "per_seed_effect_stored_rounding_convention": dict(
            zip(map(str, seeds), per_seed_stored_convention)),
        "across_training_seed_sd_ddof1": round(
            float(np.std(per_seed, ddof=1)), 5),
        "rows_paired": True,
        "pairing_unit": "development question row (identical rows both sides)",
        "analysis_status": "COMPUTED_FROM_REPRODUCED_ROW_EVIDENCE",
    }


def arm_accuracy(rows: dict, template: str, cluster_index, label: str,
                 seeds=None) -> dict | None:
    seeds = seeds or SEEDS_V2
    ids = [template.format(seed=s) for s in seeds]
    if any(i not in rows for i in ids):
        return None
    values = [rows[i] for i in ids]
    interval = cc.clustered_interval(values, cluster_index)
    per_seed = [round(float(v.mean()), 5) for v in values]
    return {
        "arm_id": label,
        "point_estimate": interval["point_estimate"],
        **{k: v for k, v in interval.items() if k != "point_estimate"},
        "training_seeds": list(seeds),
        "per_seed_accuracy": dict(zip(map(str, seeds), per_seed)),
        "across_training_seed_sd_ddof1": round(
            float(np.std(per_seed, ddof=1)), 5),
        "analysis_status": "COMPUTED_FROM_REPRODUCED_ROW_EVIDENCE",
    }


def pooled_deficit_interval(rows: dict, template: str, bucket: np.ndarray,
                            priors: dict, cluster_index, n_clusters: int,
                            label: str, seeds=None) -> dict | None:
    """Clustered interval for the pooled >=4-step deficit.

    The statistic is the E2/v2_07 definition verbatim: the mean of the four
    per-bucket lifts minus the question-weighted combined >=4 lift, under the
    fixed v2_05b priors. It is a combination of bucket means rather than a
    row mean, so it uses the custom-statistic bootstrap with the same
    clusters, generator, draw count and percentiles.
    """
    seeds = seeds or SEEDS_V2
    ids = [template.format(seed=s) for s in seeds]
    if any(i not in rows for i in ids):
        return None
    correct = np.stack([rows[i] for i in ids])

    tables = {}
    for value in STEP_ORDER:
        mask = (bucket == value).astype("float64")
        tables[f"count_{value}"] = np.bincount(
            cluster_index, weights=mask, minlength=n_clusters)
        tables[f"sum_{value}"] = np.stack([
            np.bincount(cluster_index, weights=c * mask, minlength=n_clusters)
            for c in correct])

    def statistic(totals):
        lifts, ge4_sum, ge4_count = [], 0.0, 0.0
        for value in STEP_ORDER:
            count = totals[f"count_{value}"]
            if count <= 0:
                raise AssertionError("empty step bucket in a clustered draw")
            lifts.append(totals[f"sum_{value}"] / count - priors[value])
            if value in ("4", ">=5"):
                ge4_sum = ge4_sum + totals[f"sum_{value}"]
                ge4_count = ge4_count + count
        prior_ge4 = sum(priors[v] * totals[f"count_{v}"]
                        for v in ("4", ">=5")) / ge4_count
        combined = ge4_sum / ge4_count - prior_ge4
        return float(np.mean(np.mean(np.stack(lifts), axis=0) - combined))

    per_seed = []
    for c in correct:
        lifts = [float(c[bucket == v].mean()) - priors[v] for v in STEP_ORDER]
        ge4 = (bucket == "4") | (bucket == ">=5")
        prior_row = np.array([priors[v] for v in bucket])
        combined = float(c[ge4].mean()) - float(prior_row[ge4].mean())
        per_seed.append(round(float(np.mean(lifts)) - combined, 5))

    interval = cc.clustered_custom_interval(
        tables, statistic, n_clusters, n_questions=len(bucket),
        n_training_seeds=len(seeds))
    return {
        "deficit_id": label,
        "definition": ("mean of the four per-bucket step lifts minus the "
                       "question-weighted combined >=4 lift, under the fixed "
                       "v2_05b per-bucket priors; the E2 and v2_07 pooled "
                       "definition verbatim"),
        "priors": priors,
        "point_estimate": round(float(np.mean(per_seed)), 5),
        **interval,
        "training_seeds": list(seeds),
        "per_seed_deficit": dict(zip(map(str, seeds), per_seed)),
        "across_training_seed_sd_ddof1": round(
            float(np.std(per_seed, ddof=1)), 5),
        "analysis_role": "SECONDARY",
        "comparison_status": "DESCRIPTIVE_ONLY",
        "interpretation_bound": (
            "a positive deficit means the pooled >=4-step lift falls below "
            "the average of the per-bucket lifts. It describes one model's "
            "step profile against fixed priors; it is not a contrast between "
            "models and it does not establish a cause"),
        "analysis_status": "COMPUTED_FROM_REPRODUCED_ROW_EVIDENCE",
    }


def main() -> int:
    utils.set_seed()
    manifest_path = cc.CLOSURE_DIR / "C_reconstructed_evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    stopped = set(manifest["stopped_families"])
    rows = load_rows(manifest)

    dev_cluster, dev_n = cluster_index_for("v2_dev")
    e3_cluster, e3_n = cluster_index_for("v2_1000_dev")
    types = pd.read_csv(config.DATA_DIR / "v2" / "metadata" / "dev_types.csv",
                        dtype={"questionId": str}, keep_default_na=False)
    steps = types["n_steps"].to_numpy()
    bucket = np.where(steps <= 2, "<=2", np.where(
        steps == 3, "3", np.where(steps == 4, "4", ">=5")))
    priors = json.loads((RESULTS_ROOT / "v2_05_types" / "addendum.json")
                        .read_text())["v2_05b_addendum"][
                            "prior_accuracy_by_slice"]
    step_priors = {v: priors[f"steps:{v}"] for v in STEP_ORDER}

    contrasts, arms, deficits, slices = [], [], [], []

    def add(entry):
        if entry is not None:
            contrasts.append(entry)

    # ---------------- v2_02: the headline heads at train_40k ---------------
    for arm in ("question_only", "image_only", "concat", "fusion"):
        entry = arm_accuracy(rows, f"v2_02::{arm}::train_40k::seed{{seed}}"
                                   f"::normal", dev_cluster,
                             f"v2_02/{arm}/train_40k")
        if entry:
            arms.append({**entry, "family": "v2_02", "scale": "train_40k",
                         "encoder": "CLIP ViT-B-32", "answer_set_size": 100})
    add(contrast(rows, "v2_02::fusion::train_40k::seed{seed}::normal",
                 "v2_02::concat::train_40k::seed{seed}::normal", dev_cluster,
                 "v2_02/fusion_minus_concat/train_40k", "PRIMARY",
                 "CAPACITY_CONFOUNDED",
                 "fusion carries 1,100,388 trainable parameters against "
                 "concat's 576,100; v2_03 is the capacity-matched form of "
                 "this comparison"))
    add(contrast(rows, "v2_02::concat::train_40k::seed{seed}::normal",
                 "v2_02::question_only::train_40k::seed{seed}::normal",
                 dev_cluster, "v2_02/concat_minus_question_only/train_40k",
                 "PRIMARY", "CAPACITY_CONFOUNDED",
                 "the image-using head also has more parameters (576,100 "
                 "against 313,956); v2_06's zeroed-image intervention is the "
                 "capacity-matched form of the same question"))

    # ---------------- v2_03: capacity, priority contrast A -----------------
    for arm in ("concat_wide", "fusion_narrow"):
        entry = arm_accuracy(rows, f"v2_03::{arm}::train_40k::seed{{seed}}"
                                   f"::normal", dev_cluster,
                             f"v2_03/{arm}/train_40k")
        if entry:
            arms.append({**entry, "family": "v2_03", "scale": "train_40k",
                         "encoder": "CLIP ViT-B-32", "answer_set_size": 100})
    for label, left, right, note in [
        ("fusion_minus_concat_wide",
         "v2_02::fusion", "v2_03::concat_wide",
         "matched at about 1.1M trainable parameters: fusion 1,100,388 "
         "against concat_wide 1,101,475"),
        ("fusion_narrow_minus_concat",
         "v2_03::fusion_narrow", "v2_02::concat",
         "matched at about 576k trainable parameters: fusion_narrow 576,032 "
         "against concat 576,100"),
        ("concat_wide_minus_concat",
         "v2_03::concat_wide", "v2_02::concat",
         "same input, more capacity: isolates the capacity term alone"),
        ("fusion_minus_fusion_narrow",
         "v2_02::fusion", "v2_03::fusion_narrow",
         "same input, more capacity: isolates the capacity term alone"),
    ]:
        add(contrast(rows, f"{left}::train_40k::seed{{seed}}::normal",
                     f"{right}::train_40k::seed{{seed}}::normal", dev_cluster,
                     f"v2_03/{label}/train_40k", "PRIMARY",
                     "CLEAN_PAIRED_CONTROL", note))

    # ---------------- v2_04: interaction-feature ablation ------------------
    for arm in ("product_576k", "difference_576k", "product_natural",
                "difference_natural"):
        entry = arm_accuracy(rows, f"v2_04::{arm}::train_40k::seed{{seed}}"
                                   f"::normal", dev_cluster,
                             f"v2_04/{arm}/train_40k")
        if entry:
            arms.append({**entry, "family": "v2_04", "scale": "train_40k",
                         "encoder": "CLIP ViT-B-32", "answer_set_size": 100})
    for label, left, right in [
        ("product_576k_minus_concat", "v2_04::product_576k", "v2_02::concat"),
        ("difference_576k_minus_concat", "v2_04::difference_576k",
         "v2_02::concat"),
        ("fusion_narrow_minus_product_576k", "v2_03::fusion_narrow",
         "v2_04::product_576k"),
        ("fusion_narrow_minus_difference_576k", "v2_03::fusion_narrow",
         "v2_04::difference_576k"),
        ("product_576k_minus_difference_576k", "v2_04::product_576k",
         "v2_04::difference_576k"),
        ("product_natural_minus_concat", "v2_04::product_natural",
         "v2_02::concat"),
        ("difference_natural_minus_concat", "v2_04::difference_natural",
         "v2_02::concat"),
        ("fusion_minus_product_natural", "v2_02::fusion",
         "v2_04::product_natural"),
        ("fusion_minus_difference_natural", "v2_02::fusion",
         "v2_04::difference_natural"),
    ]:
        matched = "576k" in label or label.startswith("product_576k_minus_diff")
        add(contrast(rows, f"{left}::train_40k::seed{{seed}}::normal",
                     f"{right}::train_40k::seed{{seed}}::normal", dev_cluster,
                     f"v2_04/{label}/train_40k", "SECONDARY",
                     "CLEAN_PAIRED_CONTROL" if matched
                     else "CAPACITY_CONFOUNDED",
                     "equal-budget arms at the 576k head target" if matched
                     else "natural-width ladder: the arms differ in "
                          "trainable parameters as well as in input"))

    # ---------------- v2_06: visual reliance, priority contrast B ----------
    for arm in ("question_only", "image_only", "concat", "fusion",
                "product_576k"):
        for condition, drop in (("shuffled", "drop_shuffled"),
                                ("zeroed", "drop_zeroed")):
            add(contrast(
                rows, f"v2_06::{arm}::train_40k::seed{{seed}}::normal",
                f"v2_06::{arm}::train_40k::seed{{seed}}::{condition}",
                dev_cluster, f"v2_06/{arm}/{drop}/train_40k", "PRIMARY",
                "CLEAN_PAIRED_CONTROL",
                "the same frozen head on the same rows under an image "
                "intervention; the question side is untouched. A drop shows "
                "the answer depends on the correct image. It does not show "
                "that the model reasons about the image"))
        entry = arm_accuracy(rows, f"v2_06::{arm}::train_40k::seed{{seed}}"
                                   f"::shuffled", dev_cluster,
                             f"v2_06/{arm}/shuffled/train_40k")
        if entry:
            arms.append({**entry, "family": "v2_06", "scale": "train_40k",
                         "encoder": "CLIP ViT-B-32", "answer_set_size": 100,
                         "image_condition": "shuffled"})

    # v2_06 structural slices, for the two heads the source table slices.
    for arm in ("concat", "fusion"):
        for value in STRUCTURAL_ORDER:
            mask = (types["structural"] == value).to_numpy()
            for condition, drop in (("shuffled", "drop_shuffled"),
                                    ("zeroed", "drop_zeroed")):
                ids_n = [f"v2_06::{arm}::train_40k::seed{s}::normal"
                         for s in SEEDS_V2]
                ids_c = [f"v2_06::{arm}::train_40k::seed{s}::{condition}"
                         for s in SEEDS_V2]
                if any(i not in rows for i in ids_n + ids_c):
                    continue
                differences = [rows[a] - rows[b]
                               for a, b in zip(ids_n, ids_c)]
                interval = cc.clustered_interval(differences, dev_cluster, mask)
                per_seed = [round(float(d[mask].mean()), 5)
                            for d in differences]
                slices.append({
                    "source": "closure reconstruction (v2_06)",
                    "family": "v2_06", "slice_kind": "structural",
                    "slice": value, "arm": arm, "scale": "train_40k",
                    "quantity": drop, "point_estimate": interval[
                        "point_estimate"],
                    **{k: v for k, v in interval.items()
                       if k != "point_estimate"},
                    "training_seeds": SEEDS_V2,
                    "per_seed_value": dict(zip(map(str, SEEDS_V2), per_seed)),
                    "across_training_seed_sd_ddof1": round(
                        float(np.std(per_seed, ddof=1)), 5),
                    "analysis_role": "SECONDARY",
                    "comparison_status": "CLEAN_PAIRED_CONTROL",
                    "analysis_status": "COMPUTED_FROM_REPRODUCED_ROW_EVIDENCE",
                })

    # v2_02 structural/semantic/step accuracy slices for the headline heads.
    slice_sets = ([("structural", v, (types["structural"] == v).to_numpy())
                   for v in STRUCTURAL_ORDER]
                  + [("semantic", v, (types["semantic"] == v).to_numpy())
                     for v in SEMANTIC_ORDER]
                  + [("steps", v, bucket == v) for v in STEP_ORDER])
    for arm in ("concat", "fusion"):
        for kind, value, mask in slice_sets:
            ids = [f"v2_02::{arm}::train_40k::seed{s}::normal"
                   for s in SEEDS_V2]
            if any(i not in rows for i in ids):
                continue
            values = [rows[i] for i in ids]
            interval = cc.clustered_interval(values, dev_cluster, mask)
            per_seed = [round(float(v[mask].mean()), 5) for v in values]
            slices.append({
                "source": "closure reconstruction (v2_02)",
                "family": "v2_02", "slice_kind": kind, "slice": value,
                "arm": arm, "scale": "train_40k", "quantity": "accuracy",
                "point_estimate": interval["point_estimate"],
                **{k: v for k, v in interval.items() if k != "point_estimate"},
                "training_seeds": SEEDS_V2,
                "per_seed_value": dict(zip(map(str, SEEDS_V2), per_seed)),
                "across_training_seed_sd_ddof1": round(
                    float(np.std(per_seed, ddof=1)), 5),
                "analysis_role": "DIAGNOSTIC",
                "comparison_status": "DESCRIPTIVE_ONLY",
                "analysis_status": "COMPUTED_FROM_REPRODUCED_ROW_EVIDENCE",
            })

    # ---------------- E2: SigLIP, priority contrast C ----------------------
    e2_clip_partner_40k = {"question_only": "v2_02::question_only",
                           "concat": "v2_02::concat",
                           "fusion": "v2_02::fusion",
                           "product": "v2_04::product_natural"}
    for scale in ("train_40k", "train_250k"):
        for arm in ("question_only", "concat", "product", "fusion"):
            entry = arm_accuracy(rows, f"e2::{arm}::{scale}::seed{{seed}}"
                                       f"::normal", dev_cluster,
                                 f"e2/{arm}/{scale}")
            if entry:
                arms.append({**entry, "family": "e2", "scale": scale,
                             "encoder": "SigLIP ViT-B-16",
                             "answer_set_size": 100})
    for arm, partner in e2_clip_partner_40k.items():
        add(contrast(rows, f"e2::{arm}::train_40k::seed{{seed}}::normal",
                     f"{partner}::train_40k::seed{{seed}}::normal",
                     dev_cluster, f"e2/siglip_minus_clip_{arm}/train_40k",
                     "SECONDARY", "REPRESENTATION_CONFOUNDED",
                     "the two sides are evaluated on identical development "
                     "rows, so the interval is a valid evaluation-sampling "
                     "interval on the row-paired difference. The TRAINING "
                     "streams are not paired: the 768-d and 512-d inputs make "
                     "identical initialisation and shuffle streams "
                     "impossible, so the seed label does not pair two runs. "
                     "Encoder, representation width and head width all move "
                     "together"))
    for scale in ("train_40k", "train_250k"):
        for left, right in (("fusion", "concat"), ("product", "concat"),
                            ("concat", "question_only")):
            add(contrast(rows, f"e2::{left}::{scale}::seed{{seed}}::normal",
                         f"e2::{right}::{scale}::seed{{seed}}::normal",
                         dev_cluster, f"e2/{left}_minus_{right}/{scale}",
                         "SECONDARY", "CAPACITY_CONFOUNDED",
                         "within the SigLIP path; the arms differ in "
                         "trainable parameters as well as in fusion input"))
        for arm in ("question_only", "concat", "product", "fusion"):
            entry = pooled_deficit_interval(
                rows, f"e2::{arm}::{scale}::seed{{seed}}::normal", bucket,
                step_priors, dev_cluster, dev_n,
                f"e2/pooled_ge4_deficit/{arm}/{scale}")
            if entry:
                deficits.append({**entry, "family": "e2",
                                 "encoder": "SigLIP ViT-B-16", "arm": arm,
                                 "scale": scale})

    # ---------------- v2_02 / v2_04 CLIP pooled deficits at 40k ------------
    for family, arm in (("v2_02", "question_only"), ("v2_02", "concat"),
                        ("v2_02", "fusion"), ("v2_04", "product_natural")):
        entry = pooled_deficit_interval(
            rows, f"{family}::{arm}::train_40k::seed{{seed}}::normal", bucket,
            step_priors, dev_cluster, dev_n,
            f"{family}/pooled_ge4_deficit/{arm}/train_40k")
        if entry:
            deficits.append({**entry, "family": family,
                             "encoder": "CLIP ViT-B-32", "arm": arm,
                             "scale": "train_40k"})

    # ---------------- E3: the top-1000 answer set --------------------------
    for scale in ("train_40k", "train_250k"):
        for arm in ("question_only", "concat", "product", "fusion"):
            entry = arm_accuracy(rows, f"e3::{arm}::{scale}::seed{{seed}}"
                                       f"::normal", e3_cluster,
                                 f"e3/{arm}/{scale}")
            if entry:
                arms.append({**entry, "family": "e3", "scale": scale,
                             "encoder": "CLIP ViT-B-32",
                             "answer_set_size": 1000,
                             "denominator_note": (
                                 "the E3 development view is 9,823 rows over "
                                 "776 images, not the 7,714-row top-100 view; "
                                 "an E3 accuracy is never comparable "
                                 "row-for-row with a top-100 accuracy")})
        for left, right in (("fusion", "concat"), ("product", "concat"),
                            ("concat", "question_only")):
            add(contrast(rows, f"e3::{left}::{scale}::seed{{seed}}::normal",
                         f"e3::{right}::{scale}::seed{{seed}}::normal",
                         e3_cluster, f"e3/{left}_minus_{right}/{scale}",
                         "SECONDARY", "CAPACITY_CONFOUNDED",
                         "within the top-1000 path; the arms differ in "
                         "trainable parameters as well as in fusion input"))

    # ---------------- cross-check against the stored gap tables ------------
    # Each arm accuracy already passed the reproduction gate cell by cell.
    # This is the stronger check: the reconstructed CONTRASTS must reproduce
    # the contrast values the source experiments published, seed by seed.
    stored_gaps = {}
    v2_02 = json.loads((RESULTS_ROOT / "v2_02_multiseed" / "results.json")
                       .read_text())["v2_02_multiseed"]
    v2_03 = json.loads((RESULTS_ROOT / "v2_03_param_match" / "results.json")
                       .read_text())["v2_03_param_match"]
    v2_04 = json.loads((RESULTS_ROOT / "v2_04_ablation" / "results.json")
                       .read_text())["v2_04_ablation"]
    v2_06 = json.loads((RESULTS_ROOT / "v2_06_reliance" / "results.json")
                       .read_text())["v2_06_reliance"]
    e2 = json.loads((RESULTS_ROOT / "e2_siglip" / "results.json")
                    .read_text())["e2_siglip"]
    for name, entry in v2_02["paired_gaps"].items():
        stored_gaps[f"v2_02/{name}/train_40k"] = entry
    for name, entry in v2_03["gaps"].items():
        stored_gaps[f"v2_03/{name}/train_40k"] = entry
    for group in ("gaps_576k", "gaps_natural"):
        for name, entry in v2_04[group].items():
            key = ("v2_03" if name.startswith("fusion_narrow_minus_concat")
                   else "v2_04")
            stored_gaps[f"{key}/{name}/train_40k"] = entry
    for arm, entry in v2_06["aggregate"].items():
        for drop in ("drop_shuffled", "drop_zeroed"):
            stored_gaps[f"v2_06/{arm}/{drop}/train_40k"] = entry[drop]
    for scale_key, scale in (("40k", "train_40k"), ("250k", "train_250k")):
        for name, entry in e2["siglip_minus_clip_same_seed"][scale_key].items():
            arm = name.replace("siglip_minus_clip_", "")
            stored_gaps[f"e2/siglip_minus_clip_{arm}/{scale}"] = entry

    checks, disagreements = [], []
    for entry in contrasts:
        stored = stored_gaps.get(entry["contrast_id"])
        if stored is None:
            continue
        reconstructed = entry["per_seed_effect_stored_rounding_convention"]
        per_seed_ok = all(reconstructed[str(s)] == stored["per_seed"][str(s)]
                          for s in SEEDS_V2)
        row = {
            "contrast_id": entry["contrast_id"],
            "per_seed_reproduced_exactly": per_seed_ok,
            "reconstructed_per_seed_stored_convention": reconstructed,
            "reconstructed_per_seed_single_rounded": entry["per_seed_effect"],
            "stored_per_seed": {k: stored["per_seed"][k]
                                for k in map(str, SEEDS_V2)},
            "reconstructed_mean": entry["point_estimate"],
            "stored_mean": stored["mean"],
            "mean_difference": round(
                abs(entry["point_estimate"] - stored["mean"]), 6),
        }
        checks.append(row)
        if not per_seed_ok:
            disagreements.append(row)

    record = {
        "closure_output": "reconstructed statistics (feeds B, D and E)",
        "stored_gap_cross_check": {
            "purpose": ("the reconstructed contrasts must reproduce the "
                        "contrast values the source experiments published, "
                        "seed by seed, not merely the arm accuracies"),
            "comparison_rule": (
                "per-seed effects are compared for exact equality at the "
                "canonical 5-decimal precision. The MEAN may differ from the "
                "stored mean by up to about 1e-5 without any disagreement, "
                "because the stored aggregate averages per-seed values that "
                "were already rounded to 5 decimals while the reconstructed "
                "mean is computed from unrounded per-row differences and "
                "rounded once. That is a rounding-order difference, not a "
                "reproduction failure, and it is reported rather than hidden"),
            "contrasts_checked": len(checks),
            "per_seed_exact": sum(1 for c in checks
                                  if c["per_seed_reproduced_exactly"]),
            "disagreements": disagreements,
            "max_mean_difference": (max(c["mean_difference"] for c in checks)
                                    if checks else 0.0),
            "checks": sorted(checks, key=lambda r: r["contrast_id"]),
        },
        "provenance": cc.provenance(
            "experiments/closure/build_intervals.py",
            [cc.record_source(manifest_path, "reconstructed row manifest"),
             cc.record_source(config.DATA_DIR / "v2" / "dev.csv",
                              "split manifest"),
             cc.record_source(config.DATA_DIR / "v2_1000" / "dev.csv",
                              "split manifest"),
             cc.record_source(config.DATA_DIR / "v2" / "metadata"
                              / "dev_types.csv", "slice metadata"),
             cc.record_source(RESULTS_ROOT / "v2_05_types" / "addendum.json",
                              "fixed v2_05b step priors")]),
        "uncertainty_note": (
            "every interval here is an image-clustered evaluation-sampling "
            "interval. It conditions on the fixed trained seed set and "
            "carries evaluation-sampling variation only. The across-seed "
            "standard deviation reported beside it is a different quantity: "
            "the sample sd over independent training seeds (ddof=1). Neither "
            "is a substitute for the other and neither is described as the "
            "other anywhere in the closure outputs"),
        "multiplicity_note": (
            "every interval is nominal 95 per cent, image-clustered and "
            "UNCORRECTED for multiplicity. No family-wise correction is "
            "applied, no p-value is computed, no new hypothesis test is "
            "introduced, and no result is reported or withheld on the basis "
            "of whether its interval excludes zero"),
        "cluster_units": {
            "v2_dev": {"n_questions": 7714, "n_unique_images": dev_n},
            "v2_1000_dev": {"n_questions": 9823, "n_unique_images": e3_n}},
        "stopped_families": sorted(stopped),
        "blocked_contrasts": [{
            "contrast": "e2/siglip_minus_clip_*/train_250k",
            "reason": ("the CLIP partner at 250k is v2_07, which is stopped "
                       "by a point-estimate reproduction mismatch, so no "
                       "row-paired difference can be formed at that scale"),
            "status": "NOT_COMPUTED",
        }, {
            "contrast": "v2_07/* (all scaling contrasts)",
            "reason": "family stopped by a reproduction mismatch",
            "status": "NOT_COMPUTED",
        }, {
            "contrast": "v2_07 addendum_pooled >=4 deficit",
            "reason": ("depends on v2_07 250k checkpoints; the family is "
                       "stopped. The stored value remains a seed-42, 250k, "
                       "no-interval figure and is not upgraded here"),
            "status": "NOT_COMPUTED",
        }],
        "counts": {"contrasts": len(contrasts), "arms": len(arms),
                   "pooled_deficits": len(deficits), "slices": len(slices)},
        "arm_accuracies": sorted(arms, key=lambda r: r["arm_id"]),
        "contrasts": sorted(contrasts, key=lambda r: r["contrast_id"]),
        "pooled_deficits": sorted(deficits, key=lambda r: r["deficit_id"]),
        "slices": sorted(slices, key=lambda r: (
            r["family"], r["arm"], r["slice_kind"], r["slice"],
            r["quantity"])),
        "clean_test_accessed": False,
    }
    digest = cc.write_json(record,
                           cc.CLOSURE_DIR / "reconstructed_statistics.json")
    print(f"contrasts {len(contrasts)}  arms {len(arms)}  "
          f"deficits {len(deficits)}  slices {len(slices)}")
    print(f"reconstructed_statistics.json sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
