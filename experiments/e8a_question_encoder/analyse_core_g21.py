"""E8A G21 analysis: raw and normalised statistics from prediction artefacts.

    python -B experiments/e8a_question_encoder/analyse_core_g21.py

Every number here is regenerated from the immutable per-row prediction
artefacts of predictions_g21_v1/ — never from the legacy correctness-only
NPZ files and never by copying an earlier report. The clustered bootstrap,
directional rule, step buckets, fixed per-bucket priors and pooled-deficit
definition are imported from analyse_core unchanged, so the statistical
machinery is identical to the pre-G21 analysis and only the input vectors
and the metric labelling change.

Primary metric: single-gold pinned VQA-style normalised exact match.
Secondary metric: strict raw exact match, always separately labelled.
The two coincide row for row on the top-100 support because the collision
inventory measured zero normalised collisions and zero changed strings; that
equality is verified here per cell and condition rather than assumed, and
every interval is therefore constructed once and shared by both metrics.

New in this analysis, as the remediation requires: per question-type and
per step-bucket slices with row counts, unique-image counts and
image-clustered intervals (never row-level intervals that ignore repeated
questions per image); the exact count of intervals constructed; and a
complete old-versus-new comparison against the pre-G21 core_analysis.json.

Writes results/experiments/e8a_question_encoder/core_analysis_g21.json.
Nothing is trained and the embargoed clean-test target is never read.
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

from src import utils  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import g21_scorer as g21  # noqa: E402
from experiments.e8a_question_encoder import analyse_core as core  # noqa: E402
from experiments.e8a_question_encoder import (  # noqa: E402
    analysis_integrity as integrity)

ARMS = core.ARMS
SCALES = core.SCALES
SEEDS = core.SEEDS
CONTRASTS = core.CONTRASTS
PRED_DIR = e8a.OUT_DIR / "predictions_g21_v1"
OLD_ANALYSIS = e8a.OUT_DIR / "core_analysis.json"

CONDITIONS = ("normal", "fixed_image", "fixed_question", "shuffled_image")

# One global counter so the number of intervals REPORTED is the number
# CONSTRUCTED, exactly; the earlier report's "twenty-one" was neither.
INTERVAL_COUNT = {"headline": 0, "slice": 0}


def counted_interval(kind: str, draws: np.ndarray) -> list:
    INTERVAL_COUNT[kind] += 1
    lower, upper = core.percentile_interval(draws)
    return [lower, upper]


# --- Loading and verifying the prediction artefacts --------------------------

def load_artefacts(view: integrity.DevView) -> dict:
    """Both per-row metrics for every cell and condition, verified.

    Returns data[arm][scale][condition] = {"raw": [seed arrays],
    "normalized": [seed arrays]}, plus per-cell audit facts.
    """
    scorer = g21.scorer_provenance()
    data = {arm: {scale: {c: {"raw": [], "normalized": []}
                          for c in CONDITIONS}
                  for scale in SCALES} for arm in ARMS}
    audits = {}
    for arm in ARMS:
        for scale in SCALES:
            for seed in SEEDS:
                cell = f"{arm}/{scale}/seed{seed}"
                csv_path = (PRED_DIR
                            / f"predictions_{arm}_{scale}_seed{seed}.csv.gz")
                sidecar = json.loads(
                    csv_path.with_suffix("").with_suffix(".json").read_text())
                measured = g21.sha256_file(csv_path)
                if measured != sidecar["artefact"]["sha256"]:
                    raise AssertionError(f"{cell}: prediction artefact does "
                                         f"not match its sidecar hash")
                frame = pd.read_csv(csv_path, dtype={"questionId": str,
                                                     "imageId": str},
                                    keep_default_na=False)
                if len(frame) != view.n_rows * len(CONDITIONS):
                    raise AssertionError(f"{cell}: {len(frame)} rows")
                for column, expected in (
                        ("checkpoint_sha256",
                         {sidecar["pinned"]["checkpoint_sha256"]}),
                        ("vocabulary_sha256", {view.vocabulary_sha256}),
                        ("evaluation_row_sha256", {view.row_order_sha256}),
                        ("scorer_source_sha256",
                         {scorer["vendored_source_sha256"]})):
                    if set(frame[column].unique()) != expected:
                        raise AssertionError(f"{cell}: unexpected {column}")
                for condition in CONDITIONS:
                    rows = frame[frame.condition == condition]
                    if not np.array_equal(rows["questionId"].to_numpy(),
                                          view.question_ids):
                        raise AssertionError(
                            f"{cell}/{condition}: rows are not the canonical "
                            f"development rows in order")
                    if not np.array_equal(
                            rows["gold_label_id"].to_numpy(np.int64),
                            view.labels):
                        raise AssertionError(f"{cell}/{condition}: gold "
                                             f"labels differ")
                    raw = rows["raw_correct"].to_numpy(bool)
                    normalized = rows["normalized_correct"].to_numpy(bool)
                    data[arm][scale][condition]["raw"].append(
                        raw.astype(np.float64))
                    data[arm][scale][condition]["normalized"].append(
                        normalized.astype(np.float64))
                audits[cell] = {
                    "artefact_sha256": measured,
                    "raw_equals_normalized_rowwise": all(
                        np.array_equal(
                            data[arm][scale][c]["raw"][-1],
                            data[arm][scale][c]["normalized"][-1])
                        for c in CONDITIONS),
                    "g11": sidecar["gates"][
                        "g11_repeat_bitwise_identical"],
                    "stored_correctness_reproduced": sidecar["gates"][
                        "stored_correctness_reproduced_row_for_row"],
                    "checkpoint_sha256":
                        sidecar["pinned"]["checkpoint_sha256"],
                    "timings_seconds": sidecar["timings_seconds"],
                }
    return {"data": data, "audits": audits, "scorer": scorer}


# --- Shared statistics helpers ------------------------------------------------

def seed_stats(vectors) -> dict:
    values = [round(float(v.mean()), 5) for v in vectors]
    return {"per_seed": values,
            "mean": round(float(np.mean([v.mean() for v in vectors])), 5),
            "std": round(float(np.std([v.mean() for v in vectors],
                                      ddof=1)), 5)}


def contrast_block(correct_a, correct_b, image_index, n_images) -> dict:
    per_seed = [round(float(correct_a[i].mean() - correct_b[i].mean()), 5)
                for i in range(len(SEEDS))]

    def statistic(rows):
        return np.mean([float(correct_a[s][rows].mean()
                              - correct_b[s][rows].mean())
                        for s in range(len(correct_a))])
    interval = counted_interval(
        "headline", core.clustered_draws(statistic, image_index, n_images))
    return {"per_seed_differences": {f"seed{SEEDS[i]}": per_seed[i]
                                     for i in range(len(SEEDS))},
            "mean": round(float(np.mean(per_seed)), 5),
            "std": round(float(np.std(per_seed, ddof=1)), 5),
            "ci95_image_clustered": interval,
            "n_clusters": n_images,
            "n_draws": core.BOOTSTRAP_DRAWS,
            "rng_seed": core.BOOTSTRAP_SEED,
            "directional_rule_outcome": core.directional_rule(
                per_seed, interval[0], interval[1])}


def deficit_contrast_block(correct_a, correct_b, steps, image_index,
                           n_images) -> dict:
    block = core.deficit_contrast(correct_a, correct_b, steps, image_index,
                                  n_images)
    INTERVAL_COUNT["headline"] += 1  # constructed inside deficit_contrast
    return block


# --- Slice analysis -----------------------------------------------------------

def slice_bootstrap(vectors, mask: np.ndarray, image_index: np.ndarray) -> list:
    """Image-clustered interval for a slice's three-seed mean accuracy.

    The resampling scheme is the analyse_core one restricted to the slice:
    draw as many images as the slice represents, with replacement, take every
    slice row of every drawn image, average rowwise correctness per seed and
    then across seeds. Implemented with per-image sums so 2,000 draws stay
    cheap; the statistic is identical to clustered_draws' because every row
    of a drawn image enters the mean.
    """
    slice_images = np.unique(image_index[mask])
    position = {image: i for i, image in enumerate(slice_images)}
    sums = np.zeros((len(SEEDS), len(slice_images)))
    counts = np.zeros(len(slice_images))
    rows = np.nonzero(mask)[0]
    for row in rows:
        i = position[image_index[row]]
        counts[i] += 1
        for s in range(len(SEEDS)):
            sums[s, i] += vectors[s][row]
    rng = np.random.default_rng(core.BOOTSTRAP_SEED)
    draws = []
    for _ in range(core.BOOTSTRAP_DRAWS):
        sampled = rng.integers(0, len(slice_images), size=len(slice_images))
        n = counts[sampled].sum()
        draws.append(float(np.mean([sums[s, sampled].sum() / n
                                    for s in range(len(SEEDS))])))
    draws = np.asarray(draws)
    if not np.all(np.isfinite(draws)):
        raise AssertionError("non-finite slice bootstrap draw")
    return counted_interval("slice", draws)


def slice_analysis(data, view, types: pd.DataFrame,
                   image_index: np.ndarray) -> dict:
    """Accuracy per question-type and step slice with counts and intervals."""
    slices = {}
    for column, label in (("structural", "structural_type"),
                          ("semantic", "semantic_type")):
        for value in sorted(types[column].unique()):
            slices[f"{label}:{value}"] = \
                (types[column] == value).to_numpy()
    bucket = core.bucketize(types["n_steps"].to_numpy())
    for value in core.STEP_ORDER:
        slices[f"steps:{value}"] = bucket == value

    out = {}
    for name, mask in slices.items():
        n_rows = int(mask.sum())
        n_images = int(len(np.unique(image_index[mask])))
        entry = {"n_rows": n_rows, "n_unique_images": n_images, "arms": {}}
        for arm in ARMS:
            for scale in SCALES:
                vectors = data[arm][scale]["normal"]["normalized"]
                accuracy = round(float(np.mean(
                    [v[mask].mean() for v in vectors])), 5)
                cell = {"normalized_exact_mean": accuracy,
                        "seed_std": round(float(np.std(
                            [v[mask].mean() for v in vectors], ddof=1)), 5)}
                if n_images >= 2:
                    cell["ci95_image_clustered"] = slice_bootstrap(
                        vectors, mask, image_index)
                else:
                    cell["ci95_image_clustered"] = None
                    cell["interval_note"] = ("fewer than two image clusters; "
                                             "no valid clustered interval")
                entry["arms"][f"{arm}/{scale}"] = cell
        out[name] = entry
    return out


# --- Old-versus-new comparison ------------------------------------------------

def old_new_comparison(old: dict, analyses: dict, across: dict) -> dict:
    """Every headline figure: old value, new raw, new normalised, delta."""
    comparison = {"accuracies": {}, "contrasts": {}, "scale_gains": {},
                  "deficits": {}, "deficit_contrasts": {},
                  "reliance": {}}
    for scale in SCALES:
        old_scale = old["analyses"][scale]
        new_scale = analyses[scale]
        for arm in ARMS:
            old_value = old_scale["per_arm_summary"][arm][
                "mean_dev_accuracy"]
            new_raw = new_scale["per_arm_summary_raw"][arm]["mean"]
            new_norm = new_scale["per_arm_summary_normalized"][arm]["mean"]
            comparison["accuracies"][f"{arm}/{scale}"] = {
                "old_label_argmax": old_value,
                "new_raw_exact": new_raw,
                "new_normalized_exact": new_norm,
                "delta_normalized_minus_old": round(new_norm - old_value, 5)}
        for name in CONTRASTS:
            old_c = old_scale["contrasts"][name]
            new_c = new_scale["contrasts_normalized"][name]
            comparison["contrasts"][f"{name}/{scale}"] = {
                "old": {"mean": old_c["mean"],
                        "ci": old_c["ci95_image_clustered"],
                        "rule": old_c["directional_rule_outcome"]},
                "new_normalized": {"mean": new_c["mean"],
                                   "ci": new_c["ci95_image_clustered"],
                                   "rule": new_c[
                                       "directional_rule_outcome"]},
                "delta_mean": round(new_c["mean"] - old_c["mean"], 5),
                "rule_changed": new_c["directional_rule_outcome"]
                != old_c["directional_rule_outcome"]}
        for arm in ARMS:
            old_d = old_scale["step_deficits"]["per_arm"][arm][
                "mean_pooled_ge4_deficit"]
            new_d = new_scale["step_deficits"]["per_arm"][arm][
                "mean_pooled_ge4_deficit"]
            comparison["deficits"][f"{arm}/{scale}"] = {
                "old": old_d, "new_normalized": new_d,
                "delta": round(new_d - old_d, 5)}
        for name in CONTRASTS:
            old_dc = old_scale["step_deficits"]["contrasts"][name]
            new_dc = new_scale["step_deficits"]["contrasts"][name]
            comparison["deficit_contrasts"][f"{name}/{scale}"] = {
                "old_mean": old_dc["mean"], "new_mean": new_dc["mean"],
                "delta": round(new_dc["mean"] - old_dc["mean"], 5),
                "rule_changed": new_dc["directional_rule_outcome"]
                != old_dc["directional_rule_outcome"]}
        for arm in ARMS:
            for column in core.RELIANCE_COLUMNS:
                old_r = old_scale["reliance"][arm][column]["mean"]
                new_r = new_scale["reliance_normalized"][arm][
                    column.replace("normal_minus_", "")]["mean"]
                comparison["reliance"][f"{arm}/{scale}/{column}"] = {
                    "old": old_r, "new_normalized": new_r,
                    "delta": round(new_r - old_r, 5)}
    for arm in ARMS:
        old_g = old["across_scales"]["accuracy_scale_gain_250k_minus_40k"][
            arm]
        new_g = across["accuracy_scale_gain_250k_minus_40k"][arm]
        comparison["scale_gains"][arm] = {
            "old": {"mean": old_g["mean"],
                    "ci": old_g["ci95_image_clustered"]},
            "new_normalized": {"mean": new_g["mean"],
                               "ci": new_g["ci95_image_clustered"]},
            "delta_mean": round(new_g["mean"] - old_g["mean"], 5),
            "rule_changed": new_g["directional_rule_outcome"]
            != old_g["directional_rule_outcome"]}
    changed = []
    for family, block in comparison.items():
        for key, entry in block.items():
            delta = entry.get("delta", entry.get("delta_mean"))
            if delta is not None and abs(delta) > 0:
                changed.append(f"{family}/{key}: delta {delta}")
            if entry.get("rule_changed"):
                changed.append(f"{family}/{key}: directional rule changed")
    comparison["summary"] = {
        "any_numerical_change": bool(changed),
        "changes": changed,
        "reading": (
            "no headline value, interval or directional outcome changes"
            if not changed else
            "the listed values changed and the narrative must follow them")}
    return comparison


# --- Main ---------------------------------------------------------------------

def analyse_scale_g21(scale, data, steps, image_index, n_images,
                      old_records) -> dict:
    out = {"scale": scale}
    for metric in ("raw", "normalized"):
        out[f"per_arm_summary_{metric}"] = {
            arm: seed_stats(data[arm][scale]["normal"][metric])
            for arm in ARMS}
    out["contrasts_normalized"] = {
        name: contrast_block(data[a][scale]["normal"]["normalized"],
                             data[b][scale]["normal"]["normalized"],
                             image_index, n_images)
        for name, (a, b) in CONTRASTS.items()}
    out["contrasts_raw_note"] = (
        "raw-exact contrasts are numerically identical because the rowwise "
        "vectors are identical (verified per cell); the intervals above are "
        "constructed once and shared, not re-run per metric")
    out["reliance_normalized"] = {}
    for arm in ARMS:
        normal = data[arm][scale]["normal"]["normalized"]
        out["reliance_normalized"][arm] = {}
        for condition in ("fixed_image", "fixed_question", "shuffled_image"):
            drops = [float(normal[i].mean()
                           - data[arm][scale][condition]["normalized"][i]
                           .mean()) for i in range(len(SEEDS))]
            out["reliance_normalized"][arm][condition] = {
                "per_seed": [round(d, 5) for d in drops],
                "mean": round(float(np.mean(drops)), 5),
                "std": round(float(np.std(drops, ddof=1)), 5)}
    out["degeneracy_per_arm"] = {}
    for arm in ARMS:
        per_seed = {}
        for i, seed in enumerate(SEEDS):
            vectors = data[arm][scale]
            share_source = old_records[(arm, scale, seed)]
            normal_conditions = share_source["evaluation"]["conditions"][
                "normal"]
            drop = round(float(
                vectors["normal"]["normalized"][i].mean()
                - vectors["fixed_question"]["normalized"][i].mean()), 5)
            entropy = normal_conditions["prediction_entropy_nats"]
            share = normal_conditions["maximum_class_share"]
            per_seed[f"seed{seed}"] = {
                "prediction_entropy_nats_from_validated_record": entropy,
                "maximum_class_share": share,
                "distinct_answers":
                    normal_conditions["distinct_answers_predicted"],
                "normal_minus_fixed_question": drop,
                "question_insensitivity_fires":
                    drop <= e8a.QUESTION_INSENSITIVITY_MAX,
                "collapse_fires": (entropy <= e8a.COLLAPSE_ENTROPY_MAX_NATS
                                   or share
                                   >= e8a.COLLAPSE_CLASS_SHARE_MIN)}
            per_seed[f"seed{seed}"]["rule_fires"] = (
                per_seed[f"seed{seed}"]["question_insensitivity_fires"]
                or per_seed[f"seed{seed}"]["collapse_fires"])
        out["degeneracy_per_arm"][arm] = per_seed
    out["degeneracy_note"] = (
        "maximum class share, distinct answers and the fixed-question drop "
        "are regenerated from the prediction artefacts; prediction entropy "
        "needs the full logit distribution, which no prediction artefact "
        "carries, and is quoted from the validated run records whose "
        "argmax predictions this re-inference reproduced row for row")
    out["step_deficits"] = {
        "definition": steps["definition"],
        "prior_accuracy_by_slice": steps["prior_accuracy_by_slice"],
        "bucket_counts": steps["bucket_counts"],
        "per_arm": {arm: core.deficit_block(
            data[arm][scale]["normal"]["normalized"], steps)
            for arm in ARMS},
        "contrasts": {name: deficit_contrast_block(
            data[a][scale]["normal"]["normalized"],
            data[b][scale]["normal"]["normalized"], steps, image_index,
            n_images) for name, (a, b) in CONTRASTS.items()}}
    return out


def main() -> int:
    utils.set_seed()
    view = integrity.canonical_dev_view()
    loaded = load_artefacts(view)
    data, audits = loaded["data"], loaded["audits"]

    equality = {cell: audit["raw_equals_normalized_rowwise"]
                for cell, audit in audits.items()}
    if not all(equality.values()):
        raise AssertionError(f"raw and normalised vectors differ for "
                             f"{[c for c, ok in equality.items() if not ok]}"
                             f"; the coincidence claim must not be asserted")

    types = pd.read_csv(e8a.V2_DIR / "metadata" / "dev_types.csv",
                        dtype={"questionId": str}, keep_default_na=False)
    if not np.array_equal(types["questionId"].to_numpy().astype(str),
                          view.question_ids):
        raise AssertionError("dev_types.csv is not row-aligned with dev.csv")
    steps = core.step_view(view)

    unique = sorted(set(view.image_ids))
    index_of = {value: position for position, value in enumerate(unique)}
    image_index = np.array([index_of[value] for value in view.image_ids])

    manifest = json.loads(
        (e8a.OUT_DIR / "execution_manifest.json").read_text())[
            "e8a_execution_manifest"]
    old_records = {}
    from experiments.e8a_question_encoder import cell_validator
    for arm in ARMS:
        for scale in SCALES:
            for seed in SEEDS:
                old_records[(arm, scale, seed)] = cell_validator.load_cell(
                    manifest, arm, scale, seed)

    analyses = {scale: analyse_scale_g21(scale, data, steps, image_index,
                                         len(unique), old_records)
                for scale in SCALES}

    across = {"accuracy_scale_gain_250k_minus_40k": {},
              "step_deficit_change": {}}
    small, large = SCALES
    for arm in ARMS:
        across["accuracy_scale_gain_250k_minus_40k"][arm] = contrast_block(
            data[arm][large]["normal"]["normalized"],
            data[arm][small]["normal"]["normalized"], image_index,
            len(unique))
        across["step_deficit_change"][arm] = deficit_contrast_block(
            data[arm][large]["normal"]["normalized"],
            data[arm][small]["normal"]["normalized"], steps, image_index,
            len(unique))

    slices = slice_analysis(data, view, types, image_index)

    old = json.loads(OLD_ANALYSIS.read_text())["e8a_core_analysis"]
    comparison = old_new_comparison(old, analyses, across)

    record = {"metadata": utils.run_metadata(),
              "e8a_core_analysis_g21": {
                  "metric_definitions": {
                      "primary": "single-gold pinned VQA-style normalised "
                                 "exact match (G21)",
                      "secondary": "strict raw exact match, separately "
                                   "labelled",
                      "scorer": loaded["scorer"]},
                  "artefact_audits": audits,
                  "raw_equals_normalized_rowwise_everywhere": True,
                  "coincidence_basis": (
                      "measured: the top-100 vocabulary has zero normalised "
                      "collisions and zero strings changed by normalisation "
                      "(g21_scorer_inventory.json), and the per-cell rowwise "
                      "equality above verifies it on every prediction"),
                  "analyses": analyses,
                  "across_scales": across,
                  "type_and_step_slices": {
                      "note": ("every interval is image-clustered over the "
                               "slice's own represented images; no row-level "
                               "interval is reported anywhere"),
                      "slices": slices},
                  "intervals_constructed": {
                      "headline": INTERVAL_COUNT["headline"],
                      "per_slice": INTERVAL_COUNT["slice"],
                      "total": (INTERVAL_COUNT["headline"]
                                + INTERVAL_COUNT["slice"]),
                      "disclosure": (
                          "every interval is nominal 95 per cent, "
                          "image-clustered, and uncorrected for "
                          "multiplicity; the counts above are incremented "
                          "at construction time, so the number reported is "
                          "the number constructed")},
                  "old_versus_new": comparison,
                  "clean_test_accessed": False}}
    utils.save_json(record, e8a.OUT_DIR / "core_analysis_g21.json")

    for scale in SCALES:
        block = analyses[scale]
        print(f"\n=== {scale} (normalized exact, primary) ===")
        for arm in ARMS:
            s = block["per_arm_summary_normalized"][arm]
            print(f"  {arm:4s} mean {s['mean']:.5f} sd {s['std']:.5f} "
                  f"per-seed {s['per_seed']}")
        for name, c in block["contrasts_normalized"].items():
            print(f"  {name:14s} {c['mean']:+.5f} CI "
                  f"{c['ci95_image_clustered']} -> "
                  f"{c['directional_rule_outcome']}")
    print(f"\nintervals constructed: {INTERVAL_COUNT['headline']} headline "
          f"+ {INTERVAL_COUNT['slice']} slice")
    print(f"old-versus-new changes: "
          f"{record['e8a_core_analysis_g21']['old_versus_new']['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
