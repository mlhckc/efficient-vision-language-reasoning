"""E8A SmolLM2-135M core: collation, primary statistics and analyses.

    python -B experiments/e8a_question_encoder/analyse_core.py --scale train_40k
    python -B experiments/e8a_question_encoder/analyse_core.py            # both

Loads every arm-scale-seed record that exists — the three reused Phase-1B
pilots via the execution manifest's `reuse_artefacts` pointers, and the matrix
runs from their own records — and produces, per scale:

  * the per-arm-seed table with every required field;
  * the three paired contrasts with all three per-seed differences, their mean
    and standard deviation, a 95 per cent fixed-seed-set image-clustered
    bootstrap interval, and the canonical universal directional-rule outcome;
  * the A1r degeneracy classification per seed;

and, where both scales exist, the scale, reliance and >=4-step analyses.

Before any number is computed, every cell passes the integrity assertions in
analysis_integrity.py: the evaluation rows are the canonical dev.csv rows in
canonical order, the result belongs to the arm, scale and seed it is filed
under, and every artefact hash is re-measured now rather than trusted from the
run that wrote it.

The accuracy bootstrap is the one implemented at
experiments/v3_03_scaling/run.py:397-427 and reused unchanged: 2,000 draws,
RNG seed 0, alpha 0.05, clustering by development imageId, seeds combined by
the within-draw seed mean. The >=4-step deficit is the repaired v3_02a
definition at experiments/v3_02a_refs/stats_repair.py:241-247 under the fixed
train_40k per-bucket priors of v2_05b, which are reloaded and re-derived here
rather than assumed.

Artefact names are read from the manifest, never derived: the three reused
correctness files predate the scale-qualified naming.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import (  # noqa: E402
    analysis_integrity as integrity)

ARMS = ("A0p", "A1", "A1r")
SCALES = ("train_40k", "train_250k")
SEEDS = (0, 1, 2)
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 0
ALPHA = 0.05
STEP_ORDER = ["<=2", "3", "4", ">=5"]

# The v2_05b per-bucket prior accuracies, held fixed at every scale exactly as
# v2_07, v3_03 and E2 hold them, and re-derived below rather than trusted.
PRIOR_SOURCE = (config.RESULTS_DIR / "experiments" / "v2_05_types"
                / "addendum.json")

RELIANCE_COLUMNS = ("normal_minus_fixed_image",
                    "normal_minus_fixed_question",
                    "normal_minus_shuffled_image")

CONTRASTS = {
    "A1_minus_A1r": ("A1", "A1r"),
    "A1_minus_A0p": ("A1", "A0p"),
    "A0p_minus_A1r": ("A0p", "A1r"),
}


def load_cells(manifest: dict, view: integrity.DevView) -> dict:
    """Every completed cell, keyed by (arm, scale, seed), integrity-checked."""
    cells = {}
    for spec in manifest["runs"]:
        key = (spec["arm"], spec["scale"], spec["seed"])
        artefacts = spec["expected_artefacts"]
        if spec["reuse_artefacts"]:
            reuse = spec["reuse_artefacts"]
            path = e8a.OUT_DIR / reuse["record"]["path"]
            record = json.loads(path.read_text())
            block = (record["e8a_a0p_pilot"] if spec["arm"] == "A0p"
                     else record["e8a_pilot"])
            run = (block["run"] if spec["arm"] == "A0p"
                   else block["runs"][spec["arm"]])
            evaluation = (block["evaluation"] if spec["arm"] == "A0p"
                          else block["evaluations"][spec["arm"]])
            correctness = e8a.OUT_DIR / reuse["correctness"]["path"]
            source = reuse["record"]["path"]
        else:
            path = e8a.OUT_DIR / artefacts["record"]
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            block = record["e8a_core_run"]
            run, evaluation = block["run"], block["evaluation"]
            correctness = e8a.OUT_DIR / artefacts["correctness"]
            source = artefacts["record"]
        evidence = integrity.cell_evidence(
            f"{key[0]}/{key[1]}/seed{key[2]}", key[0], key[1], key[2], path,
            record, block, run, evaluation, correctness, view)
        cells[key] = {"run": run, "evaluation": evaluation,
                      "correctness_path": correctness, "source": source,
                      "integrity": evidence}
    return cells


def per_arm_row(cell: dict) -> dict:
    run, ev = cell["run"], cell["evaluation"]
    normal = ev["conditions"]["normal"]
    diffs = ev["differences_from_normal"]
    eff = ev["efficiency"]
    evidence = cell["integrity"]
    return {
        "dev_accuracy": normal["accuracy"],
        "best_epoch": run["best_epoch"],
        "epochs_run": run["epochs_run"],
        "stopped_by": run["stopped_by"],
        "wall_clock_hours": run["wall_clock_hours"],
        "seconds_per_epoch": run["seconds_per_epoch"],
        "normal_minus_fixed_image": diffs["normal_minus_fixed_image"],
        "normal_minus_fixed_question": diffs["normal_minus_fixed_question"],
        "normal_minus_shuffled_image":
            diffs["normal_minus_shuffled_image_derangement"],
        "prediction_entropy_nats": normal["prediction_entropy_nats"],
        "maximum_class_share": normal["maximum_class_share"],
        "distinct_answers": normal["distinct_answers_predicted"],
        "degeneracy_rule_fires": ev["degeneracy_rule"]["rule_fires"],
        "peak_allocated_mib": run["peak_allocated_mib"],
        "head_latency_ms_mean": eff["head_latency_ms_mean"],
        "trainable_parameters": eff["trainable_parameters"],
        "frozen_question_encoder_parameters": eff.get(
            "frozen_question_encoder_parameters",
            eff.get("frozen_language_model_parameters")),
        "total_loaded_parameters": eff["total_loaded_parameters"],
        "checkpoint_mib": eff["checkpoint_mib"],
        "checkpoint_sha256": evidence["checkpoint_sha256"],
        "correctness_sha256": evidence["prediction_sha256"],
        "code_head": evidence["code_head"],
        "code_head_dirty": evidence["code_head_dirty"],
        "source_record": cell["source"],
    }


# --- Clustered bootstrap --------------------------------------------------

def clustered_draws(statistic, image_index, n_images) -> np.ndarray:
    """Image-clustered resampling shared by every interval reported here.

    The draw stream is the one at experiments/v3_03_scaling/run.py:397-427:
    2,000 draws from a fresh default_rng(0), each drawing as many images as
    the development set has, with replacement, and taking every row of every
    drawn image. `statistic` receives the drawn row indices and returns the
    scalar for that draw, so the accuracy and deficit intervals differ only in
    that scalar and not in the resampling.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows_by_image = [np.nonzero(image_index == i)[0] for i in range(n_images)]
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = rng.integers(0, n_images, size=n_images)
        rows = np.concatenate([rows_by_image[i] for i in sampled])
        draws.append(float(statistic(rows)))
    draws = np.asarray(draws)
    # A resampled draw that emptied a step bucket would produce a silent NaN
    # and a meaningless percentile rather than an error.
    if not np.all(np.isfinite(draws)):
        raise AssertionError(
            f"{int((~np.isfinite(draws)).sum())} of {BOOTSTRAP_DRAWS} "
            f"bootstrap draws are not finite; a resampled draw left a "
            f"required subset empty")
    return draws


def percentile_interval(draws: np.ndarray):
    return (round(float(np.percentile(draws, 100 * ALPHA / 2)), 5),
            round(float(np.percentile(draws, 100 * (1 - ALPHA / 2))), 5))


def clustered_bootstrap(correct_a, correct_b, image_index, n_images):
    """Paired image-clustered interval for an accuracy difference."""
    def statistic(rows):
        return np.mean([float(correct_a[s][rows].mean()
                              - correct_b[s][rows].mean())
                        for s in range(len(correct_a))])
    return percentile_interval(clustered_draws(statistic, image_index,
                                               n_images))


def directional_rule(per_seed, lower, upper) -> str:
    """The canonical universal primary directional rule, applied exactly."""
    mean = float(np.mean(per_seed))
    positive = (mean > 0 and lower > 0
                and sum(1 for d in per_seed if d > 0) >= 2)
    negative = (mean < 0 and upper < 0
                and sum(1 for d in per_seed if d < 0) >= 2)
    if positive:
        return "positive directional evidence"
    if negative:
        return "negative directional evidence"
    return "uncertain or mixed evidence"


# --- Step buckets and the repaired >=4-step deficit -----------------------

def bucketize(steps: np.ndarray) -> np.ndarray:
    """The canonical bucket edges of experiments/v2_05_types/addendum.py."""
    return np.where(steps <= 2, "<=2",
                    np.where(steps == 3, "3",
                             np.where(steps == 4, "4", ">=5")))


def step_view(view: integrity.DevView) -> dict:
    """Step buckets, the >=4 mask and the fixed per-bucket prior, gated.

    The priors are the v2_05b train_40k majority answers. They are reloaded
    from the stored addendum and independently recomputed from train_40k here;
    a disagreement means the deficit would be measured against a different
    reference than every earlier experiment, so it stops the analysis.
    """
    types = pd.read_csv(e8a.V2_DIR / "metadata" / "dev_types.csv",
                        dtype={"questionId": str}, keep_default_na=False)
    if not np.array_equal(types["questionId"].to_numpy().astype(str),
                          view.question_ids):
        raise AssertionError(
            "data/v2/metadata/dev_types.csv is not row-aligned with "
            "data/v2/dev.csv; the step buckets would be attached to the "
            "wrong questions")
    bucket = bucketize(types["n_steps"].to_numpy())
    ge4_mask = (bucket == "4") | (bucket == ">=5")

    stored = json.loads(PRIOR_SOURCE.read_text())["v2_05b_addendum"][
        "prior_accuracy_by_slice"]
    train = pd.read_csv(e8a.V2_DIR / "train_40k.csv",
                        dtype={"questionId": str, "imageId": str},
                        keep_default_na=False)
    train_types = pd.read_csv(e8a.V2_DIR / "metadata" / "train_40k_types.csv",
                              dtype={"questionId": str},
                              keep_default_na=False)
    if not np.array_equal(train_types["questionId"].to_numpy().astype(str),
                          train["questionId"].to_numpy().astype(str)):
        raise AssertionError("train_40k_types.csv is not row-aligned with "
                             "train_40k.csv")
    train_bucket = bucketize(train_types["n_steps"].to_numpy())
    dev_answers = pd.read_csv(
        e8a.V2_DIR / "dev.csv", dtype={"questionId": str, "imageId": str},
        keep_default_na=False)["answer"].to_numpy()
    train_answers = train["answer"].to_numpy()

    recomputed, majority = {}, {}
    for value in STEP_ORDER:
        counts = Counter(train_answers[train_bucket == value])
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        majority[f"steps:{value}"] = top
        recomputed[f"steps:{value}"] = round(
            float((dev_answers[bucket == value] == top).mean()), 5)
    disagreeing = {k: (stored.get(k), recomputed[k]) for k in recomputed
                   if stored.get(k) != recomputed[k]}
    if disagreeing:
        raise AssertionError(
            f"the fixed per-bucket priors no longer reproduce the stored "
            f"v2_05b values {disagreeing}; the deficit would be measured "
            f"against a different reference than every earlier experiment")

    prior_correct = np.zeros(len(bucket), dtype=np.float64)
    for value in STEP_ORDER:
        prior_correct[bucket == value] = recomputed[f"steps:{value}"]
    return {"bucket": bucket, "ge4_mask": ge4_mask,
            "prior_correct": prior_correct,
            "prior_accuracy_by_slice": recomputed,
            "majority_answer_by_slice": majority,
            "bucket_counts": {v: int((bucket == v).sum())
                              for v in STEP_ORDER},
            "ge4_rows": int(ge4_mask.sum()),
            "prior_source": str(PRIOR_SOURCE.relative_to(PROJECT_ROOT)),
            "definition": (
                "the repaired v3_02a pooled deficit: the unweighted mean of "
                "the four per-bucket lifts over the fixed train_40k prior, "
                "minus the question-weighted combined >=4 lift "
                "(experiments/v3_02a_refs/stats_repair.py:241-247)")}


def pooled_deficit(correct, bucket, ge4_mask, prior_correct):
    """experiments/v3_02a_refs/stats_repair.py:241-247, unchanged."""
    lifts = [float(correct[bucket == v].mean())
             - float(prior_correct[bucket == v].mean())
             for v in STEP_ORDER]
    combined = float(correct[ge4_mask].mean()) \
        - float(prior_correct[ge4_mask].mean())
    return float(np.mean(lifts)) - combined, combined


def deficit_block(correct_by_seed, steps) -> dict:
    """Per-seed pooled deficit, its >=4 lift and the per-bucket lifts."""
    per_seed = []
    for seed_index, correct in enumerate(correct_by_seed):
        deficit, combined = pooled_deficit(
            correct, steps["bucket"], steps["ge4_mask"],
            steps["prior_correct"])
        per_seed.append({
            "seed": SEEDS[seed_index],
            "pooled_ge4_deficit": round(deficit, 5),
            "combined_ge4_lift": round(combined, 5),
            "bucket_accuracy": {
                v: round(float(correct[steps["bucket"] == v].mean()), 5)
                for v in STEP_ORDER},
            "bucket_lift": {
                v: round(float(correct[steps["bucket"] == v].mean())
                         - float(steps["prior_correct"][
                             steps["bucket"] == v].mean()), 5)
                for v in STEP_ORDER}})
    values = [row["pooled_ge4_deficit"] for row in per_seed]
    return {"per_seed": per_seed,
            "mean_pooled_ge4_deficit": round(float(np.mean(values)), 5),
            "std_pooled_ge4_deficit": round(float(np.std(values, ddof=1)), 5)}


def deficit_contrast(correct_a, correct_b, steps, image_index, n_images):
    """Paired interval for one arm's pooled deficit minus another's."""
    def statistic(rows):
        bucket, mask = steps["bucket"][rows], steps["ge4_mask"][rows]
        prior = steps["prior_correct"][rows]
        values = []
        for s in range(len(correct_a)):
            first, _ = pooled_deficit(correct_a[s][rows], bucket, mask, prior)
            second, _ = pooled_deficit(correct_b[s][rows], bucket, mask, prior)
            values.append(first - second)
        return np.mean(values)
    per_seed = []
    for s in range(len(correct_a)):
        first, _ = pooled_deficit(correct_a[s], steps["bucket"],
                                  steps["ge4_mask"], steps["prior_correct"])
        second, _ = pooled_deficit(correct_b[s], steps["bucket"],
                                   steps["ge4_mask"], steps["prior_correct"])
        per_seed.append(round(first - second, 5))
    lower, upper = percentile_interval(
        clustered_draws(statistic, image_index, n_images))
    return {"per_seed_differences": {f"seed{SEEDS[i]}": per_seed[i]
                                     for i in range(len(per_seed))},
            "mean": round(float(np.mean(per_seed)), 5),
            "std": round(float(np.std(per_seed, ddof=1)), 5),
            "ci95_image_clustered": [lower, upper],
            "directional_rule_outcome": directional_rule(per_seed, lower,
                                                         upper)}


# --- Per-scale analysis ---------------------------------------------------

CONDITION_OF = {"normal_minus_fixed_image": "fixed_image",
                "normal_minus_fixed_question": "fixed_question",
                "normal_minus_shuffled_image": "shuffled_image_derangement"}


def load_correctness(cells, scale) -> dict:
    correct = {}
    for arm in ARMS:
        correct[arm] = []
        for seed in SEEDS:
            data = np.load(cells[(arm, scale, seed)]["correctness_path"],
                           allow_pickle=True)
            correct[arm].append(data["normal"].astype(np.float64))
    return correct


def load_conditions(cells, scale) -> dict:
    """Every intervention vector, so reliance is differenced before rounding.

    Taking the difference from the records' stored five-decimal fields would
    round twice, which moves the last published digit.
    """
    conditions = {}
    for arm in ARMS:
        conditions[arm] = []
        for seed in SEEDS:
            data = np.load(cells[(arm, scale, seed)]["correctness_path"],
                           allow_pickle=True)
            conditions[arm].append({
                name: data[name].astype(np.float64)
                for name in ("normal", *CONDITION_OF.values())})
    return conditions


def analyse_scale(scale: str, cells: dict, steps, image_index,
                  n_images) -> dict:
    present = {(a, s) for (a, sc, s) in cells if sc == scale}
    arms_ready = [a for a in ARMS
                  if all((a, s) in present for s in SEEDS)]
    table = {f"{a}/seed{s}": per_arm_row(cells[(a, scale, s)])
             for a in ARMS for s in SEEDS if (a, scale, s) in cells}
    if len(arms_ready) < 3:
        return {"scale": scale, "complete": False,
                "arms_with_three_seeds": arms_ready, "per_arm_seed": table}

    correct = load_correctness(cells, scale)
    contrasts = {}
    for name, (a, b) in CONTRASTS.items():
        per_seed = [round(float(correct[a][i].mean() - correct[b][i].mean()), 5)
                    for i in range(len(SEEDS))]
        lower, upper = clustered_bootstrap(correct[a], correct[b],
                                           image_index, n_images)
        contrasts[name] = {
            "per_seed_differences": {f"seed{SEEDS[i]}": per_seed[i]
                                     for i in range(len(SEEDS))},
            "mean": round(float(np.mean(per_seed)), 5),
            "std": round(float(np.std(per_seed, ddof=1)), 5),
            "ci95_image_clustered": [lower, upper],
            "n_clusters": n_images,
            "n_draws": BOOTSTRAP_DRAWS,
            "rng_seed": BOOTSTRAP_SEED,
            "directional_rule_outcome": directional_rule(per_seed, lower,
                                                         upper),
            "conditions": {
                "mean_gt_zero": bool(np.mean(per_seed) > 0),
                "ci_lower_gt_zero": bool(lower > 0),
                "at_least_two_seeds_gt_zero":
                    sum(1 for d in per_seed if d > 0) >= 2,
                "mean_lt_zero": bool(np.mean(per_seed) < 0),
                "ci_upper_lt_zero": bool(upper < 0),
                "at_least_two_seeds_lt_zero":
                    sum(1 for d in per_seed if d < 0) >= 2,
            },
        }
    contrasts["A1_minus_A1r"]["interpretation"] = (
        "the primary within-size pretrained-versus-random contrast, subject to "
        "the canonical degeneracy qualification recorded per seed below")
    contrasts["A1_minus_A0p"]["interpretation"] = (
        "a matched SYSTEM comparison under a recipe historically selected on "
        "the CLIP-question-token reasoner, which favours A0p and makes the "
        "comparison conservative with respect to the SLM arm. It jointly "
        "changes (a) the question representation semantics and (b) whether "
        "the image and question tokens originate from CLIP's shared "
        "pretrained space. It does NOT isolate language-model semantics or "
        "pretraining.")
    contrasts["A0p_minus_A1r"]["interpretation"] = (
        "descriptive only; not a pre-registered claim")

    degeneracy = {
        f"seed{s}": {
            "normal_minus_fixed_question":
                cells[("A1r", scale, s)]["evaluation"][
                    "differences_from_normal"]["normal_minus_fixed_question"],
            "prediction_entropy_nats": cells[("A1r", scale, s)]["evaluation"][
                "conditions"]["normal"]["prediction_entropy_nats"],
            "maximum_class_share": cells[("A1r", scale, s)]["evaluation"][
                "conditions"]["normal"]["maximum_class_share"],
            "rule_fires": cells[("A1r", scale, s)]["evaluation"][
                "degeneracy_rule"]["rule_fires"],
        } for s in SEEDS}

    deficits = {a: deficit_block(correct[a], steps) for a in ARMS}
    deficit_contrasts = {
        name: deficit_contrast(correct[a], correct[b], steps, image_index,
                               n_images)
        for name, (a, b) in CONTRASTS.items()}

    conditions = load_conditions(cells, scale)
    reliance = {}
    for arm in ARMS:
        reliance[arm] = {}
        for column in RELIANCE_COLUMNS:
            drops = [float(block["normal"].mean()
                           - block[CONDITION_OF[column]].mean())
                     for block in conditions[arm]]
            reliance[arm][column] = {
                "per_seed": [round(d, 5) for d in drops],
                "mean": round(float(np.mean(drops)), 5),
                "std": round(float(np.std(drops, ddof=1)), 5)}

    return {
        "scale": scale, "complete": True, "per_arm_seed": table,
        # Summarised from the per-row vectors, not from the records' already
        # rounded five-decimal fields, which would round twice.
        "per_arm_summary": {
            a: {"mean_dev_accuracy": round(float(np.mean(
                    [v.mean() for v in correct[a]])), 5),
                "std_dev_accuracy": round(float(np.std(
                    [v.mean() for v in correct[a]], ddof=1)), 5),
                "per_seed": [table[f"{a}/seed{s}"]["dev_accuracy"]
                             for s in SEEDS]} for a in ARMS},
        "contrasts": contrasts,
        "a1r_degeneracy_per_seed": degeneracy,
        "reliance": reliance,
        "step_deficits": {"definition": steps["definition"],
                          "prior_accuracy_by_slice":
                              steps["prior_accuracy_by_slice"],
                          "bucket_counts": steps["bucket_counts"],
                          "per_arm": deficits,
                          "contrasts": deficit_contrasts},
        "disclosure": (
            "The image-clustered interval conditions on the FIXED set of "
            "trained seeds and does not fully propagate training-seed "
            "uncertainty. For scale, the measured across-seed standard "
            "deviation of the stored reasoner runs is 0.0037 at 40k in v3_01, "
            "which has no 250k runs, and 0.0074 at 250k in v3_03. No "
            "equivalence language and no equivalence margin is used anywhere."),
        "multiplicity_disclosure": (
            "Every interval here is reported at a nominal 95 per cent and "
            "none is corrected for multiplicity. This analysis reports "
            "twenty-one intervals in total, so an interval whose bound sits "
            "close to zero should be read as weaker than its nominal level. "
            "The pre-registered primary contrast is A1 minus A1r; the rest "
            "are secondary or descriptive."),
        "fixed_question_caveat": (
            "The pinned neutral question changes both question content and "
            "sequence length, and by different amounts across arms, so "
            "normal minus fixed-question is NOT a pure semantic effect."),
    }


# --- Cross-scale analysis -------------------------------------------------

def analyse_across_scales(cells, analyses, steps, image_index,
                          n_images) -> dict:
    """40k against 250k for each arm, paired within seed.

    Every quantity here is a difference of a quantity already defined and
    already reported per scale. Nothing new is defined: the same rows, the
    same seeds, the same bootstrap and the same directional rule.
    """
    small, large = SCALES
    if not all(analyses.get(s, {}).get("complete") for s in SCALES):
        return {"complete": False,
                "reason": "both scales need all three seeds"}
    correct = {s: load_correctness(cells, s) for s in SCALES}
    out = {"complete": True, "scales": list(SCALES),
           "pairing": "same seed, same development rows, same bootstrap"}

    accuracy = {}
    for arm in ARMS:
        per_seed = [round(float(correct[large][arm][i].mean()
                                - correct[small][arm][i].mean()), 5)
                    for i in range(len(SEEDS))]
        lower, upper = clustered_bootstrap(correct[large][arm],
                                           correct[small][arm], image_index,
                                           n_images)
        accuracy[arm] = {
            "per_seed_differences": {f"seed{SEEDS[i]}": per_seed[i]
                                     for i in range(len(SEEDS))},
            "mean": round(float(np.mean(per_seed)), 5),
            "std": round(float(np.std(per_seed, ddof=1)), 5),
            "ci95_image_clustered": [lower, upper],
            "directional_rule_outcome": directional_rule(per_seed, lower,
                                                         upper)}
    out["accuracy_scale_gain_250k_minus_40k"] = accuracy

    out["contrast_at_each_scale"] = {
        name: {s: {"mean": analyses[s]["contrasts"][name]["mean"],
                   "ci95_image_clustered":
                       analyses[s]["contrasts"][name]["ci95_image_clustered"],
                   "directional_rule_outcome":
                       analyses[s]["contrasts"][name][
                           "directional_rule_outcome"]}
               for s in SCALES} for name in CONTRASTS}

    out["convergence"] = {
        arm: {s: {"best_epoch": [analyses[s]["per_arm_seed"][
                      f"{arm}/seed{seed}"]["best_epoch"] for seed in SEEDS],
                  "epochs_run": [analyses[s]["per_arm_seed"][
                      f"{arm}/seed{seed}"]["epochs_run"] for seed in SEEDS],
                  "seconds_per_epoch": [analyses[s]["per_arm_seed"][
                      f"{arm}/seed{seed}"]["seconds_per_epoch"]
                      for seed in SEEDS],
                  "wall_clock_hours": [analyses[s]["per_arm_seed"][
                      f"{arm}/seed{seed}"]["wall_clock_hours"]
                      for seed in SEEDS],
                  "peak_allocated_mib": [analyses[s]["per_arm_seed"][
                      f"{arm}/seed{seed}"]["peak_allocated_mib"]
                      for seed in SEEDS]}
              for s in SCALES} for arm in ARMS}

    out["reliance_change"] = {
        arm: {column: {
            s: analyses[s]["reliance"][arm][column]["mean"] for s in SCALES}
            | {"change_250k_minus_40k": round(
                analyses[large]["reliance"][arm][column]["mean"]
                - analyses[small]["reliance"][arm][column]["mean"], 5)}
            for column in RELIANCE_COLUMNS} for arm in ARMS}

    out["entropy_and_class_share_change"] = {
        arm: {field: {
            s: round(float(np.mean([analyses[s]["per_arm_seed"][
                f"{arm}/seed{seed}"][field] for seed in SEEDS])), 5)
            for s in SCALES}
            | {"change_250k_minus_40k": round(
                float(np.mean([analyses[large]["per_arm_seed"][
                    f"{arm}/seed{seed}"][field] for seed in SEEDS]))
                - float(np.mean([analyses[small]["per_arm_seed"][
                    f"{arm}/seed{seed}"][field] for seed in SEEDS])), 5)}
            for field in ("prediction_entropy_nats", "maximum_class_share",
                          "distinct_answers")} for arm in ARMS}

    deficit_by_scale = {}
    for arm in ARMS:
        per_seed = []
        for i in range(len(SEEDS)):
            large_deficit, _ = pooled_deficit(
                correct[large][arm][i], steps["bucket"], steps["ge4_mask"],
                steps["prior_correct"])
            small_deficit, _ = pooled_deficit(
                correct[small][arm][i], steps["bucket"], steps["ge4_mask"],
                steps["prior_correct"])
            per_seed.append(round(large_deficit - small_deficit, 5))
        contrast = deficit_contrast(correct[large][arm], correct[small][arm],
                                    steps, image_index, n_images)
        deficit_by_scale[arm] = {
            "pooled_ge4_deficit_by_scale": {
                s: analyses[s]["step_deficits"]["per_arm"][arm][
                    "mean_pooled_ge4_deficit"] for s in SCALES},
            "change_250k_minus_40k": contrast}
    out["step_deficit_change"] = deficit_by_scale
    out["interpretation"] = (
        "A higher overall accuracy at 250k is not by itself evidence of "
        "improved multi-step reasoning. The pooled >=4-step deficit is "
        "reported separately at each scale and as a paired change, against "
        "the same fixed train_40k per-bucket priors, so an accuracy gain and "
        "a deficit change are read independently.")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=list(SCALES))
    args = parser.parse_args()
    utils.set_seed()

    manifest = json.loads(
        (e8a.OUT_DIR / "execution_manifest.json").read_text())[
            "e8a_execution_manifest"]
    view = integrity.canonical_dev_view()
    cells = load_cells(manifest, view)
    steps = step_view(view)

    unique = sorted(set(view.image_ids))
    index_of = {value: position for position, value in enumerate(unique)}
    image_index = np.array([index_of[value] for value in view.image_ids])

    scales = [args.scale] if args.scale else list(SCALES)
    analyses = {s: analyse_scale(s, cells, steps, image_index, len(unique))
                for s in scales}
    across = (analyse_across_scales(cells, analyses, steps, image_index,
                                    len(unique))
              if len(scales) == 2 else
              {"complete": False,
               "reason": "one scale was requested; run without --scale"})

    record = {"metadata": utils.run_metadata(),
              "e8a_core_analysis": {
                  "cells_available": sorted(f"{a}/{sc}/seed{s}"
                                            for (a, sc, s) in cells),
                  "integrity": {
                      "row_order_sha256": view.row_order_sha256,
                      "labels_sha256": view.labels_sha256,
                      "vocabulary_sha256": view.vocabulary_sha256,
                      "dev_sha256": view.dev_sha256,
                      "n_rows": view.n_rows,
                      "n_clusters": len(unique),
                      "per_cell": {f"{a}/{sc}/seed{s}": cells[(a, sc, s)][
                          "integrity"] for (a, sc, s) in cells},
                      "note": (
                          "every artefact hash above was re-measured at "
                          "analysis time, not copied from the run record")},
                  "analyses": analyses,
                  "across_scales": across,
                  "a0_note": (
                      "The stored v3_01 result is arm A0, not A0p, and does "
                      "not appear in any table above. It is context only."),
                  "clean_test_accessed": False}}
    name = f"core_analysis_{args.scale}.json" if args.scale \
        else "core_analysis.json"
    utils.save_json(record, e8a.OUT_DIR / name)

    for scale, analysis in analyses.items():
        print(f"\n=== {scale} ===")
        if not analysis["complete"]:
            print(f"  incomplete: three seeds present for "
                  f"{analysis['arms_with_three_seeds']}")
            for key, row in sorted(analysis["per_arm_seed"].items()):
                print(f"    {key:14s} dev {row['dev_accuracy']:.5f} "
                      f"epoch {row['best_epoch']:2d}/{row['epochs_run']:2d}")
            continue
        for arm in ARMS:
            summary = analysis["per_arm_summary"][arm]
            print(f"  {arm:4s} mean {summary['mean_dev_accuracy']:.5f} "
                  f"sd {summary['std_dev_accuracy']:.5f}  per-seed "
                  f"{summary['per_seed']}")
        for name_, contrast in analysis["contrasts"].items():
            print(f"  {name_:14s} mean {contrast['mean']:+.5f} "
                  f"sd {contrast['std']:.5f} "
                  f"CI {contrast['ci95_image_clustered']} -> "
                  f"{contrast['directional_rule_outcome']}")
        for arm in ARMS:
            block = analysis["step_deficits"]["per_arm"][arm]
            print(f"  {arm:4s} pooled >=4-step deficit "
                  f"{block['mean_pooled_ge4_deficit']:+.5f} "
                  f"sd {block['std_pooled_ge4_deficit']:.5f}")
        print(f"  A1r degeneracy fires: "
              f"{[analysis['a1r_degeneracy_per_seed'][f'seed{s}']['rule_fires'] for s in SEEDS]}")

    if across.get("complete"):
        print("\n=== across scales (250k minus 40k) ===")
        for arm in ARMS:
            gain = across["accuracy_scale_gain_250k_minus_40k"][arm]
            change = across["step_deficit_change"][arm]["change_250k_minus_40k"]
            print(f"  {arm:4s} accuracy {gain['mean']:+.5f} "
                  f"CI {gain['ci95_image_clustered']} -> "
                  f"{gain['directional_rule_outcome']}")
            print(f"       pooled >=4-step deficit change "
                  f"{change['mean']:+.5f} CI "
                  f"{change['ci95_image_clustered']} -> "
                  f"{change['directional_rule_outcome']}")
    print(f"\nwritten to results/experiments/e8a_question_encoder/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
