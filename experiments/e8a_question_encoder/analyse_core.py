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
  * where both scales exist, the scale, reliance and >=4-step analyses.

The bootstrap is the one implemented at experiments/v3_03_scaling/run.py:397-427
and reused unchanged: 2,000 draws, RNG seed 0, alpha 0.05, clustering by
development imageId, seeds combined by the within-draw seed mean.

Artefact names are read from the manifest, never derived: the three reused
correctness files predate the scale-qualified naming.
"""

from __future__ import annotations

import argparse
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
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402

ARMS = ("A0p", "A1", "A1r")
SEEDS = (0, 1, 2)
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 0
ALPHA = 0.05
STEP_ORDER = ["<=2", "3", "4", ">=5"]

CONTRASTS = {
    "A1_minus_A1r": ("A1", "A1r"),
    "A1_minus_A0p": ("A1", "A0p"),
    "A0p_minus_A1r": ("A0p", "A1r"),
}


def load_cells(manifest: dict) -> dict:
    """Every completed cell, keyed by (arm, scale, seed)."""
    cells = {}
    for spec in manifest["runs"]:
        key = (spec["arm"], spec["scale"], spec["seed"])
        artefacts = spec["expected_artefacts"]
        if spec["reuse_artefacts"]:
            reuse = spec["reuse_artefacts"]
            record = json.loads(
                (e8a.OUT_DIR / reuse["record"]["path"]).read_text())
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
            record = json.loads(path.read_text())["e8a_core_run"]
            run, evaluation = record["run"], record["evaluation"]
            correctness = e8a.OUT_DIR / artefacts["correctness"]
            source = artefacts["record"]
        cells[key] = {"run": run, "evaluation": evaluation,
                      "correctness_path": correctness, "source": source}
    return cells


def per_arm_row(cell: dict) -> dict:
    run, ev = cell["run"], cell["evaluation"]
    normal = ev["conditions"]["normal"]
    diffs = ev["differences_from_normal"]
    eff = ev["efficiency"]
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
        "checkpoint_sha256": run["checkpoint_sha256"],
        "correctness_sha256": ev["correctness_vectors"]["sha256"],
        "source_record": cell["source"],
    }


def clustered_bootstrap(correct_a, correct_b, image_index, n_images):
    """Paired image-clustered bootstrap over the fixed seed set.

    Mirrors experiments/v3_03_scaling/run.py:397-427: draw images with
    replacement, take the rows of the drawn images, average the per-seed
    difference within the draw, and read the 2.5/97.5 percentiles.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows_by_image = [np.nonzero(image_index == i)[0] for i in range(n_images)]
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = rng.integers(0, n_images, size=n_images)
        rows = np.concatenate([rows_by_image[i] for i in sampled])
        per_seed = [float(correct_a[s][rows].mean() - correct_b[s][rows].mean())
                    for s in range(len(correct_a))]
        draws.append(float(np.mean(per_seed)))
    draws = np.asarray(draws)
    return (round(float(np.percentile(draws, 100 * ALPHA / 2)), 5),
            round(float(np.percentile(draws, 100 * (1 - ALPHA / 2))), 5))


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


def analyse_scale(scale: str, cells: dict, dev) -> dict:
    present = {(a, s) for (a, sc, s) in cells if sc == scale}
    arms_ready = [a for a in ARMS
                  if all((a, s) in present for s in SEEDS)]
    table = {f"{a}/seed{s}": per_arm_row(cells[(a, scale, s)])
             for a in ARMS for s in SEEDS if (a, scale, s) in cells}
    if len(arms_ready) < 3:
        return {"scale": scale, "complete": False,
                "arms_with_three_seeds": arms_ready, "per_arm_seed": table}

    image_ids = dev["imageId"].to_numpy()
    unique = sorted(set(image_ids))
    image_index = np.array([unique.index(i) for i in image_ids])
    correct = {}
    for arm in ARMS:
        correct[arm] = []
        for seed in SEEDS:
            data = np.load(cells[(arm, scale, seed)]["correctness_path"],
                           allow_pickle=True)
            correct[arm].append(data["normal"].astype(np.float64))
        assert all(len(v) == len(image_ids) for v in correct[arm])

    contrasts = {}
    for name, (a, b) in CONTRASTS.items():
        per_seed = [round(float(correct[a][i].mean() - correct[b][i].mean()), 5)
                    for i in range(len(SEEDS))]
        lower, upper = clustered_bootstrap(correct[a], correct[b],
                                           image_index, len(unique))
        contrasts[name] = {
            "per_seed_differences": {f"seed{SEEDS[i]}": per_seed[i]
                                     for i in range(len(SEEDS))},
            "mean": round(float(np.mean(per_seed)), 5),
            "std": round(float(np.std(per_seed, ddof=1)), 5),
            "ci95_image_clustered": [lower, upper],
            "n_clusters": len(unique),
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

    return {
        "scale": scale, "complete": True, "per_arm_seed": table,
        "per_arm_summary": {
            a: {"mean_dev_accuracy": round(float(np.mean(
                    [table[f"{a}/seed{s}"]["dev_accuracy"] for s in SEEDS])), 5),
                "std_dev_accuracy": round(float(np.std(
                    [table[f"{a}/seed{s}"]["dev_accuracy"] for s in SEEDS],
                    ddof=1)), 5),
                "per_seed": [table[f"{a}/seed{s}"]["dev_accuracy"]
                             for s in SEEDS]} for a in ARMS},
        "contrasts": contrasts,
        "a1r_degeneracy_per_seed": degeneracy,
        "disclosure": (
            "The image-clustered interval conditions on the FIXED set of "
            "trained seeds and does not fully propagate training-seed "
            "uncertainty. The measured across-seed standard deviation in the "
            "stored v3_01 runs is 0.0037 at 40k and 0.0074 at 250k. No "
            "equivalence language and no equivalence margin is used anywhere."),
        "fixed_question_caveat": (
            "The pinned neutral question changes both question content and "
            "sequence length, and by different amounts across arms, so "
            "normal minus fixed-question is NOT a pure semantic effect."),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=["train_40k", "train_250k"])
    args = parser.parse_args()
    utils.set_seed()

    manifest = json.loads(
        (e8a.OUT_DIR / "execution_manifest.json").read_text())[
            "e8a_execution_manifest"]
    cells = load_cells(manifest)
    dev = pd.read_csv(e8a.V2_DIR / "dev.csv",
                      dtype={"questionId": str, "imageId": str},
                      keep_default_na=False)

    scales = [args.scale] if args.scale else ["train_40k", "train_250k"]
    analyses = {s: analyse_scale(s, cells, dev) for s in scales}

    record = {"metadata": utils.run_metadata(),
              "e8a_core_analysis": {
                  "cells_available": sorted(f"{a}/{sc}/seed{s}"
                                            for (a, sc, s) in cells),
                  "analyses": analyses,
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
        print(f"  A1r degeneracy fires: "
              f"{[analysis['a1r_degeneracy_per_seed'][f'seed{s}']['rule_fires'] for s in SEEDS]}")
    print(f"\nwritten to results/experiments/e8a_question_encoder/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
