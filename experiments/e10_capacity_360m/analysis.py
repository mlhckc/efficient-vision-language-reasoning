"""The frozen E10 post-matrix analysis.

Read-only over published cell evidence. Nothing here trains, evaluates, loads a
model or touches a GPU, and the embargoed clean test is never named or opened.

The analysis is preregistered and unconditional. It computes exactly the
comparisons the E10 design registered:

  * B4 minus B4r within train_40k;
  * B4 minus B4r within train_250k;
  * the train_250k minus train_40k scale effect, per arm;
  * the pretraining-by-scale difference in differences;
  * per-seed summaries for every arm and scale;
  * the planned image-clustered paired uncertainty for each contrast;
  * the planned visual-reliance summary;
  * the planned system and resource summary.

The set is fixed. It does not grow because a result looked interesting, no
comparison is added or dropped after seeing a number, and nothing is conditioned
on whether an interval excludes zero. The uncertainty machinery is REUSED, never
reinvented: the image-clustered resampling stream, the 2,000 draws from a fresh
default_rng(0), the percentile interval and the universal directional rule all
come from the E8A implementation that already encodes the canonical plan.

The analysis REFUSES to produce anything unless the exact frozen twelve-cell set
exists, is pair preserved, and reconciles: every recipe digest against the frozen
contract, every pair-normalised digest within a (scale, seed), every artefact
against its recorded hash, and every evaluation against the canonical
development row order. A missing cell is never imputed and a partial matrix is
never analysed.

    python -B -m experiments.e10_capacity_360m.run analysis
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import utils  # noqa: E402
from experiments.e8a_question_encoder import analyse_core as e8a_stats  # noqa: E402
from experiments.e10_capacity_360m import e10_common as e10  # noqa: E402
from experiments.e10_capacity_360m import science  # noqa: E402


ANALYSIS_PATH = e10.CORE_OUT_DIR / "e10_core_analysis.json"
# The two fixed metrics. Both are computed for every contrast, always, so no
# metric is ever chosen after the fact.
METRICS = ("canonical_r1_correct", "g21_normalized_correct")
INTERVAL_CAVEAT = (
    "the 95 per cent interval is image-clustered over the development images "
    "and CONDITIONS ON THE FIXED TRAINED SEED SET: it propagates sampling "
    "variation over questions and images, not training variance. The three "
    "per-seed differences are reported beside it, as the canonical rule "
    "requires."
)


class AnalysisRefused(AssertionError):
    """The frozen twelve-cell evidence is incomplete or does not reconcile."""


# --- evidence loading ---------------------------------------------------------

def _dev_reference() -> dict:
    """The canonical development row order every cell must have evaluated."""
    import pandas as pd

    path = science.manifest_path("dev")
    digest = e10.sha256_file(path)
    if digest != e10.MANIFEST_SHA256["dev"]:
        raise AnalysisRefused(
            f"the development manifest is not the frozen E10 input "
            f"({digest} != {e10.MANIFEST_SHA256['dev']})")
    frame = pd.read_csv(path, dtype={"questionId": str, "imageId": str},
                        keep_default_na=False)
    if len(frame) != e10.EXPECTED_MANIFEST_ROWS["dev"]:
        raise AnalysisRefused("the development manifest row count changed")
    images = sorted(frame["imageId"].unique().tolist())
    image_of = {image: index for index, image in enumerate(images)}
    return {
        "path": str(path),
        "sha256": digest,
        "rows": int(len(frame)),
        "question_ids": frame["questionId"].astype(str).to_numpy(),
        "labels": frame["label"].to_numpy("int64"),
        "image_index": np.array(
            [image_of[image] for image in frame["imageId"]], dtype="int64"),
        "n_images": len(images),
    }


def _load_cell(arm: str, scale: str, seed: int, reference: dict) -> dict:
    """One published cell, revalidated from disk before any number is used."""
    paths = e10.core_cell_paths(arm, scale, seed)
    if not paths["result"].exists():
        raise AnalysisRefused(
            f"cell {arm}/{scale}/seed{seed} has no published result at "
            f"{paths['result']}; a missing cell is never imputed")
    record = e10.read_json_mapping(paths["result"])
    e10.assert_current_binding(record["metadata"], f"E10 cell {arm}/{scale}/{seed}")
    cell = record["e10_core_cell"]
    if cell["cell"] != [arm, scale, seed]:
        raise AnalysisRefused(
            f"{paths['result'].name} is filed under {arm}/{scale}/seed{seed} "
            f"but records {cell['cell']}")
    recipe = e10.build_recipe(arm, scale, seed)
    if cell["recipe_sha256"] != e10.recipe_sha256(recipe) \
            or cell["paired_recipe_sha256"] != e10.paired_recipe_sha256(recipe):
        raise AnalysisRefused(
            f"cell {arm}/{scale}/seed{seed} does not carry the frozen recipe "
            f"digests")
    if cell["canonical_epoch"] != science.CANONICAL_EPOCH \
            or cell["canonical_checkpoint_rule"] != "epoch_22" \
            or cell["early_stopping"] is not False \
            or cell["epochs_run"] != science.CANONICAL_EPOCH:
        raise AnalysisRefused(
            f"cell {arm}/{scale}/seed{seed} is not an epoch-22 primary result")
    if cell["dev_evaluation_epochs"] != list(science.DEV_EVALUATION_EPOCHS):
        raise AnalysisRefused(
            f"cell {arm}/{scale}/seed{seed} used a different cadence")
    per_row = PROJECT_ROOT / cell["per_row"]["path"]
    if not per_row.exists() \
            or e10.sha256_file(per_row) != cell["per_row"]["sha256"]:
        raise AnalysisRefused(
            f"cell {arm}/{scale}/seed{seed} per-row evidence is absent or "
            f"does not match its recorded digest")
    for name, checkpoint in cell["checkpoints"].items():
        path = PROJECT_ROOT / checkpoint["path"]
        if not path.exists() \
                or e10.sha256_file(path) != checkpoint["sha256"]:
            raise AnalysisRefused(
                f"cell {arm}/{scale}/seed{seed} {name} checkpoint does not "
                f"match its recorded digest")
    with np.load(per_row, allow_pickle=False) as arrays:
        question_id = arrays["question_id"].astype(str)
        labels = arrays["labels"].astype("int64")
        predictions = arrays["canonical_predictions"].astype("int64")
        raw_correct = arrays["raw_correct"].astype(np.float64)
        normalized_correct = arrays["normalized_correct"].astype(np.float64)
    if not np.array_equal(question_id, reference["question_ids"]):
        raise AnalysisRefused(
            f"cell {arm}/{scale}/seed{seed} was not evaluated on the canonical "
            f"development rows in canonical order")
    if not np.array_equal(labels, reference["labels"]):
        raise AnalysisRefused(
            f"cell {arm}/{scale}/seed{seed} carries different gold labels")
    canonical_correct = (predictions == labels).astype(np.float64)
    if not np.array_equal(canonical_correct, raw_correct):
        raise AnalysisRefused(
            f"cell {arm}/{scale}/seed{seed} label identity and raw exact "
            f"correctness disagree")
    recomputed = round(float(canonical_correct.mean()), 5)
    if recomputed != round(
            float(cell["primary_canonical_fp32_dev_accuracy"]), 5):
        raise AnalysisRefused(
            f"cell {arm}/{scale}/seed{seed} per-row evidence recomputes "
            f"{recomputed}, not the published "
            f"{cell['primary_canonical_fp32_dev_accuracy']}")
    return {
        "cell": (arm, scale, seed),
        "record": cell,
        "result_sha256": e10.sha256_file(paths["result"]),
        "correct": {
            "canonical_r1_correct": canonical_correct,
            "g21_normalized_correct": normalized_correct,
        },
    }


def load_matrix() -> dict:
    """The complete, pair-preserved, reconciled twelve-cell evidence set."""
    reference = _dev_reference()
    order = e10.pair_preserving_order()
    if len(order) != len(e10.CORE_CELLS):
        raise AnalysisRefused("the frozen execution plan is not twelve cells")
    cells = {cell: _load_cell(*cell, reference) for cell in order}
    # Pair preservation, asserted from the evidence rather than assumed: within
    # every (scale, seed) both arms must be present and their pair-normalised
    # recipe digests must be identical, which is what makes B4 minus B4r a
    # controlled contrast instead of two unrelated runs.
    for scale in e10.CORE_SCALES:
        for seed in e10.CORE_SEEDS:
            digests = set()
            for arm in e10.CORE_ARMS:
                key = (arm, scale, seed)
                if key not in cells:
                    raise AnalysisRefused(
                        f"pair preservation broken: {key} is missing")
                digests.add(cells[key]["record"]["paired_recipe_sha256"])
            if len(digests) != 1:
                raise AnalysisRefused(
                    f"the B4/B4r pair at {scale}/seed{seed} does not share a "
                    f"pair-normalised recipe")
    charged = e10.charged_core_cells()
    completed = {tuple(entry["cell"]) for entry in charged
                 if entry["outcome"] == "completed"}
    if completed != set(order):
        raise AnalysisRefused(
            f"the spend ledger records {len(completed)} completed cells, not "
            f"the frozen twelve; accounting and evidence must reconcile")
    if any(entry["outcome"] != "completed" for entry in charged):
        raise AnalysisRefused(
            "the spend ledger contains a non-completed core entry; a failed "
            "or halted cell must be resolved by the user before analysis")
    return {"reference": reference, "cells": cells, "order": order,
            "charged": charged}


# --- the preregistered statistics ---------------------------------------------

def _paired_interval(matrix: dict, metric: str, positive, negative) -> dict:
    """One contrast: per-seed differences, clustered interval, directional rule.

    `positive` and `negative` are lists of (arm, scale) sides. The per-seed
    difference is the SUM over the positive sides minus the SUM over the
    negative sides, so a simple contrast and a difference in differences use
    exactly the same resampling and the same rule.

    The sum, not the mean, is what makes the composite contrast the quantity its
    description names. Averaging the two positive and the two negative terms
    reports half of

        (B4 - B4r) at train_250k  minus  (B4 - B4r) at train_40k,

    because each of the four terms is halved. For the three one-element
    contrasts a sum and a mean over a single value are identical, so their point
    estimates, intervals and directional outcomes are unchanged.
    """
    reference = matrix["reference"]
    cells = matrix["cells"]

    def side(sides, seed, rows=None):
        values = []
        for arm, scale in sides:
            correct = cells[(arm, scale, seed)]["correct"][metric]
            values.append(correct.mean() if rows is None
                          else correct[rows].mean())
        return float(np.sum(values))

    per_seed = [round(side(positive, seed) - side(negative, seed), 5)
                for seed in e10.CORE_SEEDS]

    def statistic(rows):
        return float(np.mean([side(positive, seed, rows)
                              - side(negative, seed, rows)
                              for seed in e10.CORE_SEEDS]))

    draws = e8a_stats.clustered_draws(
        statistic, reference["image_index"], reference["n_images"])
    lower, upper = e8a_stats.percentile_interval(draws)
    return {
        "metric": metric,
        "positive": [list(side) for side in positive],
        "negative": [list(side) for side in negative],
        "aggregation": ("sum over the positive sides minus the sum over the "
                        "negative sides; identical to a mean for the "
                        "one-element contrasts"),
        "per_seed_differences": per_seed,
        "mean_difference": round(float(np.mean(per_seed)), 5),
        "sd_difference": round(float(np.std(per_seed, ddof=1)), 5),
        "interval_95": [lower, upper],
        "directional_rule": e8a_stats.directional_rule(per_seed, lower, upper),
        "bootstrap": {"draws": e8a_stats.BOOTSTRAP_DRAWS,
                      "seed": e8a_stats.BOOTSTRAP_SEED,
                      "alpha": e8a_stats.ALPHA,
                      "clustering": "development imageId",
                      "implementation":
                          "experiments/e8a_question_encoder/analyse_core.py"},
        "caveat": INTERVAL_CAVEAT,
    }


def _frozen_contrasts() -> dict:
    """The registered contrast set, written out once and never extended."""
    contrasts = {}
    for scale in e10.CORE_SCALES:
        contrasts[f"pretraining_effect_{scale}"] = {
            "description": f"B4 minus B4r within {scale}",
            "positive": [("B4", scale)], "negative": [("B4r", scale)],
        }
    for arm in e10.CORE_ARMS:
        contrasts[f"scale_effect_{arm}"] = {
            "description": f"{arm} at train_250k minus {arm} at train_40k",
            "positive": [(arm, "train_250k")], "negative": [(arm, "train_40k")],
        }
    contrasts["pretraining_by_scale_difference_in_differences"] = {
        "description": "(B4 minus B4r) at train_250k minus (B4 minus B4r) at "
                       "train_40k",
        "positive": [("B4", "train_250k"), ("B4r", "train_40k")],
        "negative": [("B4r", "train_250k"), ("B4", "train_40k")],
    }
    return contrasts


def seed_summaries(matrix: dict) -> dict:
    summaries = {}
    for arm in e10.CORE_ARMS:
        for scale in e10.CORE_SCALES:
            block = {}
            for metric in METRICS:
                values = [
                    float(matrix["cells"][(arm, scale, seed)]["correct"][metric]
                          .mean())
                    for seed in e10.CORE_SEEDS
                ]
                block[metric] = {
                    "per_seed": [round(value, 5) for value in values],
                    "mean": round(float(np.mean(values)), 5),
                    "sd": round(float(np.std(values, ddof=1)), 5),
                }
            block["secondary_diagnostic_best_of_evaluated_epochs"] = {
                str(seed): matrix["cells"][(arm, scale, seed)]["record"][
                    "secondary_diagnostic"]
                for seed in e10.CORE_SEEDS
            }
            summaries[f"{arm}_{scale}"] = block
    return summaries


def visual_reliance_summary(matrix: dict) -> dict:
    """The planned visual-reliance summary, computed only from real evidence.

    The frozen E10 per-cell budget funds one final full-development R1 pass and
    names it the final R1 evaluation; it does not fund a separate deranged- or
    fixed-image intervention pass, and no E10 record registers an executable
    intervention set or a target-cell subset. This summary therefore reports
    exactly what the published cells contain. If a future authorised protocol
    adds intervention conditions, their evidence appears here; nothing is
    estimated, imputed or carried over from another experiment in the meantime.
    """
    conditions = {}
    for cell, payload in matrix["cells"].items():
        final = payload["record"]["final_r1_evaluation"]
        conditions.setdefault(final["condition"], []).append(list(cell))
    available = sorted(conditions)
    return {
        "status": ("EVIDENCE_PRESENT" if len(available) > 1
                   else "NO_INTERVENTION_EVIDENCE_IN_THE_FROZEN_E10_CONTRACT"),
        "conditions_present": available,
        "cells_per_condition": {name: len(rows)
                                for name, rows in conditions.items()},
        "contrasts": {},
        "note": ("the frozen E10 per-cell budget funds one final full-"
                 "development R1 pass under the normal condition; no "
                 "intervention condition is registered for E10, so no "
                 "reliance contrast is computed and none is estimated"),
    }


def resource_summary(matrix: dict) -> dict:
    """The planned system and resource close-out, from the authoritative ledger."""
    charged = matrix["charged"]
    by_identity = {}
    for identity in sorted(e10.IDENTITIES):
        hours = e10.identity_hours(identity)
        by_identity[identity] = {
            "charged_hours": hours,
            "effective_identity_ceiling_hours":
                e10.effective_identity_ceiling_hours(),
            "headroom_hours": e10.effective_identity_ceiling_hours() - hours,
        }
    cells = {}
    for cell, payload in matrix["cells"].items():
        record = payload["record"]
        cells["_".join(str(part) for part in cell)] = {
            "gpu_hours": next(
                (entry["gpu_hours"] for entry in charged
                 if tuple(entry["cell"]) == cell), None),
            "peak_reserved_fraction": record["peak_memory"]["reserved_fraction"],
            "elapsed_seconds_at_publication":
                record["wall"]["elapsed_seconds_at_publication"],
            "per_cell_wall_clock_hours":
                record["wall"]["per_cell_wall_clock_hours"],
        }
    return {
        "policy": e10.resource_policy(),
        "per_identity": by_identity,
        "per_cell": cells,
        "automatic_retries_taken": 0,
    }


def analyse() -> dict:
    """Every registered analysis, computed unconditionally from the matrix."""
    matrix = load_matrix()
    contrasts = {}
    for name, definition in _frozen_contrasts().items():
        contrasts[name] = {
            "description": definition["description"],
            "metrics": {
                metric: _paired_interval(matrix, metric,
                                         definition["positive"],
                                         definition["negative"])
                for metric in METRICS
            },
        }
    return {
        "metadata": {**utils.run_metadata(seed=e8a_stats.BOOTSTRAP_SEED),
                     "task_id": e10.PHASE2_TASK_ID,
                     "protocol_family": e10.PROTOCOL_FAMILY,
                     "binding": e10.binding_record()},
        "e10_core_analysis": {
            "schema_version": 1,
            "record_type": "e10_core_matrix_analysis",
            "status": "COMPLETE_TWELVE_CELL_MATRIX",
            "utc": e10.utc_now(),
            "cells": [list(cell) for cell in matrix["order"]],
            "development_reference": {
                "path": matrix["reference"]["path"],
                "sha256": matrix["reference"]["sha256"],
                "rows": matrix["reference"]["rows"],
                "images": matrix["reference"]["n_images"],
            },
            "cell_records": {
                "_".join(str(part) for part in cell): payload["result_sha256"]
                for cell, payload in matrix["cells"].items()
            },
            "metrics": {
                "canonical_r1_correct":
                    "canonical FP32 development R1 accuracy of the epoch-22 "
                    "checkpoint (the registered primary)",
                "g21_normalized_correct":
                    "the pinned G21 VQA-style normalised exact match over the "
                    "same rows",
            },
            "seed_summaries": seed_summaries(matrix),
            "contrasts": contrasts,
            "visual_reliance": visual_reliance_summary(matrix),
            "resources": resource_summary(matrix),
            "preregistration": {
                "contrast_set_fixed_before_results": True,
                "post_hoc_tests_added": 0,
                "analyses_conditioned_on_significance": False,
                "missing_cells_imputed": 0,
            },
            "development_results_only": True,
            "clean_test_accessed": False,
        },
    }


def main(argv=None) -> int:  # pragma: no cover - thin CLI wrapper
    import json

    try:
        record = analyse()
    except AnalysisRefused as error:
        # A refusal is the expected outcome until the frozen matrix exists and
        # reconciles; it is reported as a refusal, not as a crash.
        raise SystemExit(f"E10 ANALYSIS REFUSED: {error}") from None
    e10.CORE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    digest = e10.atomic_write_json(ANALYSIS_PATH, record)
    print(json.dumps({"path": science._relative(ANALYSIS_PATH),
                      "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
