"""Read-only analysis of the frozen E8B 18-cell core matrix.

Covers the parts of the final analysis that can be computed from the
frozen artefacts alone: the immutable matrix manifest, the primary
descriptive summary, the pre-registered directional statistics, the
training-stability diagnostics and the resource close-out.

WHAT THIS MODULE DOES NOT DO. The R1/R2/R3 readout evaluation and the
three interventions are NOT computed here. `final_evaluation.py`
implements them and is validated, but nothing in the repository drives it
over the eighteen frozen checkpoints, and building that driver is new
executable scientific code producing primary results. It is reported
rather than written here.

STATISTICS ARE REUSED, NEVER INVENTED. The universal primary directional
rule of canonical plan section 5 is applied through the E8A
implementation that already encodes it -- the same image-clustered
resampling stream, the same 2,000 draws from a fresh default_rng(0), the
same percentile interval and the same three-condition rule. No test is
added, substituted or tuned here.

The interval conditions on the fixed trained seed set and does not fully
propagate training variance. Every reported interval carries that
disclosure.

NO TRAINING, NO EVALUATION, NO GPU. This module reads JSON, NPZ and CSV.
THE EMBARGOED CLEAN TEST IS NEVER TOUCHED.

    python -B experiments/e8b_readout_generation/core_analysis.py
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
DEV_MANIFEST = config.DATA_DIR / "v2" / "dev.csv"
SEEDS = (0, 1, 2)
SCALES = ("train_40k", "train_250k")
ARMS = ("B1", "B2", "B3")

# B1 selects its checkpoint by a different frozen rule from B2/B3. Any
# comparison that crosses that boundary carries this qualification.
B1_SELECTION_CAVEAT = (
    "B1 selects its primary checkpoint by the frozen section-7.3 rule "
    "(best development accuracy over a 100-epoch budget with early "
    "stopping, earliest epoch on ties), whereas B2 and B3 use the frozen "
    "epoch-22 rule with no early stopping. A B1-versus-B2/B3 difference "
    "therefore confounds architecture with checkpoint selection and is "
    "NOT a clean causal architecture comparison. B3 minus B2 is the only "
    "contrast in this matrix whose two arms share a selection rule.")

INTERVAL_CAVEAT = (
    "the 95 per cent interval is image-clustered over the development "
    "images and CONDITIONS ON THE FIXED TRAINED SEED SET: it propagates "
    "sampling variation over questions and images, not training "
    "variance. The three per-seed differences are reported beside it, as "
    "the canonical rule requires.")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cells() -> dict:
    """Every core record, with its artefacts re-hashed against disk."""
    cells = {}
    for arm in ARMS:
        for scale in SCALES:
            for seed in SEEDS:
                run_name = e8b_training.core_run_name(arm, scale, seed)
                path = OUT_DIR / f"{run_name}.json"
                if not path.exists():
                    raise AssertionError(f"{path.name} is missing; the "
                                         f"matrix is not complete")
                record = json.loads(path.read_text())["e8b_core_cell"]
                cells[(arm, scale, seed)] = record
    return cells


def manifest_entry(arm, scale, seed, record) -> dict:
    """One immutable manifest row, every hash recomputed from disk."""
    run_name = record["run"]
    result_path = OUT_DIR / f"{run_name}.json"
    entries = {}
    for role in ("canonical", "resume"):
        stored = record["checkpoints"][role]
        path = Path(stored["path"])
        if not path.exists():
            raise AssertionError(f"{run_name}: {role} checkpoint missing")
        live = sha256_file(path)
        if live != stored["sha256"]:
            raise AssertionError(
                f"{run_name}: {role} checkpoint hash drifted from its "
                f"record; the manifest refuses to describe an artefact "
                f"that has changed")
        entries[role] = {"path": str(path.relative_to(PROJECT_ROOT)),
                         "sha256": live,
                         "bytes": path.stat().st_size}
    per_row = Path(record["per_row_dump"])
    if not per_row.exists():
        raise AssertionError(f"{run_name}: per-row artefact missing")
    rebuilt = e8b_training.recipe_sha256(
        e8b_training.build_core_recipe(arm, scale, seed))
    if rebuilt != record["recipe_sha256"]:
        raise AssertionError(f"{run_name}: recipe hash does not rebuild")
    ledger = e8b_run.read_spend_ledger()
    return {
        "arm": arm, "scale": scale, "seed": seed, "run": run_name,
        "protocol_family": record["protocol_family"],
        "recipe_sha256": record["recipe_sha256"],
        "recipe_sha256_rebuilds": True,
        "canonical_checkpoint_rule": record["canonical_checkpoint_rule"],
        "canonical_epoch": record["canonical_epoch"],
        "epochs_run": record["epochs_run"],
        "resumed_from_epoch": record["resumed_from_epoch"],
        "primary_canonical_fp32_dev_accuracy":
            record["primary_canonical_fp32_dev_accuracy"],
        "checkpoints": entries,
        "per_row_artefact": {
            "path": str(per_row.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(per_row)},
        "result_record": {
            "path": str(result_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(result_path)},
        "gpu_hours_charged": e8b_run.cell_hours(ledger, arm, scale, seed),
        "gpu_hours_by_process":
            ledger["cells"][f"{arm}_{scale}_seed{seed}"]["processes"],
        "model_identity": e8b_run.model_identity(arm),
        "peak_memory_reserved_bytes":
            record["peak_memory"].get("reserved_bytes"),
        "clean_test_accessed": record["clean_test_accessed"]}


def dev_view() -> dict:
    """Development rows and their image clustering, gated against the
    per-row artefacts so a silent misalignment cannot pass."""
    frame = pd.read_csv(DEV_MANIFEST, dtype={"questionId": str,
                                             "imageId": str},
                        keep_default_na=False)
    codes, uniques = pd.factorize(frame["imageId"], sort=True)
    return {"frame": frame,
            "labels": frame["label"].to_numpy(),
            "image_index": np.asarray(codes),
            "n_images": int(len(uniques)),
            "n_rows": int(len(frame))}


def load_correct(cells, view) -> dict:
    """Per-row canonical correctness for every cell, alignment asserted.

    The development loader is built with shuffle=False, so per-row row i
    is manifest row i. That is ASSERTED rather than assumed: the stored
    labels must equal the manifest labels row for row, and the mean of
    the correctness vector must reproduce the recorded primary accuracy.
    """
    correct = {}
    for key, record in cells.items():
        data = np.load(Path(record["per_row_dump"]))
        if len(data["labels"]) != view["n_rows"]:
            raise AssertionError(f"{record['run']}: row count differs "
                                 f"from the development manifest")
        if not np.array_equal(data["labels"], view["labels"]):
            raise AssertionError(
                f"{record['run']}: stored labels are not row-aligned "
                f"with {DEV_MANIFEST.name}; image clustering would "
                f"attach rows to the wrong images")
        hits = data["canonical_correct"].astype(float)
        if round(float(hits.mean()), 5) != \
                record["primary_canonical_fp32_dev_accuracy"]:
            raise AssertionError(
                f"{record['run']}: the per-row artefact does not "
                f"reproduce the recorded primary accuracy")
        correct[key] = hits
    return correct


def describe(values) -> dict:
    array = np.asarray(values, dtype=float)
    return {"seeds": [round(float(v), 5) for v in array],
            "mean": round(float(array.mean()), 5),
            "sample_sd": round(float(array.std(ddof=1)), 5),
            "min": round(float(array.min()), 5),
            "max": round(float(array.max()), 5),
            "range": round(float(array.max() - array.min()), 5)}


def contrast(correct, arm_a, arm_b, scale, view, label) -> dict:
    """The pre-registered contrast, computed by the E8A implementation
    of the universal primary directional rule."""
    a = [correct[(arm_a, scale, s)] for s in SEEDS]
    b = [correct[(arm_b, scale, s)] for s in SEEDS]
    per_seed = [round(float(a[i].mean() - b[i].mean()), 5)
                for i in range(len(SEEDS))]
    low, high = e8a_stats.clustered_bootstrap(
        a, b, view["image_index"], view["n_images"])
    outcome = e8a_stats.directional_rule(per_seed, low, high)
    entry = {
        "contrast": label, "scale": scale,
        "mean_paired_difference": round(float(np.mean(per_seed)), 5),
        "per_seed_differences": per_seed,
        "seeds_above_zero": int(sum(1 for d in per_seed if d > 0)),
        "seeds_below_zero": int(sum(1 for d in per_seed if d < 0)),
        "ci95_low": low, "ci95_high": high,
        "excludes_zero": bool(low > 0 or high < 0),
        "directional_rule_outcome": outcome,
        "accuracy_a": round(float(np.mean([v.mean() for v in a])), 5),
        "accuracy_b": round(float(np.mean([v.mean() for v in b])), 5),
        "n_rows": view["n_rows"], "n_image_clusters": view["n_images"],
        "bootstrap": {"draws": e8a_stats.BOOTSTRAP_DRAWS,
                      "seed": e8a_stats.BOOTSTRAP_SEED,
                      "resampling": "image-clustered, with replacement",
                      "implementation":
                          "experiments/e8a_question_encoder/"
                          "analyse_core.py"},
        "interval_caveat": INTERVAL_CAVEAT}
    if "B1" in (arm_a, arm_b):
        entry["selection_rule_caveat"] = B1_SELECTION_CAVEAT
        entry["clean_causal_architecture_comparison"] = False
    else:
        entry["clean_causal_architecture_comparison"] = True
    return entry


def scale_contrast(correct, arm, view) -> dict:
    """Within-arm 250k minus 40k, paired on the same development rows."""
    a = [correct[(arm, "train_250k", s)] for s in SEEDS]
    b = [correct[(arm, "train_40k", s)] for s in SEEDS]
    per_seed = [round(float(a[i].mean() - b[i].mean()), 5)
                for i in range(len(SEEDS))]
    low, high = e8a_stats.clustered_bootstrap(
        a, b, view["image_index"], view["n_images"])
    return {
        "contrast": f"{arm} 250k - {arm} 40k", "arm": arm,
        "mean_paired_difference": round(float(np.mean(per_seed)), 5),
        "per_seed_differences": per_seed,
        "seeds_above_zero": int(sum(1 for d in per_seed if d > 0)),
        "ci95_low": low, "ci95_high": high,
        "excludes_zero": bool(low > 0 or high < 0),
        "directional_rule_outcome": e8a_stats.directional_rule(
            per_seed, low, high),
        "n_rows": view["n_rows"], "n_image_clusters": view["n_images"],
        "paired_on": "the same development rows; only the training set "
                     "size differs",
        "interval_caveat": INTERVAL_CAVEAT}


def stability(cells) -> dict:
    """Primary versus best-of-22, and seed spread, without changing the
    primary rule. Best-of-22 stays a labelled secondary diagnostic."""
    block = {"primary_rule_unchanged": "B2/B3 primary is epoch 22; "
                                       "best-of-22 is a SECONDARY "
                                       "DIAGNOSTIC and is never promoted",
             "by_cell": [], "by_arm_scale": {}}
    for arm in ARMS:
        for scale in SCALES:
            gaps, primaries = [], []
            for seed in SEEDS:
                record = cells[(arm, scale, seed)]
                primary = record["primary_canonical_fp32_dev_accuracy"]
                secondary = record.get("secondary_diagnostic")
                best = (secondary["best_of_22_dev_accuracy"]
                        if secondary else None)
                best_epoch = (secondary["best_of_22_epoch"]
                              if secondary else None)
                gap = (round(float(best - primary), 5)
                       if best is not None else None)
                primaries.append(primary)
                if gap is not None:
                    gaps.append(gap)
                block["by_cell"].append({
                    "arm": arm, "scale": scale, "seed": seed,
                    "primary": primary,
                    "canonical_epoch": record["canonical_epoch"],
                    "epochs_run": record["epochs_run"],
                    "secondary_best_of_22": best,
                    "secondary_best_epoch": best_epoch,
                    "secondary_minus_primary": gap})
            block["by_arm_scale"][f"{arm}/{scale}"] = {
                "primary": describe(primaries),
                "secondary_minus_primary_mean":
                    round(float(np.mean(gaps)), 5) if gaps else None,
                "secondary_minus_primary_max":
                    round(float(np.max(gaps)), 5) if gaps else None}
    return block


def resources(cells, manifest) -> dict:
    """Scientific successful-run compute and operational charged compute,
    kept apart. The failed first attempt is never netted out."""
    ledger = e8b_run.read_spend_ledger()
    by_cell, scientific, operational = [], 0.0, 0.0
    for entry in manifest:
        processes = entry["gpu_hours_by_process"]
        # The successful run of each cell is its LAST process; any
        # earlier process is a failed attempt that produced no result.
        success = float(processes[-1])
        failed = float(sum(processes[:-1]))
        scientific += success
        operational += float(sum(processes))
        by_cell.append({
            "arm": entry["arm"], "scale": entry["scale"],
            "seed": entry["seed"],
            "successful_run_hours": round(success, 5),
            "failed_attempt_hours": round(failed, 5),
            "total_charged_hours": round(float(sum(processes)), 5),
            "processes": processes})
    per_identity = {}
    for identity, arm in (("pretrained", "B3"), ("random", "B2"),
                          ("none", "B1")):
        gate = e8b_run.per_identity_gate(arm, 0.0)
        per_identity[identity] = {
            "charged_hours": gate["already_charged_hours"],
            "projected_total_hours": gate["projected_total_hours"],
            "ceiling_hours": gate["ceiling_hours"],
            "headroom_hours": gate["headroom_hours"],
            "headroom_floor_hours": gate["headroom_floor_hours"],
            "gate_fires": gate["fires"],
            "forced_retries_taken": e8b_run.retries_taken(identity),
            "forced_retry_allowance":
                e8b_run.MAX_FORCED_RETRIES_PER_IDENTITY}
    trainable = {"B2_B3_trunk_plus_projection": 21_343_808,
                 "B2_B3_frozen_language_model": 134_515_008,
                 "B1_readout": 52_836,
                 "source": "asserted in the E8B preflight gradient gate "
                           "and the implementation audit; not recomputed "
                           "here"}
    checkpoint_bytes = sum(
        entry["checkpoints"]["canonical"]["bytes"] for entry in manifest)
    resume_bytes = sum(
        entry["checkpoints"]["resume"]["bytes"] for entry in manifest)
    return {
        "accounting_rule": "A is scientific successful-run compute; B is "
                           "operational charged compute and includes "
                           "every failed attempt. The failed first "
                           "attempt of B3/train_40k/seed0 is charged in "
                           "B and excluded from A. Nothing is netted "
                           "out.",
        "A_scientific_successful_run_hours": round(scientific, 5),
        "B_operational_charged_hours": round(operational, 5),
        "difference_failed_attempts_hours":
            round(operational - scientific, 5),
        "by_cell": by_cell,
        "per_identity": per_identity,
        "parameters": trainable,
        "storage_bytes": {
            "canonical_checkpoints": checkpoint_bytes,
            "resume_checkpoints": resume_bytes,
            "total_gib": round((checkpoint_bytes + resume_bytes) / 2**30,
                               3)},
        "evaluation_latency": "NOT MEASURED HERE. The frozen readout "
                              "cost measurement belongs to the readout "
                              "evaluation that has not run; see the "
                              "not_computed block.",
        "e7b_integration": "deferred: an E7b serial-cost comparison is "
                           "only methodologically valid once the R1/R2/R3 "
                           "readouts have actually run on these "
                           "checkpoints under the frozen protocol."}


def main() -> int:
    utils.set_seed()
    cells = load_cells()
    view = dev_view()
    correct = load_correct(cells, view)

    manifest = [manifest_entry(arm, scale, seed, cells[(arm, scale, seed)])
                for arm in ARMS for scale in SCALES for seed in SEEDS]

    descriptive = {}
    for arm in ARMS:
        for scale in SCALES:
            values = [cells[(arm, scale, s)][
                "primary_canonical_fp32_dev_accuracy"] for s in SEEDS]
            descriptive[f"{arm}/{scale}"] = describe(values)
    for arm in ARMS:
        small = descriptive[f"{arm}/train_40k"]["mean"]
        large = descriptive[f"{arm}/train_250k"]["mean"]
        descriptive[f"{arm}/scale_change_40k_to_250k"] = {
            "mean_40k": small, "mean_250k": large,
            "change": round(large - small, 5)}

    contrasts = []
    for scale in SCALES:
        contrasts.append(contrast(correct, "B3", "B2", scale, view,
                                  "B3 - B2"))
        contrasts.append(contrast(correct, "B2", "B1", scale, view,
                                  "B2 - B1"))
        contrasts.append(contrast(correct, "B3", "B1", scale, view,
                                  "B3 - B1"))
    scale_effects = [scale_contrast(correct, arm, view) for arm in ARMS]

    record = {
        "metadata": utils.run_metadata(),
        "e8b_core_analysis": {
            "dated_utc": "2026-08-09",
            "READ_ONLY": True,
            "basis_commit": "1b4ea42",
            "cells": len(manifest),
            "matrix_manifest": manifest,
            "descriptive_summary": descriptive,
            "paired_contrasts": contrasts,
            "scale_effects": scale_effects,
            "stability": stability(cells),
            "resources": resources(cells, manifest),
            "statistical_procedure": {
                "rule": "the universal primary directional rule of "
                        "canonical plan section 5, applied verbatim",
                "conditions": [
                    "the mean paired difference is greater than zero",
                    "the 95 per cent fixed-seed-set image-clustered "
                    "interval has a lower bound greater than zero",
                    "at least two of the three individual seed "
                    "differences are greater than zero"],
                "negative_case": "the three mirrored conditions",
                "otherwise": "uncertain or mixed evidence",
                "implementation_reused":
                    "experiments/e8a_question_encoder/analyse_core.py "
                    "clustered_bootstrap, percentile_interval and "
                    "directional_rule; no test was added, substituted "
                    "or tuned for E8B",
                "caveat": INTERVAL_CAVEAT},
            "b1_selection_rule_caveat": B1_SELECTION_CAVEAT,
            "not_computed": {
                "readouts_R1_R2_R3": "NOT RUN. final_evaluation.py "
                                     "implements and validates them, but "
                                     "nothing drives it over the "
                                     "eighteen frozen checkpoints.",
                "interventions": "NOT RUN, for the same reason: "
                                 "fixed-image, fixed-question and "
                                 "deranged-image are implemented in "
                                 "final_evaluation.evaluate_condition "
                                 "and never invoked on a core "
                                 "checkpoint.",
                "raw_10004_denominator": "NOT COMPUTED. It belongs to "
                                         "the readout evaluation above; "
                                         "the primary metric here is R1 "
                                         "accuracy on the 7,714 "
                                         "in-vocabulary rows, which is "
                                         "what the core cells recorded.",
                "consequence": "this analysis is COMPLETE for the "
                               "descriptive, directional, stability and "
                               "resource questions and INCOMPLETE for "
                               "the readout and intervention questions."},
            "clean_test_accessed": False}}

    out = OUT_DIR / "e8b_core_analysis_20260809.json"
    e8b_run.atomic_write_json(out, record)
    print(f"written {out.name}")
    print(f"  manifest entries verified against disk : {len(manifest)}")
    print(f"  paired contrasts                       : {len(contrasts)}")
    print(f"  scale effects                          : {len(scale_effects)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
