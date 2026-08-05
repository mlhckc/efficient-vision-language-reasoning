"""G21 resource-accounting correction for the E8A SmolLM2-135M core.

    python -B experiments/e8a_question_encoder/resource_correction_g21.py \
        --analysis-seconds <measured wall seconds of analyse_core_g21>

Corrects the resource story without re-timing training:

  * the historical 5.78927 GPU-hour figure is labelled by exactly the
    components it timed: 4.33551 hours of training wall-clock over eighteen
    cells (which INCLUDES the per-epoch development evaluations and the
    post-training condition evaluations run inside each invocation, never
    timed separately) plus 1.45376 hours of store extraction. It contains no
    standalone evaluation or analysis component.
  * the G21 re-inference provides the first standalone evaluation timing,
    split into normal-condition, G11-repeat and intervention passes per cell
    (read from the prediction-artefact sidecars), measured on a shared GPU
    and labelled as such.
  * resource use is aggregated separately per frozen-model identity — A0p
    (frozen CLIP question tokens), A1 (pretrained SmolLM2-135M) and A1r
    (random-initialised SmolLM2-135M) — because pooling unlike frozen models
    into one per-model gate figure answers no gate. Each identity is placed
    against the 35 GPU-hour per-model ceiling on its own, with the E8B
    caveat where it applies.
  * storage is measured and placed against an actual quota where one
    exists. /scratch carries no per-project allocation this session could
    find, so the scratch storage gate is recorded UNRESOLVED rather than
    passed; the NFS backup share has a hard filesystem size and is reported
    against it.

Nothing is trained or re-timed; this reads existing measured records and
writes resource_correction_g21.json. The embargoed clean-test target is
never read.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import utils  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import cell_validator  # noqa: E402
from experiments.e8a_question_encoder import (  # noqa: E402
    analysis_integrity as integrity)

ARMS = ("A0p", "A1", "A1r")
SCALES = ("train_40k", "train_250k")
SEEDS = (0, 1, 2)
PRED_DIR = e8a.OUT_DIR / "predictions_g21_v1"

FROZEN_IDENTITY = {
    "A0p": "frozen CLIP ViT-B/32 question tokens (no language model)",
    "A1": "pretrained SmolLM2-135M, pinned revision",
    "A1r": "random-initialised SmolLM2-135M, pinned seed 20260802",
}
PER_MODEL_CEILING_HOURS = 35.0

BACKUP_FILESYSTEM_ROOT = Path("/user/HS400/mc02623")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-seconds", type=float, required=True,
                        help="measured wall seconds of the analyse_core_g21 "
                             "invocation, recorded by the caller")
    args = parser.parse_args()
    utils.set_seed()

    manifest = json.loads(
        (e8a.OUT_DIR / "execution_manifest.json").read_text())[
            "e8a_execution_manifest"]
    view = integrity.canonical_dev_view()

    training = {arm: 0.0 for arm in ARMS}
    for arm in ARMS:
        for scale in SCALES:
            for seed in SEEDS:
                run = cell_validator.load_cell(manifest, arm, scale,
                                               seed)["run"]
                training[arm] += run["wall_clock_hours"]
    training = {arm: round(hours, 5) for arm, hours in training.items()}

    stores = json.loads(
        (e8a.OUT_DIR / "efficiency_resources.json").read_text())[
            "e8a_efficiency_and_resources"]["resource_audit"]
    extraction = stores["extraction_gpu_hours_by_arm"]

    evaluation = {"per_cell": {}, "by_component_seconds": {
        "normal": 0.0, "g11_repeat": 0.0, "fixed_image": 0.0,
        "fixed_question": 0.0, "shuffled_image": 0.0}}
    summary = json.loads((PRED_DIR / "reinference_summary.json").read_text())[
        "g21_reinference"]
    for arm in ARMS:
        for scale in SCALES:
            for seed in SEEDS:
                sidecar = json.loads(
                    (PRED_DIR / f"predictions_{arm}_{scale}_seed{seed}.json")
                    .read_text())
                timings = sidecar["timings_seconds"]
                evaluation["per_cell"][f"{arm}/{scale}/seed{seed}"] = timings
                evaluation["by_component_seconds"]["normal"] += \
                    timings["normal_s"]
                evaluation["by_component_seconds"]["g11_repeat"] += \
                    timings["g11_repeat_s"]
                for condition in ("fixed_image", "fixed_question",
                                  "shuffled_image"):
                    evaluation["by_component_seconds"][condition] += \
                        timings[f"{condition}_s"]
    evaluation["by_component_seconds"] = {
        k: round(v, 1) for k, v in evaluation["by_component_seconds"].items()}
    evaluation["normal_evaluation_seconds"] = round(
        evaluation["by_component_seconds"]["normal"]
        + evaluation["by_component_seconds"]["g11_repeat"], 1)
    evaluation["intervention_evaluation_seconds"] = round(
        sum(evaluation["by_component_seconds"][c]
            for c in ("fixed_image", "fixed_question", "shuffled_image")), 1)
    evaluation["total_wall_seconds_including_loads"] = \
        summary["timing"]["total_wall_seconds"]
    evaluation["measurement_context"] = (
        "wall-clock on a GPU shared with an unrelated user's training "
        "process; these are honest upper bounds on exclusive-GPU cost and "
        "are reported separately from, and never added to, the historical "
        "training figure")

    per_identity = {}
    for arm in ARMS:
        identity_hours = round(training[arm] + (extraction.get(arm) or 0.0),
                               5)
        per_identity[arm] = {
            "frozen_model_identity": FROZEN_IDENTITY[arm],
            "training_gpu_hours": training[arm],
            "extraction_gpu_hours": extraction.get(arm) or 0.0,
            "identity_total_gpu_hours": identity_hours,
            "per_model_gate_35h": {
                "fires": identity_hours >= PER_MODEL_CEILING_HOURS,
                "margin_hours": round(PER_MODEL_CEILING_HOURS
                                      - identity_hours, 5)},
            "caveat": ("the canonical SmolLM2-135M aggregate would also "
                       "count E8B readout arms, which are unauthorised and "
                       "unrun, so this identity total is a lower bound"
                       if arm in ("A1", "A1r") else
                       "the CLIP question tower is shared with V3/v2 work; "
                       "this total covers only the E8A A0p runs"),
        }

    scratch = shutil.disk_usage(PROJECT_ROOT)
    quota_out = subprocess.run(["quota", "-s"], capture_output=True,
                               text=True)
    backup = shutil.disk_usage(BACKUP_FILESYSTEM_ROOT)
    retained = 0
    retained_files = 0
    for pattern in ("checkpoints/*.pt", "correctness_*.npz", "*.json",
                    "predictions_g21_v1/*"):
        for path in e8a.OUT_DIR.glob(pattern):
            if path.is_file():
                retained += path.stat().st_size
                retained_files += 1
    token_stores = sum(p.stat().st_size
                       for p in (PROJECT_ROOT / "data"
                                 / "v3_slm_tokens").glob("*.h5"))

    record = {
        "historical_figure_correction": {
            "figure": 5.78927,
            "correct_label": (
                "4.33551 GPU-hours of training wall-clock over eighteen "
                "cells plus 1.45376 GPU-hours of store extraction. The "
                "training wall-clock INCLUDES the per-epoch development "
                "evaluations and the in-invocation condition evaluations; "
                "no standalone evaluation component was ever timed inside "
                "this figure and none can be recovered from it. It "
                "contains no analysis time."),
            "incorrect_readings_excluded": (
                "any reading that the 5.78927 figure separately accounted "
                "for evaluation, and any per-model gate that pooled the "
                "three unlike frozen-model identities into one number")},
        "components_reported_separately": {
            "extraction_gpu_hours_total":
                stores["extraction_gpu_hours_total"],
            "training_gpu_hours_total": stores["training_gpu_hours_total"],
            "g21_evaluation": evaluation,
            "g21_analysis_wall_seconds": round(args.analysis_seconds, 1),
            "total_previously_measured_gpu_hours": stores["total_gpu_hours"],
        },
        "per_frozen_model_identity": per_identity,
        "storage": {
            "retained_e8a_result_bytes": retained,
            "retained_e8a_result_files": retained_files,
            "frozen_question_store_bytes": token_stores,
            "scratch_free_gib": round(scratch.free / 2 ** 30, 1),
            "scratch_total_gib": round(scratch.total / 2 ** 30, 1),
            "scratch_gate": {
                "status": "UNRESOLVED",
                "reason": ("no authoritative per-project /scratch "
                           "allocation is recorded anywhere in the "
                           "repository or visible to this session "
                           f"(quota -s output: "
                           f"{quota_out.stdout.strip()!r}); free space is "
                           "reported, but free space is not a quota, so "
                           "the storage gate cannot be discharged against "
                           "it")},
            "backup_share": {
                "filesystem_root": str(BACKUP_FILESYSTEM_ROOT),
                "total_gib": round(backup.total / 2 ** 30, 1),
                "free_gib": round(backup.free / 2 ** 30, 1),
                "note": ("hard filesystem size of the NFS home share "
                         "holding the off-node backup; the E8A backup fits "
                         "within current free space")},
        },
        "clean_test_accessed": False,
    }
    utils.save_json({"metadata": utils.run_metadata(),
                     "g21_resource_correction": record},
                    e8a.OUT_DIR / "resource_correction_g21.json")
    print("per-identity GPU-hours:")
    for arm, block in per_identity.items():
        print(f"  {arm:4s} {block['identity_total_gpu_hours']:8.5f} h "
              f"({block['frozen_model_identity']})")
    print(f"evaluation: normal+repeat "
          f"{evaluation['normal_evaluation_seconds']}s, interventions "
          f"{evaluation['intervention_evaluation_seconds']}s")
    print(f"scratch storage gate: UNRESOLVED (no authoritative quota)")
    print("written to results/experiments/e8a_question_encoder/"
          "resource_correction_g21.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
