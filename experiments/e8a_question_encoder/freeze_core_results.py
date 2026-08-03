"""Freeze the E8A SmolLM2-135M core into one result object.

    python -B experiments/e8a_question_encoder/freeze_core_results.py

Collects the execution manifest, every arm-scale-seed run record, the analysis
inputs, the statistical, reasoning, reliance and efficiency outputs and the
resource report into a single object, with every referenced artefact hashed at
freeze time. Reviewers verify against this object; it is not edited while a
review is open.

Nothing is recomputed here and nothing is retrained. This reads results and
writes one file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import utils  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import (  # noqa: E402
    analysis_integrity as integrity)

FROZEN = e8a.OUT_DIR / "e8a_135m_core_frozen.json"

ANALYSIS_OUTPUTS = ("core_analysis.json", "core_analysis_train_40k.json",
                    "efficiency_resources.json")
INPUT_FILES = ("data/v2/dev.csv", "data/v2/train_40k.csv",
               "data/v2/train_250k.csv", "data/v2/answer_vocab_v2.json",
               "data/v2/metadata/dev_types.csv",
               "data/v2/metadata/train_40k_types.csv",
               "data/v3/tokens/image_tokens.h5",
               "data/v3/tokens/question_tokens.h5",
               "data/v3_slm_tokens/e8a_135m_A1_train40k_dev.h5",
               "data/v3_slm_tokens/e8a_135m_A1_train250k_dev.h5",
               "data/v3_slm_tokens/e8a_135m_A1r_train40k_dev.h5",
               "data/v3_slm_tokens/e8a_135m_A1r_train250k_dev.h5")


def hashed(path: Path) -> dict:
    return {"path": path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": e8a.sha256_file(path)}


def main() -> int:
    utils.set_seed()
    manifest_path = e8a.OUT_DIR / "execution_manifest.json"
    manifest = json.loads(manifest_path.read_text())["e8a_execution_manifest"]
    view = integrity.canonical_dev_view()

    runs, missing = {}, []
    for spec in manifest["runs"]:
        arm, scale, seed = spec["arm"], spec["scale"], spec["seed"]
        cell = f"{arm}/{scale}/seed{seed}"
        reuse, artefacts = spec["reuse_artefacts"], spec["expected_artefacts"]
        record = e8a.OUT_DIR / (reuse["record"]["path"] if reuse
                                else artefacts["record"])
        correctness = e8a.OUT_DIR / (reuse["correctness"]["path"] if reuse
                                     else artefacts["correctness"])
        checkpoint = e8a.OUT_DIR / (reuse["checkpoint"]["path"] if reuse
                                    else artefacts["checkpoint"])
        if not record.exists():
            missing.append(cell)
            continue
        loaded = json.loads(record.read_text())
        block = loaded.get("e8a_core_run") or loaded.get("e8a_a0p_pilot") \
            or loaded["e8a_pilot"]
        run = (block["runs"][arm] if "runs" in block else block["run"])
        evaluation = (block["evaluations"][arm] if "evaluations" in block
                      else block["evaluation"])
        runs[cell] = {
            "arm": arm, "scale": scale, "seed": seed,
            "reused_phase_1b_pilot": bool(reuse),
            "record": hashed(record),
            "prediction_vectors": hashed(correctness),
            "checkpoint": hashed(checkpoint),
            "code_head": loaded["metadata"]["git_commit"],
            "code_head_dirty": loaded["metadata"]["git_dirty"],
            "dev_accuracy": evaluation["conditions"]["normal"]["accuracy"],
            "best_epoch": run["best_epoch"],
            "epochs_run": run["epochs_run"],
            "wall_clock_hours": run["wall_clock_hours"],
            "store_agreement_40k_vs_250k":
                block.get("gates", {}).get("store_agreement_40k_vs_250k"),
        }

    frozen = {
        "matrix": manifest["matrix"],
        "complete": not missing,
        "cells_missing": missing,
        "cells_frozen": len(runs),
        "execution_manifest": hashed(manifest_path),
        "runs": runs,
        "analysis_inputs": {
            "canonical_evaluation_rows": {
                "n_rows": view.n_rows,
                "row_order_sha256": view.row_order_sha256,
                "labels_sha256": view.labels_sha256,
                "vocabulary_sha256": view.vocabulary_sha256,
                "dev_sha256": view.dev_sha256},
            "files": [hashed(PROJECT_ROOT / p) for p in INPUT_FILES
                      if (PROJECT_ROOT / p).exists()]},
        "analysis_outputs": {name: hashed(e8a.OUT_DIR / name)
                             for name in ANALYSIS_OUTPUTS
                             if (e8a.OUT_DIR / name).exists()},
        "scope": {
            "authorised": ("E8A SmolLM2-135M question-encoder arms A0p, A1 "
                           "and A1r at train_40k and train_250k with seeds "
                           "0, 1 and 2"),
            "not_authorised": manifest["not_authorised"],
            "clean_test_accessed": False,
            "results_are_development_only": True},
    }
    utils.save_json({"metadata": utils.run_metadata(),
                     "e8a_135m_core_frozen": frozen}, FROZEN)
    print(f"frozen {frozen['cells_frozen']} cell(s), complete="
          f"{frozen['complete']}")
    for name, block in frozen["analysis_outputs"].items():
        print(f"  {name:32s} {block['sha256'][:16]}...")
    print(f"written to {FROZEN.relative_to(PROJECT_ROOT)}")
    return 0 if frozen["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
