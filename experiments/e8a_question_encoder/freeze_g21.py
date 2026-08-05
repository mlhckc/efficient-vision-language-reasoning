"""Freeze the post-G21 E8A remediation into one result object.

    python -B experiments/e8a_question_encoder/freeze_g21.py

Collects, with every referenced artefact hashed at freeze time: the
operative canonical hashes; the vendored scorer source, licence and adapter
with their pinned SHA-256 values; the scorer test module; the collision and
fixed-point inventories; the 18 cell-validity records; every per-row
prediction artefact and sidecar; the regenerated raw and normalised
statistics with the old-versus-new comparison; the v2_05c clustered
correction; the resource correction; the safety regression test module; and
the supervisor-feedback closure record. Reviewers verify against this
object; it is not edited while a review is open, and the pre-G21
e8a_135m_core_frozen.json is left untouched as the before-state.

Nothing is recomputed and nothing is retrained. This reads results and
writes one file: e8a_135m_core_frozen_g21.json. The embargoed clean-test
target is never read.
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
from experiments.e8a_question_encoder import g21_scorer as g21  # noqa: E402

FROZEN = e8a.OUT_DIR / "e8a_135m_core_frozen_g21.json"
PRED_DIR = e8a.OUT_DIR / "predictions_g21_v1"

CANONICAL_FILES = (
    ".agent-bridge/tasks/_shared/e8-master-protocol-canonical.md",
    ".agent-bridge/tasks/e8a-question-encoder/10-canonical-plan.md",
    ".agent-bridge/tasks/e8b-readout-generation/10-canonical-plan.md",
    ".agent-bridge/tasks/e9-compact-vlm/10-canonical-plan.md",
)

CODE_FILES = (
    "experiments/e8a_question_encoder/g21_scorer.py",
    "experiments/e8a_question_encoder/g21_inventories.py",
    "experiments/e8a_question_encoder/cell_validator.py",
    "experiments/e8a_question_encoder/reinfer_g21.py",
    "experiments/e8a_question_encoder/analyse_core_g21.py",
    "experiments/e8a_question_encoder/resource_correction_g21.py",
    "experiments/e8a_question_encoder/run_matrix.py",
    "experiments/e8a_question_encoder/build_artefact_manifest.py",
    "experiments/v2_05_types/addendum_clustered.py",
    "tests/test_g21_scorer.py",
    "tests/test_e8a_safety.py",
    "tests/run_all.py",
)

RESULT_FILES = (
    "results/experiments/e8a_question_encoder/g21_scorer_inventory.json",
    "results/experiments/e8a_question_encoder/cell_validation_g21.json",
    "results/experiments/e8a_question_encoder/core_analysis_g21.json",
    "results/experiments/e8a_question_encoder/resource_correction_g21.json",
    "results/experiments/e8a_question_encoder/e8a_135m_core_frozen.json",
    "results/experiments/e8a_question_encoder/core_analysis.json",
    "results/experiments/v2_05_types/addendum_clustered.json",
    "results/experiments/v2_05_types/addendum.json",
)

DOC_FILES = (
    "docs/experiments/e8a_core_135m.md",
    "docs/experiments/supervisor_feedback_closure.md",
    "docs/experiments/v3_03_scaling.md",
    "docs/experiments/v2_05_types.md",
    "docs/experiments/v2_06_reliance.md",
    "docs/experiments/e7a_efficiency.md",
    "results/experiments/e8a_question_encoder/README.md",
)


def hashed(path: Path) -> dict:
    return {"path": path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": g21.sha256_file(path)}


def main() -> int:
    utils.set_seed()
    analysis = json.loads(
        (e8a.OUT_DIR / "core_analysis_g21.json").read_text())[
            "e8a_core_analysis_g21"]
    validation = json.loads(
        (e8a.OUT_DIR / "cell_validation_g21.json").read_text())[
            "e8a_cell_validation"]
    if validation["n_valid"] != 18:
        raise AssertionError("cannot freeze: not all 18 cells valid")
    if not analysis["raw_equals_normalized_rowwise_everywhere"]:
        raise AssertionError("cannot freeze: metric coincidence unverified")

    predictions = sorted(PRED_DIR.glob("predictions_*.csv.gz"))
    sidecars = sorted(PRED_DIR.glob("predictions_*.json"))
    if len(predictions) != 18 or len(sidecars) != 18:
        raise AssertionError(
            f"cannot freeze: {len(predictions)} artefacts and "
            f"{len(sidecars)} sidecars, need 18 of each")

    frozen = {
        "object": "E8A SmolLM2-135M post-G21 remediation freeze",
        "supersedes_for_scoring": (
            "results/experiments/e8a_question_encoder/"
            "e8a_135m_core_frozen.json remains the untouched before-state; "
            "this object adds the G21 normalised scoring, the per-row "
            "prediction artefacts and the corrections, and every "
            "comparative claim now rests on the normalised metric"),
        "scorer": g21.scorer_provenance(),
        "behavioural_adaptations": list(g21.BEHAVIOURAL_ADAPTATIONS),
        "operative_canonical_files": [
            hashed(PROJECT_ROOT / p) for p in CANONICAL_FILES],
        "code": [hashed(PROJECT_ROOT / p) for p in CODE_FILES],
        "results": [hashed(PROJECT_ROOT / p) for p in RESULT_FILES],
        "documents": [hashed(PROJECT_ROOT / p) for p in DOC_FILES],
        "prediction_artefacts": [hashed(p) for p in predictions],
        "prediction_sidecars": [hashed(p) for p in sidecars],
        "reinference_summary": hashed(PRED_DIR / "reinference_summary.json"),
        "cell_validation": {
            "n_valid": validation["n_valid"],
            "failures": validation["failures"]},
        "headline": {
            "metrics": analysis["metric_definitions"],
            "normalized_per_arm": {
                scale: analysis["analyses"][scale][
                    "per_arm_summary_normalized"]
                for scale in ("train_40k", "train_250k")},
            "raw_per_arm": {
                scale: analysis["analyses"][scale]["per_arm_summary_raw"]
                for scale in ("train_40k", "train_250k")},
            "contrasts_normalized": {
                scale: analysis["analyses"][scale]["contrasts_normalized"]
                for scale in ("train_40k", "train_250k")},
            "intervals_constructed": analysis["intervals_constructed"],
            "old_versus_new_summary":
                analysis["old_versus_new"]["summary"]},
        "clean_test_accessed": False,
    }
    utils.save_json({"metadata": utils.run_metadata(),
                     "e8a_135m_core_frozen_g21": frozen}, FROZEN)
    print(f"frozen: {len(frozen['prediction_artefacts'])} prediction "
          f"artefacts, {len(frozen['code'])} code files, "
          f"{len(frozen['results'])} result records, "
          f"{len(frozen['documents'])} documents")
    print(f"written to {FROZEN.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
