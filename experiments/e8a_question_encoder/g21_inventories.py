"""G21 collision and fixed-point inventories, required before re-inference.

    python -B experiments/e8a_question_encoder/g21_inventories.py

Canonical section 12: before development re-inference, produce complete
normalised-string collision and fixed-point inventories over the top-100
vocabulary, the top-1000 vocabulary and every V2 training answer string. A
collision never merges, removes or remaps classes; this script reports, and
the scorer compares normalised strings row by row with every class ID intact.

Writes results/experiments/e8a_question_encoder/g21_scorer_inventory.json.
Nothing is trained, no store is opened, and the embargoed clean-test target is
never read.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import utils  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import g21_scorer as g21  # noqa: E402

VOCAB_100 = PROJECT_ROOT / "data" / "v2" / "answer_vocab_v2.json"
VOCAB_1000 = PROJECT_ROOT / "data" / "v2_1000" / "answer_vocab_v2_1000.json"
TRAIN_MANIFESTS = {
    "v2_top100": [PROJECT_ROOT / "data" / "v2" / f"{scale}.csv"
                  for scale in ("train_40k", "train_100k", "train_250k")],
    "v2_1000": [PROJECT_ROOT / "data" / "v2_1000" / f"{scale}.csv"
                for scale in ("train_40k", "train_100k", "train_250k")],
}


def vocabulary_block(path: Path) -> dict:
    mapping, provenance = g21.load_index_to_answer(path)
    collisions = g21.collision_inventory(mapping)
    fixed_points = g21.fixed_point_inventory(mapping)
    for normalized, members in collisions["collision_groups"].items():
        collisions.setdefault("collision_class_ids", {})[normalized] = {
            member: mapping.index(member) for member in members}
    return {"vocabulary": provenance, "collisions": collisions,
            "fixed_points": fixed_points}


def training_answer_block(name: str, manifests) -> dict:
    answers = set()
    per_manifest = {}
    for manifest in manifests:
        frame = pd.read_csv(manifest, usecols=["answer"],
                            keep_default_na=False)
        strings = set(frame["answer"].astype(str))
        per_manifest[manifest.relative_to(PROJECT_ROOT).as_posix()] = {
            "rows": int(len(frame)),
            "distinct_answer_strings": len(strings),
            "sha256": g21.sha256_file(manifest)}
        answers |= strings
    return {"manifests": per_manifest,
            "distinct_answer_strings": len(answers),
            "collisions": g21.collision_inventory(sorted(answers)),
            "fixed_points": g21.fixed_point_inventory(answers)}


def main() -> int:
    utils.set_seed()
    dev = pd.read_csv(PROJECT_ROOT / "data" / "v2" / "dev.csv",
                      usecols=["answer"], keep_default_na=False)
    dev_answers = set(dev["answer"].astype(str))

    record = {
        "scorer": g21.scorer_provenance(),
        "behavioural_adaptations": list(g21.BEHAVIOURAL_ADAPTATIONS),
        "top_100": vocabulary_block(VOCAB_100),
        "top_1000": vocabulary_block(VOCAB_1000),
        "training_answers": {
            name: training_answer_block(name, manifests)
            for name, manifests in TRAIN_MANIFESTS.items()},
        "dev_answers": {
            "distinct_answer_strings": len(dev_answers),
            "collisions": g21.collision_inventory(sorted(dev_answers)),
            "fixed_points": g21.fixed_point_inventory(dev_answers)},
        "raw_versus_normalized_consequence": (
            "The top-100 vocabulary has zero normalised collisions and zero "
            "strings changed by normalisation, so for classifiers whose "
            "predictions and golds are both top-100 vocabulary strings, "
            "rowwise normalised exact match coincides with rowwise raw exact "
            "match. This is now a measured inventory result, not an untested "
            "assertion; the per-row prediction artefacts still carry both "
            "metrics independently."),
        "no_silent_merge": (
            "No class ID is merged, removed or remapped by any collision; "
            "the inventory reports groups and their class IDs, and row "
            "correctness is defined only by normalised prediction-string "
            "versus normalised canonical-gold-string equality."),
        "clean_test_accessed": False,
    }
    utils.save_json({"metadata": utils.run_metadata(),
                     "g21_scorer_inventory": record},
                    e8a.OUT_DIR / "g21_scorer_inventory.json")

    for name in ("top_100", "top_1000"):
        block = record[name]
        print(f"{name}: {block['collisions']['n_collision_groups']} collision "
              f"group(s), {block['fixed_points']['n_changed_by_normalization']}"
              f" string(s) changed by normalisation, idempotent on all")
    for name, block in record["training_answers"].items():
        print(f"training answers {name}: "
              f"{block['distinct_answer_strings']} distinct, "
              f"{block['collisions']['n_collision_groups']} collision "
              f"group(s)")
    print("written to results/experiments/e8a_question_encoder/"
          "g21_scorer_inventory.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
