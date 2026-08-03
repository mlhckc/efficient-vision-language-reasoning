"""Integrity assertions every E8A analysis must pass before it computes a number.

Three failures are cheap to make and expensive to discover late: comparing two
prediction vectors that are not in the same row order, filing a result under the
wrong arm, scale or seed, and analysing an artefact that changed after the run
that produced it recorded its hash. Each is asserted here, from the artefacts
themselves rather than from what a record claims about them, and each assertion
has a known-negative probe in tests/test_e8a.py.

Nothing here reads, resolves or names the embargoed clean-test target.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402

# Every condition an E8A evaluation must have scored, and which the analysis is
# entitled to read. A cell missing one of these is not comparable.
REQUIRED_CONDITIONS = ("normal", "fixed_image", "fixed_question",
                       "shuffled_image_derangement")


@dataclass(frozen=True)
class DevView:
    """The one canonical evaluation-row order, derived from data/v2/dev.csv."""

    question_ids: np.ndarray
    labels: np.ndarray
    image_ids: np.ndarray
    row_order_sha256: str
    labels_sha256: str
    vocabulary_sha256: str
    dev_sha256: str
    n_rows: int


def canonical_dev_view() -> DevView:
    dev = pd.read_csv(e8a.V2_DIR / "dev.csv",
                      dtype={"questionId": str, "imageId": str},
                      keep_default_na=False)
    question_ids = dev["questionId"].to_numpy().astype(str)
    labels = dev["label"].to_numpy().astype(np.int64)
    return DevView(
        question_ids=question_ids,
        labels=labels,
        image_ids=dev["imageId"].to_numpy().astype(str),
        row_order_sha256=hashlib.sha256(
            "\n".join(question_ids).encode()).hexdigest(),
        labels_sha256=hashlib.sha256(
            labels.tobytes()).hexdigest(),
        vocabulary_sha256=e8a.sha256_file(e8a.V2_DIR / "answer_vocab_v2.json"),
        dev_sha256=e8a.sha256_file(e8a.V2_DIR / "dev.csv"),
        n_rows=len(dev))


def assert_row_order(cell: str, correctness, view: DevView) -> dict:
    """The stored per-row vectors must be the canonical dev rows, in order.

    Equal lengths prove nothing: two vectors of 7,714 rows in different orders
    would still subtract, and would produce a plausible wrong number rather
    than an error. The questionIds are compared elementwise, and the stored
    label column is compared against the vocabulary-derived labels, so the
    assertion covers row identity, row order and the scoring target together.
    """
    question_ids = np.asarray(correctness["question_ids"]).astype(str)
    if len(question_ids) != view.n_rows:
        raise AssertionError(
            f"{cell}: {len(question_ids)} evaluation rows, but the canonical "
            f"development view has {view.n_rows}")
    if not np.array_equal(question_ids, view.question_ids):
        first = int(np.nonzero(question_ids != view.question_ids)[0][0])
        raise AssertionError(
            f"{cell}: the evaluation rows are NOT in the canonical dev.csv "
            f"order; first difference at row {first}: stored "
            f"{question_ids[first]!r}, canonical {view.question_ids[first]!r}")
    labels = np.asarray(correctness["labels"]).astype(np.int64)
    if not np.array_equal(labels, view.labels):
        first = int(np.nonzero(labels != view.labels)[0][0])
        raise AssertionError(
            f"{cell}: the stored scoring targets differ from the vocabulary "
            f"labels at row {first}: {labels[first]} against "
            f"{view.labels[first]}; the cell was scored against a different "
            f"target and is not comparable")
    missing = [c for c in REQUIRED_CONDITIONS if c not in correctness]
    if missing:
        raise AssertionError(f"{cell}: correctness vectors missing "
                             f"condition(s) {missing}")
    wrong = {c: int(len(correctness[c])) for c in REQUIRED_CONDITIONS
             if len(correctness[c]) != view.n_rows}
    if wrong:
        raise AssertionError(f"{cell}: condition row counts {wrong} against "
                             f"{view.n_rows}")
    return {"row_order_verified": True,
            "row_order_sha256": view.row_order_sha256,
            "labels_sha256": view.labels_sha256,
            "n_rows": view.n_rows}


def remeasure(path: Path, recorded: str | None, cell: str, what: str) -> str:
    """Hash the artefact now, not when the run finished.

    A hash stored at run completion proves what existed then. The analysis
    reads the file as it is today, so that is what has to be hashed.
    """
    if not path.exists():
        raise AssertionError(f"{cell}: {what} is missing at {path}")
    live = e8a.sha256_file(path)
    if recorded is not None and live != recorded:
        raise AssertionError(
            f"{cell}: {what} changed since the run recorded it: "
            f"{recorded} -> {live}. STOP: the artefact under analysis is not "
            f"the artefact that was produced.")
    return live


def assert_cell_identity(cell: str, arm: str, scale: str, seed: int,
                         record: dict, block: dict, run: dict,
                         evaluation: dict, view: DevView) -> dict:
    """A result must not be able to arrive under the wrong arm, scale or seed.

    Everything the analysis will key on is checked against the record itself:
    the record's own arm/scale/seed, the seed the metadata was saved with, the
    checkpoint filename, and the vocabulary the run hashed.
    """
    claimed = (block.get("arm", arm), block.get("scale", scale),
               block.get("seed", seed))
    if claimed != (arm, scale, seed):
        raise AssertionError(
            f"{cell}: the record identifies itself as {claimed[0]}/"
            f"{claimed[1]}/seed{claimed[2]}; it is filed as {arm}/{scale}/"
            f"seed{seed}")
    metadata_seed = record.get("metadata", {}).get("seed")
    if metadata_seed is not None and metadata_seed != seed:
        raise AssertionError(f"{cell}: run metadata was saved with seed "
                             f"{metadata_seed}, not {seed}")
    # The run block carries arm, scale and seed in every record shape, pilot
    # and matrix alike, so it is the one identity source that is always there.
    if (run.get("arm", arm), run.get("scale", scale),
            run.get("seed", seed)) != (arm, scale, seed):
        raise AssertionError(
            f"{cell}: the run block says {run.get('arm')}/{run.get('scale')}/"
            f"seed{run.get('seed')}")
    if evaluation.get("arm", arm) != arm:
        raise AssertionError(f"{cell}: the evaluation block says "
                             f"{evaluation.get('arm')}")
    expected_stem = f"e8a_{arm}_{scale}_seed{seed}.pt"
    if Path(run["checkpoint"]).name != expected_stem:
        raise AssertionError(f"{cell}: the checkpoint is named "
                             f"{Path(run['checkpoint']).name}, not "
                             f"{expected_stem}")
    recorded_vocabulary = (block.get("data_hashes") or {}).get(
        "data/v2/answer_vocab_v2.json")
    if recorded_vocabulary and recorded_vocabulary != view.vocabulary_sha256:
        raise AssertionError(
            f"{cell}: scored against vocabulary {recorded_vocabulary[:12]}, "
            f"the analysis uses {view.vocabulary_sha256[:12]}")
    return {
        "arm": arm, "scale": scale, "seed": seed,
        "metadata_seed": metadata_seed,
        "code_head": record.get("metadata", {}).get("git_commit"),
        "code_head_dirty": record.get("metadata", {}).get("git_dirty"),
        "vocabulary_sha256": view.vocabulary_sha256,
        "dev_sha256": view.dev_sha256,
    }


def assert_scoring_agrees(cell: str, correctness, evaluation: dict,
                          run: dict) -> dict:
    """The reported accuracy must be the mean of the per-row vectors it cites.

    This is what makes "the same scoring code" checkable across cells whose
    records were written at different commits: whatever code produced them, the
    stored vectors and the stored scalar have to agree under one definition.
    """
    recomputed = {c: round(float(np.asarray(correctness[c]).mean()), 5)
                  for c in REQUIRED_CONDITIONS}
    reported = {c: evaluation["conditions"][c]["accuracy"]
                for c in REQUIRED_CONDITIONS}
    disagreeing = {c: (reported[c], recomputed[c]) for c in REQUIRED_CONDITIONS
                   if abs(reported[c] - recomputed[c]) > 1e-5}
    if disagreeing:
        raise AssertionError(
            f"{cell}: reported accuracy disagrees with the mean of the stored "
            f"per-row vectors for {disagreeing}")
    if abs(run["best_dev_accuracy"] - recomputed["normal"]) > 1e-5:
        raise AssertionError(
            f"{cell}: best_dev_accuracy {run['best_dev_accuracy']} is not the "
            f"evaluated normal accuracy {recomputed['normal']}")
    return {"recomputed_from_stored_vectors": recomputed,
            "agrees_within": 1e-5}


def cell_evidence(cell: str, arm: str, scale: str, seed: int,
                  record_path: Path, record: dict, block: dict, run: dict,
                  evaluation: dict, correctness_path: Path,
                  view: DevView) -> dict:
    """Every integrity check for one statistical cell, in one place."""
    checkpoint_sha = remeasure(PROJECT_ROOT / run["checkpoint"],
                               run.get("checkpoint_sha256"), cell,
                               "the checkpoint")
    correctness_sha = remeasure(
        correctness_path,
        evaluation.get("correctness_vectors", {}).get("sha256"), cell,
        "the correctness vectors")
    record_sha = remeasure(record_path, None, cell, "the run record")
    with np.load(correctness_path, allow_pickle=True) as correctness:
        rows = assert_row_order(cell, correctness, view)
        scoring = assert_scoring_agrees(cell, correctness, evaluation, run)
    identity = assert_cell_identity(cell, arm, scale, seed, record, block, run,
                                    evaluation, view)
    return {**identity, **rows, "scoring": scoring,
            "checkpoint_sha256": checkpoint_sha,
            "prediction_sha256": correctness_sha,
            "record_sha256": record_sha,
            "record_path": record_path.name,
            "correctness_path": correctness_path.name,
            "hashes_remeasured_at_analysis_time": True}
