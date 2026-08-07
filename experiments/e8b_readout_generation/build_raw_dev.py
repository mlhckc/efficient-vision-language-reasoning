"""Build the 10,004-row RAW development manifest, with gold answers.

Canonical plan section 6: "Every readout evaluation runs once over the
10,004-row raw denominator", and section 6.1 requires every final
checkpoint to receive raw-distribution evaluation. `data/v2/dev.csv`
holds only the 7,714 IN-VOCABULARY rows, and
`data/v2/dev_raw_question_ids.json` holds the raw identifiers with no
answers, so the raw evaluation had nothing to score against.

This rebuilds the raw rows from the same GQA source and the same
partition the Day-1 build used, and VERIFIES the result against the two
artefacts already on record: the questionId set must equal
`dev_raw_question_ids.json` exactly, and the in-vocabulary subset must
reproduce `dev.csv` row for row. A mismatch is a refusal, not a warning.

Why the raw denominator matters, stated plainly: for R1 and R2, which
are closed over the answer vocabulary, an out-of-vocabulary row can
never be correct, so raw accuracy is in-vocabulary accuracy times
coverage and needs no gold string. R3 generates freely and COULD emit a
correct out-of-vocabulary answer, so R3 genuinely needs these gold
answers and cannot be scored on the raw denominator without them.

THE EMBARGOED CLEAN TEST IS NEVER TOUCHED. This module reads the GQA
training questions and the development partition only.

NO OPTIMIZER STEP.

    python -B experiments/e8b_readout_generation/build_raw_dev.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402
from experiments.e8b_readout_generation import run as e8b_run  # noqa: E402

V2_DIR = PROJECT_ROOT / "data" / "v2"
RAW_DEV = V2_DIR / "dev_raw.csv"
RECORD = e8b_run.OUT_DIR / "raw_dev_manifest_20260807.json"


def load_v1_prepare_module():
    """The Day-1 loader, reused verbatim so the parse cannot drift."""
    import importlib.util
    path = PROJECT_ROOT / "1_prepare_gqa.py"
    spec = importlib.util.spec_from_file_location("v1_prepare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    utils.set_seed()
    expected_ids = set(json.loads(
        (V2_DIR / "dev_raw_question_ids.json").read_text()))
    in_vocab = pd.read_csv(V2_DIR / "dev.csv", dtype={"questionId": str,
                                                      "imageId": str},
                           keep_default_na=False, na_filter=False)

    v1 = load_v1_prepare_module()
    raw_train = v1.load_questions(
        config.GQA_RAW_DIR / config.GQA_TRAIN_QUESTIONS)
    raw_train["questionId"] = raw_train["questionId"].astype(str)
    raw_train["imageId"] = raw_train["imageId"].astype(str)

    # Select by the RECORDED identifiers rather than by re-deriving the
    # partition: the identifiers are the artefact of record, and using
    # them makes the selection independent of any partition code drift.
    raw_dev = raw_train[raw_train["questionId"].isin(expected_ids)].copy()
    raw_dev = raw_dev.sort_values("questionId").reset_index(drop=True)

    if set(raw_dev["questionId"]) != expected_ids:
        missing = sorted(expected_ids - set(raw_dev["questionId"]))[:5]
        raise AssertionError(
            f"RAW DEV REFUSED: {len(expected_ids - set(raw_dev['questionId']))} "
            f"recorded questionIds are absent from the GQA source "
            f"(e.g. {missing}); the manifest would silently cover less "
            f"than the raw denominator it claims")

    vocabulary = json.loads(
        (V2_DIR / "answer_vocab_v2.json").read_text())
    answer_to_index = vocabulary["answer_to_index"]
    raw_dev["label"] = raw_dev["answer"].map(
        lambda a: answer_to_index.get(a, -1))
    raw_dev["in_vocabulary"] = raw_dev["label"] >= 0

    # The in-vocabulary subset MUST reproduce dev.csv exactly. This is
    # the check that proves the rebuild is the same data, not merely a
    # plausible one.
    rebuilt = raw_dev[raw_dev["in_vocabulary"]][
        ["questionId", "imageId", "question", "answer", "label"]
    ].sort_values("questionId").reset_index(drop=True)
    reference = in_vocab.sort_values("questionId").reset_index(drop=True)
    reference["label"] = reference["label"].astype(int)
    if len(rebuilt) != len(reference):
        raise AssertionError(
            f"RAW DEV REFUSED: the in-vocabulary subset has "
            f"{len(rebuilt)} rows against dev.csv's {len(reference)}")
    for column in ("questionId", "imageId", "question", "answer", "label"):
        if not rebuilt[column].equals(reference[column]):
            differing = int((rebuilt[column] != reference[column]).sum())
            raise AssertionError(
                f"RAW DEV REFUSED: column {column!r} differs from "
                f"dev.csv on {differing} rows")

    columns = ["questionId", "imageId", "question", "answer", "label",
               "in_vocabulary"]
    raw_dev[columns].to_csv(RAW_DEV, index=False)

    coverage = float(raw_dev["in_vocabulary"].mean())
    oov = raw_dev[~raw_dev["in_vocabulary"]]
    record = {"metadata": utils.run_metadata(),
              "e8b_raw_dev_manifest": {
        "dated_utc": "2026-08-07",
        "purpose": "the 10,004-row RAW denominator required by canonical "
                   "plan section 6, with gold answers, so R3 can be "
                   "scored on it. R1 and R2 are closed over the "
                   "vocabulary, so their raw accuracy is in-vocabulary "
                   "accuracy times coverage and needs no gold string; "
                   "R3 generates freely and genuinely needs these.",
        "path": str(RAW_DEV.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(RAW_DEV),
        "rows": int(len(raw_dev)),
        "in_vocabulary_rows": int(raw_dev["in_vocabulary"].sum()),
        "out_of_vocabulary_rows": int((~raw_dev["in_vocabulary"]).sum()),
        "coverage": round(coverage, 6),
        "verification": {
            "questionid_set_equals_recorded_manifest": True,
            "in_vocabulary_subset_reproduces_dev_csv": True,
            "dev_csv_sha256": sha256_file(V2_DIR / "dev.csv"),
            "recorded_ids_sha256": sha256_file(
                V2_DIR / "dev_raw_question_ids.json"),
            "note": "both checks are refusals, not warnings: a mismatch "
                    "raises before anything is written"},
        "out_of_vocabulary_answer_head": [
            [answer, int(count)] for answer, count in
            Counter(oov["answer"]).most_common(10)],
        "source": {
            "questions": str((config.GQA_RAW_DIR
                              / config.GQA_TRAIN_QUESTIONS
                              ).relative_to(PROJECT_ROOT)),
            "loader": "1_prepare_gqa.load_questions, reused verbatim",
            "selection": "by the RECORDED questionIds in "
                         "dev_raw_question_ids.json, not by re-deriving "
                         "the partition"},
        "clean_test_accessed": False}}
    e8b_run.atomic_write_json(RECORD, record) if not RECORD.exists() else \
        RECORD.write_text(json.dumps(record, indent=2, default=str) + "\n")

    body = record["e8b_raw_dev_manifest"]
    print(f"  rows          : {body['rows']} "
          f"({body['in_vocabulary_rows']} in vocabulary, "
          f"{body['out_of_vocabulary_rows']} out)")
    print(f"  coverage      : {body['coverage']}")
    print(f"  dev.csv match : reproduced row for row")
    print(f"  sha256        : {body['sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
