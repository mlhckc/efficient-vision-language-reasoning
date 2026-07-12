"""Build the V2 evaluation protocol manifests (Day 1: manifests only).

This script partitions the raw balanced GQA questions into a development set,
a training pool with nested training manifests, and a blinded clean test set,
and writes the manifests, membership artefacts, vocabulary, a deterministic
build summary and a hash record under data/v2. It trains nothing, extracts no
embeddings and makes no predictions.

Test blinding: training and development code must never load
data/v2/test_clean_targets.csv before final evaluation. The inputs file
(questionId, imageId, question) is the only test file such code may read.

Embargo: no output of this script states clean-test vocabulary coverage, OOV
counts, answer distribution, yes/no share or question-type statistics.
Clean-test reporting is structural only: question count and unique image count.

data/val.csv is the legacy V1 validation manifest and is referred to as
legacy_v1_validation throughout. It is read only to exclude its images from
the clean test.

Determinism: three independent numpy.random.default_rng streams, each
constructed independently and used exactly once, applied after canonical
lexicographic string sorting. The threshold constants live here rather than in
config.py because config.py is outside this run's output allowlist; the seeds
derive from config.RANDOM_SEED.
"""

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402

DEV_IMAGE_SEED = config.RANDOM_SEED
TRAIN_QUESTION_SEED = config.RANDOM_SEED + 1
TEST_IMAGE_SEED = config.RANDOM_SEED + 2

DEV_MIN_RAW_QUESTIONS = 10000
TEST_MIN_RAW_QUESTIONS = 8000
TRAIN_PREFIX_SIZES = (40000, 100000, 250000)

V2_DIR = config.DATA_DIR / "v2"
OUT_RESULTS_DIR = config.RESULTS_DIR / "experiments" / "v2_00_protocol"
MANIFEST_COLUMNS = ["questionId", "imageId", "question", "answer", "label"]


def load_v1_prepare_module():
    """Load 1_prepare_gqa.py via importlib; its filename is not importable."""
    path = PROJECT_ROOT / "1_prepare_gqa.py"
    spec = importlib.util.spec_from_file_location("v1_prepare_gqa", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(obj, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(frame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_complete_images(sorted_image_ids, questions_per_image, seed, minimum):
    """Assign complete images, in an rng-permuted order over the sorted ids,
    until the accumulated raw question count first reaches minimum."""
    rng = np.random.default_rng(seed)
    chosen = []
    accumulated = 0
    for index in rng.permutation(len(sorted_image_ids)):
        image_id = sorted_image_ids[int(index)]
        chosen.append(image_id)
        accumulated += questions_per_image[image_id]
        if accumulated >= minimum:
            break
    return set(chosen), accumulated


def answer_distribution(frame) -> dict:
    """Top-20 answers with shares and the combined yes/no share."""
    counts = Counter(frame["answer"])
    total = len(frame)
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
    return {
        "top20_by_frequency": [[a, int(c), round(c / total, 6)] for a, c in top],
        "yes_no_share": round(
            (counts.get("yes", 0) + counts.get("no", 0)) / total, 6),
    }


def main() -> None:
    utils.set_seed()
    v1 = load_v1_prepare_module()

    raw_train = v1.load_questions(config.GQA_RAW_DIR / config.GQA_TRAIN_QUESTIONS)
    raw_val = v1.load_questions(config.GQA_RAW_DIR / config.GQA_VAL_QUESTIONS)
    for frame in (raw_train, raw_val):
        frame["questionId"] = frame["questionId"].astype(str)
        frame["imageId"] = frame["imageId"].astype(str)
    print(f"raw train_balanced questions: {len(raw_train)}")
    print(f"raw val_balanced questions:   {len(raw_val)}")

    train_images_sorted = sorted(set(raw_train["imageId"]))
    val_images_set = set(raw_val["imageId"])
    raw_image_overlap = len(set(train_images_sorted) & val_images_set)
    print(f"unique train images: {len(train_images_sorted)}  "
          f"unique val images: {len(val_images_set)}  "
          f"raw train/val image overlap: {raw_image_overlap}")

    # Dev partition: complete images from the permuted sorted train image list.
    train_q_per_image = Counter(raw_train["imageId"])
    dev_images, dev_raw_count = assign_complete_images(
        train_images_sorted, train_q_per_image,
        DEV_IMAGE_SEED, DEV_MIN_RAW_QUESTIONS)
    pool_images = set(train_images_sorted) - dev_images

    dev_raw = raw_train[raw_train["imageId"].isin(dev_images)]
    pool_raw = raw_train[raw_train["imageId"].isin(pool_images)]
    assert len(dev_raw) == dev_raw_count
    assert dev_raw_count >= DEV_MIN_RAW_QUESTIONS
    print(f"dev: {len(dev_images)} images, {dev_raw_count} raw questions")
    print(f"pool: {len(pool_images)} images, {len(pool_raw)} raw questions")

    # V2 vocabulary from raw training-pool answers only (same rule as V1).
    vocab = v1.build_answer_vocab(pool_raw["answer"], config.TOP_K_ANSWERS)
    answers_by_index = sorted(vocab, key=vocab.get)

    # dev.csv: in-vocab dev questions with labels, sorted by questionId.
    dev_in = dev_raw[dev_raw["answer"].isin(vocab)].copy()
    dev_in["label"] = dev_in["answer"].map(vocab).astype(int)
    dev_in = dev_in.sort_values("questionId")[MANIFEST_COLUMNS]
    write_csv(dev_in, V2_DIR / "dev.csv")
    print(f"dev.csv: {len(dev_in)} in-vocab of {dev_raw_count} raw dev questions")

    # Eligible pool, then one permutation and nested prefixes.
    eligible = pool_raw[pool_raw["answer"].isin(vocab)].copy()
    if len(eligible) < TRAIN_PREFIX_SIZES[-1]:
        sys.exit(f"ABORT: eligible pool has {len(eligible)} in-vocab questions, "
                 f"fewer than the required {TRAIN_PREFIX_SIZES[-1]}")
    eligible["label"] = eligible["answer"].map(vocab).astype(int)
    eligible = eligible.sort_values("questionId").reset_index(drop=True)
    train_order = np.random.default_rng(TRAIN_QUESTION_SEED).permutation(len(eligible))
    eligible = eligible.iloc[train_order].reset_index(drop=True)
    for size in TRAIN_PREFIX_SIZES:
        write_csv(eligible.head(size)[MANIFEST_COLUMNS],
                  V2_DIR / f"train_{size // 1000}k.csv")
    print(f"eligible pool: {len(eligible)} questions "
          f"(margin over {TRAIN_PREFIX_SIZES[-1]}: "
          f"{len(eligible) - TRAIN_PREFIX_SIZES[-1]}); "
          f"wrote nested prefixes {TRAIN_PREFIX_SIZES}")

    # Clean test: val_balanced images not touched by legacy_v1_validation.
    legacy_v1_validation = pd.read_csv(
        config.VAL_SPLIT_PATH,
        dtype={"questionId": str, "imageId": str}, keep_default_na=False)
    legacy_images = set(legacy_v1_validation["imageId"])
    candidates = [i for i in sorted(val_images_set) if i not in legacy_images]
    val_q_per_image = Counter(raw_val["imageId"])
    available = sum(val_q_per_image[i] for i in candidates)
    if available < TEST_MIN_RAW_QUESTIONS:
        sys.exit(f"ABORT: only {available} raw questions on val images outside "
                 f"legacy_v1_validation, fewer than {TEST_MIN_RAW_QUESTIONS}")
    test_images, test_raw_count = assign_complete_images(
        candidates, val_q_per_image, TEST_IMAGE_SEED, TEST_MIN_RAW_QUESTIONS)

    test_raw = raw_val[raw_val["imageId"].isin(test_images)].copy()
    test_raw = test_raw.sort_values("questionId")
    assert len(test_raw) == test_raw_count
    test_raw["label"] = [vocab.get(a, -1) for a in test_raw["answer"]]
    write_csv(test_raw[["questionId", "imageId", "question"]],
              V2_DIR / "test_clean_inputs.csv")
    write_csv(test_raw[["questionId", "answer", "label"]],
              V2_DIR / "test_clean_targets.csv")
    print(f"test_clean: {len(test_raw)} questions on {len(test_images)} images "
          f"(structural counts only; all other clean-test statistics are "
          f"embargoed until final evaluation)")

    # Membership artefacts: sorted JSON string lists.
    write_json(sorted(dev_images), V2_DIR / "dev_image_ids.json")
    write_json(sorted(pool_images), V2_DIR / "train_pool_image_ids.json")
    write_json(sorted(test_images), V2_DIR / "test_clean_image_ids.json")
    write_json(sorted(dev_raw["questionId"]), V2_DIR / "dev_raw_question_ids.json")
    write_json(sorted(test_raw["questionId"]),
               V2_DIR / "test_clean_raw_question_ids.json")

    # Vocabulary, same schema as V1.
    write_json({"answer_to_index": vocab, "answers": answers_by_index,
                "top_k": len(vocab)}, V2_DIR / "answer_vocab_v2.json")

    # Deterministic build summary: no timestamps, runtimes, hostnames,
    # versions or absolute paths.
    with open(config.ANSWER_VOCAB_PATH, encoding="utf-8") as handle:
        v1_answers = set(json.load(handle)["answers"])
    v2_answers = set(answers_by_index)
    train_frames = {f"train_{s // 1000}k": eligible.head(s)
                    for s in TRAIN_PREFIX_SIZES}
    summary = {
        "seeds": {
            "DEV_IMAGE_SEED": DEV_IMAGE_SEED,
            "TRAIN_QUESTION_SEED": TRAIN_QUESTION_SEED,
            "TEST_IMAGE_SEED": TEST_IMAGE_SEED,
        },
        "raw": {
            "n_train_balanced_questions": int(len(raw_train)),
            "n_val_balanced_questions": int(len(raw_val)),
            "n_train_balanced_unique_images": len(train_images_sorted),
            "n_val_balanced_unique_images": len(val_images_set),
            "train_val_image_overlap": raw_image_overlap,
        },
        "dev": {
            "n_images": len(dev_images),
            "n_raw_questions": int(dev_raw_count),
            "n_invocab_questions": int(len(dev_in)),
            "v2_coverage": round(len(dev_in) / dev_raw_count, 6),
        },
        "pool": {
            "n_images": len(pool_images),
            "n_raw_questions": int(len(pool_raw)),
            "n_eligible_questions": int(len(eligible)),
            "margin_over_250k": int(len(eligible) - TRAIN_PREFIX_SIZES[-1]),
            "v2_coverage": round(len(eligible) / len(pool_raw), 6),
        },
        "vocabulary": {
            "top_k": len(vocab),
            "n_common_with_v1": len(v1_answers & v2_answers),
            "v1_only": sorted(v1_answers - v2_answers),
            "v2_only": sorted(v2_answers - v1_answers),
        },
        "manifests": {
            "dev": {"n_questions": int(len(dev_in)),
                    "n_unique_images": int(dev_in["imageId"].nunique())},
            **{name: {"n_questions": int(len(frame)),
                      "n_unique_images": int(frame["imageId"].nunique())}
               for name, frame in train_frames.items()},
            "test_clean": {"n_questions": int(len(test_raw)),
                           "n_unique_images": int(test_raw["imageId"].nunique())},
        },
        "answer_distributions": {
            "dev": answer_distribution(dev_in),
            **{name: answer_distribution(frame)
               for name, frame in train_frames.items()},
        },
    }
    write_json(summary, V2_DIR / "protocol_build_summary.json")

    # Hash record. manifest_hashes.json never hashes itself and never includes
    # the volatile build_run_metadata.json.
    generated_names = [
        "dev.csv", "train_40k.csv", "train_100k.csv", "train_250k.csv",
        "test_clean_inputs.csv", "test_clean_targets.csv",
        "dev_image_ids.json", "train_pool_image_ids.json",
        "test_clean_image_ids.json", "dev_raw_question_ids.json",
        "test_clean_raw_question_ids.json", "answer_vocab_v2.json",
        "protocol_build_summary.json",
    ]
    raw_source_paths = [config.GQA_RAW_DIR / config.GQA_TRAIN_QUESTIONS,
                        config.GQA_RAW_DIR / config.GQA_VAL_QUESTIONS]
    legacy_paths = [config.TRAIN_SPLIT_PATH, config.VAL_SPLIT_PATH,
                    config.ANSWER_VOCAB_PATH]
    hashes = {
        "generated": {(V2_DIR / name).relative_to(PROJECT_ROOT).as_posix():
                      sha256_file(V2_DIR / name) for name in generated_names},
        "raw_sources": {p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
                        for p in raw_source_paths},
        "legacy_v1": {p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
                      for p in legacy_paths},
    }
    write_json(hashes, V2_DIR / "manifest_hashes.json")

    # Volatile run metadata goes only here.
    utils.save_json(utils.run_metadata(),
                    OUT_RESULTS_DIR / "build_run_metadata.json")
    print("build complete: manifests, membership artefacts, vocabulary, "
          "summary and hashes written to data/v2")


if __name__ == "__main__":
    main()
