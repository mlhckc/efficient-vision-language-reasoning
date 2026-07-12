"""Independent verification of the V2 protocol manifests.

This verifier reloads the raw GQA JSONs with its own inline loading logic and
recomputes the partitions, vocabulary, counts and hashes with inline
implementations. It imports nothing from build_manifests.py and does not
import 1_prepare_gqa.py, so it shares the build's specification but not its
code. It trusts no value in protocol_build_summary.json; every compared number
is recomputed here.

Embargo: clean-test reporting is structural only (question count, unique image
count, overlaps, duplicates, hashes, image availability). No clean-test
vocabulary coverage, OOV count, answer distribution, yes/no share or
question-type statistic is computed or printed. test_clean_targets.csv is read
here only for row-level consistency checks; training and development code must
never load it before final evaluation.

Every check prints PASS or FAIL with expected and actual values. The report is
written to results/experiments/v2_00_protocol/protocol_report.json and the
exit code is nonzero on any failure. Missing image files set status BLOCKED.
"""

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

DEV_IMAGE_SEED = config.RANDOM_SEED
TRAIN_QUESTION_SEED = config.RANDOM_SEED + 1
TEST_IMAGE_SEED = config.RANDOM_SEED + 2
DEV_MIN_RAW_QUESTIONS = 10000
TEST_MIN_RAW_QUESTIONS = 8000
TRAIN_PREFIX_SIZES = (40000, 100000, 250000)

V2_DIR = config.DATA_DIR / "v2"
OUT_RESULTS_DIR = config.RESULTS_DIR / "experiments" / "v2_00_protocol"

OUTPUT_ALLOWLIST = (
    "experiments/v2_00_protocol/build_manifests.py",
    "experiments/v2_00_protocol/verify_protocol.py",
    "data/v2/dev.csv",
    "data/v2/train_40k.csv",
    "data/v2/train_100k.csv",
    "data/v2/train_250k.csv",
    "data/v2/test_clean_inputs.csv",
    "data/v2/test_clean_targets.csv",
    "data/v2/dev_image_ids.json",
    "data/v2/train_pool_image_ids.json",
    "data/v2/test_clean_image_ids.json",
    "data/v2/dev_raw_question_ids.json",
    "data/v2/test_clean_raw_question_ids.json",
    "data/v2/answer_vocab_v2.json",
    "data/v2/protocol_build_summary.json",
    "data/v2/manifest_hashes.json",
    "results/experiments/v2_00_protocol/build_run_metadata.json",
    "results/experiments/v2_00_protocol/protocol_report.json",
    "results/experiments/v2_00_protocol/preservation_before.json",
    "results/experiments/v2_00_protocol/preservation_after.json",
    "results/experiments/v2_00_protocol/git_before.txt",
    "results/experiments/v2_00_protocol/git_after.txt",
    "docs/experiments/v2_00_protocol.md",
)

SIZE_LIMIT = 100 * 1024 * 1024
REDUCED_PREFIX = "data/gqa/images/"
SCOPE_TEXT = (
    "Inventory scope: every regular file under the project root, walked with "
    "directory symlinks not followed, excluding only the .git directory. Files "
    "whose repository-relative path starts with data/gqa/images/ and files "
    "larger than 100 MiB (104857600 bytes) are recorded by size and mtime_ns "
    "only; every other file is recorded by sha256. The preservation comparison "
    "excludes exactly the files on the output allowlist of experiment "
    "v2_00_protocol."
)

CHECKS = []
FAILURES = []
BLOCKED = False


def check(name, ok, expected, actual):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name} | expected: {expected} | actual: {actual}")
    CHECKS.append({"name": name, "status": status,
                   "expected": str(expected), "actual": str(actual)})
    if not ok:
        FAILURES.append(name)
    return ok


def load_raw_questions(path):
    """Inline raw GQA loader, independent of the build script's loader."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    records = [(str(question_id), str(entry["imageId"]),
                entry["question"], entry["answer"])
               for question_id, entry in data.items()]
    del data
    return records


def permute_and_accumulate(sorted_ids, per_image, seed, minimum):
    """Recompute a complete-image assignment: permute the sorted ids with an
    independent rng stream and accumulate until minimum is first reached."""
    rng = np.random.default_rng(seed)
    chosen = []
    total = 0
    for index in rng.permutation(len(sorted_ids)):
        image_id = sorted_ids[int(index)]
        chosen.append(image_id)
        total += per_image[image_id]
        if total >= minimum:
            break
    return set(chosen), total


def read_manifest(name):
    return pd.read_csv(V2_DIR / name,
                       dtype={"questionId": str, "imageId": str},
                       keep_default_na=False)


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_diff_text(recomputed, stored):
    return (f"|recomputed_only|={len(recomputed - stored)}, "
            f"|stored_only|={len(stored - recomputed)}")


def main() -> None:
    global BLOCKED

    # Artefact existence.
    expected_files = [p for p in OUTPUT_ALLOWLIST
                      if p.startswith("data/v2/")] + \
                     ["results/experiments/v2_00_protocol/build_run_metadata.json"]
    missing_files = [p for p in expected_files
                     if not (PROJECT_ROOT / p).exists()]
    if not check("all build artefacts exist", not missing_files,
                 "0 missing", f"{len(missing_files)} missing {missing_files}"):
        sys.exit(1)

    summary = load_json(V2_DIR / "protocol_build_summary.json")
    vocab_file = load_json(V2_DIR / "answer_vocab_v2.json")
    hashes_file = load_json(V2_DIR / "manifest_hashes.json")

    check("summary seeds match config-derived seeds",
          summary["seeds"] == {"DEV_IMAGE_SEED": DEV_IMAGE_SEED,
                               "TRAIN_QUESTION_SEED": TRAIN_QUESTION_SEED,
                               "TEST_IMAGE_SEED": TEST_IMAGE_SEED},
          f"{DEV_IMAGE_SEED}/{TRAIN_QUESTION_SEED}/{TEST_IMAGE_SEED}",
          str(summary["seeds"]))

    # Reload raw data with the inline loader.
    print("loading raw train_balanced and val_balanced with the inline loader")
    raw_train = load_raw_questions(config.GQA_RAW_DIR / config.GQA_TRAIN_QUESTIONS)
    raw_val = load_raw_questions(config.GQA_RAW_DIR / config.GQA_VAL_QUESTIONS)
    train_by_qid = {q: (img, text, ans) for q, img, text, ans in raw_train}
    val_by_qid = {q: (img, text, ans) for q, img, text, ans in raw_val}

    train_image_counts = Counter(img for _, img, _, _ in raw_train)
    val_image_counts = Counter(img for _, img, _, _ in raw_val)
    train_images_sorted = sorted(train_image_counts)
    val_images_sorted = sorted(val_image_counts)
    raw_overlap = len(set(train_images_sorted) & set(val_images_sorted))

    check("raw train_balanced question count matches summary",
          len(raw_train) == summary["raw"]["n_train_balanced_questions"],
          summary["raw"]["n_train_balanced_questions"], len(raw_train))
    check("raw val_balanced question count matches summary",
          len(raw_val) == summary["raw"]["n_val_balanced_questions"],
          summary["raw"]["n_val_balanced_questions"], len(raw_val))
    check("unique train image count matches summary",
          len(train_images_sorted) == summary["raw"]["n_train_balanced_unique_images"],
          summary["raw"]["n_train_balanced_unique_images"], len(train_images_sorted))
    check("unique val image count matches summary",
          len(val_images_sorted) == summary["raw"]["n_val_balanced_unique_images"],
          summary["raw"]["n_val_balanced_unique_images"], len(val_images_sorted))
    check("raw train/val image overlap matches summary",
          raw_overlap == summary["raw"]["train_val_image_overlap"],
          summary["raw"]["train_val_image_overlap"], raw_overlap)

    # Check 1: dev image set.
    dev_set, dev_raw_count = permute_and_accumulate(
        train_images_sorted, train_image_counts,
        DEV_IMAGE_SEED, DEV_MIN_RAW_QUESTIONS)
    dev_ids_file = load_json(V2_DIR / "dev_image_ids.json")
    check("dev image set matches dev_image_ids.json",
          set(dev_ids_file) == dev_set and dev_ids_file == sorted(dev_set),
          f"sorted list of {len(dev_set)} recomputed ids",
          f"{len(dev_ids_file)} ids, {set_diff_text(dev_set, set(dev_ids_file))}")

    # Check 2: dev and pool partition the train image set.
    pool_set = set(train_images_sorted) - dev_set
    pool_ids_file = load_json(V2_DIR / "train_pool_image_ids.json")
    check("dev and pool partition the unique train images",
          dev_set | pool_set == set(train_images_sorted)
          and not (dev_set & pool_set),
          "union = all train images, intersection = 0",
          f"union {len(dev_set | pool_set)} of {len(train_images_sorted)}, "
          f"intersection {len(dev_set & pool_set)}")
    check("pool image set matches train_pool_image_ids.json",
          set(pool_ids_file) == pool_set and pool_ids_file == sorted(pool_set),
          f"sorted list of {len(pool_set)} recomputed ids",
          f"{len(pool_ids_file)} ids, {set_diff_text(pool_set, set(pool_ids_file))}")

    # Check 3: clean-test image set.
    legacy_v1_validation = pd.read_csv(
        config.VAL_SPLIT_PATH,
        dtype={"questionId": str, "imageId": str}, keep_default_na=False)
    legacy_images = set(legacy_v1_validation["imageId"])
    candidates = [i for i in val_images_sorted if i not in legacy_images]
    test_set, test_raw_count = permute_and_accumulate(
        candidates, val_image_counts, TEST_IMAGE_SEED, TEST_MIN_RAW_QUESTIONS)
    test_ids_file = load_json(V2_DIR / "test_clean_image_ids.json")
    check("clean-test image set matches test_clean_image_ids.json",
          set(test_ids_file) == test_set and test_ids_file == sorted(test_set),
          f"sorted list of {len(test_set)} recomputed ids",
          f"{len(test_ids_file)} ids, {set_diff_text(test_set, set(test_ids_file))}")

    # Check 4: V2 vocabulary, inline Counter.
    pool_answer_counts = Counter(a for _, img, _, a in raw_train
                                 if img in pool_set)
    ordered = sorted(pool_answer_counts.items(),
                     key=lambda kv: (-kv[1], kv[0]))[:config.TOP_K_ANSWERS]
    recomputed_answers = [a for a, _ in ordered]
    recomputed_index = {a: i for i, a in enumerate(recomputed_answers)}
    check("V2 vocabulary answers and indices match answer_vocab_v2.json",
          vocab_file["answers"] == recomputed_answers
          and vocab_file["answer_to_index"] == recomputed_index
          and vocab_file["top_k"] == len(recomputed_answers),
          f"top {config.TOP_K_ANSWERS} pool answers, frequency then alphabetical",
          "exact match" if vocab_file["answers"] == recomputed_answers
          else "mismatch")

    v1_vocab = load_json(config.ANSWER_VOCAB_PATH)
    v1_answers = set(v1_vocab["answers"])
    v2_answers = set(recomputed_answers)
    check("V1/V2 vocabulary symmetric difference matches summary",
          summary["vocabulary"]["v1_only"] == sorted(v1_answers - v2_answers)
          and summary["vocabulary"]["v2_only"] == sorted(v2_answers - v1_answers)
          and summary["vocabulary"]["n_common_with_v1"] == len(v1_answers & v2_answers),
          f"v1_only={len(v1_answers - v2_answers)}, "
          f"v2_only={len(v2_answers - v1_answers)}, "
          f"common={len(v1_answers & v2_answers)}",
          f"summary lists v1_only={len(summary['vocabulary']['v1_only'])}, "
          f"v2_only={len(summary['vocabulary']['v2_only'])}, "
          f"common={summary['vocabulary']['n_common_with_v1']}")

    # Check 5: recomputed counts and coverages against the summary.
    dev_raw_qids = sorted(q for q, img, _, _ in raw_train if img in dev_set)
    dev_invocab_qids = sorted(q for q, img, _, a in raw_train
                              if img in dev_set and a in recomputed_index)
    pool_raw_count = sum(1 for _, img, _, _ in raw_train if img in pool_set)
    eligible_qids_sorted = sorted(q for q, img, _, a in raw_train
                                  if img in pool_set and a in recomputed_index)
    check("dev raw question count matches summary",
          dev_raw_count == summary["dev"]["n_raw_questions"]
          and dev_raw_count == len(dev_raw_qids),
          summary["dev"]["n_raw_questions"], dev_raw_count)
    check("dev image count matches summary",
          len(dev_set) == summary["dev"]["n_images"],
          summary["dev"]["n_images"], len(dev_set))
    check("dev in-vocab question count matches summary",
          len(dev_invocab_qids) == summary["dev"]["n_invocab_questions"],
          summary["dev"]["n_invocab_questions"], len(dev_invocab_qids))
    check("dev v2 coverage matches summary",
          round(len(dev_invocab_qids) / dev_raw_count, 6) == summary["dev"]["v2_coverage"],
          summary["dev"]["v2_coverage"],
          round(len(dev_invocab_qids) / dev_raw_count, 6))
    check("pool image count matches summary",
          len(pool_set) == summary["pool"]["n_images"],
          summary["pool"]["n_images"], len(pool_set))
    check("pool raw question count matches summary",
          pool_raw_count == summary["pool"]["n_raw_questions"],
          summary["pool"]["n_raw_questions"], pool_raw_count)
    check("eligible question count matches summary",
          len(eligible_qids_sorted) == summary["pool"]["n_eligible_questions"],
          summary["pool"]["n_eligible_questions"], len(eligible_qids_sorted))
    check("margin over 250k matches summary",
          len(eligible_qids_sorted) - TRAIN_PREFIX_SIZES[-1]
          == summary["pool"]["margin_over_250k"],
          summary["pool"]["margin_over_250k"],
          len(eligible_qids_sorted) - TRAIN_PREFIX_SIZES[-1])
    check("pool v2 coverage matches summary",
          round(len(eligible_qids_sorted) / pool_raw_count, 6)
          == summary["pool"]["v2_coverage"],
          summary["pool"]["v2_coverage"],
          round(len(eligible_qids_sorted) / pool_raw_count, 6))

    test_raw_qids = sorted(q for q, img, _, _ in raw_val if img in test_set)
    dev_qids_file = load_json(V2_DIR / "dev_raw_question_ids.json")
    test_qids_file = load_json(V2_DIR / "test_clean_raw_question_ids.json")
    check("dev_raw_question_ids.json matches recomputation",
          dev_qids_file == dev_raw_qids,
          f"sorted list of {len(dev_raw_qids)} ids",
          f"{len(dev_qids_file)} ids, "
          f"{set_diff_text(set(dev_raw_qids), set(dev_qids_file))}")
    check("test_clean_raw_question_ids.json matches recomputation",
          test_qids_file == test_raw_qids,
          f"sorted list of {len(test_raw_qids)} ids",
          f"{len(test_qids_file)} ids, "
          f"{set_diff_text(set(test_raw_qids), set(test_qids_file))}")

    # Check 6: manifest membership, row integrity and labels.
    dev = read_manifest("dev.csv")
    t40 = read_manifest("train_40k.csv")
    t100 = read_manifest("train_100k.csv")
    t250 = read_manifest("train_250k.csv")
    inputs = read_manifest("test_clean_inputs.csv")
    targets = read_manifest("test_clean_targets.csv")
    trains = {"train_40k": t40, "train_100k": t100, "train_250k": t250}

    check("dev.csv columns", list(dev.columns) ==
          ["questionId", "imageId", "question", "answer", "label"],
          "questionId,imageId,question,answer,label", ",".join(dev.columns))
    for name, frame in trains.items():
        check(f"{name}.csv columns", list(frame.columns) ==
              ["questionId", "imageId", "question", "answer", "label"],
              "questionId,imageId,question,answer,label", ",".join(frame.columns))
    check("test_clean_inputs.csv columns", list(inputs.columns) ==
          ["questionId", "imageId", "question"],
          "questionId,imageId,question", ",".join(inputs.columns))
    check("test_clean_targets.csv columns", list(targets.columns) ==
          ["questionId", "answer", "label"],
          "questionId,answer,label", ",".join(targets.columns))

    dev_qids = dev["questionId"].tolist()
    check("dev.csv sorted by questionId and complete for in-vocab dev questions",
          dev_qids == dev_invocab_qids,
          f"{len(dev_invocab_qids)} in-vocab dev questionIds in sorted order",
          f"{len(dev_qids)} rows, equal: {dev_qids == dev_invocab_qids}")
    check("dev.csv rows on dev images",
          set(dev["imageId"]) <= dev_set,
          "subset of dev image set",
          f"{len(set(dev['imageId']) - dev_set)} images outside")
    dev_row_ok = all(train_by_qid[q] == (img, text, ans)
                     for q, img, text, ans in
                     zip(dev_qids, dev["imageId"], dev["question"], dev["answer"]))
    check("dev.csv rows match raw imageId/question/answer",
          dev_row_ok, "all rows equal raw records", f"all equal: {dev_row_ok}")
    dev_label_ok = all(recomputed_index[a] == l
                       for a, l in zip(dev["answer"], dev["label"]))
    check("dev.csv labels are the V2 indices and in-vocab",
          dev_label_ok, "label == vocab index for every row",
          f"all correct: {dev_label_ok}")

    # Recompute the train permutation and its prefixes.
    train_order = np.random.default_rng(TRAIN_QUESTION_SEED).permutation(
        len(eligible_qids_sorted))
    expected_250k_qids = [eligible_qids_sorted[int(i)]
                          for i in train_order[:TRAIN_PREFIX_SIZES[-1]]]
    check("train_250k questionId sequence matches recomputed permuted prefix",
          t250["questionId"].tolist() == expected_250k_qids,
          "first 250000 of the seeded permutation over sorted eligible ids",
          f"equal: {t250['questionId'].tolist() == expected_250k_qids}")

    for name, frame in trains.items():
        expected_n = int(name.split("_")[1].replace("k", "")) * 1000
        check(f"{name} row count", len(frame) == expected_n,
              expected_n, len(frame))
        check(f"{name} rows on pool images",
              set(frame["imageId"]) <= pool_set,
              "subset of pool image set",
              f"{len(set(frame['imageId']) - pool_set)} images outside")
        label_ok = all(recomputed_index.get(a) == l
                       for a, l in zip(frame["answer"], frame["label"]))
        check(f"{name} labels are the V2 indices and in-vocab",
              label_ok, "label == vocab index for every row",
              f"all correct: {label_ok}")
    t250_row_ok = all(train_by_qid[q] == (img, text, ans)
                      for q, img, text, ans in
                      zip(t250["questionId"], t250["imageId"],
                          t250["question"], t250["answer"]))
    check("train_250k rows match raw imageId/question/answer "
          "(covers the nested prefixes)",
          t250_row_ok, "all rows equal raw records", f"all equal: {t250_row_ok}")

    inputs_qids = inputs["questionId"].tolist()
    check("test_clean_inputs sorted by questionId and complete for raw "
          "clean-test questions",
          inputs_qids == test_raw_qids,
          f"{len(test_raw_qids)} raw clean-test questionIds in sorted order",
          f"{len(inputs_qids)} rows, equal: {inputs_qids == test_raw_qids}")
    check("test_clean_inputs image set equals clean-test image set",
          set(inputs["imageId"]) == test_set,
          "set equality", set_diff_text(test_set, set(inputs["imageId"])))
    inputs_row_ok = all(val_by_qid[q][:2] == (img, text)
                        for q, img, text in
                        zip(inputs_qids, inputs["imageId"], inputs["question"]))
    check("test_clean_inputs rows match raw imageId/question",
          inputs_row_ok, "all rows equal raw records",
          f"all equal: {inputs_row_ok}")
    check("inputs and targets carry identical questionId sequences",
          targets["questionId"].tolist() == inputs_qids,
          "identical order and membership",
          f"equal: {targets['questionId'].tolist() == inputs_qids}")
    targets_answer_ok = all(val_by_qid[q][2] == a
                            for q, a in zip(targets["questionId"],
                                            targets["answer"]))
    check("test_clean_targets answers match raw answers",
          targets_answer_ok, "all rows equal raw records",
          f"all equal: {targets_answer_ok}")
    targets_label_ok = all(recomputed_index.get(a, -1) == l
                           for a, l in zip(targets["answer"], targets["label"]))
    check("test_clean_targets labels consistent per row (V2 index or -1)",
          targets_label_ok, "row-level consistency for every row",
          f"all consistent: {targets_label_ok}")

    # Manifest counts against the summary.
    for name, frame in [("dev", dev), ("train_40k", t40),
                        ("train_100k", t100), ("train_250k", t250)]:
        check(f"summary manifest counts for {name}",
              summary["manifests"][name]["n_questions"] == len(frame)
              and summary["manifests"][name]["n_unique_images"]
              == int(frame["imageId"].nunique()),
              f"{len(frame)} questions / {int(frame['imageId'].nunique())} images",
              str(summary["manifests"][name]))
    check("summary manifest counts for test_clean (structural only)",
          summary["manifests"]["test_clean"]["n_questions"] == len(inputs)
          and summary["manifests"]["test_clean"]["n_unique_images"]
          == int(inputs["imageId"].nunique())
          and set(summary["manifests"]["test_clean"])
          == {"n_questions", "n_unique_images"},
          f"{len(inputs)} questions / {int(inputs['imageId'].nunique())} images, "
          "no other keys",
          str(summary["manifests"]["test_clean"]))
    check("summary answer_distributions cover dev and train manifests only",
          set(summary["answer_distributions"])
          == {"dev", "train_40k", "train_100k", "train_250k"},
          "dev, train_40k, train_100k, train_250k",
          ",".join(sorted(summary["answer_distributions"])))
    for name, frame in [("dev", dev), ("train_40k", t40),
                        ("train_100k", t100), ("train_250k", t250)]:
        counts = Counter(frame["answer"])
        total = len(frame)
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
        expected_dist = {
            "top20_by_frequency": [[a, int(c), round(c / total, 6)]
                                   for a, c in top],
            "yes_no_share": round(
                (counts.get("yes", 0) + counts.get("no", 0)) / total, 6),
        }
        check(f"summary answer distribution for {name} matches recomputation",
              summary["answer_distributions"][name] == expected_dist,
              "top-20 with shares and yes/no share recomputed from the file",
              "exact match" if summary["answer_distributions"][name]
              == expected_dist else "mismatch")

    # Check 7: overlaps and nesting.
    dev_qid_set = set(dev_qids)
    dev_img_set = set(dev["imageId"])
    test_qid_set = set(inputs_qids)
    test_img_set = set(inputs["imageId"])
    for name, frame in trains.items():
        frame_qids = set(frame["questionId"])
        frame_imgs = set(frame["imageId"])
        check(f"{name} vs dev questionId overlap",
              not (frame_qids & dev_qid_set), 0, len(frame_qids & dev_qid_set))
        check(f"{name} vs dev imageId overlap",
              not (frame_imgs & dev_img_set), 0, len(frame_imgs & dev_img_set))
        check(f"{name} vs test_clean questionId overlap",
              not (frame_qids & test_qid_set), 0,
              len(frame_qids & test_qid_set))
        check(f"{name} vs test_clean imageId overlap",
              not (frame_imgs & test_img_set), 0,
              len(frame_imgs & test_img_set))
    check("dev vs test_clean questionId overlap",
          not (dev_qid_set & test_qid_set), 0, len(dev_qid_set & test_qid_set))
    check("dev vs test_clean imageId overlap",
          not (dev_img_set & test_img_set), 0, len(dev_img_set & test_img_set))
    check("legacy_v1_validation imageIds vs clean-test imageIds overlap",
          not (legacy_images & test_img_set), 0,
          len(legacy_images & test_img_set))

    check("nested intersection train_40k within train_100k",
          len(set(t40["questionId"]) & set(t100["questionId"])) == 40000,
          40000, len(set(t40["questionId"]) & set(t100["questionId"])))
    check("nested intersection train_100k within train_250k",
          len(set(t100["questionId"]) & set(t250["questionId"])) == 100000,
          100000, len(set(t100["questionId"]) & set(t250["questionId"])))
    check("nested intersection train_40k within train_250k",
          len(set(t40["questionId"]) & set(t250["questionId"])) == 40000,
          40000, len(set(t40["questionId"]) & set(t250["questionId"])))
    check("row-for-row prefix equality: train_100k[:40000] == train_40k",
          t100.head(40000).reset_index(drop=True).equals(
              t40.reset_index(drop=True)),
          "byte-equal rows in prefix order", "equal" if
          t100.head(40000).reset_index(drop=True).equals(
              t40.reset_index(drop=True)) else "differs")
    check("row-for-row prefix equality: train_250k[:100000] == train_100k",
          t250.head(100000).reset_index(drop=True).equals(
              t100.reset_index(drop=True)),
          "byte-equal rows in prefix order", "equal" if
          t250.head(100000).reset_index(drop=True).equals(
              t100.reset_index(drop=True)) else "differs")

    # Check 8: duplicates. questionId duplicates are hard failures;
    # (imageId, question[, answer]) duplicates are diagnostic only.
    manifest_frames = {"dev": dev, "train_40k": t40, "train_100k": t100,
                       "train_250k": t250, "test_clean_inputs": inputs,
                       "test_clean_targets": targets}
    for name, frame in manifest_frames.items():
        n_dup = len(frame) - frame["questionId"].nunique()
        check(f"no duplicate questionId within {name}",
              n_dup == 0, 0, n_dup)

    test_joined = inputs.merge(targets, on="questionId")
    diagnostics = {"within": {}, "across": {}}
    triple_frames = {"dev": dev, "train_250k": t250,
                     "test_clean": test_joined}
    for name, frame in triple_frames.items():
        pairs = len(frame) - len(frame.drop_duplicates(["imageId", "question"]))
        triples = len(frame) - len(
            frame.drop_duplicates(["imageId", "question", "answer"]))
        diagnostics["within"][name] = {
            "image_question_duplicate_rows": int(pairs),
            "image_question_answer_duplicate_rows": int(triples),
        }
        print(f"[DIAG] duplicates within {name}: "
              f"(imageId,question) extra rows {pairs}, "
              f"(imageId,question,answer) extra rows {triples}")
    pair_sets = {name: set(zip(frame["imageId"], frame["question"]))
                 for name, frame in triple_frames.items()}
    triple_sets = {name: set(zip(frame["imageId"], frame["question"],
                                 frame["answer"]))
                   for name, frame in triple_frames.items()}
    for a, b in [("dev", "train_250k"), ("dev", "test_clean"),
                 ("train_250k", "test_clean")]:
        shared_pairs = len(pair_sets[a] & pair_sets[b])
        shared_triples = len(triple_sets[a] & triple_sets[b])
        diagnostics["across"][f"{a}__{b}"] = {
            "shared_image_question_pairs": int(shared_pairs),
            "shared_image_question_answer_triples": int(shared_triples),
        }
        print(f"[DIAG] duplicates across {a} and {b}: "
              f"shared (imageId,question) pairs {shared_pairs}, "
              f"shared triples {shared_triples}")

    # Check 9: image availability via the canonical path rule.
    availability = {}
    for name, frame in [("dev", dev), ("train_40k", t40),
                        ("train_100k", t100), ("train_250k", t250),
                        ("test_clean", inputs)]:
        unique_images = sorted(set(frame["imageId"]))
        missing = [i for i in unique_images
                   if not (config.GQA_IMAGES_DIR / f"{i}.jpg").exists()]
        availability[name] = {"n_unique_images": len(unique_images),
                              "n_found": len(unique_images) - len(missing),
                              "n_missing": len(missing)}
        ok = check(f"image files available for {name}",
                   not missing, f"0 missing of {len(unique_images)}",
                   f"{len(missing)} missing of {len(unique_images)}")
        if not ok:
            BLOCKED = True
            print(f"        missing examples (up to 20): {missing[:20]}")

    # Check 10: re-hash every file in all three groups.
    for group in ("generated", "raw_sources", "legacy_v1"):
        stored = hashes_file[group]
        recomputed = {p: sha256_file(PROJECT_ROOT / p) for p in stored}
        mismatched = sorted(p for p in stored if stored[p] != recomputed[p])
        check(f"hash group {group} matches on disk ({len(stored)} files)",
              not mismatched, "0 mismatches",
              f"{len(mismatched)} mismatches {mismatched[:5]}")
    expected_generated = {f"data/v2/{n}" for n in (
        "dev.csv", "train_40k.csv", "train_100k.csv", "train_250k.csv",
        "test_clean_inputs.csv", "test_clean_targets.csv",
        "dev_image_ids.json", "train_pool_image_ids.json",
        "test_clean_image_ids.json", "dev_raw_question_ids.json",
        "test_clean_raw_question_ids.json", "answer_vocab_v2.json",
        "protocol_build_summary.json")}
    check("hash group path sets are exactly as specified",
          set(hashes_file["generated"]) == expected_generated
          and set(hashes_file["raw_sources"])
          == {"data/gqa/raw/train_balanced_questions.json",
              "data/gqa/raw/val_balanced_questions.json"}
          and set(hashes_file["legacy_v1"])
          == {"data/train.csv", "data/val.csv", "data/answer_vocab.json"},
          "13 generated, 2 raw sources, 3 legacy files",
          f"{len(hashes_file['generated'])}/{len(hashes_file['raw_sources'])}/"
          f"{len(hashes_file['legacy_v1'])}")

    # Check 11: preservation against the exact output allowlist.
    print("building preservation_after inventory (full repository walk)")
    inventory = {}
    for dirpath, dirnames, filenames in os.walk(PROJECT_ROOT, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d != ".git")
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            st = os.stat(path)
            if rel.startswith(REDUCED_PREFIX) or st.st_size > SIZE_LIMIT:
                inventory[rel] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
            else:
                digest = hashlib.sha256()
                with open(path, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                inventory[rel] = {"sha256": digest.hexdigest()}

    before = load_json(OUT_RESULTS_DIR / "preservation_before.json")["files"]
    allow = set(OUTPUT_ALLOWLIST)
    changed = sorted(p for p in before
                     if p not in allow and p in inventory
                     and inventory[p] != before[p])
    deleted = sorted(p for p in before if p not in allow and p not in inventory)
    new_bad = sorted(p for p in inventory if p not in allow and p not in before)
    check("preservation: no non-allowlisted file changed",
          not changed, 0, f"{len(changed)} {changed[:20]}")
    check("preservation: no non-allowlisted file deleted",
          not deleted, 0, f"{len(deleted)} {deleted[:20]}")
    check("preservation: no new file outside the output allowlist",
          not new_bad, 0, f"{len(new_bad)} {new_bad[:20]}")

    with open(OUT_RESULTS_DIR / "preservation_after.json", "w",
              encoding="utf-8") as handle:
        json.dump({"scope": SCOPE_TEXT, "files": inventory}, handle,
                  indent=2, sort_keys=True)
        handle.write("\n")

    git_status = subprocess.run(["git", "status", "--porcelain"],
                                cwd=PROJECT_ROOT, capture_output=True,
                                text=True, check=True).stdout
    git_diff = subprocess.run(["git", "diff"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True,
                              check=True).stdout
    with open(OUT_RESULTS_DIR / "git_after.txt", "w",
              encoding="utf-8") as handle:
        handle.write("# git status --porcelain\n")
        handle.write(git_status)
        handle.write("# git diff\n")
        handle.write(git_diff)

    # Report.
    status = "BLOCKED" if BLOCKED else ("FAIL" if FAILURES else "PASS")
    report = {
        "status": status,
        "n_checks": len(CHECKS),
        "n_failed": len(FAILURES),
        "failed_checks": FAILURES,
        "checks": CHECKS,
        "duplicate_diagnostics": diagnostics,
        "image_availability": availability,
        "preservation": {
            "scope": SCOPE_TEXT,
            "note": ("The preservation claim covers exactly the inventoried "
                     "scope described above; the comparison excludes exactly "
                     "the output allowlist of this experiment."),
            "n_files_before": len(before),
            "n_files_after": len(inventory),
            "changed_non_allowlisted": changed[:200],
            "deleted_non_allowlisted": deleted[:200],
            "new_non_allowlisted": new_bad[:200],
        },
        "embargo": ("Clean-test reporting in this file is structural only: "
                    "question count, unique image count, overlaps, duplicates, "
                    "hashes and image availability. No clean-test vocabulary "
                    "coverage, OOV count, answer distribution, yes/no share or "
                    "question-type statistic is stated."),
    }
    with open(OUT_RESULTS_DIR / "protocol_report.json", "w",
              encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"verification status: {status} "
          f"({len(CHECKS)} checks, {len(FAILURES)} failed)")
    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
