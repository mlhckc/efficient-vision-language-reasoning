"""Stage 1: prepare the GQA subset and the answer vocabulary.

This stage reads the raw GQA question and answer files, keeps a fixed-size
random subset for training and validation (config.N_TRAIN and config.N_VAL),
and builds the answer vocabulary as the config.TOP_K_ANSWERS most frequent
answers. Questions whose answer falls outside that vocabulary are dropped,
because the task is answer classification over a closed set. The resulting
splits and the answer-to-index mapping are written to data/ for later stages.

This is kept separate from embedding extraction so the subset and the
vocabulary are decided once and stay fixed across every experiment.

Order of operations:
  1. download the balanced GQA question files if they are not already present,
  2. build the answer vocabulary from the full training answers,
  3. drop out-of-vocabulary questions and attach an integer label,
  4. sample N_TRAIN and N_VAL examples under the fixed seed,
  5. write answer_vocab.json, train.csv and val.csv, plus a run-metadata record.
"""

import json
import shutil
import zipfile
from collections import Counter
from urllib.request import urlretrieve

import pandas as pd
from tqdm import tqdm

import config
from src import utils


def _download_with_progress(url: str, dest) -> None:
    """Download url to dest, showing a progress bar."""
    with tqdm(unit="B", unit_scale=True, miniters=1, desc=dest.name) as bar:
        def hook(block_count, block_size, total_size):
            if total_size > 0:
                bar.total = total_size
            bar.update(block_count * block_size - bar.n)

        urlretrieve(url, dest, reporthook=hook)


def _find_member(names, filename):
    """Locate a file inside the zip by name, tolerating a directory prefix."""
    for name in names:
        if name == filename or name.endswith("/" + filename):
            return name
    raise FileNotFoundError(f"{filename} not found in archive")


def ensure_questions() -> None:
    """Download and extract the balanced GQA question files if missing."""
    raw_dir = config.GQA_RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    train_path = raw_dir / config.GQA_TRAIN_QUESTIONS
    val_path = raw_dir / config.GQA_VAL_QUESTIONS
    if train_path.exists() and val_path.exists():
        print(f"Found GQA question files in {raw_dir}")
        return

    zip_path = raw_dir / "questions1.2.zip"
    if not zip_path.exists():
        print(f"Downloading GQA questions (about 1.4 GB) from "
              f"{config.GQA_QUESTIONS_URL}")
        _download_with_progress(config.GQA_QUESTIONS_URL, zip_path)

    print("Extracting balanced train and val question files")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        for filename, dest in ((config.GQA_TRAIN_QUESTIONS, train_path),
                               (config.GQA_VAL_QUESTIONS, val_path)):
            member = _find_member(names, filename)
            with archive.open(member) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
    print(f"Question files ready in {raw_dir}")


def load_questions(path) -> pd.DataFrame:
    """Load a GQA questions JSON into a DataFrame of the fields we need."""
    with open(path) as handle:
        data = json.load(handle)
    rows = [
        {
            "questionId": qid,
            "imageId": entry["imageId"],
            "question": entry["question"],
            "answer": entry["answer"],
        }
        for qid, entry in data.items()
    ]
    return pd.DataFrame(rows)


def build_answer_vocab(answers, top_k: int) -> dict:
    """Return an answer-to-index map for the top_k most frequent answers.

    Ties are broken alphabetically so the vocabulary is deterministic.
    """
    counts = Counter(answers)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top_answers = [answer for answer, _ in ordered[:top_k]]
    return {answer: index for index, answer in enumerate(top_answers)}


def filter_and_label(frame: pd.DataFrame, vocab: dict) -> pd.DataFrame:
    """Keep only in-vocabulary answers and attach the integer label."""
    kept = frame[frame["answer"].isin(vocab)].copy()
    kept["label"] = kept["answer"].map(vocab).astype(int)
    return kept


def sample(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    """Sample n rows under the fixed seed, or keep all if fewer are available."""
    if len(frame) <= n:
        print(f"  requested {n} but only {len(frame)} available; keeping all")
        return frame.reset_index(drop=True)
    return frame.sample(n=n, random_state=config.RANDOM_SEED).reset_index(drop=True)


def main() -> None:
    utils.set_seed()
    ensure_questions()

    train_full = load_questions(config.GQA_RAW_DIR / config.GQA_TRAIN_QUESTIONS)
    val_full = load_questions(config.GQA_RAW_DIR / config.GQA_VAL_QUESTIONS)
    print(f"Loaded {len(train_full)} train and {len(val_full)} val questions "
          f"(full balanced split)")

    vocab = build_answer_vocab(train_full["answer"], config.TOP_K_ANSWERS)
    train_coverage = train_full["answer"].isin(vocab).mean()
    val_coverage = val_full["answer"].isin(vocab).mean()
    print(f"Answer vocabulary: {len(vocab)} answers covering "
          f"{train_coverage:.1%} of train and {val_coverage:.1%} of val answers")

    train_df = sample(filter_and_label(train_full, vocab), config.N_TRAIN)
    val_df = sample(filter_and_label(val_full, vocab), config.N_VAL)
    print(f"Sampled {len(train_df)} train and {len(val_df)} val examples")

    # Save the vocabulary (answer order = label index) and the splits.
    answers_by_index = sorted(vocab, key=vocab.get)
    utils.save_json(
        {"answer_to_index": vocab,
         "answers": answers_by_index,
         "top_k": len(vocab)},
        config.ANSWER_VOCAB_PATH,
    )
    train_df.to_csv(config.TRAIN_SPLIT_PATH, index=False)
    val_df.to_csv(config.VAL_SPLIT_PATH, index=False)
    print(f"Wrote {config.ANSWER_VOCAB_PATH.name}, "
          f"{config.TRAIN_SPLIT_PATH.name} and {config.VAL_SPLIT_PATH.name} "
          f"to {config.DATA_DIR}")

    # Record what was produced so the numbers are traceable.
    most_common = Counter(train_full["answer"]).most_common(10)
    metadata = utils.run_metadata()
    metadata["stage1"] = {
        "n_train_full": int(len(train_full)),
        "n_val_full": int(len(val_full)),
        "answer_vocab_size": len(vocab),
        "train_answer_coverage": round(float(train_coverage), 4),
        "val_answer_coverage": round(float(val_coverage), 4),
        "n_train_final": int(len(train_df)),
        "n_val_final": int(len(val_df)),
        "most_common_answers": most_common,
    }
    utils.save_json(metadata, config.RESULTS_DIR / "stage1_prepare_gqa.json")
    print("Stage 1 complete.")


if __name__ == "__main__":
    main()
