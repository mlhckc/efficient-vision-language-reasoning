"""V2 zero-shot CLIP baseline on the development set only. No training.

For each dev question, the score of each vocabulary answer is the cosine
similarity between the dev image embedding and the answer-text embedding
(both L2-normalised, so the dot product), and the prediction is the argmax.
Accuracy is reported for the three prompt variants stored in answers.h5:
the raw answer string, "a photo of {answer}", and their normalised mean
(prompt ensembling as in the CLIP paper).

This is the CLIP-paper zero-shot classification protocol applied to our
answer vocabulary. It uses the image only, so it is a training-free lower
anchor. A question-conditioned zero-shot variant is not possible with plain
CLIP, because the question and the answers are both text: CLIP can score
text against an image, but it has no mechanism to condition an image-text
score on a second text input.

Blinding: this script reads dev data only. The clean test is never touched.
The majority-class reference predicts the most frequent train_40k label, as
in the V1 baselines, and is evaluated on dev for context.
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402

V2_DIR = config.DATA_DIR / "v2"
EMB_DIR = V2_DIR / "embeddings"
OUT_RESULTS_DIR = config.RESULTS_DIR / "experiments" / "v2_01_embeddings"


def main() -> None:
    utils.set_seed()

    with h5py.File(EMB_DIR / "dev.h5", "r") as store:
        image = store["image"][:]
        label = store["label"][:]
    with h5py.File(EMB_DIR / "answers.h5", "r") as store:
        stored_answers = [a.decode("utf-8") for a in store["answers"][:]]
        variants = {name: store[name][:] for name in ("raw", "photo", "ensembled")}

    vocab = json.loads((V2_DIR / "answer_vocab_v2.json").read_text())
    assert stored_answers == vocab["answers"], "answers.h5 order != vocabulary"

    accuracies = {}
    for name, matrix in variants.items():
        prediction = (image @ matrix.T).argmax(axis=1)
        accuracies[name] = round(float((prediction == label).mean()), 5)

    train_40k = pd.read_csv(V2_DIR / "train_40k.csv",
                            dtype={"questionId": str, "imageId": str},
                            keep_default_na=False)
    majority_label = int(train_40k["label"].value_counts().idxmax())
    majority_answer = vocab["answers"][majority_label]
    majority_accuracy = round(float((label == majority_label).mean()), 5)

    print(f"dev rows: {len(label)}")
    print(f"majority reference (most frequent train_40k label "
          f"'{majority_answer}'): dev accuracy {majority_accuracy:.4f}")
    for name in ("raw", "photo", "ensembled"):
        print(f"zero-shot dev accuracy ({name}): {accuracies[name]:.4f}")

    metadata = utils.run_metadata()
    metadata["v2_01_zero_shot"] = {
        "split": "dev",
        "n_dev": int(len(label)),
        "accuracy": accuracies,
        "majority_reference": {
            "source": "most frequent train_40k label",
            "label": majority_label,
            "answer": majority_answer,
            "dev_accuracy": majority_accuracy,
        },
        "note": ("Image-only zero-shot per the CLIP protocol; a training-free "
                 "lower anchor. A question-conditioned zero-shot variant is "
                 "not possible with plain CLIP because the question and the "
                 "answers are both text. The clean test was never touched."),
    }
    utils.save_json(metadata, OUT_RESULTS_DIR / "zero_shot.json")
    print("Zero-shot baseline complete.")


if __name__ == "__main__":
    main()
