"""v2_07 part 1: materialise the 100k and 250k aligned embedding views.

Pure I/O, no encoding: rows are gathered from the canonical keyed stores
(data/v2/embeddings/images.h5 and questions.h5) in the exact row order of
train_100k.csv and train_250k.csv, with labels from the manifests, and
written in the same V1 format as train_40k.h5 (image (N, 512), question
(N, 512), label (N,)) so src/data.py reads them unchanged.

Verification: row counts, unit norms, no NaN/Inf, labels equal to the
manifests, and byte-identity of the first 40,000 rows of each view with
train_40k.h5 (the manifest nesting made visible in the arrays); as an extra
check, the first 100,000 rows of the 250k view must equal the 100k view.

Blinding: no test_clean_* file is read.
"""

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
OUT_RESULTS_DIR = config.RESULTS_DIR / "experiments" / "v2_07_scaling"
SCALES = {"train_100k": 100000, "train_250k": 250000}


def load_keyed_store(name: str):
    with h5py.File(EMB_DIR / name, "r") as store:
        ids = [i.decode("utf-8") for i in store["ids"][:]]
        embeddings = store["embeddings"][:]
    return {key: index for index, key in enumerate(ids)}, embeddings


def check(name, ok, detail) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def main() -> None:
    utils.set_seed()
    image_row, image_matrix = load_keyed_store("images.h5")
    qid_row, question_matrix = load_keyed_store("questions.h5")

    with h5py.File(EMB_DIR / "train_40k.h5", "r") as store:
        ref_image = store["image"][:]
        ref_question = store["question"][:]
        ref_label = store["label"][:]

    ok = True
    counts = {}
    views = {}
    for name, expected in SCALES.items():
        frame = pd.read_csv(V2_DIR / f"{name}.csv",
                            dtype={"questionId": str, "imageId": str},
                            keep_default_na=False)
        image = image_matrix[np.array([image_row[i] for i in frame["imageId"]])]
        question = question_matrix[np.array([qid_row[q]
                                             for q in frame["questionId"]])]
        label = frame["label"].to_numpy(dtype="int64")
        with h5py.File(EMB_DIR / f"{name}.h5", "w") as store:
            store.create_dataset("image", data=image)
            store.create_dataset("question", data=question)
            store.create_dataset("label", data=label)
            store.attrs["clip_model"] = config.CLIP_MODEL_NAME
            store.attrs["clip_pretrained"] = config.CLIP_PRETRAINED
            store.attrs["embed_dim"] = config.EMBED_DIM
            store.attrs["normalized"] = True
        views[name] = (image, question, label)
        counts[name] = int(image.shape[0])

        ok &= check(f"{name} row count", image.shape[0] == expected,
                    f"{image.shape[0]} (expected {expected})")
        norms_i = np.linalg.norm(image, axis=1)
        norms_q = np.linalg.norm(question, axis=1)
        ok &= check(f"{name} unit norms and finite",
                    bool(np.isfinite(image).all() and np.isfinite(question).all()
                         and np.abs(norms_i - 1).max() < 1e-3
                         and np.abs(norms_q - 1).max() < 1e-3),
                    f"image norms [{norms_i.min():.6f}, {norms_i.max():.6f}], "
                    f"question norms [{norms_q.min():.6f}, {norms_q.max():.6f}]")
        ok &= check(f"{name} labels equal manifest",
                    bool(np.array_equal(label,
                                        frame["label"].to_numpy("int64"))),
                    "exact")
        prefix_ok = (np.array_equal(image[:40000], ref_image)
                     and np.array_equal(question[:40000], ref_question)
                     and np.array_equal(label[:40000], ref_label))
        ok &= check(f"{name} first 40,000 rows byte-identical to train_40k.h5",
                    prefix_ok, "exact" if prefix_ok else "differs")

    i100, q100, l100 = views["train_100k"]
    i250, q250, l250 = views["train_250k"]
    nested = (np.array_equal(i250[:100000], i100)
              and np.array_equal(q250[:100000], q100)
              and np.array_equal(l250[:100000], l100))
    ok &= check("train_250k first 100,000 rows equal train_100k view",
                nested, "exact" if nested else "differs")

    metadata = utils.run_metadata()
    metadata["v2_07_materialise"] = {"rows": counts,
                                     "verification_passed": bool(ok)}
    utils.save_json(metadata, OUT_RESULTS_DIR / "materialise.json")
    if not ok:
        sys.exit("materialisation verification FAILED")
    print("Materialisation complete and verified.")


if __name__ == "__main__":
    main()
