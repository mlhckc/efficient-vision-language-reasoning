"""Schema checks for the keyed global stores, aligned views and token
stores: shapes, dtypes, ID uniqueness and packed-span bounds."""

from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"


def run() -> None:
    with h5py.File(DATA / "v2/embeddings/images.h5", "r") as store:
        ids = store["ids"][:]
        assert ids.shape == (63_599,), ids.shape
        assert len(np.unique(ids)) == 63_599, "duplicate image ids"
        assert store["embeddings"].shape == (63_599, 512)
        assert store["embeddings"].dtype == np.float32

    with h5py.File(DATA / "v2/embeddings/questions.h5", "r") as store:
        assert store["ids"].shape == (265_727,)
        assert store["embeddings"].shape == (265_727, 512)
        assert store["embeddings"].dtype == np.float32

    with h5py.File(DATA / "v2/embeddings/dev.h5", "r") as store:
        assert store["image"].shape == (7_714, 512)
        assert store["question"].shape == (7_714, 512)
        assert store["image"].dtype == np.float32
        assert store["label"].shape == (7_714,)
        assert store["label"].dtype == np.int64
        labels = store["label"][:]
        assert labels.min() >= 0 and labels.max() <= 99, "label out of range"

    with h5py.File(DATA / "v3/tokens/image_tokens.h5", "r") as store:
        assert store["ids"].shape == (63_599,)
        assert store["tokens"].shape == (63_599, 50, 512)
        assert store["tokens"].dtype == np.float16

    with h5py.File(DATA / "v3/tokens/question_tokens.h5", "r") as store:
        offsets = store["offsets"][:].astype(np.int64)
        lengths = store["lengths"][:].astype(np.int64)
        assert store["ids"].shape == (265_727,)
        assert offsets.shape == (265_727,) and lengths.shape == (265_727,)
        assert store["tokens"].shape == (2_435_691, 512)
        assert store["tokens"].dtype == np.float16
        assert lengths.min() >= 2, "a question span lacks SOT/EOT"
        assert offsets.min() >= 0
        assert int((offsets + lengths).max()) <= 2_435_691, "span out of bounds"
