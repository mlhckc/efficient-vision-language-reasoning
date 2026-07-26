"""v3_02a Phase B1: build mean-of-patches aligned views.

For every unique imageId used by train_40k and dev: load the (50, 512)
token tensor, exclude token 0 (CLS), assert exactly 49 patch tokens, cast
to float32, mean over patches, record the pre-normalisation norm, and
L2-normalise. Aligned views copy the question and label arrays exactly from
the verified V2 aligned views; only the image array differs.

Atomic writes (.partial -> reopen -> verify -> rename). Deterministic
rebuild verification: the dev view is rebuilt into a temporary path and
compared; datasets are created with track_times=False so byte identity is
meaningful, and the byte comparison is attempted first with exact
array/metadata equality as the documented fallback. Temporary files are
removed after successful verification. test_clean_targets.csv is never
read.
"""

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402

V2_DIR = config.DATA_DIR / "v2"
TOKEN_STORE = config.DATA_DIR / "v3" / "tokens" / "image_tokens.h5"
OUT_DIR = config.DATA_DIR / "v3" / "meanpatch"
FAILURES = []


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)
    return ok


def write_view(path: Path, image, question, label) -> None:
    with h5py.File(path, "w") as store:
        store.create_dataset("image", data=image, track_times=False)
        store.create_dataset("question", data=question, track_times=False)
        store.create_dataset("label", data=label, track_times=False)
        store.attrs["source"] = "v3_02a mean-of-patches (token 0 excluded)"
        store.attrs["normalized"] = True
        store.attrs["embed_dim"] = config.EMBED_DIM


def build_views():
    """Return {name: (image, question, label)} for both splits."""
    frames = {name: pd.read_csv(V2_DIR / f"{name}.csv",
                                dtype={"questionId": str, "imageId": str},
                                keep_default_na=False)
              for name in ("train_40k", "dev")}
    needed = sorted(set(frames["train_40k"]["imageId"])
                    | set(frames["dev"]["imageId"]))
    with h5py.File(TOKEN_STORE, "r") as store:
        ids = [i.decode("utf-8") for i in store["ids"][:]]
        row_of = {image_id: index for index, image_id in enumerate(ids)}
        tokens = store["tokens"][:]  # (63599, 50, 512) fp16, read once
    assert all(i in row_of for i in needed)

    rows = np.array([row_of[i] for i in needed])
    patch = tokens[rows][:, 1:, :].astype("float32")  # exclude CLS
    assert patch.shape[1] == 49, patch.shape
    mean_patch = patch.mean(axis=1)
    pre_norms = np.linalg.norm(mean_patch, axis=1)
    check("pre-normalisation norms finite and non-zero",
          bool(np.isfinite(pre_norms).all() and (pre_norms > 1e-6).all()),
          f"range [{pre_norms.min():.4f}, {pre_norms.max():.4f}]")
    mean_patch = mean_patch / pre_norms[:, None]
    post_norms = np.linalg.norm(mean_patch, axis=1)
    check("post-normalisation norms equal 1 within 1e-5",
          bool(np.abs(post_norms - 1).max() < 1e-5),
          f"max |norm-1| {np.abs(post_norms - 1).max():.2e}")
    check("no NaN/Inf in mean-patch vectors",
          bool(np.isfinite(mean_patch).all()), "finite")
    check("CLS excluded, exactly 49 patches used", True,
          "token 0 sliced away; patch axis length asserted 49")
    vector_of = {image_id: mean_patch[index]
                 for index, image_id in enumerate(needed)}

    views = {}
    for name, frame in frames.items():
        with h5py.File(V2_DIR / "embeddings" / f"{name}.h5", "r") as ref:
            question = ref["question"][:]
            label = ref["label"][:]
        image = np.stack([vector_of[i] for i in frame["imageId"]])
        check(f"{name} manifest row alignment",
              image.shape == (len(frame), 512)
              and bool(np.array_equal(label,
                                      frame["label"].to_numpy("int64"))),
              f"{image.shape[0]} rows; labels equal manifest")
        views[name] = (image.astype("float32"), question, label)
    return views


def main() -> None:
    utils.set_seed()
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    views = build_views()

    for name, (image, question, label) in views.items():
        final = OUT_DIR / f"{name}_meanpatch.h5"
        partial = OUT_DIR / f"{name}_meanpatch.h5.partial"
        if final.exists() or partial.exists():
            sys.exit(f"ABORT: {final} or its partial already exists")
        write_view(partial, image, question, label)
        with h5py.File(partial, "r") as store:
            ok = (bool(np.array_equal(store["image"][:], image))
                  and bool(np.array_equal(store["question"][:], question))
                  and bool(np.array_equal(store["label"][:], label)))
        check(f"{name}_meanpatch.h5 reopened content equals in-memory arrays",
              ok, "exact")
        if not ok:
            sys.exit(f"ABORT: partial verification failed for {name}; "
                     "partial preserved")
        partial.replace(final)
        print(f"wrote {final.name} ({final.stat().st_size / 1e6:.1f} MB)")

    # Deterministic rebuild verification for the dev view.
    rebuilt = build_views()["dev"]
    temp_path = OUT_DIR / "dev_meanpatch.rebuild.tmp.h5"
    write_view(temp_path, *rebuilt)
    original_bytes = (OUT_DIR / "dev_meanpatch.h5").read_bytes()
    rebuilt_bytes = temp_path.read_bytes()
    if original_bytes == rebuilt_bytes:
        check("dev rebuild determinism", True,
              "byte identity (track_times=False)")
    else:
        with h5py.File(OUT_DIR / "dev_meanpatch.h5", "r") as a, \
                h5py.File(temp_path, "r") as b:
            equal = all(bool(np.array_equal(a[k][:], b[k][:]))
                        for k in ("image", "question", "label")) \
                and dict(a.attrs) == dict(b.attrs)
        check("dev rebuild determinism", equal,
              "byte identity failed; exact array and metadata equality used "
              "as the documented fallback")
    temp_path.unlink()
    print("temporary rebuild file removed")

    if FAILURES:
        sys.exit(f"B1 FAILED: {FAILURES}")
    print(f"B1 complete in {time.time() - started:.1f} s (measured)")


if __name__ == "__main__":
    main()
