"""Extract the token features the RAW denominator needs and the pinned
stores do not have.

The v3_00 token stores were built for the 7,714 IN-VOCABULARY
development rows. The raw denominator required by canonical plan section
6 has 10,004 rows, so the 2,290 out-of-vocabulary rows have no cached
question tokens, and nine of their images have no cached image tokens.
Without them the raw-denominator R3 evaluation cannot run at all.

The extension is written to a SEPARATE store. The training path pins
`image_store_sha256` and `question_store_sha256` and re-checks them on
every resume, so appending to the existing files would change those
hashes and invalidate the pinning for every core cell. The pinned stores
are left byte-identical and verified so afterwards.

This is deterministic frozen-encoder feature extraction on the
DEVELOPMENT partition: the same CLIP ViT-B/32 used everywhere else, in
eval mode, under no_grad. It is not training and it is not a scientific
measurement, but it does use the GPU, so its cost is measured and
charged.

THE EMBARGOED CLEAN TEST IS NEVER TOUCHED.

NO OPTIMIZER STEP.

    python -B experiments/e8b_readout_generation/extend_raw_tokens.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils, tokens_data  # noqa: E402
from experiments.e8b_readout_generation import run as e8b_run  # noqa: E402

V2_DIR = PROJECT_ROOT / "data" / "v2"
RAW_DEV = V2_DIR / "dev_raw.csv"
EXTENSION_DIR = config.DATA_DIR / "v3" / "tokens"
QUESTION_EXTENSION = EXTENSION_DIR / "question_tokens_raw_dev_extension.h5"
IMAGE_EXTENSION = EXTENSION_DIR / "image_tokens_raw_dev_extension.h5"
RECORD = e8b_run.OUT_DIR / "raw_token_extension_20260807.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    utils.set_seed()
    e8b_run.enable_strict_determinism()
    e8b_run.pin_fp32_precision()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = pd.read_csv(RAW_DEV, dtype={"questionId": str, "imageId": str},
                      keep_default_na=False, na_filter=False)
    stores = tokens_data.TokenStores()
    pinned_before = {
        "image": sha256_file(EXTENSION_DIR / "image_tokens.h5"),
        "question": sha256_file(EXTENSION_DIR / "question_tokens.h5")}

    missing_questions = raw[~raw["questionId"].isin(stores.question_index)]
    missing_images = sorted({i for i in raw["imageId"]
                             if i not in stores.image_row})
    print(f"missing question tokens: {len(missing_questions)}")
    print(f"missing image tokens   : {len(missing_images)}")
    if len(missing_questions) == 0 and not missing_images:
        print("nothing to extract")
        return 0

    # Reuse the v3_00 extractor's OWN functions and constants, so the
    # extension cannot drift from the store it extends.
    import open_clip
    from experiments.v3_00_tokens import extract_tokens as v3

    started = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED)
    tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if utils.count_parameters(model) != 0:
        raise AssertionError("CLIP must be frozen")
    visual = model.visual
    # The same architectural guards the original extraction asserted, so
    # a different open_clip build cannot silently change the features.
    if not (visual.attn_pool is None and not visual.final_ln_after_pool
            and visual.pool_type == "tok"):
        raise AssertionError("visual path differs from the extraction")
    if tuple(visual.proj.shape) != (768, 512) \
            or tuple(model.text_projection.shape) != (512, 512) \
            or model.text_pool_type != "argmax":
        raise AssertionError("projection heads differ from the extraction")
    load_seconds = time.time() - started

    # --- questions: tokenise, then pack exactly as the store does ---
    started = time.time()
    texts = list(missing_questions["question"])
    ids = list(missing_questions["questionId"])
    token_ids = np.zeros((len(texts), 77), dtype="int32")
    for start in range(0, len(texts), 8192):
        chunk = texts[start:start + 8192]
        token_ids[start:start + len(chunk)] = tokenizer(chunk).numpy()
    eot = token_ids.argmax(axis=1)
    lengths = (eot + 1).astype("int32")
    offsets = np.zeros(len(texts), dtype="int64")
    offsets[1:] = np.cumsum(lengths[:-1])
    packed = np.zeros((int(lengths.sum()), 512), dtype=np.float16)
    with torch.no_grad():
        for start in range(0, len(texts), v3.TEXT_BATCH):
            batch = torch.from_numpy(
                token_ids[start:start + v3.TEXT_BATCH].astype("int64")
            ).to(device)
            states = v3.text_tokens_batch(model, batch).cpu().numpy()
            for row in range(states.shape[0]):
                index = start + row
                length = int(lengths[index])
                packed[offsets[index]:offsets[index] + length] = \
                    states[row, :length].astype(np.float16)
    question_seconds = time.time() - started

    EXTENSION_DIR.mkdir(parents=True, exist_ok=True)
    with h5py.File(QUESTION_EXTENSION, "w") as store:
        store.create_dataset("ids", data=np.array(ids, dtype="S32"))
        store.create_dataset("tokens", data=packed)
        store.create_dataset("offsets", data=offsets)
        store.create_dataset("lengths", data=lengths.astype("int64"))

    # --- images ---
    started = time.time()
    image_stack = []
    with torch.no_grad():
        for chunk_start in range(0, len(missing_images), v3.IMAGE_BATCH):
            chunk = missing_images[chunk_start:chunk_start + v3.IMAGE_BATCH]
            tensors = torch.stack(
                [v3.load_image_tensor(preprocess, i) for i in chunk]
            ).to(device)
            image_stack.append(
                v3.image_tokens_batch(visual, tensors).cpu().numpy()
                .astype(np.float16))
    image_seconds = time.time() - started
    with h5py.File(IMAGE_EXTENSION, "w") as store:
        store.create_dataset("ids",
                             data=np.array(missing_images, dtype="S32"))
        store.create_dataset(
            "tokens",
            data=(np.concatenate(image_stack) if image_stack
                  else np.zeros((0, 50, 512), dtype=np.float16)))

    pinned_after = {
        "image": sha256_file(EXTENSION_DIR / "image_tokens.h5"),
        "question": sha256_file(EXTENSION_DIR / "question_tokens.h5")}
    if pinned_after != pinned_before:
        raise AssertionError(
            "EXTENSION REFUSED: the PINNED token stores changed. They "
            "are hashed by the training path and re-checked on every "
            "resume; an extension must never touch them.")

    total_seconds = load_seconds + question_seconds + image_seconds
    record = {"metadata": utils.run_metadata(),
              "e8b_raw_token_extension": {
        "dated_utc": "2026-08-07",
        "NON_TRAINING": True,
        "optimizer_steps": 0,
        "purpose": "the raw denominator's 2,290 out-of-vocabulary rows "
                   "have no cached question tokens in the v3_00 stores, "
                   "which were built for the in-vocabulary rows only. "
                   "Without these the raw-denominator R3 evaluation "
                   "cannot run.",
        "questions_extracted": int(len(missing_questions)),
        "images_extracted": len(missing_images),
        "extension_stores": {
            "questions": {"path": str(QUESTION_EXTENSION.relative_to(
                PROJECT_ROOT)), "sha256": sha256_file(QUESTION_EXTENSION)},
            "images": {"path": str(IMAGE_EXTENSION.relative_to(
                PROJECT_ROOT)), "sha256": sha256_file(IMAGE_EXTENSION)}},
        "pinned_stores_unchanged": {
            "verified": True,
            "image_tokens_sha256": pinned_after["image"],
            "question_tokens_sha256": pinned_after["question"],
            "why_separate": "the training path pins these hashes and "
                            "re-checks them on every resume, so an "
                            "append would invalidate the pinning for "
                            "every core cell"},
        "gpu_cost": {
            "model_load_seconds": round(load_seconds, 2),
            "question_seconds": round(question_seconds, 2),
            "image_seconds": round(image_seconds, 2),
            "total_seconds": round(total_seconds, 2),
            "total_hours": round(total_seconds / 3600, 5),
            "charged_to": "no frozen-LM identity: this uses the frozen "
                          "CLIP encoders only and loads neither "
                          "SmolLM2 model, so it touches neither 35-hour "
                          "ceiling. It IS charged to the 180-hour core "
                          "ceiling as spent compute."},
        "encoder": "frozen CLIP ViT-B/32, eval mode, no_grad, the same "
                   "instance and preprocessing as the v3_00 stores",
        "clean_test_accessed": False}}
    if RECORD.exists():
        RECORD.unlink()
    RECORD.write_text(json.dumps(record, indent=2, default=str) + "\n")
    body = record["e8b_raw_token_extension"]
    print(f"  questions   : {body['questions_extracted']}")
    print(f"  images      : {body['images_extracted']}")
    print(f"  GPU cost    : {body['gpu_cost']['total_seconds']} s "
          f"({body['gpu_cost']['total_hours']} h)")
    print(f"  pinned safe : {body['pinned_stores_unchanged']['verified']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
