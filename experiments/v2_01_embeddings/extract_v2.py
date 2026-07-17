"""V2 Day 2: extract and cache frozen CLIP embeddings for the V2 manifests.

Reuses the V1 extraction approach (2_extract_embeddings.py): the frozen CLIP
model named in config, eval mode, requires_grad False and verified to have
zero trainable parameters, batched encoding under torch.no_grad(), and
L2-normalised float32 vectors of size config.EMBED_DIM.

Outputs, all under data/v2/embeddings/:

- images.h5, questions.h5: canonical keyed stores. Each unique image across
  the union of train_40k, train_100k, train_250k, dev and test_clean_inputs
  is encoded once; each unique questionId likewise (identical question texts
  are encoded once and shared). Both stores use an id-index scheme: row i of
  "embeddings" belongs to "ids"[i], with ids sorted lexicographically.
- train_40k.h5, dev.h5: materialised row-aligned views in the V1 format
  (image (N, 512), question (N, 512), label (N,)) matching the manifest row
  order, so src/data.py works unchanged. The 100k and 250k views are
  materialised later when the scaling study starts.
- answers.h5: for each of the 100 vocabulary answers, the L2-normalised text
  embedding of the raw answer string, of the prompt "a photo of {answer}",
  and their L2-normalised mean (prompt ensembling as in CLIP), each stored as
  a (100, 512) array in vocabulary order.

Blinding: this script reads data/v2/test_clean_inputs.csv only.
data/v2/test_clean_targets.csv is never opened, and no clean-test label
information exists anywhere in the outputs. No model is trained.
"""

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402

V2_DIR = config.DATA_DIR / "v2"
EMB_DIR = V2_DIR / "embeddings"
OUT_RESULTS_DIR = config.RESULTS_DIR / "experiments" / "v2_01_embeddings"

MANIFESTS = {
    "train_40k": V2_DIR / "train_40k.csv",
    "train_100k": V2_DIR / "train_100k.csv",
    "train_250k": V2_DIR / "train_250k.csv",
    "dev": V2_DIR / "dev.csv",
    "test_clean_inputs": V2_DIR / "test_clean_inputs.csv",
}
VIEW_NAMES = ("train_40k", "dev")
STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def read_manifest(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"questionId": str, "imageId": str},
                       keep_default_na=False)


def load_clip(device):
    """Load the frozen CLIP model; verified to have zero trainable parameters."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED)
    tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert utils.count_parameters(model) == 0, "CLIP must be frozen"
    return model, preprocess, tokenizer


def normalize(features: torch.Tensor) -> np.ndarray:
    features = features / features.norm(dim=-1, keepdim=True)
    return features.float().cpu().numpy()


@torch.no_grad()
def encode_images(model, preprocess, device, image_ids) -> np.ndarray:
    """Encode the given image ids in order; a missing file is a hard error,
    because Day 1 verified every referenced image exists."""
    rows = []
    for start in tqdm(range(0, len(image_ids), config.BATCH_SIZE),
                      desc="images"):
        batch_ids = image_ids[start:start + config.BATCH_SIZE]
        tensors = []
        for image_id in batch_ids:
            path = config.GQA_IMAGES_DIR / f"{image_id}.jpg"
            if not path.exists():
                raise RuntimeError(f"image file missing: {image_id} "
                                   "(Day 1 verified availability; aborting)")
            tensors.append(preprocess(Image.open(path).convert("RGB")))
        rows.append(normalize(model.encode_image(
            torch.stack(tensors).to(device))))
    return np.concatenate(rows, axis=0)


@torch.no_grad()
def encode_texts(model, tokenizer, device, texts) -> np.ndarray:
    """Encode the given texts in order."""
    rows = []
    for start in tqdm(range(0, len(texts), config.BATCH_SIZE), desc="texts"):
        batch = texts[start:start + config.BATCH_SIZE]
        rows.append(normalize(model.encode_text(tokenizer(batch).to(device))))
    return np.concatenate(rows, axis=0)


def write_keyed_store(path, ids, embeddings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as store:
        store.create_dataset("ids", data=np.array(ids, dtype=object),
                             dtype=STRING_DTYPE)
        store.create_dataset("embeddings", data=embeddings)
        store.attrs["scheme"] = "row i of embeddings belongs to ids[i]"
        store.attrs["clip_model"] = config.CLIP_MODEL_NAME
        store.attrs["clip_pretrained"] = config.CLIP_PRETRAINED
        store.attrs["embed_dim"] = config.EMBED_DIM
        store.attrs["normalized"] = True


def write_view(path, image, question, label) -> None:
    """V1-format aligned view so src/data.py reads it unchanged."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as store:
        store.create_dataset("image", data=image)
        store.create_dataset("question", data=question)
        store.create_dataset("label", data=label)
        store.attrs["clip_model"] = config.CLIP_MODEL_NAME
        store.attrs["clip_pretrained"] = config.CLIP_PRETRAINED
        store.attrs["embed_dim"] = config.EMBED_DIM
        store.attrs["normalized"] = True


def check(name, ok, detail) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def main() -> None:
    utils.set_seed()
    device = utils.get_device()

    frames = {name: read_manifest(path) for name, path in MANIFESTS.items()}
    manifest_rows = {name: int(len(frame)) for name, frame in frames.items()}

    unique_images = sorted(set().union(*(set(f["imageId"]) for f in frames.values())))
    question_text = {}
    for frame in frames.values():
        for qid, text in zip(frame["questionId"], frame["question"]):
            if qid in question_text and question_text[qid] != text:
                raise RuntimeError(f"questionId {qid} has inconsistent text")
            question_text[qid] = text
    unique_qids = sorted(question_text)
    unique_texts = sorted(set(question_text.values()))
    print(f"manifests: { {n: manifest_rows[n] for n in MANIFESTS} }")
    print(f"unique images: {len(unique_images)}  unique questionIds: "
          f"{len(unique_qids)}  unique question texts: {len(unique_texts)}")

    print(f"Loading CLIP {config.CLIP_MODEL_NAME} ({config.CLIP_PRETRAINED}) "
          f"on {device}")
    model, preprocess, tokenizer = load_clip(device)

    started = time.time()
    image_matrix = encode_images(model, preprocess, device, unique_images)
    text_matrix = encode_texts(model, tokenizer, device, unique_texts)
    text_row = {text: index for index, text in enumerate(unique_texts)}
    question_matrix = text_matrix[
        np.array([text_row[question_text[q]] for q in unique_qids])]

    # Answer-text embeddings in vocabulary order, two prompts plus ensemble.
    vocab = json.loads((V2_DIR / "answer_vocab_v2.json").read_text())
    answers = vocab["answers"]
    raw = encode_texts(model, tokenizer, device, answers)
    photo = encode_texts(model, tokenizer, device,
                         [f"a photo of {answer}" for answer in answers])
    ensembled = (raw + photo) / 2.0
    ensembled = ensembled / np.linalg.norm(ensembled, axis=1, keepdims=True)
    encode_seconds = round(time.time() - started, 1)

    # Canonical keyed stores.
    write_keyed_store(EMB_DIR / "images.h5", unique_images, image_matrix)
    write_keyed_store(EMB_DIR / "questions.h5", unique_qids, question_matrix)
    with h5py.File(EMB_DIR / "answers.h5", "w") as store:
        store.create_dataset("answers", data=np.array(answers, dtype=object),
                             dtype=STRING_DTYPE)
        store.create_dataset("raw", data=raw)
        store.create_dataset("photo", data=photo)
        store.create_dataset("ensembled", data=ensembled)
        store.attrs["scheme"] = ("row i of each array is the answer with "
                                 "vocabulary index i")
        store.attrs["clip_model"] = config.CLIP_MODEL_NAME
        store.attrs["clip_pretrained"] = config.CLIP_PRETRAINED
        store.attrs["prompt_raw"] = "{answer}"
        store.attrs["prompt_photo"] = "a photo of {answer}"

    # Materialised aligned views for train_40k and dev only.
    image_row = {image_id: index for index, image_id in enumerate(unique_images)}
    qid_row = {qid: index for index, qid in enumerate(unique_qids)}
    view_rows = {}
    for name in VIEW_NAMES:
        frame = frames[name]
        image = image_matrix[np.array([image_row[i] for i in frame["imageId"]])]
        question = question_matrix[np.array([qid_row[q] for q in frame["questionId"]])]
        label = frame["label"].to_numpy(dtype="int64")
        write_view(EMB_DIR / f"{name}.h5", image, question, label)
        view_rows[name] = int(image.shape[0])
        print(f"view {name}.h5: {image.shape[0]} rows")

    # Verification.
    print("--- verification ---")
    ok = True
    ok &= check("trainable parameters", utils.count_parameters(model) == 0,
                f"{utils.count_parameters(model)} (expected 0)")
    for name in VIEW_NAMES:
        ok &= check(f"{name} view rows match manifest",
                    view_rows[name] == manifest_rows[name],
                    f"{view_rows[name]} vs {manifest_rows[name]}")
        with h5py.File(EMB_DIR / f"{name}.h5", "r") as store:
            labels = store["label"][:]
        ok &= check(f"{name} view labels match manifest",
                    bool(np.array_equal(
                        labels, frames[name]["label"].to_numpy("int64"))),
                    "exact")
    for name, frame in frames.items():
        missing_img = set(frame["imageId"]) - set(unique_images)
        missing_qid = set(frame["questionId"]) - set(unique_qids)
        ok &= check(f"{name} ids present in stores",
                    not missing_img and not missing_qid,
                    f"missing images {len(missing_img)}, "
                    f"missing questionIds {len(missing_qid)}")
    for label_name, matrix in [("images", image_matrix),
                               ("questions", question_matrix),
                               ("answers raw", raw), ("answers photo", photo),
                               ("answers ensembled", ensembled)]:
        norms = np.linalg.norm(matrix, axis=1)
        finite = bool(np.isfinite(matrix).all())
        unit = bool(np.abs(norms - 1.0).max() < 1e-3)
        ok &= check(f"{label_name} unit norms and finite", finite and unit,
                    f"norm range [{norms.min():.6f}, {norms.max():.6f}], "
                    f"finite {finite}, shape {matrix.shape}")

    # Spot re-encode: recompute a few embeddings individually and compare.
    with torch.no_grad():
        picks = [0, len(unique_images) // 4, len(unique_images) // 2,
                 3 * len(unique_images) // 4, len(unique_images) - 1]
        image_diff = 0.0
        for index in picks:
            path = config.GQA_IMAGES_DIR / f"{unique_images[index]}.jpg"
            single = normalize(model.encode_image(
                preprocess(Image.open(path).convert("RGB"))
                .unsqueeze(0).to(device)))[0]
            image_diff = max(image_diff,
                             float(np.abs(single - image_matrix[index]).max()))
        picks = [0, len(unique_qids) // 2, len(unique_qids) - 1]
        text_diff = 0.0
        for index in picks:
            single = normalize(model.encode_text(
                tokenizer([question_text[unique_qids[index]]]).to(device)))[0]
            text_diff = max(text_diff,
                            float(np.abs(single - question_matrix[index]).max()))
    ok &= check("spot re-encode agreement",
                image_diff < 1e-3 and text_diff < 1e-3,
                f"max abs diff images {image_diff:.2e}, texts {text_diff:.2e}")

    summary = {
        "manifest_rows": manifest_rows,
        "n_unique_images": len(unique_images),
        "n_unique_question_ids": len(unique_qids),
        "n_unique_question_texts": len(unique_texts),
        "n_answers": len(answers),
        "views": view_rows,
        "missing_images": 0,
        "trainable_parameters": 0,
        "encode_seconds": encode_seconds,
        "verification_passed": bool(ok),
        "test_clean_targets_read": False,
    }
    metadata = utils.run_metadata()
    metadata["v2_01_extraction"] = summary
    utils.save_json(metadata, OUT_RESULTS_DIR / "extraction.json")
    print(f"extraction summary: {json.dumps(summary)}")
    if not ok:
        sys.exit("extraction verification FAILED")
    print("Extraction complete and verified.")


if __name__ == "__main__":
    main()
