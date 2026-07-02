"""Stage 2: run the frozen CLIP encoder once and cache the vectors to disk.

This stage loads the CLIP model named in config (frozen, never trained), then
encodes every image and every question in the Stage 1 subset into fixed vectors
of size config.EMBED_DIM. The image and question vectors, together with the
answer-class labels, are written to embeddings/ as arrays on disk.

Encoding happens once here so that the training stages read precomputed vectors
instead of running CLIP repeatedly. This is the main efficiency idea of the
project and it keeps every later stage cheap.

Within a split each distinct image and each distinct question is encoded only
once; the per-example arrays are then assembled by lookup, in the row order of
train.csv / val.csv. Rows whose image file is missing are skipped and counted.
"""

import zipfile
from urllib.request import urlretrieve

import h5py
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import config
from src import utils


def _download_with_progress(url: str, dest) -> None:
    """Download url to dest, showing a progress bar (as in ensure_questions)."""
    with tqdm(unit="B", unit_scale=True, miniters=1, desc=dest.name) as bar:
        def hook(block_count, block_size, total_size):
            if total_size > 0:
                bar.total = total_size
            bar.update(block_count * block_size - bar.n)

        urlretrieve(url, dest, reporthook=hook)


def ensure_images() -> None:
    """Download and extract the GQA image archive if it is not already present.

    Mirrors ensure_questions() in Stage 1: skip if the extracted directory
    already holds files, otherwise download the archive to data/gqa (reusing it
    if it is already there) and extract it.
    """
    images_dir = config.GQA_IMAGES_DIR
    if images_dir.exists() and any(images_dir.iterdir()):
        print(f"Found extracted images in {images_dir}")
        return

    parent = images_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    zip_path = parent / "images.zip"
    if not zip_path.exists():
        print(f"Downloading GQA images (about 20 GB) from {config.GQA_IMAGES_URL}")
        _download_with_progress(config.GQA_IMAGES_URL, zip_path)

    print(f"Extracting image archive into {parent} (this takes a few minutes)")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(parent)
    print(f"Images ready in {images_dir}")


def load_clip(device):
    """Load the frozen CLIP model, its eval transform and its tokenizer.

    The model is put in eval mode, all parameters have requires_grad set to
    False, and the freeze is verified by asserting it has zero trainable
    parameters.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED)
    tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert utils.count_parameters(model) == 0, "CLIP must be frozen"
    return model, preprocess, tokenizer


def _maybe_normalize(features: torch.Tensor) -> torch.Tensor:
    if config.NORMALIZE_EMBEDDINGS:
        features = features / features.norm(dim=-1, keepdim=True)
    return features


@torch.no_grad()
def encode_images(model, preprocess, device, image_ids):
    """Encode each unique image once. Returns (id -> vector, set of missing ids).

    An image whose file is not on disk is recorded as missing and skipped.
    """
    vectors = {}
    missing = set()
    image_ids = list(image_ids)
    for start in tqdm(range(0, len(image_ids), config.BATCH_SIZE),
                      desc="images"):
        batch_ids = image_ids[start:start + config.BATCH_SIZE]
        tensors, kept = [], []
        for image_id in batch_ids:
            path = config.GQA_IMAGES_DIR / f"{image_id}.jpg"
            if not path.exists():
                missing.add(image_id)
                continue
            image = Image.open(path).convert("RGB")
            tensors.append(preprocess(image))
            kept.append(image_id)
        if not tensors:
            continue
        features = _maybe_normalize(model.encode_image(torch.stack(tensors).to(device)))
        features = features.float().cpu().numpy()
        for image_id, vector in zip(kept, features):
            vectors[image_id] = vector
    return vectors, missing


@torch.no_grad()
def encode_questions(model, tokenizer, device, questions):
    """Encode each unique question once. Returns question -> vector."""
    questions = list(questions)
    vectors = {}
    for start in tqdm(range(0, len(questions), config.BATCH_SIZE),
                      desc="questions"):
        batch = questions[start:start + config.BATCH_SIZE]
        tokens = tokenizer(batch).to(device)
        features = _maybe_normalize(model.encode_text(tokens))
        features = features.float().cpu().numpy()
        for text, vector in zip(batch, features):
            vectors[text] = vector
    return vectors


def assemble(frame, image_vectors, missing, question_vectors):
    """Build row-aligned image, question and label arrays, skipping missing images."""
    images, questions, labels = [], [], []
    skipped = 0
    for row in frame.itertuples(index=False):
        if row.imageId in missing or row.imageId not in image_vectors:
            skipped += 1
            continue
        images.append(image_vectors[row.imageId])
        questions.append(question_vectors[row.question])
        labels.append(row.label)
    image = np.stack(images).astype("float32")
    question = np.stack(questions).astype("float32")
    label = np.asarray(labels, dtype="int64")
    return image, question, label, skipped


def save_embeddings(path, image, question, label) -> None:
    """Write the three arrays to an HDF5 file, one dataset each."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as store:
        store.create_dataset("image", data=image)
        store.create_dataset("question", data=question)
        store.create_dataset("label", data=label)
        store.attrs["clip_model"] = config.CLIP_MODEL_NAME
        store.attrs["clip_pretrained"] = config.CLIP_PRETRAINED
        store.attrs["embed_dim"] = config.EMBED_DIM
        store.attrs["normalized"] = config.NORMALIZE_EMBEDDINGS


def process_split(name, csv_path, out_path, model, preprocess, tokenizer, device):
    """Encode one split and cache it, returning the counts for the metadata."""
    frame = pd.read_csv(csv_path, dtype={"imageId": str, "questionId": str})
    unique_images = sorted(set(frame["imageId"]))
    unique_questions = sorted(set(frame["question"]))
    print(f"[{name}] {len(frame)} rows, {len(unique_images)} unique images, "
          f"{len(unique_questions)} unique questions")

    image_vectors, missing = encode_images(model, preprocess, device, unique_images)
    question_vectors = encode_questions(model, tokenizer, device, unique_questions)

    image, question, label, skipped = assemble(
        frame, image_vectors, missing, question_vectors)
    save_embeddings(out_path, image, question, label)
    print(f"[{name}] wrote {image.shape[0]} rows to {out_path.name} "
          f"(skipped {skipped}, missing images {len(missing)})")

    return {
        "rows_written": int(image.shape[0]),
        "images_encoded": len(image_vectors),
        "questions_encoded": len(question_vectors),
        "rows_skipped": int(skipped),
        "images_missing": len(missing),
        "embed_dim": int(image.shape[1]),
    }


def main() -> None:
    utils.set_seed()
    device = utils.get_device()
    ensure_images()

    print(f"Loading CLIP {config.CLIP_MODEL_NAME} ({config.CLIP_PRETRAINED}) "
          f"on {device}")
    model, preprocess, tokenizer = load_clip(device)

    splits = [
        ("train", config.TRAIN_SPLIT_PATH, config.TRAIN_EMB_PATH),
        ("val", config.VAL_SPLIT_PATH, config.VAL_EMB_PATH),
    ]
    counts = {}
    for name, csv_path, out_path in splits:
        counts[name] = process_split(
            name, csv_path, out_path, model, preprocess, tokenizer, device)

    metadata = utils.run_metadata()
    metadata["stage2"] = counts
    utils.save_json(metadata, config.RESULTS_DIR / "stage2_extract_embeddings.json")
    print("Stage 2 complete.")


if __name__ == "__main__":
    main()
