"""Token-level dataset and loaders for V3 experiments.

Reads the v3_00 token stores (data/v3/tokens/image_tokens.h5 and
question_tokens.h5) fully into memory as fp16 arrays once, shared between
datasets, and serves fp32 tensors per example.

Padding-mask contract (T6): collate_tokens returns key_padding_mask in the
PyTorch nn.MultiheadAttention convention: True marks a PADDED position that
attention must IGNORE; False marks a valid token. A unit test in
experiments/v3_00_tokens/loader_bench.py verifies that padded positions do
not affect attention output.

Label isolation (T5): labels are explicitly optional. A dataset built with
with_labels=True requires a labelled train/dev manifest and reads only that
manifest. test_clean_targets.csv is never opened by this module; unlabelled
use (the clean-test inputs at final evaluation) passes with_labels=False and
receives -1 placeholders.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import config
from src import utils

TOKEN_DIR = config.DATA_DIR / "v3" / "tokens"


class TokenStores:
    """The image and question token stores, loaded into memory once."""

    def __init__(self, image_tokens_path=None, question_tokens_path=None):
        image_tokens_path = Path(image_tokens_path
                                 or TOKEN_DIR / "image_tokens.h5")
        question_tokens_path = Path(question_tokens_path
                                    or TOKEN_DIR / "question_tokens.h5")
        with h5py.File(image_tokens_path, "r") as store:
            ids = [i.decode("utf-8") for i in store["ids"][:]]
            self.image_tokens = store["tokens"][:]  # (N, 50, 512) fp16
        self.image_row = {image_id: index for index, image_id in enumerate(ids)}
        with h5py.File(question_tokens_path, "r") as store:
            ids = [i.decode("utf-8") for i in store["ids"][:]]
            self.question_tokens = store["tokens"][:]  # (T, 512) fp16 packed
            self._offsets = store["offsets"][:]
            self._lengths = store["lengths"][:]
        self.question_index = {qid: index for index, qid in enumerate(ids)}

    def question_span(self, qid: str):
        index = self.question_index[qid]
        return int(self._offsets[index]), int(self._lengths[index])


class TokenDataset(Dataset):
    """Rows of a V2 manifest served as token tensors.

    Returns (image_tokens (50, 512) fp32, question_tokens (L, 512) fp32,
    length, label); label is -1 when with_labels is False.
    """

    def __init__(self, manifest_path, stores: TokenStores,
                 with_labels: bool = True):
        frame = pd.read_csv(manifest_path,
                            dtype={"questionId": str, "imageId": str},
                            keep_default_na=False)
        if with_labels and "label" not in frame.columns:
            raise ValueError(f"{manifest_path} has no label column; pass "
                             "with_labels=False for unlabelled manifests")
        self.stores = stores
        self.image_rows = np.array([stores.image_row[i]
                                    for i in frame["imageId"]])
        spans = [stores.question_span(q) for q in frame["questionId"]]
        self.question_offsets = np.array([s[0] for s in spans], dtype="int64")
        self.question_lengths = np.array([s[1] for s in spans], dtype="int64")
        self.labels = (frame["label"].to_numpy("int64") if with_labels
                       else None)

    def __len__(self):
        return len(self.image_rows)

    def __getitem__(self, index):
        image = torch.from_numpy(
            self.stores.image_tokens[self.image_rows[index]]
            .astype(np.float32))
        offset = self.question_offsets[index]
        length = int(self.question_lengths[index])
        question = torch.from_numpy(
            self.stores.question_tokens[offset:offset + length]
            .astype(np.float32))
        label = int(self.labels[index]) if self.labels is not None else -1
        return image, question, length, label


def collate_tokens(batch):
    """Pad question tokens to the batch maximum.

    Returns (images (B, 50, 512), questions (B, L_max, 512), lengths (B,),
    key_padding_mask (B, L_max) bool, labels (B,)). Mask convention: True
    marks a padded position to IGNORE (the PyTorch key_padding_mask
    convention); False marks a valid token.
    """
    images = torch.stack([item[0] for item in batch])
    lengths = torch.tensor([item[2] for item in batch], dtype=torch.long)
    max_length = int(lengths.max())
    questions = torch.zeros(len(batch), max_length, images.shape[-1])
    key_padding_mask = torch.ones(len(batch), max_length, dtype=torch.bool)
    for row, item in enumerate(batch):
        questions[row, :item[2]] = item[1]
        key_padding_mask[row, :item[2]] = False
    labels = torch.tensor([item[3] for item in batch], dtype=torch.long)
    return images, questions, lengths, key_padding_mask, labels


def make_token_loaders(train_manifest, dev_manifest, stores=None,
                       batch_size: int = 128):
    """Train and dev token loaders mirroring the V2 conventions: seeded
    shuffle on train only, seed_worker, num_workers=0 first, pinned memory
    on CUDA."""
    stores = stores if stores is not None else TokenStores()
    train_dataset = TokenDataset(train_manifest, stores, with_labels=True)
    dev_dataset = TokenDataset(dev_manifest, stores, with_labels=True)
    pin_memory = config.DEVICE == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        generator=utils.make_generator(), worker_init_fn=utils.seed_worker,
        num_workers=0, pin_memory=pin_memory, collate_fn=collate_tokens)
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=pin_memory, collate_fn=collate_tokens)
    return train_loader, dev_loader
